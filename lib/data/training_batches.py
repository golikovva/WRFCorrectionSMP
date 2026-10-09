"""Strict train batches and a common failure boundary before DDP forward."""

from collections.abc import Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate

from lib.distributed import distributed_is_initialized, rank, world_size


def _date_context(dates):
    try:
        return ", ".join(str(date) for date in np.asarray(dates).reshape(-1))
    except Exception:
        return repr(dates)


def _collate_values(values, path, *, missing_as_nan=False):
    """Pad variable time axes without ever dropping a sample."""
    first = values[0]
    if any(value is None for value in values):
        if all(value is None for value in values):
            return None
        raise ValueError(f"{path}: only part of the batch is None")
    if isinstance(first, np.datetime64):
        return np.asarray(values)
    if isinstance(first, np.ndarray):
        return _collate_values([torch.as_tensor(value) for value in values], path, missing_as_nan=missing_as_nan)
    if isinstance(first, torch.Tensor):
        if first.ndim == 0:
            return default_collate(values)
        # Scatter represents a date with no observations by shape (0,).
        # Match its trailing dimensions to observed dates before time padding.
        exemplar = next((value for value in values if value.shape[0] > 0), first)
        normalized = [
            value.new_empty((0, *exemplar.shape[1:]))
            if value.shape[0] == 0 and value.ndim == 1 and exemplar.ndim > 1
            else value
            for value in values
        ]
        if any(value.shape[1:] != exemplar.shape[1:] for value in normalized):
            raise ValueError(f"{path}: inconsistent non-temporal dimensions")
        # Missing scatter observations must not become artificial zero-wind
        # targets. The observation loss already masks non-finite values.
        padding = float("nan") if missing_as_nan and exemplar.is_floating_point() else 0.0
        return pad_sequence(normalized, batch_first=True, padding_value=padding)
    if isinstance(first, Mapping):
        keys = set(first)
        if any(not isinstance(value, Mapping) or set(value) != keys for value in values):
            raise ValueError(f"{path}: inconsistent dictionary keys")
        return {
            key: _collate_values(
                [value[key] for value in values],
                f"{path}.{key}",
                missing_as_nan=missing_as_nan or key == "Scatter",
            )
            for key in first
        }
    if isinstance(first, (tuple, list)):
        if any(not isinstance(value, (tuple, list)) or len(value) != len(first) for value in values):
            raise ValueError(f"{path}: inconsistent tuple lengths")
        return [
            _collate_values([value[index] for value in values], f"{path}[{index}]", missing_as_nan=missing_as_nan)
            for index in range(len(first))
        ]
    return default_collate(values)


def strict_train_collate(batch, *, required_sources, sequence_length, expected_batch_size):
    """Collate (source dictionary, date) samples, rejecting incomplete sequences."""
    if len(batch) != expected_batch_size:
        dates = [sample[1] for sample in batch if isinstance(sample, (tuple, list)) and len(sample) == 2]
        raise ValueError(
            f"Training batch has {len(batch)} samples, expected {expected_batch_size}; "
            f"dates={_date_context(dates)}"
        )
    for position, sample in enumerate(batch):
        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            raise ValueError(f"Training sample {position} is not (data, date)")
        data, date = sample
        if not isinstance(data, Mapping):
            raise ValueError(f"date={date}: training data must be a source dictionary")
        for source in required_sources:
            value = data.get(source)
            if value is None:
                raise ValueError(f"date={date}: required source {source!r} is missing")
            if not isinstance(value, (np.ndarray, torch.Tensor)) or value.ndim == 0:
                raise ValueError(f"date={date}: required source {source!r} is not a sequence")
            if sequence_length is not None and value.shape[0] != sequence_length:
                raise ValueError(
                    f"date={date}: source {source!r} has {value.shape[0]} time steps, "
                    f"expected {sequence_length}"
                )
    try:
        result = _collate_values(batch, "batch")
        validate_train_batch(
            result,
            required_sources=required_sources,
            sequence_length=sequence_length,
            expected_batch_size=expected_batch_size,
        )
        return result
    except Exception as exc:
        dates = [sample[1] for sample in batch]
        raise ValueError(f"dates={_date_context(dates)}: {exc}") from exc


def validate_train_batch(batch, *, required_sources, sequence_length, expected_batch_size):
    """Check collated batch size and mandatory sources before the common DDP gate."""
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError("Training batch is not (data, dates)")
    data, dates = batch
    context = f"dates={_date_context(dates)}"
    if not isinstance(data, Mapping):
        raise ValueError(f"{context}: training data must be a source dictionary")
    date_array = np.asarray(dates)
    if date_array.ndim != 1 or len(date_array) != expected_batch_size:
        raise ValueError(f"{context}: expected {expected_batch_size} dates")
    if not np.issubdtype(date_array.dtype, np.datetime64) or np.isnat(date_array).any():
        raise ValueError(f"{context}: training dates must be valid datetime64 values")
    for source in required_sources:
        value = data.get(source)
        if value is None:
            raise ValueError(f"{context}: required source {source!r} is missing")
        if not isinstance(value, torch.Tensor) or value.ndim < 2:
            raise ValueError(f"{context}: required source {source!r} is not a batched sequence")
        if value.shape[0] != expected_batch_size:
            raise ValueError(
                f"{context}: source {source!r} has {value.shape[0]} samples, "
                f"expected {expected_batch_size}"
            )
        if sequence_length is not None and value.shape[1] != sequence_length:
            raise ValueError(
                f"{context}: source {source!r} has {value.shape[1]} time steps, "
                f"expected {sequence_length}"
            )


def _raise_if_failed(error, device, stage):
    message = f"rank {rank()}, {stage}: {error}" if error is not None else None
    if not distributed_is_initialized():
        if message is not None:
            raise RuntimeError(f"Training data failure: {message}")
        return
    collective_device = device if dist.get_backend() == "nccl" else torch.device("cpu")
    failed = torch.tensor(int(error is not None), dtype=torch.int32, device=collective_device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        messages = [None] * world_size()
        dist.all_gather_object(messages, message)
        details = "\n".join(value for value in messages if value is not None)
        raise RuntimeError(f"Training data failure before forward:\n{details}")


def guarded_train_batches(
    dataloader,
    device,
    *,
    required_sources=None,
    sequence_length=None,
    expected_batch_size=None,
    validator=None,
):
    """Yield only batches every rank obtained and validated successfully.

    All ranks must iterate this generator to exhaustion. A validator may be
    supplied for legacy batch formats; it must raise on invalid local data.
    Unrecoverable process death and indefinitely blocked I/O remain subject
    to the DataLoader/process-group timeout.
    """
    error = None
    iterator = None
    steps = None
    try:
        steps = len(dataloader)
        if steps <= 0:
            raise ValueError("No full training batches are available")
        if validator is None:
            if required_sources is None:
                raise ValueError("required_sources or a batch validator is required")
            if expected_batch_size is None:
                expected_batch_size = dataloader.batch_size
            if expected_batch_size is None or expected_batch_size <= 0:
                raise ValueError("A positive expected_batch_size is required")
        iterator = iter(dataloader)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _raise_if_failed(error, device, "loader initialization")
    if distributed_is_initialized():
        counts = [None] * world_size()
        dist.all_gather_object(counts, steps)
        if len(set(counts)) != 1:
            raise RuntimeError(f"Training loaders have different batch counts by rank: {counts}")

    for step in range(steps):
        batch, error = None, None
        try:
            batch = next(iterator)
            if validator is not None:
                validator(batch)
            else:
                validate_train_batch(
                    batch,
                    required_sources=required_sources,
                    sequence_length=sequence_length,
                    expected_batch_size=expected_batch_size,
                )
        except StopIteration:
            error = f"Loader exhausted early; expected {steps} batches"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _raise_if_failed(error, device, f"batch {step + 1}/{steps}")
        yield batch

    error = None
    try:
        next(iterator)
        error = f"Loader produced more than the advertised {steps} batches"
    except StopIteration:
        pass
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _raise_if_failed(error, device, "loader exhaustion")
