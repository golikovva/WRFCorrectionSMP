"""Prepare one audited, shared list of valid training dates before DDP training."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np

from lib.data.availability import filter_valid_dates
from lib.distributed import broadcast_object, is_main_process, rank, world_size


_POLL_INTERVAL_SECONDS = 0.1


def _date_array(dates):
    dates = np.asarray(dates)
    if dates.size == 0 and dates.ndim == 1:
        if dates.dtype.kind != "M":
            dates = dates.astype("datetime64[ns]")
    if dates.ndim != 1 or dates.dtype.kind != "M":
        raise ValueError("Training dates must be a one-dimensional datetime64 array")
    if np.isnat(dates).any():
        raise ValueError("Training dates must not contain NaT")
    return dates


def _dates_digest(dates):
    digest = hashlib.sha256(dates.dtype.name.encode("ascii"))
    digest.update(dates.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)
    return hashlib.sha256(payload).hexdigest()


def _check_deadline(deadline, description):
    if time.monotonic() >= deadline:
        raise TimeoutError(f"Timed out while {description}")


def _wait_for_file(path, deadline):
    path = Path(path)
    while not path.is_file():
        _check_deadline(deadline, f"waiting for training-date preflight file {path.name}")
        time.sleep(min(_POLL_INTERVAL_SECONDS, max(0, deadline - time.monotonic())))
    _check_deadline(deadline, f"waiting for training-date preflight file {path.name}")


def _assert_ordered_subset(valid, candidates):
    if valid.dtype != candidates.dtype:
        raise ValueError("Valid training dates changed the candidate datetime resolution")
    position = 0
    for date in valid:
        while position < len(candidates) and candidates[position] != date:
            position += 1
        if position == len(candidates):
            raise ValueError("Valid training dates must preserve candidate membership and order")
        position += 1


def _build_manifest(token, candidates, valid, rejected, batch_size, replicas):
    valid = _date_array(valid)
    _assert_ordered_subset(valid, candidates)
    if len(valid) + len(rejected) != len(candidates):
        raise ValueError("Valid and rejected date counts do not match the candidate count")
    global_batch_size = batch_size * replicas
    steps = (len(valid) + global_batch_size - 1) // global_batch_size
    padded_count = steps * global_batch_size
    return {
        "schema_version": 1,
        "token": token,
        "datetime_dtype": valid.dtype.name,
        "candidate_count": len(candidates),
        "candidate_digest": _dates_digest(candidates),
        "valid_count": len(valid),
        "valid_digest": _dates_digest(valid),
        "valid_date_ticks": valid.astype("int64").tolist(),
        "valid_dates": [str(date) for date in valid],
        "rejected_count": len(rejected),
        "rejected": rejected,
        "rejection_counts": dict(Counter(str(item["reason"]) for item in rejected)),
        "batch_size_per_rank": batch_size,
        "world_size": replicas,
        "global_batch_size": global_batch_size,
        "padded_sample_count": padded_count,
        "repeated_sample_count": padded_count - len(valid),
        "samples_per_rank": steps * batch_size,
        "steps_per_rank": steps,
        "tail_policy": "repeat",
    }


def _read_manifest(path, status, token, candidates, batch_size, replicas):
    payload = Path(path).read_bytes()
    payload_digest = hashlib.sha256(payload).hexdigest()
    if payload_digest != status["manifest_digest"]:
        raise ValueError("Training-date manifest file digest mismatch")
    manifest = json.loads(payload)
    if manifest["schema_version"] != 1 or manifest["token"] != token:
        raise ValueError("Training-date manifest belongs to a different preflight run")
    if manifest["candidate_count"] != len(candidates) or manifest["candidate_digest"] != _dates_digest(candidates):
        raise ValueError("Ranks supplied different candidate training dates")
    if manifest["batch_size_per_rank"] != batch_size or manifest["world_size"] != replicas:
        raise ValueError("Ranks supplied different training batch sizes or world sizes")
    dtype = np.dtype(manifest["datetime_dtype"])
    if dtype.kind != "M":
        raise ValueError("Training-date manifest does not contain datetime64 dates")
    valid = np.asarray(manifest["valid_date_ticks"], dtype="int64").view(dtype)
    _assert_ordered_subset(valid, candidates)
    if manifest["valid_count"] != len(valid) or manifest["valid_digest"] != _dates_digest(valid):
        raise ValueError("Training-date manifest count or date digest mismatch")
    if manifest["rejected_count"] != len(manifest["rejected"]) or len(valid) + manifest["rejected_count"] != len(candidates):
        raise ValueError("Training-date manifest rejected count mismatch")
    if len(valid) == 0:
        raise ValueError("No valid training dates remain after the availability check")
    return valid, payload_digest


def prepare_valid_train_dates(dataset, dates, save_dir, *, batch_size, timeout_seconds=3600):
    """Scan on rank 0 and return an identical ordered date list on every rank.

    Each call creates a fresh audit directory; prior results are never reused.
    During the potentially long scan, peers wait on shared files, not NCCL.
    The deadline covers preparation, publication, and peer acknowledgements.
    A blocking dataset metadata read cannot be interrupted here; if it returns
    after the deadline, its result is rejected and peers have already timed out.
    """
    started_at = time.monotonic()
    start = None
    if is_main_process():
        try:
            if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size <= 0:
                raise ValueError("Training batch_size must be a positive integer")
            if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
                raise ValueError("Training-date timeout_seconds must be positive and finite")
            candidates = _date_array(dates)
            token = uuid.uuid4().hex
            run_dir = Path(save_dir) / "train_dates" / token
            run_dir.mkdir(parents=True, exist_ok=False)
            start = {"token": token, "run_dir": str(run_dir), "error": None}
        except Exception as exc:
            start = {"error": f"{type(exc).__name__}: {exc}"}
    start = broadcast_object(start, src=0)
    if start["error"] is not None:
        raise RuntimeError(f"Training-date preflight could not start: {start['error']}")

    deadline = started_at + timeout_seconds
    run_dir = Path(start["run_dir"])
    manifest_path = run_dir / "manifest.json"
    status_path = run_dir / "status.json"
    replicas = world_size()
    process_rank = rank()

    if is_main_process():
        try:
            _check_deadline(deadline, "starting the training-date scan")
            valid, rejected = filter_valid_dates(dataset, candidates)
            _check_deadline(deadline, "scanning training-date availability")
            manifest = _build_manifest(start["token"], candidates, valid, rejected, int(batch_size), replicas)
            manifest_digest = _atomic_json(manifest_path, manifest)
            summary = {key: value for key, value in manifest.items() if key not in {"valid_date_ticks", "valid_dates", "rejected"}}
            summary["manifest_path"] = str(manifest_path)
            _atomic_json(Path(save_dir) / "train_dates_summary.json", summary)
            _check_deadline(deadline, "publishing the training-date manifest")
            status = {"state": "complete", "token": start["token"], "manifest_digest": manifest_digest}
        except Exception as exc:
            status = {"state": "error", "token": start["token"], "error": f"{type(exc).__name__}: {exc}"}
        _atomic_json(status_path, status)

    _wait_for_file(status_path, deadline)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["token"] != start["token"]:
        raise RuntimeError("Training-date status belongs to a different preflight run")
    if status["state"] != "complete":
        raise RuntimeError(f"Training-date preflight failed: {status['error']}")

    valid = None
    try:
        candidates = _date_array(dates)
        valid, manifest_digest = _read_manifest(
            manifest_path, status, start["token"], candidates, batch_size, replicas,
        )
        agreement = {
            "rank": process_rank, "error": None, "valid_count": len(valid),
            "valid_digest": _dates_digest(valid), "manifest_digest": manifest_digest,
        }
    except Exception as exc:
        agreement = {"rank": process_rank, "error": f"{type(exc).__name__}: {exc}"}
    _atomic_json(run_dir / f"rank_{process_rank}_ready.json", agreement)

    # Do not enter a collective while another rank is still waiting for files.
    for other_rank in range(replicas):
        _wait_for_file(run_dir / f"rank_{other_rank}_ready.json", deadline)
    agreements = [broadcast_object(agreement if process_rank == source else None, src=source)
                  for source in range(replicas)]
    errors = [f"rank {item['rank']}: {item['error']}" for item in agreements if item["error"]]
    if errors:
        raise RuntimeError("Training-date preflight agreement failed: " + "; ".join(errors))
    signatures = {(item["valid_count"], item["valid_digest"], item["manifest_digest"]) for item in agreements}
    if len(signatures) != 1:
        raise RuntimeError("Training-date preflight agreement failed: rank date counts or digests differ")
    _check_deadline(deadline, "completing training-date preflight agreement")
    if is_main_process():
        print(
            f"Training dates: {len(valid)}/{len(candidates)} valid; "
            f"{manifest['repeated_sample_count']} repeated samples for full batches; "
            f"{manifest['steps_per_rank']} steps per rank. Audit: {manifest_path}",
            flush=True,
        )
    return valid
