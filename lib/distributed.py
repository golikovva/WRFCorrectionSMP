import os
import time
import copy

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def distributed_is_initialized():
    return dist.is_available() and dist.is_initialized()


def setup_distributed():
    """Initialize torchrun's process group, or keep the regular single-process mode."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False

    if not distributed_is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        if local_rank() >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank()} but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
        torch.cuda.set_device(local_rank())
    return True


def world_size():
    return dist.get_world_size() if distributed_is_initialized() else 1


def rank():
    return dist.get_rank() if distributed_is_initialized() else 0


def local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process():
    return rank() == 0


def local_device(configured_device=None):
    if distributed_is_initialized() and torch.cuda.is_available():
        return torch.device("cuda", local_rank())
    if configured_device is None:
        configured_device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(configured_device)


def barrier():
    if distributed_is_initialized():
        dist.barrier()


def prepare_file_signal(path):
    """Remove a stale signal on rank 0, then synchronize before a long rank-0 task."""
    if is_main_process() and os.path.exists(path):
        os.remove(path)
    barrier()


def signal_file(path):
    """Atomically publish completion through a shared filesystem."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write("complete\n")
    os.replace(tmp_path, path)


def wait_for_file_signal(path, poll_interval=5.0):
    """Wait without an NCCL collective, so the process-group timeout is irrelevant."""
    while not os.path.exists(path):
        time.sleep(poll_interval)


def broadcast_object(value, src=0):
    if not distributed_is_initialized():
        return value
    values = [value]
    dist.broadcast_object_list(values, src=src)
    return values[0]


def reduce_mean(value, device):
    if not distributed_is_initialized():
        return float(value)
    tensor = torch.as_tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size()
    return tensor.item()


def reduce_sum(values, device):
    """Sum a short sequence of numeric values over all ranks."""
    tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
    if distributed_is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def wrap_model(model, device):
    if not distributed_is_initialized():
        return model
    kwargs = {}
    if device.type == "cuda":
        kwargs = {"device_ids": [device.index], "output_device": device.index}
    return DistributedDataParallel(model, **kwargs)


def unwrap_model(model):
    """Return the original module beneath DDP and torch.compile wrappers."""
    seen = set()
    while id(model) not in seen:
        seen.add(id(model))
        if isinstance(model, DistributedDataParallel):
            model = model.module
            continue
        original_model = getattr(model, "_orig_mod", None)
        if original_model is not None:
            model = original_model
            continue
        break
    return model


def _strip_checkpoint_wrapper_prefixes(state_dict):
    """Normalize checkpoints saved from any nesting of DDP/torch.compile."""
    prefixes = ("module.", "_orig_mod.")
    normalized = copy.copy(state_dict)
    while normalized:
        keys = list(normalized)
        prefix = next(
            (
                candidate
                for candidate in prefixes
                if all(
                    isinstance(key, str) and key.startswith(candidate)
                    for key in keys
                )
            ),
            None,
        )
        if prefix is None:
            break
        consume_prefix_in_state_dict_if_present(normalized, prefix)
    return normalized


def load_model_state_dict(model, state_dict, strict=True):
    """Load regular, DDP, torch.compile, or nested wrapper checkpoints."""
    target = unwrap_model(model)
    state_dict = _strip_checkpoint_wrapper_prefixes(state_dict)
    return target.load_state_dict(state_dict, strict=strict)
