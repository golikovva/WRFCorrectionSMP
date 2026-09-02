from types import SimpleNamespace
import importlib.machinery
import sys

import torch

from lib.distributed import load_model_state_dict, unwrap_model
from lib.pipeline.train import _restore_rng_state, load_training_state, save_training_state


def test_training_state_roundtrip_restores_state(tmp_path):
    if "xarray" in sys.modules and getattr(sys.modules["xarray"], "__spec__", None) is None:
        sys.modules["xarray"].__spec__ = importlib.machinery.ModuleSpec("xarray", loader=None)

    checkpoint_path = tmp_path / "training_state.pth"

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.5)
    logger = SimpleNamespace(train_loss=[1.25], loss_evolution=[0.75], best_epoch=0)

    x = torch.ones(1, 2)
    model(x).sum().backward()
    optimizer.step()
    scheduler.step()

    expected_weight = model.weight.detach().clone()
    save_training_state(checkpoint_path, model, optimizer, scheduler, logger, epoch=0)

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.1)
    restored_scheduler = torch.optim.lr_scheduler.MultiStepLR(restored_optimizer, milestones=[1], gamma=0.5)
    restored_logger = SimpleNamespace(train_loss=[], loss_evolution=[], best_epoch=-1)

    state = load_training_state(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        restored_scheduler,
        restored_logger,
        map_location="cpu",
    )

    assert state["next_epoch"] == 1
    assert restored_logger.train_loss == [1.25]
    assert restored_logger.loss_evolution == [0.75]
    assert restored_logger.best_epoch == 0
    assert torch.allclose(restored_model.weight, expected_weight)
    assert restored_scheduler.last_epoch == scheduler.last_epoch


def test_restore_rng_state_accepts_non_tensor_torch_state():
    rng_state = {
        "torch": torch.get_rng_state().tolist(),
    }

    _restore_rng_state(rng_state)


class _CompiledModelStub(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self._orig_mod = model


def test_load_model_state_dict_strips_compile_and_ddp_prefixes():
    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    prefixed_state = {
        f"module._orig_mod.{key}": value.clone()
        for key, value in source.state_dict().items()
    }

    load_model_state_dict(_CompiledModelStub(target), prefixed_state)

    assert torch.allclose(target.weight, source.weight)
    assert torch.allclose(target.bias, source.bias)


def test_unwrap_model_removes_compile_wrapper():
    model = torch.nn.Linear(2, 1)

    assert unwrap_model(_CompiledModelStub(model)) is model
