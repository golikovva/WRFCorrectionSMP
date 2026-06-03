from pathlib import Path
from types import SimpleNamespace

import pytest

from lib.data.logger import WRFLogger


def _allocate_pipeline_run_dir(*args, **kwargs):
    pytest.importorskip("addict")
    from experiments.multi_domain.main import _allocate_pipeline_run_dir as allocate

    return allocate(*args, **kwargs)


def _cfg(run_mode="train"):
    return SimpleNamespace(
        run_config=SimpleNamespace(run_mode=run_mode),
        test_config=SimpleNamespace(run_id=3),
    )


def test_allocate_named_pipeline_run_dir(tmp_path):
    run_dir = Path(_allocate_pipeline_run_dir(tmp_path, "RoPEUNet", "zero shot/borey"))

    assert run_dir == tmp_path / "RoPEUNet" / "zero_shot_borey"
    assert run_dir.is_dir()


def test_allocate_anonymous_pipeline_run_dir_uses_next_misc(tmp_path):
    model_dir = tmp_path / "RoPEUNet"
    (model_dir / "misc_1").mkdir(parents=True)
    (model_dir / "misc_3").mkdir()
    (model_dir / "notes").mkdir()

    run_dir = Path(_allocate_pipeline_run_dir(tmp_path, "RoPEUNet"))

    assert run_dir == model_dir / "misc_4"
    assert run_dir.is_dir()


def test_wrf_logger_accepts_explicit_stage_save_dir(tmp_path):
    stage_dir = tmp_path / "RoPEUNet" / "experiment" / "stage_00_pretrain_borey"

    logger = WRFLogger(_cfg(), save_dir=stage_dir)

    assert Path(logger.save_dir) == stage_dir
    assert Path(logger.model_save_dir) == stage_dir / "models"
    assert Path(logger.log_dir) == stage_dir / "logs"
    assert Path(logger.plots_dir) == stage_dir / "plots"
    assert Path(logger.model_save_dir).is_dir()
    assert Path(logger.log_dir).is_dir()
    assert Path(logger.plots_dir).is_dir()
