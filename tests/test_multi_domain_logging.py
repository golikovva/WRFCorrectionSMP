from pathlib import Path
from types import SimpleNamespace

import pytest

from lib.data.logger import WRFLogger


def _allocate_pipeline_run_dir(*args, **kwargs):
    pytest.importorskip("addict")
    from experiments.multi_domain.main import _allocate_pipeline_run_dir as allocate

    return allocate(*args, **kwargs)


def _resolve_init_weights(*args, **kwargs):
    pytest.importorskip("addict")
    from experiments.multi_domain.main import _resolve_init_weights as resolve

    return resolve(*args, **kwargs)


def _resolve_pipeline_run_dir(*args, **kwargs):
    pytest.importorskip("addict")
    from experiments.multi_domain.main import _resolve_pipeline_run_dir as resolve

    return resolve(*args, **kwargs)


def _completed_stage_results(*args, **kwargs):
    pytest.importorskip("addict")
    from experiments.multi_domain.main import _completed_stage_results as completed

    return completed(*args, **kwargs)


def _resolve_stage_selector(*args, **kwargs):
    pytest.importorskip("addict")
    from experiments.multi_domain.main import _resolve_stage_selector as resolve

    return resolve(*args, **kwargs)


def _validate_stage_rerun_options(*args, **kwargs):
    pytest.importorskip("addict")
    from experiments.multi_domain.main import _validate_stage_rerun_options as validate

    return validate(*args, **kwargs)


def _resolve_resume_training(*args, **kwargs):
    pytest.importorskip("addict")
    from experiments.multi_domain.main import _resolve_resume_training as resolve

    return resolve(*args, **kwargs)


def _prepare_resume_stage_results(*args, **kwargs):
    pytest.importorskip("addict")
    from experiments.multi_domain.main import _prepare_resume_stage_results as prepare

    return prepare(*args, **kwargs)


def _record_stage_result(*args, **kwargs):
    pytest.importorskip("addict")
    from experiments.multi_domain.main import _record_stage_result as record

    return record(*args, **kwargs)


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


def test_resolve_init_weights_accepts_previous_stage_name():
    finished = {"pretrain_borey": {"best_model_path": "/home/logs/run/models/model_5.pth"}}

    resolved = _resolve_init_weights(
        {"name": "finetune_nestp", "init_weights_from": "pretrain_borey"},
        finished,
    )

    assert resolved == "/home/logs/run/models/model_5.pth"


def test_resolve_init_weights_accepts_direct_checkpoint_path():
    path = "/home/logs/multi_domain_experiment__01_finetune_nestp/misc_1/models/model_5.pth"

    resolved = _resolve_init_weights(
        {"name": "test_nestp", "init_weights_from": path},
        {},
    )

    assert resolved == path


def test_resolve_pipeline_run_dir_resumes_named_experiment(tmp_path):
    run_dir = tmp_path / "RoPEUNet" / "zero_shot"
    run_dir.mkdir(parents=True)

    resolved = Path(
        _resolve_pipeline_run_dir(
            tmp_path,
            "RoPEUNet",
            "zero shot",
            resume=True,
        )
    )

    assert resolved == run_dir


def test_resolve_pipeline_run_dir_requires_dir_for_anonymous_resume(tmp_path):
    with pytest.raises(ValueError, match="--resume-dir"):
        _resolve_pipeline_run_dir(tmp_path, "RoPEUNet", resume=True)


def test_completed_stage_results_accepts_matching_prefix():
    summary = {
        "stages": [
            {"stage_name": "pretrain_borey", "best_model_path": "/home/logs/model_1.pth"},
        ],
    }
    configured = [
        {"name": "pretrain_borey"},
        {"name": "finetune_nestp"},
    ]

    completed = _completed_stage_results(summary, configured)

    assert completed == summary["stages"]


def test_completed_stage_results_rejects_mismatched_prefix():
    summary = {
        "stages": [
            {"stage_name": "test_nestp", "best_model_path": "/home/logs/model_1.pth"},
        ],
    }
    configured = [
        {"name": "pretrain_borey"},
        {"name": "test_nestp"},
    ]

    with pytest.raises(ValueError, match="current config expects"):
        _completed_stage_results(summary, configured)


def _stages():
    return [
        {"name": "pretrain_borey"},
        {"name": "test_nestp_zeroshot"},
        {"name": "test_nestp_finetune"},
    ]


def _summary():
    return {
        "stages": [
            {
                "stage_index": 0,
                "stage_name": "pretrain_borey",
                "best_model_path": "/home/logs/model_0.pth",
            },
            {
                "stage_index": 1,
                "stage_name": "test_nestp_zeroshot",
                "best_model_path": "/home/logs/model_1.pth",
            },
            {
                "stage_index": 2,
                "stage_name": "test_nestp_finetune",
                "best_model_path": "/home/logs/model_2.pth",
            },
        ],
    }


def test_resolve_stage_selector_accepts_zero_based_index_and_name():
    assert _resolve_stage_selector("1", _stages()) == 1
    assert _resolve_stage_selector("test_nestp_finetune", _stages()) == 2


def test_resolve_stage_selector_rejects_invalid_selector():
    with pytest.raises(ValueError, match="Unknown stage selector"):
        _resolve_stage_selector("missing_stage", _stages())

    with pytest.raises(ValueError, match="out of range"):
        _resolve_stage_selector("3", _stages())


def test_validate_stage_rerun_options_rejects_only_and_from_together():
    with pytest.raises(ValueError, match="either --only-stage or --from-stage"):
        _validate_stage_rerun_options(only_stage="1", from_stage="2")


def test_resolve_resume_training_defaults_to_pipeline_resume_and_allows_override():
    assert _resolve_resume_training(resume=True) is True
    assert _resolve_resume_training(resume=False) is False
    assert _resolve_resume_training(resume=True, resume_training=False) is False


def test_prepare_resume_stage_results_trims_summary_for_from_stage():
    summary_stages, dependency_stages = _prepare_resume_stage_results(
        _summary(),
        _stages(),
        from_stage_idx=1,
    )

    assert [stage["stage_name"] for stage in summary_stages] == ["pretrain_borey"]
    assert dependency_stages == summary_stages


def test_prepare_resume_stage_results_preserves_dependency_context_for_only_stage():
    summary_stages, dependency_stages = _prepare_resume_stage_results(
        _summary(),
        _stages(),
        only_stage_idx=2,
    )

    assert [stage["stage_name"] for stage in summary_stages] == [
        "pretrain_borey",
        "test_nestp_zeroshot",
        "test_nestp_finetune",
    ]
    assert [stage["stage_name"] for stage in dependency_stages] == [
        "pretrain_borey",
        "test_nestp_zeroshot",
    ]


def test_prepare_resume_stage_results_requires_completed_prefix_for_selected_stage():
    short_summary = {"stages": _summary()["stages"][:1]}

    with pytest.raises(ValueError, match="contains only 1 completed"):
        _prepare_resume_stage_results(
            short_summary,
            _stages(),
            only_stage_idx=2,
        )


def test_record_stage_result_replaces_only_selected_stage_result():
    summary = _summary()
    replacement = {
        "stage_index": 1,
        "stage_name": "test_nestp_zeroshot",
        "best_model_path": "/home/logs/replacement.pth",
    }

    _record_stage_result(summary, 1, replacement)

    assert summary["stages"][0]["best_model_path"] == "/home/logs/model_0.pth"
    assert summary["stages"][1] == replacement
    assert summary["stages"][2]["best_model_path"] == "/home/logs/model_2.pth"
