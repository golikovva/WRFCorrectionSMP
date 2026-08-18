import os
import sys
import copy
import yaml
import re
import argparse
import importlib.util
from collections.abc import Mapping

import torch

sys.path.insert(0, '/home')
sys.path.insert(0, '../../')

from lib.config.cfg import Config
from lib.distributed import (
    barrier,
    broadcast_object,
    is_main_process,
    setup_distributed,
)


_MISC_DIR_RE = re.compile(r'^misc_(\d+)$')


def _to_serializable(obj):
    try:
        import numpy as np
    except ImportError:
        np = None

    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()

    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()

    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]

    return obj


def _deep_update(dst, src):
    for key, value in src.items():
        if (
            isinstance(value, Mapping)
            and key in dst
            and isinstance(dst[key], Mapping)
        ):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _load_single_stage_main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        "/home/experiments/train_test/main.py",
        os.path.abspath(os.path.join(current_dir, "..", "train_test", "main.py")),
    ]

    for path in candidates:
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location("single_stage_train_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.main, path

    raise FileNotFoundError(
        "Could not find the single-stage trainer. Tried:\n"
        + "\n".join(candidates)
    )


def _sanitize_path_part(value):
    value = str(value).strip()
    value = re.sub(r'[<>:"/\\|?*\s]+', '_', value)
    value = re.sub(r'_+', '_', value).strip('._')
    return value or 'unnamed'


def _next_misc_name(parent_dir):
    if not os.path.isdir(parent_dir):
        return 'misc_1'

    numbers = []
    for name in os.listdir(parent_dir):
        match = _MISC_DIR_RE.match(name)
        if match is not None:
            numbers.append(int(match.group(1)))

    return f"misc_{max(numbers, default=0) + 1}"


def _allocate_pipeline_run_dir(logs_folder, model_name, experiment_name=None):
    model_dir = os.path.join(logs_folder, _sanitize_path_part(model_name))
    os.makedirs(model_dir, exist_ok=True)

    if experiment_name not in (None, ''):
        run_name = _sanitize_path_part(experiment_name)
    else:
        run_name = _next_misc_name(model_dir)

    run_dir = os.path.join(model_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _pipeline_run_dir(logs_folder, model_name, experiment_name=None):
    return os.path.join(
        logs_folder,
        _sanitize_path_part(model_name),
        _sanitize_path_part(experiment_name),
    )


def _resolve_pipeline_run_dir(
    logs_folder,
    model_name,
    experiment_name=None,
    *,
    resume=False,
    resume_dir=None,
):
    if not resume:
        return _allocate_pipeline_run_dir(logs_folder, model_name, experiment_name)

    if resume_dir not in (None, ''):
        run_dir = os.fspath(resume_dir)
    else:
        if experiment_name in (None, ''):
            raise ValueError("--resume requires --resume-dir when experiment_name is empty.")
        run_dir = _pipeline_run_dir(logs_folder, model_name, experiment_name)

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(
            f"Cannot resume: pipeline run directory does not exist: {run_dir}"
        )
    return run_dir


def _load_pipeline_summary(pipeline_dir):
    summary_path = os.path.join(pipeline_dir, "pipeline_summary.yaml")
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _completed_stage_results(summary, configured_stages):
    completed = []
    for stage_idx, stage_result in enumerate(summary.get('stages', [])):
        if stage_idx >= len(configured_stages):
            raise ValueError(
                "Cannot resume: pipeline_summary.yaml has more stages than the current config."
            )

        expected_name = configured_stages[stage_idx]['name']
        actual_name = stage_result.get('stage_name')
        if actual_name != expected_name:
            raise ValueError(
                f"Cannot resume: completed stage {stage_idx} is {actual_name!r}, "
                f"but current config expects {expected_name!r}."
            )
        if 'best_model_path' not in stage_result:
            raise ValueError(
                f"Cannot resume: completed stage {actual_name!r} has no best_model_path."
            )
        completed.append(stage_result)
    return completed


def _has_stage_selector(value):
    return value not in (None, '')


def _validate_stage_rerun_options(*, only_stage=None, from_stage=None):
    if _has_stage_selector(only_stage) and _has_stage_selector(from_stage):
        raise ValueError("Use either --only-stage or --from-stage, not both.")


def _resolve_resume_training(resume, resume_training=None):
    if resume_training is None:
        return bool(resume)
    return bool(resume_training)


def _resolve_stage_selector(selector, configured_stages):
    if not _has_stage_selector(selector):
        return None

    selector_text = str(selector).strip()
    try:
        stage_idx = int(selector_text)
    except ValueError:
        matches = [
            idx
            for idx, stage in enumerate(configured_stages)
            if stage['name'] == selector_text
        ]
        if not matches:
            available = ', '.join(stage['name'] for stage in configured_stages)
            raise ValueError(
                f"Unknown stage selector {selector!r}. "
                f"Use a zero-based stage index or one of: {available}."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Stage selector {selector!r} matches multiple stages. "
                "Use a zero-based stage index instead."
            )
        return matches[0]

    if stage_idx < 0 or stage_idx >= len(configured_stages):
        raise ValueError(
            f"Stage index {stage_idx} is out of range. "
            f"Valid indexes are 0..{len(configured_stages) - 1}."
        )
    return stage_idx


def _build_finished_stages(stage_results):
    return {
        stage_result['stage_name']: stage_result
        for stage_result in stage_results
    }


def _require_completed_prefix(completed, stage_idx, option_name):
    if stage_idx > len(completed):
        raise ValueError(
            f"{option_name} cannot start from stage {stage_idx}: "
            f"pipeline_summary.yaml contains only {len(completed)} completed "
            "stage(s) before the selected stage."
        )


def _prepare_resume_stage_results(
    previous_summary,
    configured_stages,
    *,
    only_stage_idx=None,
    from_stage_idx=None,
):
    completed = _completed_stage_results(previous_summary, configured_stages)

    if only_stage_idx is not None:
        _require_completed_prefix(completed, only_stage_idx, "--only-stage")
        return list(completed), completed[:only_stage_idx]

    if from_stage_idx is not None:
        _require_completed_prefix(completed, from_stage_idx, "--from-stage")
        prefix = completed[:from_stage_idx]
        return prefix, prefix

    return completed, completed


def _record_stage_result(summary, stage_idx, stage_result):
    stages = summary.setdefault('stages', [])
    if stage_idx < len(stages):
        stages[stage_idx] = stage_result
        return

    if stage_idx == len(stages):
        stages.append(stage_result)
        return

    raise ValueError(
        f"Cannot record stage {stage_idx}: pipeline_summary.yaml has only "
        f"{len(stages)} stage result(s)."
    )


def _write_pipeline_summary(summary, pipeline_dir):
    summary_path = os.path.join(pipeline_dir, "pipeline_summary.yaml")
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)
    return summary_path


def _looks_like_checkpoint_path(value):
    if not isinstance(value, (str, os.PathLike)):
        return False

    path = os.fspath(value)
    suffix = os.path.splitext(path)[1].lower()
    return (
        os.path.isabs(path)
        or '/' in path
        or '\\' in path
        or suffix in {'.pt', '.pth', '.ckpt', '.bin'}
    )


def _resolve_init_weights(stage_cfg, finished_stages):
    init_from = stage_cfg.get('init_weights_from', None)

    if init_from in (None, '', 'none'):
        return None

    if init_from == 'previous':
        if not finished_stages:
            raise ValueError("init_weights_from='previous' but there is no previous stage yet.")
        last_stage_name = list(finished_stages.keys())[-1]
        return finished_stages[last_stage_name]['best_model_path']

    if init_from in finished_stages:
        return finished_stages[init_from]['best_model_path']

    if _looks_like_checkpoint_path(init_from):
        return os.fspath(init_from)

    if init_from not in finished_stages:
        raise KeyError(
            f"Stage {stage_cfg.get('name')} wants init_weights_from={init_from!r}, "
            f"but finished stages are only: {list(finished_stages.keys())}. "
            "Use a stage name, 'previous', 'none', or a checkpoint path."
        )


def run_pipeline(
    multi_cfg_path="/home/configs/multi_domain.yaml",
    *,
    resume=False,
    resume_dir=None,
    only_stage=None,
    from_stage=None,
    resume_training=None,
):
    setup_distributed()
    pipeline_cfg = Config.fromfile(multi_cfg_path)
    single_stage_main, single_stage_path = _load_single_stage_main()
    resume_training = _resolve_resume_training(resume, resume_training)

    _validate_stage_rerun_options(only_stage=only_stage, from_stage=from_stage)
    only_stage_idx = _resolve_stage_selector(only_stage, pipeline_cfg.stages)
    from_stage_idx = _resolve_stage_selector(from_stage, pipeline_cfg.stages)
    selected_start_idx = only_stage_idx
    if selected_start_idx is None:
        selected_start_idx = from_stage_idx

    base_config_path = pipeline_cfg.base_config_path
    base_cfg = Config.fromfile(base_config_path)
    experiment_name = pipeline_cfg.get('experiment_name', None)
    pipeline_logs_folder = pipeline_cfg.get('pipeline_logs_folder', base_cfg.data.logs_folder)
    model_name = pipeline_cfg.get('model_name', base_cfg.model_type)

    pipeline_dir = None
    if is_main_process():
        pipeline_dir = _resolve_pipeline_run_dir(
            pipeline_logs_folder,
            model_name,
            experiment_name,
            resume=resume,
            resume_dir=resume_dir,
        )
    pipeline_dir = broadcast_object(pipeline_dir, src=0)
    pipeline_cfg_path = os.path.join(
            pipeline_dir,
            f"pipeline_config_used.yaml"
        )
    if is_main_process() and not (resume and os.path.exists(pipeline_cfg_path)):
        pipeline_cfg.save_config(pipeline_cfg_path)

    resolved_configs_dir = os.path.join(pipeline_dir, "resolved_configs")
    if is_main_process():
        os.makedirs(resolved_configs_dir, exist_ok=True)
    barrier()

    summary = {
        'experiment_name': experiment_name,
        'model_name': model_name,
        'multi_config_path': multi_cfg_path,
        'base_config_path': base_config_path,
        'save_dir': pipeline_dir,
        'resolved_configs_dir': resolved_configs_dir,
        'single_stage_trainer_path': single_stage_path,
        # 'device': "cuda" if torch.cuda.is_available() else "cpu",
        'stages': [],
    }

    finished_stages = {}
    completed_stage_count = 0
    if resume:
        previous_summary = _load_pipeline_summary(pipeline_dir)
        if previous_summary is not None:
            summary_stages, dependency_stages = _prepare_resume_stage_results(
                previous_summary,
                pipeline_cfg.stages,
                only_stage_idx=only_stage_idx,
                from_stage_idx=from_stage_idx,
            )
            summary.update(previous_summary)
            summary.update({
                'experiment_name': experiment_name,
                'model_name': model_name,
                'multi_config_path': multi_cfg_path,
                'base_config_path': base_config_path,
                'save_dir': pipeline_dir,
                'resolved_configs_dir': resolved_configs_dir,
                'single_stage_trainer_path': single_stage_path,
            })
            summary['stages'] = summary_stages
            finished_stages = _build_finished_stages(dependency_stages)
            if only_stage_idx is None and from_stage_idx is None:
                completed_stage_count = len(summary_stages)
            elif from_stage_idx is not None:
                completed_stage_count = from_stage_idx
            print(f"Resuming pipeline from {pipeline_dir}")
            if only_stage_idx is not None:
                print(
                    f"Rerunning only stage {only_stage_idx}: "
                    f"{pipeline_cfg.stages[only_stage_idx]['name']}"
                )
            elif from_stage_idx is not None:
                discarded_count = (
                    len(previous_summary.get('stages', [])) - len(summary_stages)
                )
                print(
                    f"Rerunning from stage {from_stage_idx}: "
                    f"{pipeline_cfg.stages[from_stage_idx]['name']}"
                )
                if discarded_count > 0:
                    print(f"Replacing {discarded_count} previous stage result(s).")
            else:
                print(f"Skipping {completed_stage_count} completed stage(s).")
        else:
            if selected_start_idx not in (None, 0):
                raise ValueError(
                    "Cannot skip earlier stages without pipeline_summary.yaml. "
                    "Run without a stage selector, start from stage 0, or provide a "
                    "resume directory with completed prerequisite stages."
                )
            print(f"Resuming pipeline from {pipeline_dir}; no summary found yet.")
    elif selected_start_idx not in (None, 0):
        raise ValueError(
            "--only-stage/--from-stage can skip earlier stages only with --resume, "
            "so prerequisite stage results can be loaded from pipeline_summary.yaml."
        )

    for stage_idx, stage in enumerate(pipeline_cfg.stages):
        stage_name = stage['name']
        if only_stage_idx is not None and stage_idx != only_stage_idx:
            print(
                "Skipping stage outside --only-stage selection "
                f"{stage_idx + 1}/{len(pipeline_cfg.stages)}: {stage_name}"
            )
            continue

        if from_stage_idx is not None and stage_idx < from_stage_idx:
            print(
                "Skipping stage before --from-stage selection "
                f"{stage_idx + 1}/{len(pipeline_cfg.stages)}: {stage_name}"
            )
            continue

        if only_stage_idx is None and from_stage_idx is None and stage_idx < completed_stage_count:
            print(
                "Skipping completed stage "
                f"{stage_idx + 1}/{len(pipeline_cfg.stages)}: {stage_name}"
            )
            continue

        print(f"\n{'=' * 80}")
        print(f"Running stage {stage_idx + 1}/{len(pipeline_cfg.stages)}: {stage_name}")
        print(f"{'=' * 80}\n")

        stage_base_cfg = Config.fromfile(base_config_path)
        stage_overrides = stage.get('overrides', {})
        _deep_update(stage_base_cfg, stage_overrides)

        stage_base_cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"

        init_weights = _resolve_init_weights(stage, finished_stages)
        if init_weights is not None:
            stage_base_cfg['pretrained_weights'] = init_weights
        elif 'pretrained_weights' not in stage_overrides:
            stage_base_cfg['pretrained_weights'] = None

        resolved_stage_cfg_path = os.path.join(
            resolved_configs_dir,
            f"stage_{stage_idx:02d}_{_sanitize_path_part(stage_name)}.yaml"
        )
        if is_main_process():
            stage_base_cfg.save_config(resolved_stage_cfg_path)
        barrier()

        stage_dir = os.path.join(
            pipeline_dir,
            f"stage_{stage_idx:02d}_{_sanitize_path_part(stage_name)}",
        )

        stage_result = single_stage_main(
            stage_base_cfg,
            results=None,
            folder_name=stage_name,
            stage_name=stage_name,
            save_metadata=True,
            baselines_only=False,
            save_dir=stage_dir,
            resume_training=resume_training,
        )

        stage_result = _to_serializable(stage_result)
        stage_result['stage_index'] = stage_idx
        stage_result['stage_name'] = stage_name
        stage_result['stage_dir'] = stage_dir
        stage_result['resolved_config_path'] = resolved_stage_cfg_path
        stage_result['init_weights_from'] = stage.get('init_weights_from', None)
        stage_result['resolved_init_weights_path'] = init_weights

        finished_stages[stage_name] = stage_result
        _record_stage_result(summary, stage_idx, stage_result)

        if is_main_process():
            _write_pipeline_summary(summary, pipeline_dir)
        barrier()

        print(f"Stage {stage_name} finished.")
        print(f"Best model: {stage_result['best_model_path']}")

    summary_path = os.path.join(pipeline_dir, "pipeline_summary.yaml")
    print(f"\nSaved pipeline summary to {summary_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a multi-domain training/testing pipeline.")
    parser.add_argument(
        "cfg_path",
        nargs="?",
        default="/home/configs/multi_domain.yaml",
        help="Path to the multi-domain YAML config.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing pipeline run and continue interrupted training stages.",
    )
    parser.add_argument(
        "--resume-dir",
        default=None,
        help="Existing pipeline run directory to resume. Required for anonymous runs.",
    )
    parser.add_argument(
        "--only-stage",
        default=None,
        help="Run exactly one configured stage by zero-based index or exact stage name.",
    )
    parser.add_argument(
        "--from-stage",
        default=None,
        help=(
            "Run the selected configured stage and every later stage by "
            "zero-based index or exact stage name."
        ),
    )
    parser.add_argument(
        "--no-resume-training",
        action="store_true",
        help=(
            "Do not load stage training_state.pth when --resume is used. "
            "The pipeline run is still resumed for summary/dependency context."
        ),
    )
    args = parser.parse_args()

    summary = run_pipeline(
        args.cfg_path,
        resume=args.resume,
        resume_dir=args.resume_dir,
        only_stage=args.only_stage,
        from_stage=args.from_stage,
        resume_training=False if args.no_resume_training else None,
    )
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))
