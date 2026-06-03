import os
import sys
import copy
import yaml
import re
import importlib.util
from collections.abc import Mapping

import torch

sys.path.insert(0, '/home')
sys.path.insert(0, '../../')

from lib.config.cfg import Config


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


def _resolve_init_weights(stage_cfg, finished_stages):
    init_from = stage_cfg.get('init_weights_from', None)

    if init_from in (None, '', 'none'):
        return None

    if init_from == 'previous':
        if not finished_stages:
            raise ValueError("init_weights_from='previous' but there is no previous stage yet.")
        last_stage_name = list(finished_stages.keys())[-1]
        return finished_stages[last_stage_name]['best_model_path']

    if init_from not in finished_stages:
        raise KeyError(
            f"Stage {stage_cfg.get('name')} wants init_weights_from={init_from!r}, "
            f"but finished stages are only: {list(finished_stages.keys())}"
        )

    return finished_stages[init_from]['best_model_path']


def run_pipeline(multi_cfg_path="/home/configs/multi_domain.yaml"):
    pipeline_cfg = Config.fromfile(multi_cfg_path)
    single_stage_main, single_stage_path = _load_single_stage_main()

    base_config_path = pipeline_cfg.base_config_path
    base_cfg = Config.fromfile(base_config_path)
    experiment_name = pipeline_cfg.get('experiment_name', None)
    pipeline_logs_folder = pipeline_cfg.get('pipeline_logs_folder', base_cfg.data.logs_folder)
    model_name = pipeline_cfg.get('model_name', base_cfg.model_type)

    pipeline_dir = _allocate_pipeline_run_dir(
        pipeline_logs_folder,
        model_name,
        experiment_name,
    )
    resolved_configs_dir = os.path.join(pipeline_dir, "resolved_configs")
    os.makedirs(resolved_configs_dir, exist_ok=True)

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

    for stage_idx, stage in enumerate(pipeline_cfg.stages):
        stage_name = stage['name']
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
        stage_base_cfg.save_config(resolved_stage_cfg_path)

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
        )

        stage_result = _to_serializable(stage_result)
        stage_result['stage_index'] = stage_idx
        stage_result['stage_name'] = stage_name
        stage_result['stage_dir'] = stage_dir
        stage_result['resolved_config_path'] = resolved_stage_cfg_path
        stage_result['init_weights_from'] = stage.get('init_weights_from', None)

        finished_stages[stage_name] = stage_result
        summary['stages'].append(stage_result)

        summary_path = os.path.join(pipeline_dir, "pipeline_summary.yaml")
        with open(summary_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)

        print(f"Stage {stage_name} finished.")
        print(f"Best model: {stage_result['best_model_path']}")

    summary_path = os.path.join(pipeline_dir, "pipeline_summary.yaml")
    print(f"\nSaved pipeline summary to {summary_path}")
    return summary


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "/home/configs/multi_domain.yaml"
    summary = run_pipeline(cfg_path)
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))
