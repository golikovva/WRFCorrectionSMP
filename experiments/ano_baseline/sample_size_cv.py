import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.config.cfg import Config
from lib.data.logger import WRFLogger
from lib.data.train_test_split import (
    blocked_cv_splits,
    is_all_train_size,
    parse_timedelta64,
    split_dates_dispatch,
)
from lib.models.ano_corr_accumulator import ANOCorrector
from experiments.ano_baseline.cli import add_mean_period_arguments, validate_mean_period_arguments

from experiments.ano_baseline.common import (
    IdentityScaler,
    build_dataset,
    build_era_to_wrf_interpolator,
    build_loader,
    build_test_extra_datasets,
    build_test_losses,
    build_wrf_era_datasets,
    ensure_dir,
    fit_ano_model,
    non_overlapping_sequence_starts,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run ANO train-size cross-validation.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "train_test.yaml"),
        help="Path to a YAML config.",
    )
    add_mean_period_arguments(parser)
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        default=["90D", "180D", "365D", "all"],
        help="Train block durations, for example 7D 30D 180D, or all/rest.",
    )
    parser.add_argument(
        "--test-size",
        default="90D",
        help="Fixed test block duration.",
    )
    parser.add_argument(
        "--step-size",
        default=None,
        help="Distance between test block starts. Defaults to --test-size.",
    )
    parser.add_argument(
        "--gap",
        default="0D",
        help="Gap between train and test blocks.",
    )
    parser.add_argument(
        "--train-position",
        default="around",
        choices=["before", "around"],
        help="Place train data before the test block or around it with edge fill.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CV summary. Defaults to cfg.data.logs_folder/ano_baseline_sample_size_cv.",
    )
    parser.add_argument("--disable-stations", action="store_true")
    parser.add_argument("--disable-scatter", action="store_true")
    return validate_mean_period_arguments(parser, parser.parse_args(argv))


def configured_dates(cfg):
    train_dates, val_dates, test_dates = split_dates_dispatch(**cfg.split_config)
    parts = [np.asarray(x) for x in (train_dates, val_dates, test_dates) if len(x) > 0]
    return np.sort(np.concatenate(parts))


def duration_tag(value):
    return str(value).strip().replace(" ", "").replace("/", "_")


def duration_days(value):
    if is_all_train_size(value):
        return float("inf")
    delta = parse_timedelta64(value)
    return float(delta / np.timedelta64(1, "D"))


def add_metric_summary(row, prefix, values):
    if values is None:
        return
    arr = np.asarray(values, dtype=float).squeeze()
    row[f"{prefix}_mean"] = float(np.nanmean(arr))

    flat = arr.ravel()
    if flat.size == 3:
        labels = ["u10", "v10", "t2"]
    elif flat.size == 2:
        labels = ["u10", "v10"]
    elif flat.size == 1:
        labels = ["value"]
    else:
        return

    for label, value in zip(labels, flat):
        row[f"{prefix}_{label}"] = float(value)


def summarize_results(res_dict, model_name):
    row = {}
    metric_names = ["era_mae", "era_mse", "era_err_norm", "era_angle_norm", "era_norm_dif", "mesoscale_loss"]
    for metric_name in metric_names:
        add_metric_summary(row, f"ano_{metric_name}", res_dict.get(f"{model_name}_{metric_name}"))
        add_metric_summary(row, f"orig_{metric_name}", res_dict.get(f"orig_{metric_name}"))

    for metric_name in ["era_mae", "era_mse", "era_err_norm", "era_angle_norm", "era_norm_dif"]:
        ano = row.get(f"ano_{metric_name}_mean")
        orig = row.get(f"orig_{metric_name}_mean")
        if ano is not None and orig not in (None, 0):
            row[f"{metric_name}_skill_vs_orig"] = float(1.0 - ano / orig)
    return row


def write_summaries(rows, summary_path, aggregate_path):
    df = pd.DataFrame(rows)
    df.to_csv(summary_path, index=False)

    if df.empty:
        return

    metadata_cols = {
        "fold",
        "test_start",
        "test_end",
        "train_start",
        "train_end",
        "train_left_start",
        "train_left_end",
        "train_right_start",
        "train_right_end",
        "train_position",
        "train_size",
        "train_size_days",
        "train_sequence_starts",
        "test_sequence_starts",
        "train_dates",
        "test_dates",
        "model_name",
        "save_dir",
    }
    metric_cols = [
        col for col in df.columns
        if col not in metadata_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not metric_cols:
        return

    grouped = df.groupby(["train_size", "train_size_days"], sort=False)[metric_cols]
    parts = []
    for stat_name, func in [
        ("mean", "mean"),
        ("std", "std"),
        ("median", "median"),
        ("q25", lambda x: x.quantile(0.25)),
        ("q75", lambda x: x.quantile(0.75)),
    ]:
        part = grouped.agg(func)
        part.columns = [f"{col}_{stat_name}" for col in part.columns]
        parts.append(part)

    pd.concat(parts, axis=1).reset_index().to_csv(aggregate_path, index=False)


def prepare_cfg(args):
    cfg = Config.fromfile(args.config)
    cfg.device = torch.device(cfg.device if "device" in cfg else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg.run_config.run_mode = "test"
    cfg.test_config.save_outputs = False
    cfg.test_config.draw_spectrum = False
    if args.disable_stations:
        cfg.test_config.use_stations = False
        cfg.run_config.use_stations = False
    if args.disable_scatter:
        cfg.run_config.use_scatter = False
    return cfg


def main():
    args = parse_args()
    from lib.pipeline.test import test
    cfg = prepare_cfg(args)

    output_dir = Path(args.output_dir or Path(cfg.data.logs_folder) / "ano_baseline_sample_size_cv")
    ensure_dir(output_dir)

    all_dates = configured_dates(cfg)
    wrf_dataset, era_dataset = build_wrf_era_datasets(cfg)
    train_dataset = build_dataset(wrf_dataset, era_dataset)

    extra_datasets = build_test_extra_datasets(cfg)
    for dataset in extra_datasets:
        if getattr(dataset, "name", None) == "Stations" and getattr(dataset, "wind_format", None) == "uv":
            dataset.wind_format = "wd"
    test_dataset = build_dataset(wrf_dataset, era_dataset, *extra_datasets)

    era_to_wrf = build_era_to_wrf_interpolator(wrf_dataset, era_dataset, cfg.device)
    losses = build_test_losses(cfg, wrf_dataset, era_dataset)
    identity_scaler = IdentityScaler()

    train_sizes = list(args.train_sizes)
    max_train_size = max(train_sizes, key=duration_days)
    common_splits = blocked_cv_splits(
        all_dates,
        train_size=max_train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        gap=args.gap,
        train_position=args.train_position,
    )
    if not common_splits:
        raise ValueError(
            f"No blocked CV splits for max_train_size={max_train_size}, test_size={args.test_size}, gap={args.gap}."
        )

    test_starts = [split["test_start"] for split in common_splits]
    print(
        f"Using {len(test_starts)} common test blocks: "
        f"test_size={args.test_size}, step_size={args.step_size or args.test_size}, "
        f"max_train_size={max_train_size}, train_position={args.train_position}."
    )

    rows = []
    summary_path = output_dir / "summary.csv"
    aggregate_path = output_dir / "summary_by_train_size.csv"

    for train_size in train_sizes:
        splits = blocked_cv_splits(
            all_dates,
            train_size=train_size,
            test_size=args.test_size,
            step_size=args.step_size,
            gap=args.gap,
            train_position=args.train_position,
            test_starts=test_starts,
        )
        if len(splits) != len(test_starts):
            raise RuntimeError(
                f"Expected {len(test_starts)} folds for train_size={train_size}, got {len(splits)}."
            )

        size_tag = duration_tag(train_size)
        model_name = f"ANO_{args.mean_period}_{size_tag}"

        for split in splits:
            fold_name = f"fold_{split['fold']:03d}_{str(split['test_start'])[:10]}"
            folder_name = f"ano_baseline/blocked_cv/{args.mean_period}/train_{size_tag}/{fold_name}"
            logger = WRFLogger(cfg, cfg.data.logs_folder, folder_name)

            train_dates_full = split["train_dates"]
            test_dates_full = split["test_dates"]
            selected_train = non_overlapping_sequence_starts(train_dates_full, cfg.s2s.sequence_len)
            test_dates = non_overlapping_sequence_starts(test_dates_full, cfg.s2s.sequence_len)
            if len(selected_train) == 0 or len(test_dates) == 0:
                continue

            train_loader = build_loader(train_dataset, selected_train, cfg, shuffle=False)
            test_loader = build_loader(test_dataset, test_dates, cfg, shuffle=False)

            model = ANOCorrector(
                time_mean_period=args.mean_period,
                season_months=args.season_months,
                time_step_hours=getattr(wrf_dataset, "time_res_h", 1),
            )
            fit_ano_model(
                model,
                train_loader,
                era_to_wrf,
                cfg,
                time_step_hours=getattr(wrf_dataset, "time_res_h", 1),
            )
            suffix = ".npz" if model.period == "season" else ".npy"
            model.save_correction_fields(Path(logger.save_dir) / f"correction_fields{suffix}")

            res_dict = test(
                {model_name: model},
                losses,
                identity_scaler,
                identity_scaler,
                test_loader,
                logger,
                cfg,
            )

            row = {
                "fold": fold_name,
                "train_size": train_size,
                "train_size_days": duration_days(train_size),
                "train_sequence_starts": len(selected_train),
                "test_sequence_starts": len(test_dates),
                "train_dates": len(train_dates_full),
                "test_dates": len(test_dates_full),
                "train_start": str(split["train_start"]),
                "train_end": str(split["train_end"]),
                "train_left_start": str(split["train_left_start"]),
                "train_left_end": str(split["train_left_end"]),
                "train_right_start": str(split["train_right_start"]),
                "train_right_end": str(split["train_right_end"]),
                "train_position": split["train_position"],
                "test_start": str(split["test_start"]),
                "test_end": str(split["test_end"]),
                "model_name": model_name,
                "save_dir": logger.save_dir,
            }
            row.update(summarize_results(res_dict, model_name))
            rows.append(row)
            write_summaries(rows, summary_path, aggregate_path)
            print(f"Updated CV summaries: {summary_path}, {aggregate_path}")


if __name__ == "__main__":
    main()
