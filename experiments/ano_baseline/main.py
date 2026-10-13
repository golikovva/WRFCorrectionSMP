import argparse
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.config.cfg import Config
from lib.data.train_test_split import split_dates_dispatch, split_given_dates_by_percents
from lib.models.ano_corr_accumulator import ANOCorrector
from experiments.ano_baseline.cli import add_mean_period_arguments, validate_mean_period_arguments

from experiments.ano_baseline.common import (
    build_dataset,
    build_era_to_wrf_interpolator,
    build_loader,
    build_wrf_era_datasets,
    ensure_dir,
    fit_ano_model,
    non_overlapping_sequence_starts,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fit ANO correction fields.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "train_test.yaml"),
        help="Path to a YAML config.",
    )
    add_mean_period_arguments(parser)
    parser.add_argument(
        "--train-size",
        type=float,
        default=1.0,
        help="Fraction of cfg split train dates used to accumulate ANO fields.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for saved correction fields. Defaults to cfg.data.logs_folder/ano_baseline.",
    )
    return validate_mean_period_arguments(parser, parser.parse_args(argv))


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.device = torch.device(cfg.device if "device" in cfg else ("cuda" if torch.cuda.is_available() else "cpu"))

    wrf_dataset, era_dataset = build_wrf_era_datasets(cfg)
    dataset = build_dataset(wrf_dataset, era_dataset)

    train_dates, _, _ = split_dates_dispatch(**cfg.split_config)
    if args.train_size < 1.0:
        train_dates, _, _ = split_given_dates_by_percents(
            train_dates,
            train_size=args.train_size,
            validation_size=0.0,
            test_size=None,
        )
    train_dates = non_overlapping_sequence_starts(np.asarray(train_dates), cfg.s2s.sequence_len)

    train_loader = build_loader(dataset, train_dates, cfg, shuffle=False)
    era_to_wrf = build_era_to_wrf_interpolator(wrf_dataset, era_dataset, cfg.device)

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

    output_dir = args.output_dir or str(Path(cfg.data.logs_folder) / "ano_baseline")
    ensure_dir(output_dir)

    size_tag = f"{args.train_size:g}".replace(".", "p")
    suffix = ".npz" if model.period == "season" else ".npy"
    output_path = Path(output_dir) / f"{model.period}_train_{size_tag}_correction_fields{suffix}"
    model.save_correction_fields(output_path)
    print(f"Saved ANO correction fields to {output_path}")


if __name__ == "__main__":
    main()
