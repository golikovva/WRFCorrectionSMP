"""Seasonal ANO learning curves on fixed, nested weeks-per-season splits.

Dataset and evaluation imports deliberately stay inside run_experiment: dry-run
and plot-only need no WRF/ERA files or optional meteorological libraries.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.config.cfg import Config
from lib.data.train_test_split import parse_timedelta64, split_dates_dispatch

DEFAULT_SEASON_MONTHS = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
METRIC = "lat_weighted_mae"
TABLE_PATH = "mean_era_metrics_table.csv"
MODELS = ("ANO", "orig")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "ano_train_size.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="CPU by default; CUDA must be requested explicitly.")
    parser.add_argument("--weeks-per-season", type=int, nargs="+", default=None)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Save and verify splits without weather data.")
    modes.add_argument("--plot-only", action="store_true", help="Plot existing standard metric tables.")
    return parser.parse_args(argv)


def _weeks(values):
    values = list(values)
    if not values or any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= 9 for k in values):
        raise ValueError("weeks_per_season must contain integers from 1 to 9.")
    if len(set(values)) != len(values):
        raise ValueError("weeks_per_season must not contain duplicates.")
    return sorted(int(k) for k in values)


def prepare_cfg(args):
    cfg = Config.fromfile(str(args.config))
    cfg.device = args.device
    cfg.weeks_per_season = _weeks(
        args.weeks_per_season if args.weeks_per_season is not None
        else getattr(cfg, "weeks_per_season", list(range(1, 10)))
    )
    if cfg.split_config.split_mode != "seasonal_nested":
        raise ValueError("ANO train-size experiments require split_mode='seasonal_nested'.")
    if parse_timedelta64(cfg.split_config.time_step) != np.timedelta64(1, "h"):
        raise ValueError("ANO train-size experiments require hourly (1h) data.")
    if cfg.s2s.sequence_len != 4:
        raise ValueError("The paper protocol requires sequence_len=4.")
    if cfg.ano.time_mean_period not in {"season", "seasonal"}:
        raise ValueError("The train-size experiment requires seasonal ANO.")
    cfg.ano.time_mean_period = "season"
    # Model validation is also applied before fitting; this early check works
    # without importing torch or meteorological data dependencies.
    groups = cfg.ano.season_months
    try:
        months = [month for group in groups for month in group]
        valid = (bool(groups) and all(bool(group) for group in groups)
                 and all(type(month) is int for month in months)
                 and sorted(months) == list(range(1, 13)))
    except TypeError:
        valid = False
    if not valid:
        raise ValueError("season_months must be nonempty groups covering months 1..12 exactly once.")
    if cfg.reference_dataset != "WRF" or cfg.target_dataset != "ERA5":
        raise ValueError("The shared ANO evaluation requires WRF and ERA5 datasets.")
    if cfg.reference_region.lower() != "nestp":
        raise ValueError("This learning-curve protocol uses reference_region='nestp'.")
    cfg.reference_region = "nestp"
    cfg.run_config.run_mode = "test"
    cfg.run_config.use_stations = False
    cfg.run_config.use_scatter = False
    cfg.run_config.variable_sequence_length = False
    if args.device == "cpu":
        cfg.run_config.num_workers = 0
    if int(cfg.run_config.batch_size) < 1 or int(cfg.run_config.num_workers) < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative.")
    for key in ("use_stations", "save_outputs", "save_losses", "draw_spectrum",
                "plot_samples", "draw_plots", "debug_mode"):
        cfg.test_config[key] = False
    cfg.test_config.profiling = {"enabled": False}
    cfg.test_config.regional_metrics = {"enabled": False}
    return cfg


def build_splits(cfg, weeks_per_season):
    """Return complete, disjoint splits and all non-overlapping sequence starts."""
    from experiments.ano_baseline.common import non_overlapping_sequence_starts, sequence_dates

    splits = {}
    previous_train = None
    fixed_validation = fixed_test = None
    for size in _weeks(weeks_per_season):
        options = dict(cfg.split_config)
        options["blocks_per_season"] = size
        arrays = split_dates_dispatch(**options)
        split = {}
        for name, dates in zip(("train", "validation", "test"), arrays):
            dates = np.sort(np.asarray(dates).astype("datetime64[h]"))
            if not len(dates) or np.isnat(dates).any() or len(np.unique(dates)) != len(dates):
                raise ValueError(f"{size}wps: {name} must contain unique, nonempty valid dates.")
            starts = non_overlapping_sequence_starts(dates, cfg.s2s.sequence_len)
            represented = np.sort(sequence_dates(starts, cfg.s2s.sequence_len, 1).ravel())
            if not np.array_equal(represented, dates):
                raise ValueError(f"{size}wps: {name} cannot be fully covered by non-overlapping sequences.")
            split[f"{name}_dates"] = dates
            split[f"{name}_starts"] = starts
        train, validation, test = (split[f"{name}_dates"] for name in ("train", "validation", "test"))
        if any(np.intersect1d(a, b).size for a, b in ((train, validation), (train, test), (validation, test))):
            raise ValueError(f"{size}wps: train, validation and test overlap.")
        if previous_train is not None and not np.isin(previous_train, train).all():
            raise ValueError(f"{size}wps: training selections are not nested.")
        if fixed_validation is not None and (
            not np.array_equal(fixed_validation, validation) or not np.array_equal(fixed_test, test)
        ):
            raise ValueError(f"{size}wps: validation/test must stay fixed across training sizes.")
        previous_train, fixed_validation, fixed_test = train, validation, test
        counts = _season_counts(train, cfg.ano.season_months)
        if any(count == 0 for count in counts):
            raise ValueError(f"{size}wps: empty training season group; selected hour counts are {counts}.")
        splits[size] = split
    return splits


def _season_counts(dates, groups):
    months = np.asarray(dates).astype("datetime64[M]").astype(int) % 12 + 1
    return [int(np.isin(months, group).sum()) for group in groups]


def _validate_fields(data, cfg, label, date, *, batched, batch_size=None):
    if not isinstance(data, dict):
        raise ValueError(f"{label}: missing mandatory WRF/ERA5 data at {date}.")
    for name in (cfg.reference_dataset, cfg.target_dataset):
        field = data.get(name)
        shape = getattr(field, "shape", ())
        expected_ndim = 5 if batched else 4
        time_axis = 1 if batched else 0
        if field is None:
            raise ValueError(f"{label}: missing mandatory {name} data at {date}.")
        if len(shape) != expected_ndim or shape[time_axis] != cfg.s2s.sequence_len:
            raise ValueError(f"{label}: incomplete {name} sequence at {date}: shape={shape}, expected {cfg.s2s.sequence_len} hours.")
        if batched and shape[0] != batch_size:
            raise ValueError(f"{label}: {name} batch/date count mismatch at {date}.")
        channels = shape[-3]
        if (name == cfg.target_dataset and channels != 3) or (name == cfg.reference_dataset and channels < 3):
            raise ValueError(f"{label}: {name} requires u10, v10, t2m channels at {date}, got {channels}.")
        if any(length == 0 for length in shape[-2:]):
            raise ValueError(f"{label}: empty {name} spatial grid at {date}.")
        if hasattr(field, "detach"):
            import torch
            finite = bool(torch.isfinite(field).all().item())
        else:
            finite = bool(np.isfinite(np.asarray(field)).all())
        if not finite or (np.ma.isMaskedArray(field) and np.ma.getmaskarray(field).any()):
            raise ValueError(f"{label}: nonfinite or masked mandatory {name} data at {date}.")


class StrictDataset:
    """Reject missing/short samples before variable-length collation can pad them."""

    def __init__(self, dataset, cfg, label):
        self.dataset, self.cfg, self.label = dataset, cfg, label
        self.datasets = dataset.datasets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            data, date = self.dataset[index]
        except Exception as exc:
            raise ValueError(f"{self.label}: cannot read mandatory WRF/ERA5 data at {index}: {exc}") from exc
        if np.datetime64(date, "h") != np.datetime64(index, "h"):
            raise ValueError(f"{self.label}: dataset returned date {date} for requested {index}.")
        _validate_fields(data, self.cfg, self.label, date, batched=False)
        return data, date


class StrictLoader:
    """Require every requested sequence exactly once, with complete finite tensors."""

    def __init__(self, loader, expected_starts, cfg, label):
        self.loader, self.cfg, self.label = loader, cfg, label
        self.dataset = loader.dataset
        self.expected_starts = np.asarray(expected_starts).astype("datetime64[h]")
        self.observed_sequences = 0
        self.observed_hours = 0

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.observed_sequences = self.observed_hours = 0
        for data, dates in self.loader:
            if dates is None:
                raise ValueError(f"{self.label}: missing mandatory sequence data.")
            dates = np.asarray(dates).astype("datetime64[h]")
            offset = self.observed_sequences
            expected = self.expected_starts[offset:offset + len(dates)]
            if dates.ndim != 1 or not len(dates) or not np.array_equal(dates, expected):
                raise ValueError(f"{self.label}: missing, duplicated or reordered sequence starts near index {offset}.")
            _validate_fields(data, self.cfg, self.label, dates[0], batched=True, batch_size=len(dates))
            self.observed_sequences += len(dates)
            self.observed_hours += len(dates) * self.cfg.s2s.sequence_len
            yield data, dates
        self.assert_complete()

    def assert_complete(self):
        if self.observed_sequences != len(self.expected_starts):
            raise ValueError(f"{self.label}: incomplete coverage; read {self.observed_sequences} of {len(self.expected_starts)} sequences.")


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _split_metadata(cfg, size, split):
    return {
        "weeks_per_season": size,
        "season_months": cfg.ano.season_months,
        "seasonal_train_hours": _season_counts(split["train_dates"], cfg.ano.season_months),
        **{f"{part}_hours": len(split[f"{part}_dates"]) for part in ("train", "validation", "test")},
        **{f"{part}_sequences": len(split[f"{part}_starts"]) for part in ("train", "validation", "test")},
        "date_arrays": "split_dates.npz",
        "time_step_hours": 1,
        "sequence_len": cfg.s2s.sequence_len,
    }


def _prepare_output(output_dir, cfg, splits, *, dry_run):
    """Permit reuse only of an identical dry run; never overwrite fitted results."""
    manifest_path = output_dir / "run_manifest.json"
    signature = {"config": cfg.to_dict(), "weeks_per_season": list(splits)}
    if output_dir.exists() and any(output_dir.iterdir()):
        if not manifest_path.is_file():
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Choose a new run directory.")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") != "dry_run" or any(existing.get(key) != value for key, value in signature.items()):
            raise FileExistsError(f"Output directory contains a completed, started or different run: {output_dir}.")
        allowed_top = {"run_manifest.json", "resolved_config.yaml", "split_manifest.json", *[f"{size}wps" for size in splits]}
        if any(path.name not in allowed_top for path in output_dir.iterdir()):
            raise FileExistsError(f"Output directory has artifacts beyond the reusable dry run: {output_dir}.")
        for size in splits:
            folder = output_dir / f"{size}wps"
            if folder.exists() and any(path.name not in {"split_dates.npz", "split_manifest.json"} for path in folder.iterdir()):
                raise FileExistsError(f"Refusing to overwrite existing results in {folder}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {**signature, "status": "dry_run" if dry_run else "running",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model": "ANO", "baseline": "orig", "metric_table": TABLE_PATH}
    _write_json(manifest_path, manifest)
    cfg.save_config(str(output_dir / "resolved_config.yaml"))
    metadata = {}
    for size, split in splits.items():
        folder = output_dir / f"{size}wps"
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder / "split_dates.npz", **split)
        metadata[f"{size}wps"] = _split_metadata(cfg, size, split)
        _write_json(folder / "split_manifest.json", metadata[f"{size}wps"])
    _write_json(output_dir / "split_manifest.json", metadata)
    return manifest


def plot_results(output_dir, weeks_per_season=range(1, 10)):
    """Validate standard ERA-grid tables, then export three curves and raw values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from experiments.compare_models.plot_train_size_dependence import (
        collect_train_size_metrics, plot_error_vs_train_size,
    )

    output_dir = Path(output_dir)
    weeks = _weeks(weeks_per_season)
    size_pattern = r"^(?P<size>" + "|".join(map(str, weeks)) + r")wps$"
    parts = []
    # Validate all variables before producing any new plots.
    for label, column in (("u10", "u10"), ("v10", "v10"), ("t2m", "t2")):
        raw = collect_train_size_metrics(output_dir, METRIC, column, table_path=TABLE_PATH,
                                         model_names=MODELS, size_pattern=size_pattern)
        counts = raw.groupby(["nwps", "model"]).size()
        expected = pd.MultiIndex.from_product([weeks, MODELS], names=["nwps", "model"])
        if not counts.reindex(expected, fill_value=0).eq(1).all() or len(raw) != len(expected):
            raise ValueError(f"{label}: expected one ANO and one orig row at every requested training size.")
        if not np.isfinite(raw["value"].to_numpy()).all():
            raise ValueError(f"{label}: metrics contain nonfinite values.")
        baseline = raw.loc[raw["model"] == "orig", "value"].to_numpy()
        if not np.allclose(baseline, baseline[0], rtol=1e-7, atol=1e-10):
            raise ValueError(f"{label}: WRF baseline changes across training sizes; verify common test coverage.")
        raw["variable"], raw["metric"] = label, METRIC.upper()
        parts.append(raw)

    for label, column in (("u10", "u10"), ("v10", "v10"), ("t2m", "t2")):
        figure, _, _ = plot_error_vs_train_size(
            output_dir, METRIC, column, table_path=TABLE_PATH,
            model_names=MODELS, model_labels={"ANO": "ANO seasonal", "orig": "WRF"},
            size_pattern=size_pattern, show_std=False,
            title=f"LAT_WEIGHTED_MAE for {label} vs training-set size",
            ylabel=f"LAT_WEIGHTED_MAE ({label})",
            output_path=output_dir / f"{METRIC}_{label}.png",
        )
        try:
            figure.savefig(output_dir / f"{METRIC}_{label}.pdf", bbox_inches="tight")
        finally:
            plt.close(figure)
    summary = pd.concat(parts, ignore_index=True)[
        ["nwps", "model", "variable", "metric", "value", "experiment", "source_file"]
    ].sort_values(["variable", "model", "nwps"])
    summary.to_csv(output_dir / "summary.csv", index=False, float_format="%.17g")
    return summary


def run_experiment(cfg, output_dir, splits):
    """Fit independent fields and use precisely the shared neural-model evaluation."""
    # Resolve mandatory folders before importing any weather reader.
    required_folders = [Path(cfg.data.nestp_folder),
                        Path(cfg.data.era_folder) / cfg.data.era_wind_subdir,
                        Path(cfg.data.era_folder) / "t2"]
    for folder in required_folders:
        if not folder.is_dir():
            raise FileNotFoundError(f"Mandatory weather-data directory does not exist: {folder}")
    import torch
    from lib.data.logger import WRFLogger
    from lib.models.ano_corr_accumulator import ANOCorrector
    from lib.pipeline.test import test
    from experiments.ano_baseline.common import (
        IdentityScaler, build_dataset, build_era_to_wrf_interpolator, build_loader,
        build_test_losses, build_wrf_era_datasets, fit_ano_model,
    )

    torch.set_num_threads(1)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested explicitly but is unavailable.")
    np.random.seed(int(cfg.global_seed))
    torch.manual_seed(int(cfg.global_seed))
    try:
        wrf_dataset, era_dataset = build_wrf_era_datasets(cfg)
    except Exception as exc:
        raise RuntimeError(f"Cannot initialize mandatory WRF/ERA5 data from {required_folders}: {exc}") from exc
    for dataset in (wrf_dataset, *era_dataset.datasets):
        if getattr(dataset, "time_res_h", None) != 1:
            raise ValueError(f"{dataset.name}: dataset time resolution must match the hourly split.")
    dataset = build_dataset(wrf_dataset, era_dataset)
    era_to_wrf = build_era_to_wrf_interpolator(wrf_dataset, era_dataset, cfg.device)
    losses = build_test_losses(cfg, wrf_dataset, era_dataset)
    scaler = IdentityScaler()
    for size, split in splits.items():
        folder = output_dir / f"{size}wps"
        logger = WRFLogger(cfg, save_dir=str(folder))
        loaders = {}
        for part in ("train", "test"):
            label = f"{size}wps {part}"
            checked_dataset = StrictDataset(dataset, cfg, label)
            loader = build_loader(checked_dataset, split[f"{part}_starts"], cfg, shuffle=False)
            loaders[part] = StrictLoader(loader, split[f"{part}_starts"], cfg, label)
        model = ANOCorrector(time_mean_period="season", season_months=cfg.ano.season_months,
                             time_step_hours=1).to(cfg.device)
        fit_ano_model(model, loaders["train"], era_to_wrf, cfg, time_step_hours=1)
        loaders["train"].assert_complete()
        counts = model.period_counts.detach().cpu().numpy()
        expected_counts = np.asarray(_season_counts(split["train_dates"], cfg.ano.season_months))
        if not np.array_equal(counts, expected_counts) or np.any(counts == 0):
            raise ValueError(f"{size}wps: fitted seasonal counts {counts.tolist()} do not match planned {expected_counts.tolist()}.")
        model.save_correction_fields(folder / "correction_fields.npz")
        _write_json(folder / "season_counts.json",
                    {"season_months": cfg.ano.season_months, "counts": counts.astype(int).tolist()})
        test({"ANO": model}, losses, scaler, scaler, loaders["test"], logger, cfg)
        loaders["test"].assert_complete()
        _write_json(folder / "coverage.json", {
            f"{part}_{measure}": getattr(loaders[part], f"observed_{measure}")
            for part in ("train", "test") for measure in ("sequences", "hours")
        })
        print(f"Finished {size}wps: {folder}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    cfg = prepare_cfg(args)
    if args.plot_only:
        if args.output_dir is None:
            raise ValueError("--plot-only requires --output-dir containing existing metric tables.")
        summary = plot_results(args.output_dir, cfg.weeks_per_season)
        print(f"Saved three PNG/PDF curves and {len(summary)} values to {args.output_dir}")
        return

    # Keep local validation and even CPU production runs to one compute thread.
    # These are set before the lazy torch/dataset/evaluation imports.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    splits = build_splits(cfg, cfg.weeks_per_season)
    output_dir = args.output_dir or Path(cfg.data.logs_folder) / "ano_train_size" / datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = output_dir.resolve()
    manifest = _prepare_output(output_dir, cfg, splits, dry_run=args.dry_run)
    if args.dry_run:
        for size, split in splits.items():
            print(f"{size}wps: train={len(split['train_dates'])}h, validation={len(split['validation_dates'])}h, test={len(split['test_dates'])}h")
        print(f"Dry-run manifests saved to {output_dir}; no weather data were opened.")
        return
    try:
        run_experiment(cfg, output_dir, splits)
        plot_results(output_dir, cfg.weeks_per_season)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write_json(output_dir / "run_manifest.json", manifest)
        raise
    manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat())
    _write_json(output_dir / "run_manifest.json", manifest)
    print(f"Seasonal ANO learning curves saved to {output_dir}")


if __name__ == "__main__":
    main()
