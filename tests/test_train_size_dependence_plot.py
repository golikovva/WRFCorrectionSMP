from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from experiments.compare_models.plot_train_size_dependence import (
    collect_train_size_metrics,
    plot_error_vs_train_size,
)


def _write_metrics(root: Path, nwps: int, model_mae: float, orig_mae: float) -> None:
    experiment = root / f"stage_train_from_scratch_{nwps}wps"
    experiment.mkdir()
    pd.DataFrame(
        {
            "model": ["RoPEUNet", "orig"],
            "mae_t2": [model_mae, orig_mae],
            "mse_t2": [model_mae**2, orig_mae**2],
        }
    ).to_csv(experiment / "era_metrics_table.csv", index=False)


def test_plot_error_vs_train_size_collects_sorts_and_saves(tmp_path: Path) -> None:
    _write_metrics(tmp_path, 3, 2.0, 4.0)
    _write_metrics(tmp_path, 1, 3.0, 5.0)
    output = tmp_path / "plots" / "mae_t2.png"

    figure, axes, summary = plot_error_vs_train_size(
        tmp_path,
        "MAE",
        "temperature",
        output_path=output,
    )

    assert output.is_file()
    assert sorted(summary["nwps"].unique().tolist()) == [1, 3]
    model_values = summary[summary["model"] == "RoPEUNet"].sort_values("nwps")
    assert model_values["mean"].tolist() == [3.0, 2.0]
    assert axes.get_xticks().tolist() == [1, 3]
    figure.clear()


def test_collect_train_size_metrics_reports_missing_metric(tmp_path: Path) -> None:
    _write_metrics(tmp_path, 1, 3.0, 5.0)

    with pytest.raises(ValueError, match="mae_u10"):
        collect_train_size_metrics(tmp_path, "mae", "u10")
