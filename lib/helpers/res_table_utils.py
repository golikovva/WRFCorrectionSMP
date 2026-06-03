from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


def _to_1d_float_array(x) -> np.ndarray:
    """Convert metric value to a flat float array."""
    arr = np.asarray(x, dtype=float).squeeze()
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.ravel()


def build_metrics_dataframe(
    res_dict: Mapping[str, object],
    models: Sequence[str],
    metrics: Sequence[str],
    target: str = "era",
    key_template: str = "{model}_{target}_{metric}",
    lead_labels: Sequence[str] | None = None,
    model_display_names: Mapping[str, str] | None = None,
    missing_as_nan: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Build a DataFrame from a metrics dictionary.

    Parameters
    ----------
    res_dict
        Dictionary like:
        {
            'RoPEUNet_era_mse': array([5.35, 5.39, 8.13]),
            'RoPEUNet_era_mae': array([1.61, 1.61, 2.01]),
            ...
        }
    models
        Model names to extract.
    metrics
        Metric names to extract, e.g. ['mse', 'mae'].
    target
        Target name used in the key.
    key_template
        Template for lookup key in res_dict.
    lead_labels
        Optional names for array positions, e.g. ['lead1', 'lead2', 'lead3'].
        If None, columns will be named metric_1, metric_2, ...
    model_display_names
        Optional mapping for prettier row names.
    missing_as_nan
        If True, missing keys are filled with NaN.
        If False, missing keys raise KeyError.

    Returns
    -------
    df
        DataFrame with models as rows and metrics/leads as columns.
    column_to_metric
        Mapping from column name back to metric name.
    """
    rows: list[dict[str, float]] = []
    row_names: list[str] = []
    column_order: list[str] = []
    column_to_metric: dict[str, str] = {}

    for model in models:
        row: dict[str, float] = {}

        for metric in metrics:
            key = key_template.format(model=model, target=target, metric=metric)

            if key not in res_dict:
                if missing_as_nan:
                    values = np.array([np.nan], dtype=float)
                else:
                    raise KeyError(f"Missing key in res_dict: {key}")
            else:
                values = _to_1d_float_array(res_dict[key])

            if len(values) == 1:
                col = metric
                row[col] = float(values[0])
                if col not in column_order:
                    column_order.append(col)
                column_to_metric[col] = metric
            else:
                if lead_labels is not None and len(lead_labels) != len(values):
                    raise ValueError(
                        f"lead_labels has length {len(lead_labels)}, "
                        f"but metric '{key}' has length {len(values)}"
                    )

                labels = lead_labels if lead_labels is not None else [str(i + 1) for i in range(len(values))]
                for lab, val in zip(labels, values):
                    col = f"{metric}_{lab}"
                    row[col] = float(val)
                    if col not in column_order:
                        column_order.append(col)
                    column_to_metric[col] = metric

        rows.append(row)
        row_names.append(model_display_names.get(model, model) if model_display_names else model)

    df = pd.DataFrame(rows, index=row_names)

    # Keep columns in the order encountered
    ordered_cols = [c for c in column_order if c in df.columns]
    df = df.reindex(columns=ordered_cols)

    return df, column_to_metric


def save_colored_table_image(
    df: pd.DataFrame,
    image_path: str | Path,
    column_to_metric: Mapping[str, str] | None = None,
    higher_is_better_metrics: Iterable[str] = (),
    precision: int = 4,
    cmap_low_better: str = "RdYlGn_r",
    cmap_high_better: str = "RdYlGn",
    dpi: int = 220,
    title: str | None = None,
) -> None:
    """
    Save a PNG image of a DataFrame as a colored table.

    Coloring is done column-wise:
    - for lower-is-better metrics: lower = greener, higher = redder
    - for higher-is-better metrics: higher = greener, lower = redder
    """
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    higher_is_better_metrics = set(higher_is_better_metrics)
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    display_df = numeric_df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].map(
            lambda x: "—" if pd.isna(x) else f"{x:.{precision}f}"
        )

    nrows, ncols = display_df.shape
    fig_w = max(8, 1.35 * (ncols + 1.5))
    fig_h = max(1.8, 0.55 * (nrows + 2))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        rowLabels=list(display_df.index),
        colLabels=list(display_df.columns),
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Header styling
    for c in range(ncols):
        cell = table[(0, c)]
        cell.set_facecolor("#D9D9D9")
        cell.set_text_props(weight="bold")

    # Row label styling
    for r in range(1, nrows + 1):
        if (r, -1) in table.get_celld():
            cell = table[(r, -1)]
            cell.set_facecolor("#EFEFEF")
            cell.set_text_props(weight="bold")

    # Color metric cells column-wise
    for c, col in enumerate(numeric_df.columns):
        vals = numeric_df[col].to_numpy(dtype=float)
        finite_mask = np.isfinite(vals)

        if not finite_mask.any():
            continue

        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        if np.isclose(vmin, vmax):
            # identical values -> neutral coloring
            for r in range(nrows):
                table[(r + 1, c)].set_facecolor("#FFFFFF")
            continue

        metric_name = column_to_metric[col] if column_to_metric is not None else col
        lower_is_better = metric_name not in higher_is_better_metrics
        cmap = cm.get_cmap(cmap_low_better if lower_is_better else cmap_high_better)

        # Best row index for bold formatting
        best_row = np.nanargmin(vals) if lower_is_better else np.nanargmax(vals)

        for r, val in enumerate(vals):
            cell = table[(r + 1, c)]

            if not np.isfinite(val):
                cell.set_facecolor("#F5F5F5")
                continue

            t = (val - vmin) / (vmax - vmin)  # 0..1
            rgba = cmap(float(t))
            cell.set_facecolor(rgba)

            if r == best_row:
                cell.get_text().set_weight("bold")

    if title:
        ax.set_title(title, fontsize=12, pad=12)

    plt.tight_layout()
    plt.savefig(image_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def export_metrics_table(
    res_dict: Mapping[str, object],
    models: Sequence[str],
    metrics: Sequence[str],
    target: str = "era",
    csv_path: str | Path = "metrics_table.csv",
    image_path: str | Path = "metrics_table.png",
    lead_labels: Sequence[str] | None = None,
    key_template: str = "{model}_{target}_{metric}",
    model_display_names: Mapping[str, str] | None = None,
    higher_is_better_metrics: Iterable[str] = (),
    precision: int = 4,
    missing_as_nan: bool = True,
    image_title: str | None = None,
) -> pd.DataFrame:
    """
    Main convenience function:
    1) build DataFrame
    2) save CSV
    3) save colored table image
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df, column_to_metric = build_metrics_dataframe(
        res_dict=res_dict,
        models=models,
        metrics=metrics,
        target=target,
        key_template=key_template,
        lead_labels=lead_labels,
        model_display_names=model_display_names,
        missing_as_nan=missing_as_nan,
    )

    df.to_csv(csv_path, index=True, index_label="model")

    save_colored_table_image(
        df=df,
        image_path=image_path,
        column_to_metric=column_to_metric,
        higher_is_better_metrics=higher_is_better_metrics,
        precision=precision,
        title=image_title,
    )

    return df