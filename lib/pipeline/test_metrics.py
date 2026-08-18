from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from lib.helpers.metrics import MetricMeta, NamedDictMetric


ERA_CHANNELS = ("u10", "v10", "t2")
WIND_CHANNEL = ("w10",)
WIND_TEMP_CHANNELS = ("w10", "t2")
WIND_UV_CHANNELS = ("u10", "v10")
DIR_CHANNEL = ("dir",)


def _add_metric(
    metrics_dict: dict[str, NamedDictMetric],
    key: str,
    fn: Any,
    names: list[str],
    *,
    grid: str | None,
    model: str | None,
    target: str | None,
    metric: str,
    kind: str = "field",
    channels: tuple[str, ...] | None = None,
) -> None:
    metrics_dict[key] = NamedDictMetric(
        fn,
        names,
        meta=MetricMeta(
            grid=grid,
            model=model,
            target=target,
            metric=metric,
            kind=kind,
            channels=channels,
            sources=tuple(names),
        ),
    )


def build_test_metrics(
    model_names: Iterable[str],
    metric_fns: Mapping[str, Any],
    *,
    include_stations: bool = False,
    stations_has_dir: bool = False,
    include_scatter: bool = False,
    include_spectrum: bool = True,
) -> dict[str, NamedDictMetric]:
    metrics_dict: dict[str, NamedDictMetric] = {}
    model_names = list(model_names)

    for model_name in model_names:
        _add_metric(
            metrics_dict,
            f"{model_name}_mesoscale_loss",
            metric_fns["mesoscale_loss"],
            ["wrf", model_name],
            grid="wrf",
            model=model_name,
            target="mesoscale",
            metric="mesoscale_loss",
            channels=ERA_CHANNELS,
        )
        for metric_name, fn_key in [
            ("mse", "mse"),
            ("mae", "mae"),
            ("err_norm", "wind_err_norm"),
            ("angle_norm", "wind_angle_norm"),
            ("norm_dif", "wind_norm_diff"),
        ]:
            _add_metric(
                metrics_dict,
                f"{model_name}_era_{metric_name}",
                metric_fns[fn_key],
                [model_name, "era_up"],
                grid="wrf",
                model=model_name,
                target="era",
                metric=metric_name,
                channels=ERA_CHANNELS,
            )

        mean_era_metrics = [("mse", "mse", "field"), ("mae", "mae", "field")]
        if "era_lat_weighted_mse" in metric_fns:
            mean_era_metrics.append(("lat_weighted_mse", "era_lat_weighted_mse", "global"))
        if "era_lat_weighted_mae" in metric_fns:
            mean_era_metrics.append(("lat_weighted_mae", "era_lat_weighted_mae", "global"))
        for metric_name, fn_key, kind in mean_era_metrics:
            _add_metric(
                metrics_dict,
                f"{model_name}_mean_era_{metric_name}",
                metric_fns[fn_key],
                [f"{model_name}_meaned", "era"],
                grid="era",
                model=model_name,
                target="era",
                metric=metric_name,
                kind=kind,
                channels=ERA_CHANNELS,
            )

        for key_suffix, fn_key, names in [
            ("ssim_custom_211", "ssim_211", [model_name, "wrf", "era_up"]),
            ("ssim_custom", "ssim_111", [model_name, "wrf", "era_up"]),
            ("ssim_wrf", "ssim_111", [model_name, "wrf", "wrf"]),
            ("ssim_wrf_011", "ssim_011", [model_name, "wrf", "wrf"]),
            ("ssim_era", "ssim_111", [model_name, "era_up", "era_up"]),
        ]:
            _add_metric(
                metrics_dict,
                f"{model_name}_{key_suffix}",
                metric_fns[fn_key],
                names,
                grid=None,
                model=model_name,
                target="ssim",
                metric=key_suffix,
                kind="similarity",
            )

        if include_spectrum:
            _add_metric(
                metrics_dict,
                f"{model_name}_spectrum",
                metric_fns["identity"],
                [f"{model_name}_spectrum"],
                grid=None,
                model=model_name,
                target="spectrum",
                metric="spectrum",
                kind="spectrum",
            )

    for metric_name, fn_key in [
        ("mse", "mse"),
        ("mae", "mae"),
        ("err_norm", "wind_err_norm"),
        ("angle_norm", "wind_angle_norm"),
        ("norm_dif", "wind_norm_diff"),
    ]:
        _add_metric(
            metrics_dict,
            f"orig_era_{metric_name}",
            metric_fns[fn_key],
            ["wrf", "era_up"],
            grid="wrf",
            model="orig",
            target="era",
            metric=metric_name,
            channels=ERA_CHANNELS,
        )

    mean_era_metrics = [("mse", "mse", "field"), ("mae", "mae", "field")]
    if "era_lat_weighted_mse" in metric_fns:
        mean_era_metrics.append(("lat_weighted_mse", "era_lat_weighted_mse", "global"))
    if "era_lat_weighted_mae" in metric_fns:
        mean_era_metrics.append(("lat_weighted_mae", "era_lat_weighted_mae", "global"))
    for metric_name, fn_key, kind in mean_era_metrics:
        _add_metric(
            metrics_dict,
            f"mean_orig_era_{metric_name}",
            metric_fns[fn_key],
            ["wrf_meaned", "era"],
            grid="era",
            model="orig",
            target="era",
            metric=metric_name,
            kind=kind,
            channels=ERA_CHANNELS,
        )

    for key, fn_key, names in [
        ("orig_ssim_era", "ssim_111", ["wrf", "era_up", "era_up"]),
        ("orig_ssim_custom", "ssim_111", ["wrf", "wrf", "era_up"]),
        ("orig_ssim_custom_211", "ssim_211", ["wrf", "wrf", "era_up"]),
    ]:
        _add_metric(
            metrics_dict,
            key,
            metric_fns[fn_key],
            names,
            grid=None,
            model="orig",
            target="ssim",
            metric=key.replace("orig_", ""),
            kind="similarity",
        )

    if include_spectrum:
        for model_name in ["wrf", "era"]:
            _add_metric(
                metrics_dict,
                f"{model_name}_spectrum",
                metric_fns["identity"],
                [f"{model_name}_spectrum"],
                grid=None,
                model=model_name,
                target="spectrum",
                metric="spectrum",
                kind="spectrum",
            )

    if include_stations:
        _add_station_metrics(metrics_dict, model_names, metric_fns, stations_has_dir=stations_has_dir)

    if include_scatter:
        _add_scatter_metrics(metrics_dict, model_names, metric_fns)

    return metrics_dict


def _add_station_metrics(
    metrics_dict: dict[str, NamedDictMetric],
    model_names: list[str],
    metric_fns: Mapping[str, Any],
    *,
    stations_has_dir: bool,
) -> None:
    if stations_has_dir:
        for model_name in model_names:
            _add_metric(
                metrics_dict,
                f"{model_name}_stations_accuracy",
                metric_fns["mc_acc"],
                [f"{model_name}_stations_dir", "stations_dir"],
                grid="stations",
                model=model_name,
                target="stations",
                metric="accuracy",
                channels=DIR_CHANNEL,
            )
            _add_metric(
                metrics_dict,
                f"{model_name}_stations_dir_mae",
                metric_fns["st_dir_mae"],
                [f"{model_name}_stations_dir", "stations_dir"],
                grid="stations",
                model=model_name,
                target="stations",
                metric="dir_mae",
                channels=DIR_CHANNEL,
            )
        for model_name, source_prefix in [("orig", "wrf"), ("era", "era")]:
            _add_metric(
                metrics_dict,
                f"{model_name}_stations_accuracy",
                metric_fns["mc_acc"],
                [f"{source_prefix}_stations_dir", "stations_dir"],
                grid="stations",
                model=model_name,
                target="stations",
                metric="accuracy",
                channels=DIR_CHANNEL,
            )
            _add_metric(
                metrics_dict,
                f"{model_name}_stations_dir_mae",
                metric_fns["st_dir_mae"],
                [f"{source_prefix}_stations_dir", "stations_dir"],
                grid="stations",
                model=model_name,
                target="stations",
                metric="dir_mae",
                channels=DIR_CHANNEL,
            )

    for model_name in model_names:
        for metric_name, fn_key in [("mse", "mse"), ("mae", "mae")]:
            _add_metric(
                metrics_dict,
                f"{model_name}_stations_{metric_name}",
                metric_fns[fn_key],
                [f"{model_name}_stations_wt", "stations_wt"],
                grid="stations",
                model=model_name,
                target="stations",
                metric=metric_name,
                channels=WIND_TEMP_CHANNELS,
            )

    for model_name, source_prefix in [("orig", "wrf"), ("era", "era")]:
        for metric_name, fn_key in [("mse", "mse"), ("mae", "mae")]:
            _add_metric(
                metrics_dict,
                f"{model_name}_stations_{metric_name}",
                metric_fns[fn_key],
                [f"{source_prefix}_stations_wt", "stations_wt"],
                grid="stations",
                model=model_name,
                target="stations",
                metric=metric_name,
                channels=WIND_TEMP_CHANNELS,
            )


def _add_scatter_metrics(
    metrics_dict: dict[str, NamedDictMetric],
    model_names: list[str],
    metric_fns: Mapping[str, Any],
) -> None:
    scatter_metrics = [
        ("mse", "mse", WIND_UV_CHANNELS, "field"),
        ("mae", "mae", WIND_UV_CHANNELS, "field"),
        ("err_norm", "wind_err_norm", WIND_CHANNEL, "field"),
        ("angle_norm", "wind_angle_norm", WIND_CHANNEL, "field"),
        ("norm_diff", "wind_norm_diff", WIND_CHANNEL, "field"),
    ]
    if "scatter_lat_weighted_mse" in metric_fns:
        scatter_metrics.insert(1, ("lat_weighted_mse", "scatter_lat_weighted_mse", WIND_UV_CHANNELS, "global"))
    if "scatter_lat_weighted_mae" in metric_fns:
        scatter_metrics.insert(2, ("lat_weighted_mae", "scatter_lat_weighted_mae", WIND_UV_CHANNELS, "global"))

    for model_name in model_names:
        for metric_name, fn_key, channels, kind in scatter_metrics:
            _add_metric(
                metrics_dict,
                f"{model_name}_scatter_{metric_name}",
                metric_fns[fn_key],
                [f"{model_name}_scatter", "scatter"],
                grid="scatter",
                model=model_name,
                target="scatter",
                metric=metric_name,
                kind=kind,
                channels=channels,
            )

    for model_name, source_name in [("orig", "wrf_scatter"), ("era", "era_scatter")]:
        for metric_name, fn_key, channels, kind in scatter_metrics:
            _add_metric(
                metrics_dict,
                f"{model_name}_scatter_{metric_name}",
                metric_fns[fn_key],
                [source_name, "scatter"],
                grid="scatter",
                model=model_name,
                target="scatter",
                metric=metric_name,
                kind=kind,
                channels=channels,
            )
