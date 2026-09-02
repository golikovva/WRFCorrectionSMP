from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.helpers.aggregators import RegionalTemporalAggregator
from lib.helpers.metrics import MetricMeta, NamedDictMetric
from lib.helpers.region_masks import build_region_masks, grid_lat_lon_arrays
from lib.pipeline.test_metrics import build_test_metrics


def test_regional_temporal_aggregator_keeps_sum_count_by_expanded_dates():
    masks = {"all": np.array([[True, True]])}
    field = torch.tensor(
        [
            [[[[1.0, 3.0]]]],
            [[[[5.0, float("nan")]]]],
        ]
    )  # T, B, C, H, W
    dates = np.array([np.datetime64("2020-01-01T00")])

    agg = RegionalTemporalAggregator(masks, date_step=np.timedelta64(1, "h"))
    acc = agg.init_accumulator(field.shape[2:])
    agg.accumulate(acc, field, dates)
    result = agg.finalize(acc)

    first = np.datetime64("2020-01-01T00", "ns")
    second = np.datetime64("2020-01-01T01", "ns")
    assert set(result["all"]) == {first, second}
    assert result["all"][first]["value"].tolist() == [2.0]
    assert result["all"][first]["valid_count"].tolist() == [2]
    assert result["all"][second]["value"].tolist() == [5.0]
    assert result["all"][second]["valid_count"].tolist() == [1]


@pytest.mark.parametrize(
    ("temporal_resolution", "dates", "expected_date"),
    [
        ("6h", ("2020-01-01T01", "2020-01-01T05"), "2020-01-01T00"),
        ("D", ("2020-01-01T01", "2020-01-01T23"), "2020-01-01"),
        ("M", ("2020-01-01", "2020-01-31"), "2020-01"),
        ("Y", ("2020-01-01", "2020-12-31"), "2020"),
    ],
)
def test_regional_temporal_aggregator_sums_entries_in_temporal_bucket(
    temporal_resolution,
    dates,
    expected_date,
):
    masks = {"all": np.array([[True, True]])}
    agg = RegionalTemporalAggregator(masks, temporal_resolution=temporal_resolution)
    acc = agg.init_accumulator((1, 1, 2))

    agg.accumulate(acc, torch.tensor([[[1.0, 3.0]]]), np.datetime64(dates[0]))
    agg.accumulate(acc, torch.tensor([[[5.0, float("nan")]]]), np.datetime64(dates[1]))

    expected_key = np.datetime64(expected_date).astype("datetime64[ns]")
    assert set(acc["all"]) == {expected_key}
    assert acc["all"][expected_key]["sum"].tolist() == [9.0]
    assert acc["all"][expected_key]["count"].tolist() == [3]

    result = agg.finalize(acc)
    assert result["all"][expected_key]["value"].tolist() == [3.0]
    assert result["all"][expected_key]["valid_count"].tolist() == [3]


def test_regional_temporal_aggregator_can_reduce_time_after_collection():
    masks = {"all": np.array([[True, True]])}
    field = torch.tensor(
        [
            [[[[1.0, 3.0]]]],
            [[[[5.0, float("nan")]]]],
        ]
    )
    dates = np.array([np.datetime64("2020-01-01T00")])

    temporal = RegionalTemporalAggregator(masks, date_step=np.timedelta64(1, "h"))
    temporal_acc = temporal.init_accumulator(field.shape[2:])
    temporal.accumulate(temporal_acc, field, dates)

    reduced_from_temporal = RegionalTemporalAggregator.finalize_reduced_time(temporal_acc)

    assert reduced_from_temporal["all"]["value"].tolist() == [3.0]
    assert reduced_from_temporal["all"]["valid_count"].tolist() == [3]


def test_build_region_masks_bbox_intersection_and_difference_on_2d_grid():
    lon, lat = np.meshgrid(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))
    grid = {"latitude": lat, "longitude": lon}
    cfg = {
        "region_sources": {
            "left": {"type": "bbox", "bbox": [-0.5, 1.5, -0.5, 0.5]},
            "north": {"type": "bbox", "bbox": [0.5, 1.5, -0.5, 2.5]},
        },
        "masks": [
            {"name": "full_domain", "op": "all"},
            {"name": "left_only", "op": "region", "region": "left"},
            {"name": "left_north", "op": "intersection", "regions": ["left", "north"]},
            {"name": "without_left", "op": "difference", "base": "all", "subtract": "left"},
        ],
    }

    masks = build_region_masks(grid, cfg)

    np.testing.assert_array_equal(masks["full_domain"], np.ones((2, 3), dtype=bool))
    np.testing.assert_array_equal(masks["left_only"], np.array([[True, False, False], [True, False, False]]))
    np.testing.assert_array_equal(masks["left_north"], np.array([[False, False, False], [True, False, False]]))
    np.testing.assert_array_equal(masks["without_left"], np.array([[False, True, True], [False, True, True]]))


def test_build_region_masks_polygon_on_2d_grid():
    pytest.importorskip("geopandas")
    lon, lat = np.meshgrid(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))
    grid = {"latitude": lat, "longitude": lon}
    cfg = {
        "region_sources": {
            "box": {
                "type": "polygon",
                "coords_order": "latlon",
                "coords": [[-0.5, -0.5], [-0.5, 1.5], [1.5, 1.5], [1.5, -0.5]],
            }
        },
        "masks": [{"name": "box", "op": "region", "region": "box"}],
    }

    masks = build_region_masks(grid, cfg)

    np.testing.assert_array_equal(masks["box"], np.array([[True, True, False], [True, True, False]]))


def test_point_grid_1d_lat_lon_are_paired_not_meshed():
    grid = {
        "latitude": np.array([70.0, 70.0, 72.0]),
        "longitude": np.array([30.0, 40.0, 30.0]),
    }
    lat, lon = grid_lat_lon_arrays(grid)
    cfg = {
        "region_sources": {
            "first_point": {"type": "bbox", "bbox": [69.5, 70.5, 29.5, 30.5]},
        },
        "masks": [{"name": "first_point", "op": "region", "region": "first_point"}],
    }

    masks = build_region_masks(grid, cfg)

    assert lat.shape == (3,)
    assert lon.shape == (3,)
    np.testing.assert_array_equal(masks["first_point"], np.array([True, False, False]))


def test_named_dict_metric_stores_explicit_meta_and_sources():
    metric = NamedDictMetric(
        torch.nn.Identity(),
        ["prediction"],
        meta=MetricMeta(grid="wrf", model="m1", target="era", metric="mae"),
    )

    assert metric.meta.grid == "wrf"
    assert metric.meta.model == "m1"
    assert metric.meta.sources == ("prediction",)


def test_regional_aggregator_supports_metrics_by_meta_grid_and_kind():
    grid = {
        "latitude": np.array([[70.0, 70.0], [71.0, 71.0]]),
        "longitude": np.array([[30.0, 31.0], [30.0, 31.0]]),
    }
    cfg = {
        "region_sources": {
            "all_box": {"type": "bbox", "bbox": [69.0, 72.0, 29.0, 32.0]},
        },
        "masks": [{"name": "all_box", "op": "region", "region": "all_box"}],
    }
    agg = RegionalTemporalAggregator(grids={"wrf": grid}, regional_config=cfg)

    field_metric = NamedDictMetric(torch.nn.Identity(), ["x"], meta=MetricMeta(grid="wrf"))
    spectrum_metric = NamedDictMetric(
        torch.nn.Identity(),
        ["x"],
        meta=MetricMeta(grid=None, kind="spectrum"),
    )
    global_metric = NamedDictMetric(
        torch.nn.Identity(),
        ["x"],
        meta=MetricMeta(grid="wrf", kind="global"),
    )

    assert agg.supports(field_metric)
    assert not agg.supports(spectrum_metric)
    assert not agg.supports(global_metric)


def test_build_test_metrics_sets_grid_metadata_without_name_routing():
    metric_fns = {
        "mesoscale_loss": torch.nn.Identity(),
        "mse": torch.nn.Identity(),
        "mae": torch.nn.Identity(),
        "wind_err_norm": torch.nn.Identity(),
        "wind_angle_norm": torch.nn.Identity(),
        "wind_norm_diff": torch.nn.Identity(),
        "ssim_111": torch.nn.Identity(),
        "ssim_211": torch.nn.Identity(),
        "ssim_011": torch.nn.Identity(),
        "identity": torch.nn.Identity(),
        "mc_acc": torch.nn.Identity(),
        "st_dir_mae": torch.nn.Identity(),
        "era_lat_weighted_mse": torch.nn.Identity(),
        "era_lat_weighted_mae": torch.nn.Identity(),
        "scatter_lat_weighted_mse": torch.nn.Identity(),
        "scatter_lat_weighted_mae": torch.nn.Identity(),
    }

    metrics = build_test_metrics(
        ["model_a"],
        metric_fns,
        include_stations=True,
        stations_has_dir=True,
        include_scatter=True,
    )

    assert metrics["model_a_era_mae"].meta.grid == "wrf"
    assert metrics["model_a_mean_era_mae"].meta.grid == "era"
    assert metrics["model_a_stations_mae"].meta.grid == "stations"
    assert metrics["model_a_scatter_mae"].meta.grid == "scatter"
    assert metrics["model_a_mean_era_lat_weighted_mse"].meta.grid == "era"
    assert metrics["model_a_mean_era_lat_weighted_mse"].meta.kind == "global"
    assert metrics["model_a_mean_era_lat_weighted_mae"].meta.grid == "era"
    assert metrics["model_a_mean_era_lat_weighted_mae"].meta.kind == "global"
    assert metrics["model_a_scatter_lat_weighted_mse"].meta.grid == "scatter"
    assert metrics["model_a_scatter_lat_weighted_mse"].meta.kind == "global"
    assert metrics["model_a_scatter_lat_weighted_mae"].meta.grid == "scatter"
    assert metrics["model_a_scatter_lat_weighted_mae"].meta.kind == "global"
    assert metrics["model_a_spectrum"].meta.kind == "spectrum"
    assert metrics["model_a_ssim_custom"].meta.kind == "similarity"


def test_build_test_metrics_can_skip_spectrum_metrics():
    metric_fns = {
        "mesoscale_loss": torch.nn.Identity(),
        "mse": torch.nn.Identity(),
        "mae": torch.nn.Identity(),
        "wind_err_norm": torch.nn.Identity(),
        "wind_angle_norm": torch.nn.Identity(),
        "wind_norm_diff": torch.nn.Identity(),
        "ssim_111": torch.nn.Identity(),
        "ssim_211": torch.nn.Identity(),
        "ssim_011": torch.nn.Identity(),
        "identity": torch.nn.Identity(),
    }

    metrics = build_test_metrics(
        ["model_a"],
        metric_fns,
        include_spectrum=False,
    )

    assert "model_a_era_mae" in metrics
    assert "model_a_spectrum" not in metrics
    assert "wrf_spectrum" not in metrics
    assert "era_spectrum" not in metrics


def test_regional_metrics_config_has_expected_shape():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "train_test.yaml"

    cfg = yaml.safe_load(config_path.read_text())
    regional_cfg = cfg["test_config"]["regional_metrics"]

    assert isinstance(regional_cfg["enabled"], bool)
    assert isinstance(regional_cfg["temporal_resolution"], str)
    assert isinstance(regional_cfg["region_sources"], dict)
    assert isinstance(regional_cfg["masks"], list)
