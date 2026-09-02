from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lib.helpers.shapefile_to_mask as stm


def test_module_imports_without_optional_gis_dependencies():
    assert hasattr(stm, "make_mask_from_shapefile")
    assert hasattr(stm, "make_grid_boundary_shapefile")


def test_grid_lat_lon_arrays_meshes_axis_coordinates():
    grid = {
        "latitude": np.array([70.0, 71.0]),
        "longitude": np.array([30.0, 40.0, 50.0]),
    }

    lat, lon = stm.grid_lat_lon_arrays(grid)

    assert lat.shape == (2, 3)
    assert lon.shape == (2, 3)
    np.testing.assert_array_equal(lat[:, 0], np.array([70.0, 71.0]))
    np.testing.assert_array_equal(lon[0], np.array([30.0, 40.0, 50.0]))


def test_grid_lat_lon_arrays_preserves_paired_point_coordinates():
    grid = {
        "latitude": np.array([70.0, 71.0]),
        "longitude": np.array([30.0, 40.0]),
    }

    lat, lon = stm.grid_lat_lon_arrays(grid)

    assert lat.shape == (2,)
    assert lon.shape == (2,)
    np.testing.assert_array_equal(lat, grid["latitude"])
    np.testing.assert_array_equal(lon, grid["longitude"])


def test_outer_boundary_ring_from_grid_is_closed_lonlat_ring():
    lat = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    lon = np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])

    ring = stm._outer_boundary_ring_from_grid(lat, lon)

    assert ring.shape == (7, 2)
    np.testing.assert_array_equal(ring[0], np.array([10.0, 0.0]))
    np.testing.assert_array_equal(ring[-1], ring[0])


@pytest.mark.parametrize(
    ("suffix", "driver"),
    [
        (".shp", "ESRI Shapefile"),
        (".geojson", "GeoJSON"),
        (".json", "GeoJSON"),
        (".gpkg", "GPKG"),
    ],
)
def test_vector_driver_from_path(suffix, driver):
    assert stm._vector_driver_from_path(Path("region").with_suffix(suffix)) == driver


def test_make_grid_boundary_shapefile_writes_when_gis_stack_is_available(tmp_path):
    for package in ("geopandas", "shapely", "pyproj"):
        pytest.importorskip(package)

    lon, lat = np.meshgrid(np.array([30.0, 40.0, 50.0]), np.array([70.0, 71.0]))
    grid = {"latitude": lat, "longitude": lon}
    out_path = tmp_path / "grid_boundary.geojson"

    gdf = stm.make_grid_boundary_shapefile(
        grid,
        out_path,
        attrs={"name": "toy"},
        return_gdf=True,
    )

    assert out_path.exists()
    assert len(gdf) == 1
    assert gdf.iloc[0]["name"] == "toy"
