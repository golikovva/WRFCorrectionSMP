from __future__ import annotations

from pathlib import Path
from typing import Union, Literal
import tempfile
import zipfile
import urllib.request

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from matplotlib.path import Path as MplPath
from pyproj import CRS, Transformer

from libs.validation.visualization import lat_lon_from_grid


NE_URLS = {
    "land": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_land.zip",
    "ocean": "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_ocean.zip",
}

PathLike = Union[str, Path]


def get_default_arctic_crs(lon_0: float = 0.0) -> CRS:
    """
    A robust planar CRS for Arctic domains that include the pole / antimeridian.
    """
    return CRS.from_proj4(
        f"+proj=laea +lat_0=90 +lon_0={lon_0} +datum=WGS84 +units=m +no_defs"
    )


def circular_mean_deg(lon_deg: np.ndarray) -> float:
    """
    Circular mean of longitudes in degrees, robust to antimeridian crossing.
    Returns value in [-180, 180).
    """
    lon_deg = np.asarray(lon_deg, dtype=float)
    lon_deg = lon_deg[np.isfinite(lon_deg)]
    if lon_deg.size == 0:
        return 0.0

    ang = np.deg2rad(lon_deg)
    mean_ang = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
    lon0 = np.rad2deg(mean_ang)
    lon0 = ((lon0 + 180.0) % 360.0) - 180.0
    return float(lon0)


def _grid_lat_lon_2d(grid):
    """
    Uses lat_lon_from_grid(grid) helper and returns 2D lat/lon arrays.
    """
    lat, lon = lat_lon_from_grid(grid)

    if lat is None or lon is None:
        raise ValueError("Could not extract latitude/longitude from grid.")

    lat = np.asarray(lat).squeeze()
    lon = np.asarray(lon).squeeze()

    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    elif lat.ndim == 2 and lon.ndim == 2:
        if lat.shape != lon.shape:
            raise ValueError(
                f"2D lat/lon shapes do not match: lat={lat.shape}, lon={lon.shape}"
            )
        lat2d, lon2d = lat, lon
    else:
        raise ValueError(
            "lat/lon must be either both 1D or both 2D after squeeze(). "
            f"Got lat.ndim={lat.ndim}, lon.ndim={lon.ndim}"
        )

    return lat2d, lon2d


def _coords_to_lat_lon(coords: np.ndarray, *, order: Literal["latlon", "lonlat"] = "latlon"):
    """
    Convert an array of shape (..., 2) into separate lat/lon arrays.

    Parameters
    ----------
    coords : np.ndarray
        Shape (..., 2)
    order : "latlon" or "lonlat"
        Interpretation of the last axis.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim < 2 or coords.shape[-1] != 2:
        raise ValueError(f"Expected coords with shape (..., 2), got {coords.shape}")

    if order == "latlon":
        lat = coords[..., 0]
        lon = coords[..., 1]
    elif order == "lonlat":
        lon = coords[..., 0]
        lat = coords[..., 1]
    else:
        raise ValueError(f"Unsupported order={order!r}")

    return np.asarray(lat), np.asarray(lon)


def _project_lonlat_to_xy(
    lon: np.ndarray,
    lat: np.ndarray,
    dst_crs: CRS,
    src_crs: CRS = CRS.from_epsg(4326),
):
    """
    Project lon/lat arrays to x/y.
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def _mask_from_single_polygon_xy(
    polygon: Polygon,
    points_xy: np.ndarray,
    *,
    boundary_radius: float = 1e-9,
) -> np.ndarray:
    """
    Rasterize one projected polygon onto projected point locations.
    """
    exterior = np.asarray(polygon.exterior.coords)
    mask = MplPath(exterior[:, :2]).contains_points(points_xy, radius=boundary_radius)

    for interior in polygon.interiors:
        hole = np.asarray(interior.coords)
        hole_mask = MplPath(hole[:, :2]).contains_points(points_xy, radius=boundary_radius)
        mask &= ~hole_mask

    return mask


def _geometry_to_mask_xy(
    geometry,
    x: np.ndarray,
    y: np.ndarray,
    *,
    boundary_radius: float = 1e-9,
) -> np.ndarray:
    """
    Convert a projected shapely geometry into a mask on projected point locations.

    Works for:
      - 2D grids (x.shape == y.shape == (H, W))
      - 1D point clouds (x.shape == y.shape == (N,))
      - any arbitrary common shape
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape, got {x.shape} vs {y.shape}")

    points_xy = np.column_stack([x.ravel(), y.ravel()])
    flat_mask = np.zeros(points_xy.shape[0], dtype=bool)

    def _accumulate(geom):
        nonlocal flat_mask

        if geom.is_empty:
            return

        if isinstance(geom, Polygon):
            flat_mask |= _mask_from_single_polygon_xy(
                geom, points_xy, boundary_radius=boundary_radius
            )
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                flat_mask |= _mask_from_single_polygon_xy(
                    poly, points_xy, boundary_radius=boundary_radius
                )
        elif isinstance(geom, GeometryCollection):
            for subgeom in geom.geoms:
                _accumulate(subgeom)

    _accumulate(geometry)
    return flat_mask.reshape(x.shape)


def make_mask_from_shapefile(
    grid,
    mask_shapefile: PathLike,
    *,
    work_crs: CRS | str | None = None,
    inside_value: float = 1.0,
    outside_value: float = 0.0,
    invert: bool = False,
    dtype=np.float32,
    boundary_radius: float = 1e-9,
    shapefile_crs_if_missing: CRS | str | None = None,
) -> np.ndarray:
    """
    Create a gridded mask from a shapefile on the given grid.
    """
    lat2d, lon2d = _grid_lat_lon_2d(grid)

    gdf = gpd.read_file(mask_shapefile)
    if len(gdf) == 0:
        raise ValueError(f"No geometries found in shapefile: {mask_shapefile}")

    if gdf.crs is None:
        if shapefile_crs_if_missing is None:
            raise ValueError(
                f"Shapefile {mask_shapefile} has no CRS. "
                "Pass shapefile_crs_if_missing=... if you know it."
            )
        gdf = gdf.set_crs(shapefile_crs_if_missing)

    if work_crs is None:
        shp_crs = CRS.from_user_input(gdf.crs)
        work_crs = shp_crs if shp_crs.is_projected else get_default_arctic_crs()

    work_crs = CRS.from_user_input(work_crs)
    gdf = gdf.to_crs(work_crs)

    x, y = _project_lonlat_to_xy(lon2d, lat2d, work_crs)

    geometry = unary_union(gdf.geometry.values)
    inside = _geometry_to_mask_xy(
        geometry,
        x,
        y,
        boundary_radius=boundary_radius,
    )

    if invert:
        inside = ~inside

    return np.where(inside, inside_value, outside_value).astype(dtype)


def points_in_shapefile_lonlat(
    points: np.ndarray,
    mask_shapefile: PathLike,
    *,
    order: Literal["latlon", "lonlat"] = "latlon",
    work_crs: CRS | str | None = None,
    boundary_radius: float = 1e-9,
    shapefile_crs_if_missing: CRS | str | None = None,
) -> np.ndarray:
    """
    Check whether lon/lat points are inside a shapefile geometry.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 2) or (..., 2).
    order : "latlon" or "lonlat"
        Column order in `points`.
    Returns
    -------
    inside : np.ndarray
        Boolean array of shape points.shape[:-1]
    """
    lat, lon = _coords_to_lat_lon(points, order=order)

    gdf = gpd.read_file(mask_shapefile)
    if len(gdf) == 0:
        raise ValueError(f"No geometries found in shapefile: {mask_shapefile}")

    if gdf.crs is None:
        if shapefile_crs_if_missing is None:
            raise ValueError(
                f"Shapefile {mask_shapefile} has no CRS. "
                "Pass shapefile_crs_if_missing=... if you know it."
            )
        gdf = gdf.set_crs(shapefile_crs_if_missing)

    if work_crs is None:
        shp_crs = CRS.from_user_input(gdf.crs)
        if shp_crs.is_projected:
            work_crs = shp_crs
        else:
            work_crs = get_default_arctic_crs(lon_0=circular_mean_deg(lon))

    work_crs = CRS.from_user_input(work_crs)
    gdf = gdf.to_crs(work_crs)

    x, y = _project_lonlat_to_xy(lon, lat, work_crs)
    geometry = unary_union(gdf.geometry.values)

    return _geometry_to_mask_xy(
        geometry,
        x,
        y,
        boundary_radius=boundary_radius,
    )


def points_in_polygon_lonlat(
    points: np.ndarray,
    polygon_coords: np.ndarray,
    *,
    points_order: Literal["latlon", "lonlat"] = "latlon",
    polygon_order: Literal["latlon", "lonlat"] = "latlon",
    work_crs: CRS | str | None = None,
    boundary_radius: float = 1e-9,
    fix_geometry: bool = True,
) -> np.ndarray:
    """
    Check whether lon/lat points are inside a single polygon given by coordinates.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 2) or (..., 2)
    polygon_coords : np.ndarray
        Shape (M, 2), polygon boundary vertices
    points_order, polygon_order : "latlon" or "lonlat"
        Input coordinate order
    work_crs : projected CRS or None
        If None, uses Arctic LAEA centered near the data longitude.
    Returns
    -------
    inside : np.ndarray
        Boolean array of shape points.shape[:-1]
    """
    pts_lat, pts_lon = _coords_to_lat_lon(points, order=points_order)
    poly_lat, poly_lon = _coords_to_lat_lon(polygon_coords, order=polygon_order)

    if work_crs is None:
        lon0 = circular_mean_deg(np.concatenate([pts_lon.ravel(), poly_lon.ravel()]))
        work_crs = get_default_arctic_crs(lon_0=lon0)

    work_crs = CRS.from_user_input(work_crs)

    poly_x, poly_y = _project_lonlat_to_xy(poly_lon, poly_lat, work_crs)
    ring_xy = np.column_stack([poly_x.ravel(), poly_y.ravel()])

    if ring_xy.shape[0] < 3:
        raise ValueError("polygon_coords must contain at least 3 vertices")

    if not np.allclose(ring_xy[0], ring_xy[-1]):
        ring_xy = np.vstack([ring_xy, ring_xy[0]])

    polygon = Polygon(ring_xy)
    if fix_geometry:
        polygon = polygon.buffer(0)

    if polygon.is_empty:
        raise ValueError("Projected polygon is empty or invalid.")

    pts_x, pts_y = _project_lonlat_to_xy(pts_lon, pts_lat, work_crs)

    return _geometry_to_mask_xy(
        polygon,
        pts_x,
        pts_y,
        boundary_radius=boundary_radius,
    )