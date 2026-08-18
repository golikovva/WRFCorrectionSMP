"""Radius graphs and cached edge geometry for sphere convolutions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple
import warnings

import torch
from torch import Tensor

from .sphere_geometry import (
    TangentFrameStrategy,
    build_tangent_frames,
    chord_radius_from_km,
    latlon_to_xyz,
    log_map_sphere_km,
    project_to_frame,
    rotate_tangent_frames,
    transport_matrix_q_to_p,
)


def _limit_neighbors(
    points_xyz: Tensor,
    rows: list[list[int]],
    *,
    max_neighbors: Optional[int],
) -> list[list[int]]:
    if max_neighbors is None:
        return rows

    limited: list[list[int]] = []
    points_cpu = points_xyz.detach().cpu()
    for center, neighbs in enumerate(rows):
        if len(neighbs) <= max_neighbors:
            limited.append(neighbs)
            continue
        neighb_t = torch.as_tensor(neighbs, dtype=torch.long)
        chord_sq = ((points_cpu[neighb_t] - points_cpu[center]) ** 2).sum(dim=-1)
        order = torch.argsort(chord_sq)[:max_neighbors]
        limited.append([int(neighbs[int(i)]) for i in order])
    return limited


def _rows_to_edges(rows: list[list[int]]) -> tuple[Tensor, Tensor]:
    centers = []
    neighbors = []
    for center, row in enumerate(rows):
        centers.extend([center] * len(row))
        neighbors.extend(row)
    return torch.tensor(centers, dtype=torch.long), torch.tensor(neighbors, dtype=torch.long)


def _build_radius_graph_scipy(
    points_xyz: Tensor,
    chord_radius: float,
    *,
    max_neighbors: Optional[int],
) -> tuple[Tensor, Tensor]:
    from scipy.spatial import cKDTree  # type: ignore

    points_np = points_xyz.detach().cpu().numpy()
    tree = cKDTree(points_np)
    rows = tree.query_ball_point(points_np, r=chord_radius)
    rows = [[int(i) for i in row] for row in rows]
    rows = _limit_neighbors(points_xyz, rows, max_neighbors=max_neighbors)
    return _rows_to_edges(rows)


def _build_radius_graph_torch(
    points_xyz: Tensor,
    chord_radius: float,
    *,
    max_neighbors: Optional[int],
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    points = points_xyz.detach().cpu()
    n_points = points.shape[0]
    if n_points > 20000:
        warnings.warn(
            "Falling back to chunked all-pairs radius search for more than 20k "
            "points. Install scipy for cKDTree graph construction or use a "
            "regional subset for the MVP.",
            RuntimeWarning,
            stacklevel=2,
        )

    chord_sq_radius = chord_radius * chord_radius
    rows: list[list[int]] = []
    for start in range(0, n_points, chunk_size):
        stop = min(start + chunk_size, n_points)
        d2 = torch.cdist(points[start:stop], points, p=2.0).square()
        mask = d2 <= chord_sq_radius
        for local_i in range(stop - start):
            neighbs = torch.nonzero(mask[local_i], as_tuple=False).flatten()
            if max_neighbors is not None and neighbs.numel() > max_neighbors:
                order = torch.argsort(d2[local_i, neighbs])[:max_neighbors]
                neighbs = neighbs[order]
            rows.append([int(i) for i in neighbs])
    return _rows_to_edges(rows)


def build_radius_graph_sphere(
    points_xyz: Tensor,
    radius_km: float,
    *,
    earth_radius_km: float = 6371.0,
    max_neighbors: Optional[int] = None,
    prefer_scipy: bool = True,
    chunk_size: int = 2048,
) -> tuple[Tensor, Tensor]:
    """Build a radius graph on unit-sphere points.

    The search radius is specified as a geodesic distance in kilometers.  For
    efficient neighbor lookup we convert it to the equivalent chord radius on
    the unit sphere:

        chord = 2 sin((radius_km / R_earth) / 2).

    This is not an approximation of the set of neighbors; for points on the
    unit sphere it is exactly equivalent to a great-circle radius threshold.
    """

    if points_xyz.ndim != 2 or points_xyz.shape[-1] != 3:
        raise ValueError("points_xyz must have shape [N, 3]")
    if radius_km <= 0.0:
        raise ValueError("radius_km must be positive")
    if max_neighbors is not None and max_neighbors < 1:
        raise ValueError("max_neighbors must be positive when provided")

    points = torch.nn.functional.normalize(points_xyz.detach().cpu().to(torch.float64), dim=-1)
    chord_radius = chord_radius_from_km(radius_km, earth_radius_km)

    if points.shape[0] > 250000:
        warnings.warn(
            "Building a graph for a very large sphere point set. A full ERA5 "
            "0.25 degree grid has about one million points; prefer regional "
            "slices, max_neighbors, or an offline/scalable graph builder for "
            "serious experiments.",
            RuntimeWarning,
            stacklevel=2,
        )

    if prefer_scipy:
        try:
            return _build_radius_graph_scipy(points, chord_radius, max_neighbors=max_neighbors)
        except ImportError:
            warnings.warn(
                "scipy is not available; using a chunked torch all-pairs radius search.",
                RuntimeWarning,
                stacklevel=2,
            )

    return _build_radius_graph_torch(
        points,
        chord_radius,
        max_neighbors=max_neighbors,
        chunk_size=chunk_size,
    )


def _edge_geometry(
    points_xyz: Tensor,
    frames_e1: Tensor,
    frames_e2: Tensor,
    center_idx: Tensor,
    neighbor_idx: Tensor,
    *,
    earth_radius_km: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    p = points_xyz[center_idx]
    q = points_xyz[neighbor_idx]
    e1_p = frames_e1[center_idx]
    e2_p = frames_e2[center_idx]
    e1_q = frames_e1[neighbor_idx]
    e2_q = frames_e2[neighbor_idx]

    log_km = log_map_sphere_km(p, q, earth_radius_km=earth_radius_km)
    u = project_to_frame(log_km, e1_p, e2_p)
    r = u.norm(dim=-1)
    phi = torch.atan2(u[:, 1], u[:, 0])
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    transport = transport_matrix_q_to_p(p, q, e1_p, e2_p, e1_q, e2_q)
    return u, r, phi, cos_phi, sin_phi, transport


def _rotation_angle_from_matrix(matrix: Tensor, eps: float = 1e-8) -> Tensor:
    """Extract SO(2) angles once when immutable graph geometry is built."""

    cos_value = 0.5 * (matrix[..., 0, 0] + matrix[..., 1, 1])
    sin_value = 0.5 * (matrix[..., 1, 0] - matrix[..., 0, 1])
    norm = torch.sqrt(cos_value * cos_value + sin_value * sin_value).clamp_min(eps)
    return torch.atan2(sin_value / norm, cos_value / norm)


def _neighbor_counts(center_idx: Tensor, n_points: int) -> Tensor:
    counts = torch.zeros(n_points, dtype=torch.float32)
    ones = torch.ones_like(center_idx, dtype=torch.float32)
    counts.scatter_add_(0, center_idx.cpu(), ones.cpu())
    return counts.clamp_min(1.0)


@dataclass(frozen=True)
class SphereGraphGeometry:
    """Packed graph and edge geometry consumed by steerable sphere layers."""

    points_xyz: Tensor
    frames_e1: Tensor
    frames_e2: Tensor
    center_idx: Tensor
    neighbor_idx: Tensor
    u: Tensor
    r: Tensor
    phi: Tensor
    cos_phi: Tensor
    sin_phi: Tensor
    transport: Tensor
    neighbor_count: Tensor
    radius_km: float = 100.0
    earth_radius_km: float = 6371.0
    lat_shape: Optional[Tuple[int, int]] = None
    transport_angle: Tensor | None = None

    def __post_init__(self) -> None:
        if self.transport_angle is None:
            object.__setattr__(
                self,
                "transport_angle",
                _rotation_angle_from_matrix(self.transport),
            )

    @property
    def n_points(self) -> int:
        return int(self.points_xyz.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.center_idx.shape[0])

    @property
    def device(self) -> torch.device:
        return self.points_xyz.device

    @property
    def dtype(self) -> torch.dtype:
        return self.points_xyz.dtype

    @classmethod
    def from_latlon(
        cls,
        lat: Tensor | Sequence[float],
        lon: Tensor | Sequence[float],
        radius_km: float = 100.0,
        *,
        earth_radius_km: float = 6371.0,
        max_neighbors: Optional[int] = None,
        degrees: bool = True,
        dtype: torch.dtype = torch.float32,
        prefer_scipy: bool = True,
        frame_strategy: TangentFrameStrategy = "robust",
    ) -> "SphereGraphGeometry":
        points_xyz, lat_shape = latlon_to_xyz(lat, lon, degrees=degrees, dtype=dtype)
        return cls.from_points(
            points_xyz,
            radius_km=radius_km,
            earth_radius_km=earth_radius_km,
            max_neighbors=max_neighbors,
            lat_shape=lat_shape,
            prefer_scipy=prefer_scipy,
            frame_strategy=frame_strategy,
        )

    @classmethod
    def from_points(
        cls,
        points_xyz: Tensor,
        radius_km: float = 100.0,
        *,
        earth_radius_km: float = 6371.0,
        max_neighbors: Optional[int] = None,
        lat_shape: Optional[Tuple[int, int]] = None,
        prefer_scipy: bool = True,
        frame_strategy: TangentFrameStrategy = "robust",
    ) -> "SphereGraphGeometry":
        points = torch.nn.functional.normalize(points_xyz.detach().cpu(), dim=-1)
        center_idx, neighbor_idx = build_radius_graph_sphere(
            points,
            radius_km,
            earth_radius_km=earth_radius_km,
            max_neighbors=max_neighbors,
            prefer_scipy=prefer_scipy,
        )
        frames_e1, frames_e2 = build_tangent_frames(points, frame_strategy)
        return cls.from_existing_graph(
            points,
            frames_e1,
            frames_e2,
            center_idx,
            neighbor_idx,
            radius_km=radius_km,
            earth_radius_km=earth_radius_km,
            lat_shape=lat_shape,
        )

    @classmethod
    def from_existing_graph(
        cls,
        points_xyz: Tensor,
        frames_e1: Tensor,
        frames_e2: Tensor,
        center_idx: Tensor,
        neighbor_idx: Tensor,
        *,
        radius_km: float = 100.0,
        earth_radius_km: float = 6371.0,
        lat_shape: Optional[Tuple[int, int]] = None,
    ) -> "SphereGraphGeometry":
        points = torch.nn.functional.normalize(points_xyz, dim=-1)
        center_idx = center_idx.to(dtype=torch.long, device=points.device)
        neighbor_idx = neighbor_idx.to(dtype=torch.long, device=points.device)
        frames_e1 = frames_e1.to(device=points.device, dtype=points.dtype)
        frames_e2 = frames_e2.to(device=points.device, dtype=points.dtype)
        u, r, phi, cos_phi, sin_phi, transport = _edge_geometry(
            points,
            frames_e1,
            frames_e2,
            center_idx,
            neighbor_idx,
            earth_radius_km=earth_radius_km,
        )
        transport_angle = _rotation_angle_from_matrix(transport)
        counts = _neighbor_counts(center_idx, points.shape[0]).to(device=points.device, dtype=points.dtype)
        return cls(
            points_xyz=points,
            frames_e1=frames_e1,
            frames_e2=frames_e2,
            center_idx=center_idx,
            neighbor_idx=neighbor_idx,
            u=u,
            r=r,
            phi=phi,
            cos_phi=cos_phi,
            sin_phi=sin_phi,
            transport=transport,
            transport_angle=transport_angle,
            neighbor_count=counts,
            radius_km=float(radius_km),
            earth_radius_km=float(earth_radius_km),
            lat_shape=lat_shape,
        )

    def with_rotated_gauge(self, gamma: Tensor) -> "SphereGraphGeometry":
        """Return a graph with frames rotated pointwise by ``gamma`` radians."""

        gamma = gamma.to(device=self.points_xyz.device, dtype=self.points_xyz.dtype)
        if gamma.shape != (self.n_points,):
            raise ValueError(f"gamma must have shape [{self.n_points}]")
        e1, e2 = rotate_tangent_frames(self.frames_e1, self.frames_e2, gamma)
        return type(self).from_existing_graph(
            self.points_xyz,
            e1,
            e2,
            self.center_idx,
            self.neighbor_idx,
            radius_km=self.radius_km,
            earth_radius_km=self.earth_radius_km,
            lat_shape=self.lat_shape,
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SphereGraphGeometry":
        """Move graph tensors to a device/dtype, preserving integer indices."""

        device = torch.device(device) if device is not None else self.device
        dtype = dtype or self.dtype
        if self.device == device and self.dtype == dtype:
            return self

        def move_float(t: Tensor) -> Tensor:
            return t.to(device=device, dtype=dtype)

        def move_long(t: Tensor) -> Tensor:
            return t.to(device=device, dtype=torch.long)

        assert self.transport_angle is not None
        return replace(
            self,
            points_xyz=move_float(self.points_xyz),
            frames_e1=move_float(self.frames_e1),
            frames_e2=move_float(self.frames_e2),
            center_idx=move_long(self.center_idx),
            neighbor_idx=move_long(self.neighbor_idx),
            u=move_float(self.u),
            r=move_float(self.r),
            phi=move_float(self.phi),
            cos_phi=move_float(self.cos_phi),
            sin_phi=move_float(self.sin_phi),
            transport=move_float(self.transport),
            transport_angle=move_float(self.transport_angle),
            neighbor_count=move_float(self.neighbor_count),
        )
