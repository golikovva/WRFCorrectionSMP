"""Multi-resolution sphere graph hierarchies and equivariant pooling."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal, Sequence

import torch
from torch import Tensor

from .sphere_geometry import TangentFrameStrategy, transport_matrix_q_to_p
from .sphere_graph import SphereGraphGeometry, _rotation_angle_from_matrix
# from .spherical_mnist import (
#     cube_sphere_points,
#     healpix_ring_sphere_points,
#     icosahedral_grid_coordinates,
#     icosahedral_vertex_coordinates,
# )


def _scatter_mean(
    values: Tensor,
    assignment: Tensor,
    count: Tensor,
    n_coarse: int,
) -> Tensor:
    batch = int(values.shape[0])
    trailing = values.shape[2:]
    out = values.new_zeros((batch, n_coarse, *trailing))
    index_shape = (1, int(assignment.shape[0]), *([1] * len(trailing)))
    index = assignment.view(index_shape).expand(batch, int(assignment.shape[0]), *trailing)
    out.scatter_add_(1, index, values)
    norm_shape = (1, n_coarse, *([1] * len(trailing)))
    return out / count.to(device=values.device, dtype=values.dtype).clamp_min(1.0).view(norm_shape)


def _assignment_counts(assignment: Tensor, n_coarse: int) -> Tensor:
    count = torch.zeros(n_coarse, dtype=torch.float32)
    ones = torch.ones_like(assignment, dtype=torch.float32)
    count.scatter_add_(0, assignment.cpu(), ones.cpu())
    return count.clamp_min(1.0)


def _nearest_assignment(points_xyz: Tensor, centers_xyz: Tensor, *, chunk_size: int = 4096) -> Tensor:
    points = torch.nn.functional.normalize(points_xyz.detach().cpu().to(torch.float64), dim=-1)
    centers = torch.nn.functional.normalize(centers_xyz.detach().cpu().to(torch.float64), dim=-1)
    chunks = []
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        d2 = torch.cdist(points[start:stop], centers, p=2.0).square()
        chunks.append(torch.argmin(d2, dim=1).to(dtype=torch.long))
    return torch.cat(chunks, dim=0)


def _nearest_tie_pool_edges(
    points_xyz: Tensor,
    centers_xyz: Tensor,
    *,
    tie_tol: float = 1e-5,
    chunk_size: int = 4096,
) -> tuple[Tensor, Tensor, Tensor]:
    points = torch.nn.functional.normalize(points_xyz.detach().cpu().to(torch.float64), dim=-1)
    centers = torch.nn.functional.normalize(centers_xyz.detach().cpu().to(torch.float64), dim=-1)
    fine_parts = []
    coarse_parts = []
    weight_parts = []
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        distances = torch.cdist(points[start:stop], centers, p=2.0)
        nearest = distances.min(dim=1).values
        mask = distances <= nearest[:, None] + float(tie_tol)
        local_fine, coarse_idx = torch.nonzero(mask, as_tuple=True)
        tie_count = mask.sum(dim=1).to(dtype=torch.float32)
        fine_idx = local_fine.to(dtype=torch.long) + int(start)
        weight = 1.0 / tie_count[local_fine].clamp_min(1.0)
        fine_parts.append(fine_idx)
        coarse_parts.append(coarse_idx.to(dtype=torch.long))
        weight_parts.append(weight)
    return torch.cat(fine_parts), torch.cat(coarse_parts), torch.cat(weight_parts)


def farthest_point_indices(points_xyz: Tensor, n_samples: int) -> Tensor:
    """Deterministic farthest-point sampling on normalized sphere points."""

    points = torch.nn.functional.normalize(points_xyz.detach().cpu().to(torch.float64), dim=-1)
    n_points = int(points.shape[0])
    n_samples = int(n_samples)
    if n_samples < 1 or n_samples > n_points:
        raise ValueError("n_samples must be in [1, n_points]")

    selected = torch.empty(n_samples, dtype=torch.long)
    distances = torch.full((n_points,), float("inf"), dtype=torch.float64)
    current = 0
    for sample_i in range(n_samples):
        selected[sample_i] = current
        d2 = (points - points[current]).square().sum(dim=-1)
        distances = torch.minimum(distances, d2)
        current = int(torch.argmax(distances).item())
    return selected


@dataclass(frozen=True)
class SpherePooler:
    """Pool features from a fine sphere graph onto a coarse sphere graph."""

    fine_graph: SphereGraphGeometry
    coarse_graph: SphereGraphGeometry
    assignment: Tensor
    transport_fine_to_coarse: Tensor
    count: Tensor
    transport_angle: Tensor | None = None

    def __post_init__(self) -> None:
        if self.transport_angle is None:
            object.__setattr__(
                self,
                "transport_angle",
                _rotation_angle_from_matrix(self.transport_fine_to_coarse),
            )

    @classmethod
    def from_assignment(
        cls,
        fine_graph: SphereGraphGeometry,
        coarse_graph: SphereGraphGeometry,
        assignment: Tensor,
    ) -> "SpherePooler":
        assignment = assignment.to(dtype=torch.long, device=fine_graph.points_xyz.device)
        if assignment.shape != (fine_graph.n_points,):
            raise ValueError(f"assignment must have shape [{fine_graph.n_points}]")
        if int(assignment.min().item()) < 0 or int(assignment.max().item()) >= coarse_graph.n_points:
            raise ValueError("assignment contains an out-of-range coarse index")

        coarse_idx = assignment.to(dtype=torch.long, device=coarse_graph.points_xyz.device)
        p = coarse_graph.points_xyz[coarse_idx].to(device=fine_graph.points_xyz.device)
        q = fine_graph.points_xyz
        e1_p = coarse_graph.frames_e1[coarse_idx].to(device=fine_graph.points_xyz.device)
        e2_p = coarse_graph.frames_e2[coarse_idx].to(device=fine_graph.points_xyz.device)
        e1_q = fine_graph.frames_e1
        e2_q = fine_graph.frames_e2
        transport = transport_matrix_q_to_p(p, q, e1_p, e2_p, e1_q, e2_q)
        transport_angle = _rotation_angle_from_matrix(transport)
        count = _assignment_counts(assignment, coarse_graph.n_points).to(device=fine_graph.points_xyz.device)
        return cls(
            fine_graph,
            coarse_graph,
            assignment,
            transport,
            count,
            transport_angle=transport_angle,
        )

    @classmethod
    def nearest(
        cls,
        fine_graph: SphereGraphGeometry,
        coarse_graph: SphereGraphGeometry,
    ) -> "SpherePooler":
        assignment = _nearest_assignment(fine_graph.points_xyz, coarse_graph.points_xyz)
        return cls.from_assignment(fine_graph, coarse_graph, assignment)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SpherePooler":
        device = torch.device(device) if device is not None else self.fine_graph.device
        dtype = dtype or self.fine_graph.dtype
        assert self.transport_angle is not None
        if (
            self.fine_graph.device == device
            and self.fine_graph.dtype == dtype
            and self.coarse_graph.device == device
            and self.coarse_graph.dtype == dtype
            and self.assignment.device == device
            and self.transport_fine_to_coarse.device == device
            and self.transport_fine_to_coarse.dtype == dtype
            and self.transport_angle.device == device
            and self.transport_angle.dtype == dtype
            and self.count.device == device
            and self.count.dtype == dtype
        ):
            return self
        return replace(
            self,
            fine_graph=self.fine_graph.to(device=device, dtype=dtype),
            coarse_graph=self.coarse_graph.to(device=device, dtype=dtype),
            assignment=self.assignment.to(device=device, dtype=torch.long),
            transport_fine_to_coarse=self.transport_fine_to_coarse.to(device=device, dtype=dtype),
            transport_angle=self.transport_angle.to(device=device, dtype=dtype),
            count=self.count.to(device=device, dtype=dtype),
        )

    def pool_scalar(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, N_fine, C]")
        if int(x.shape[1]) != self.fine_graph.n_points:
            raise ValueError(f"x has {x.shape[1]} points, expected {self.fine_graph.n_points}")
        assignment = self.assignment.to(device=x.device)
        count = self.count.to(device=x.device, dtype=x.dtype)
        return _scatter_mean(x, assignment, count, self.coarse_graph.n_points)

    def pool_vector(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[-1] != 2:
            raise ValueError("x must have shape [B, N_fine, C, 2]")
        if int(x.shape[1]) != self.fine_graph.n_points:
            raise ValueError(f"x has {x.shape[1]} points, expected {self.fine_graph.n_points}")
        transport = self.transport_fine_to_coarse.to(device=x.device, dtype=x.dtype)
        transported = torch.einsum("ndc,bnic->bnid", transport, x)
        assignment = self.assignment.to(device=x.device)
        count = self.count.to(device=x.device, dtype=x.dtype)
        return _scatter_mean(transported, assignment, count, self.coarse_graph.n_points)


@dataclass(frozen=True)
class SphereWeightedPooler:
    """Equivariant local weighted pooling from a fine graph to a coarse graph."""

    fine_graph: SphereGraphGeometry
    coarse_graph: SphereGraphGeometry
    fine_idx: Tensor
    coarse_idx: Tensor
    weight: Tensor
    transport_fine_to_coarse: Tensor
    count: Tensor
    transport_angle: Tensor | None = None

    def __post_init__(self) -> None:
        if self.transport_angle is None:
            object.__setattr__(
                self,
                "transport_angle",
                _rotation_angle_from_matrix(self.transport_fine_to_coarse),
            )

    @classmethod
    def from_edges(
        cls,
        fine_graph: SphereGraphGeometry,
        coarse_graph: SphereGraphGeometry,
        fine_idx: Tensor,
        coarse_idx: Tensor,
        weight: Tensor,
    ) -> "SphereWeightedPooler":
        fine_idx = fine_idx.to(dtype=torch.long, device=fine_graph.points_xyz.device)
        coarse_idx = coarse_idx.to(dtype=torch.long, device=fine_graph.points_xyz.device)
        weight = weight.to(dtype=fine_graph.points_xyz.dtype, device=fine_graph.points_xyz.device)
        if fine_idx.shape != coarse_idx.shape or fine_idx.shape != weight.shape:
            raise ValueError("fine_idx, coarse_idx, and weight must have matching shapes")
        if int(fine_idx.min().item()) < 0 or int(fine_idx.max().item()) >= fine_graph.n_points:
            raise ValueError("fine_idx contains an out-of-range fine index")
        if int(coarse_idx.min().item()) < 0 or int(coarse_idx.max().item()) >= coarse_graph.n_points:
            raise ValueError("coarse_idx contains an out-of-range coarse index")

        p = coarse_graph.points_xyz[coarse_idx].to(device=fine_graph.points_xyz.device)
        q = fine_graph.points_xyz[fine_idx]
        e1_p = coarse_graph.frames_e1[coarse_idx].to(device=fine_graph.points_xyz.device)
        e2_p = coarse_graph.frames_e2[coarse_idx].to(device=fine_graph.points_xyz.device)
        e1_q = fine_graph.frames_e1[fine_idx]
        e2_q = fine_graph.frames_e2[fine_idx]
        transport = transport_matrix_q_to_p(p, q, e1_p, e2_p, e1_q, e2_q)
        transport_angle = _rotation_angle_from_matrix(transport)
        count = torch.zeros(coarse_graph.n_points, dtype=weight.dtype, device=weight.device)
        count.scatter_add_(0, coarse_idx, weight)
        return cls(
            fine_graph,
            coarse_graph,
            fine_idx,
            coarse_idx,
            weight,
            transport,
            count.clamp_min(1.0),
            transport_angle=transport_angle,
        )

    @classmethod
    def nearest_ties(
        cls,
        fine_graph: SphereGraphGeometry,
        coarse_graph: SphereGraphGeometry,
        *,
        tie_tol: float = 1e-5,
    ) -> "SphereWeightedPooler":
        fine_idx, coarse_idx, weight = _nearest_tie_pool_edges(
            fine_graph.points_xyz,
            coarse_graph.points_xyz,
            tie_tol=tie_tol,
        )
        return cls.from_edges(fine_graph, coarse_graph, fine_idx, coarse_idx, weight)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SphereWeightedPooler":
        device = torch.device(device) if device is not None else self.fine_graph.device
        dtype = dtype or self.fine_graph.dtype
        assert self.transport_angle is not None
        if (
            self.fine_graph.device == device
            and self.fine_graph.dtype == dtype
            and self.coarse_graph.device == device
            and self.coarse_graph.dtype == dtype
            and self.fine_idx.device == device
            and self.coarse_idx.device == device
            and self.weight.device == device
            and self.weight.dtype == dtype
            and self.transport_fine_to_coarse.device == device
            and self.transport_fine_to_coarse.dtype == dtype
            and self.transport_angle.device == device
            and self.transport_angle.dtype == dtype
            and self.count.device == device
            and self.count.dtype == dtype
        ):
            return self
        return type(self)(
            fine_graph=self.fine_graph.to(device=device, dtype=dtype),
            coarse_graph=self.coarse_graph.to(device=device, dtype=dtype),
            fine_idx=self.fine_idx.to(device=device, dtype=torch.long),
            coarse_idx=self.coarse_idx.to(device=device, dtype=torch.long),
            weight=self.weight.to(device=device, dtype=dtype),
            transport_fine_to_coarse=self.transport_fine_to_coarse.to(device=device, dtype=dtype),
            transport_angle=self.transport_angle.to(device=device, dtype=dtype),
            count=self.count.to(device=device, dtype=dtype),
        )

    def pool_scalar(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, N_fine, C]")
        if int(x.shape[1]) != self.fine_graph.n_points:
            raise ValueError(f"x has {x.shape[1]} points, expected {self.fine_graph.n_points}")
        fine_idx = self.fine_idx.to(device=x.device)
        coarse_idx = self.coarse_idx.to(device=x.device)
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        values = x[:, fine_idx, :] * weight.view(1, -1, 1)
        out = x.new_zeros((int(x.shape[0]), self.coarse_graph.n_points, int(x.shape[2])))
        index = coarse_idx.view(1, -1, 1).expand_as(values)
        out.scatter_add_(1, index, values)
        return out / self.count.to(device=x.device, dtype=x.dtype).view(1, -1, 1)

    def pool_vector(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[-1] != 2:
            raise ValueError("x must have shape [B, N_fine, C, 2]")
        if int(x.shape[1]) != self.fine_graph.n_points:
            raise ValueError(f"x has {x.shape[1]} points, expected {self.fine_graph.n_points}")
        fine_idx = self.fine_idx.to(device=x.device)
        coarse_idx = self.coarse_idx.to(device=x.device)
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        transport = self.transport_fine_to_coarse.to(device=x.device, dtype=x.dtype)
        transported = torch.einsum("ndc,bnic->bnid", transport, x[:, fine_idx, :, :])
        values = transported * weight.view(1, -1, 1, 1)
        out = x.new_zeros((int(x.shape[0]), self.coarse_graph.n_points, int(x.shape[2]), 2))
        index = coarse_idx.view(1, -1, 1, 1).expand_as(values)
        out.scatter_add_(1, index, values)
        return out / self.count.to(device=x.device, dtype=x.dtype).view(1, -1, 1, 1)


@dataclass(frozen=True)
class SphereGraphHierarchy:
    """A list of sphere graphs plus equivariant poolers between them."""

    graphs: tuple[SphereGraphGeometry, ...]
    poolers: tuple[SpherePooler | SphereWeightedPooler, ...]

    def __post_init__(self) -> None:
        if len(self.graphs) < 1:
            raise ValueError("graphs must contain at least one level")
        if len(self.poolers) != len(self.graphs) - 1:
            raise ValueError("poolers must connect consecutive graph levels")

    @property
    def n_levels(self) -> int:
        return len(self.graphs)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SphereGraphHierarchy":
        target_device = torch.device(device) if device is not None else self.graphs[0].device
        target_dtype = dtype or self.graphs[0].dtype

        def pooler_matches(pooler: SpherePooler | SphereWeightedPooler) -> bool:
            assert pooler.transport_angle is not None
            index_tensors = (
                (pooler.assignment,)
                if isinstance(pooler, SpherePooler)
                else (pooler.fine_idx, pooler.coarse_idx)
            )
            float_tensors = (
                (pooler.transport_fine_to_coarse, pooler.transport_angle, pooler.count)
                if isinstance(pooler, SpherePooler)
                else (
                    pooler.weight,
                    pooler.transport_fine_to_coarse,
                    pooler.transport_angle,
                    pooler.count,
                )
            )
            return (
                pooler.fine_graph.device == target_device
                and pooler.fine_graph.dtype == target_dtype
                and pooler.coarse_graph.device == target_device
                and pooler.coarse_graph.dtype == target_dtype
                and all(tensor.device == target_device for tensor in index_tensors)
                and all(
                    tensor.device == target_device and tensor.dtype == target_dtype
                    for tensor in float_tensors
                )
            )

        if (
            all(graph.device == target_device and graph.dtype == target_dtype for graph in self.graphs)
            and all(pooler_matches(pooler) for pooler in self.poolers)
        ):
            return self
        graphs = tuple(graph.to(device=target_device, dtype=target_dtype) for graph in self.graphs)

        def move_pooler(
            level: int,
            pooler: SpherePooler | SphereWeightedPooler,
        ) -> SpherePooler | SphereWeightedPooler:
            assert pooler.transport_angle is not None
            common = {
                "fine_graph": graphs[level],
                "coarse_graph": graphs[level + 1],
                "transport_fine_to_coarse": pooler.transport_fine_to_coarse.to(
                    device=target_device,
                    dtype=target_dtype,
                ),
                "transport_angle": pooler.transport_angle.to(
                    device=target_device,
                    dtype=target_dtype,
                ),
                "count": pooler.count.to(device=target_device, dtype=target_dtype),
            }
            if isinstance(pooler, SpherePooler):
                return replace(
                    pooler,
                    assignment=pooler.assignment.to(device=target_device, dtype=torch.long),
                    **common,
                )
            return replace(
                pooler,
                fine_idx=pooler.fine_idx.to(device=target_device, dtype=torch.long),
                coarse_idx=pooler.coarse_idx.to(device=target_device, dtype=torch.long),
                weight=pooler.weight.to(device=target_device, dtype=target_dtype),
                **common,
            )

        poolers = tuple(move_pooler(level, pooler) for level, pooler in enumerate(self.poolers))
        return type(self)(graphs=graphs, poolers=poolers)


def _ico_assignment(fine_resolution: int) -> Tensor:
    if fine_resolution <= 1:
        raise ValueError("fine_resolution must be greater than 1")
    fine_h = 2**fine_resolution
    fine_w = 2 ** (fine_resolution + 1)
    coarse_h = fine_h // 2
    coarse_w = fine_w // 2
    assignment = torch.empty(5 * fine_h * fine_w, dtype=torch.long)
    for chart in range(5):
        for h in range(fine_h):
            for w in range(fine_w):
                fine_idx = chart * fine_h * fine_w + h * fine_w + w
                coarse_idx = chart * coarse_h * coarse_w + (h // 2) * coarse_w + (w // 2)
                assignment[fine_idx] = coarse_idx
    return assignment


def _halving_sizes(initial_size: int, levels: int, *, min_size: int) -> tuple[int, ...]:
    if levels < 1:
        raise ValueError("levels must be positive")
    if initial_size < min_size:
        raise ValueError(f"initial_size must be at least {min_size}")

    sizes = [int(initial_size)]
    for _level_i in range(1, levels):
        previous = sizes[-1]
        if previous <= min_size:
            sizes.append(previous)
        else:
            sizes.append(max(int(min_size), int(math.ceil(previous / 2.0))))
    return tuple(sizes)


def _build_weighted_point_sequence_hierarchy(
    level_points: Sequence[Tensor],
    *,
    radius_km: float,
    max_neighbors: int | None,
    earth_radius_km: float,
    prefer_scipy: bool,
    frame_strategy: TangentFrameStrategy,
) -> SphereGraphHierarchy:
    if len(level_points) < 1:
        raise ValueError("level_points must contain at least one level")

    n0 = float(level_points[0].shape[0])
    graphs = []
    for points in level_points:
        level_radius = float(radius_km) * math.sqrt(n0 / float(points.shape[0]))
        graphs.append(
            SphereGraphGeometry.from_points(
                points,
                radius_km=level_radius,
                earth_radius_km=earth_radius_km,
                max_neighbors=max_neighbors,
                prefer_scipy=prefer_scipy,
                frame_strategy=frame_strategy,
            )
        )

    poolers = [SphereWeightedPooler.nearest_ties(graphs[i], graphs[i + 1]) for i in range(len(graphs) - 1)]
    return SphereGraphHierarchy(graphs=tuple(graphs), poolers=tuple(poolers))


# def build_icosahedral_sphere_hierarchy(
#     resolution: int,
#     *,
#     levels: int,
#     radius_km: float,
#     max_neighbors: int | None,
#     earth_radius_km: float = 6371.0,
#     prefer_scipy: bool = True,
#     radius_scale: float = 2.0,
#     layout: Literal["vertex", "chart"] = "vertex",
#     frame_strategy: TangentFrameStrategy = "robust",
# ) -> SphereGraphHierarchy:
#     """Build a multi-resolution icosahedral hierarchy."""

#     if levels < 1:
#         raise ValueError("levels must be positive")
#     if resolution - (levels - 1) < 1:
#         raise ValueError("requested more hierarchy levels than the icosahedral resolution supports")
#     if layout not in ("vertex", "chart"):
#         raise ValueError("layout must be 'vertex' or 'chart'")

#     graphs = []
#     for level_i in range(levels):
#         level_resolution = int(resolution) - level_i
#         if layout == "vertex":
#             points = icosahedral_vertex_coordinates(level_resolution)
#         else:
#             points = icosahedral_grid_coordinates(level_resolution).reshape(-1, 3)
#         graphs.append(
#             SphereGraphGeometry.from_points(
#                 points,
#                 radius_km=float(radius_km) * (float(radius_scale) ** level_i),
#                 earth_radius_km=earth_radius_km,
#                 max_neighbors=max_neighbors,
#                 prefer_scipy=prefer_scipy,
#                 frame_strategy=frame_strategy,
#             )
#         )

#     poolers = []
#     for level_i in range(levels - 1):
#         if layout == "vertex":
#             poolers.append(SphereWeightedPooler.nearest_ties(graphs[level_i], graphs[level_i + 1]))
#         else:
#             assignment = _ico_assignment(int(resolution) - level_i)
#             poolers.append(SpherePooler.from_assignment(graphs[level_i], graphs[level_i + 1], assignment))
#     return SphereGraphHierarchy(graphs=tuple(graphs), poolers=tuple(poolers))


# def build_healpix_sphere_hierarchy(
#     nside: int,
#     *,
#     levels: int,
#     radius_km: float,
#     max_neighbors: int | None,
#     earth_radius_km: float = 6371.0,
#     prefer_scipy: bool = True,
#     frame_strategy: TangentFrameStrategy = "robust",
# ) -> SphereGraphHierarchy:
#     """Build a HEALPix hierarchy whose levels preserve the HEALPix D4 symmetries."""

#     nsides = _halving_sizes(int(nside), levels, min_size=1)
#     level_points = [healpix_ring_sphere_points(level_nside) for level_nside in nsides]
#     return _build_weighted_point_sequence_hierarchy(
#         level_points,
#         radius_km=radius_km,
#         max_neighbors=max_neighbors,
#         earth_radius_km=earth_radius_km,
#         prefer_scipy=prefer_scipy,
#         frame_strategy=frame_strategy,
#     )


# def build_cube_sphere_hierarchy(
#     face_size: int,
#     *,
#     levels: int,
#     radius_km: float,
#     max_neighbors: int | None,
#     earth_radius_km: float = 6371.0,
#     prefer_scipy: bool = True,
#     frame_strategy: TangentFrameStrategy = "robust",
# ) -> SphereGraphHierarchy:
#     """Build a cube-sphere hierarchy whose levels preserve cube rotations."""

#     face_sizes = _halving_sizes(int(face_size), levels, min_size=2)
#     level_points = [cube_sphere_points(level_face_size) for level_face_size in face_sizes]
#     return _build_weighted_point_sequence_hierarchy(
#         level_points,
#         radius_km=radius_km,
#         max_neighbors=max_neighbors,
#         earth_radius_km=earth_radius_km,
#         prefer_scipy=prefer_scipy,
#         frame_strategy=frame_strategy,
#     )


def build_fps_sphere_hierarchy(
    points_xyz: Tensor,
    *,
    levels: int,
    radius_km: float,
    max_neighbors: int | None,
    earth_radius_km: float = 6371.0,
    prefer_scipy: bool = True,
    pool_ratio: float = 0.25,
    min_points: int = 16,
    frame_strategy: TangentFrameStrategy = "robust",
) -> SphereGraphHierarchy:
    """Build a point-cloud hierarchy with farthest-point sampling."""

    if levels < 1:
        raise ValueError("levels must be positive")
    if not (0.0 < pool_ratio < 1.0):
        raise ValueError("pool_ratio must be in (0, 1)")

    points = torch.nn.functional.normalize(points_xyz.detach().cpu(), dim=-1)
    level_points = [points]
    for _level_i in range(1, levels):
        previous = level_points[-1]
        n_next = max(int(min_points), int(math.ceil(previous.shape[0] * pool_ratio)))
        n_next = min(n_next, int(previous.shape[0]))
        if n_next == int(previous.shape[0]):
            level_points.append(previous)
        else:
            level_points.append(previous[farthest_point_indices(previous, n_next)])

    n0 = float(level_points[0].shape[0])
    graphs = []
    for pts in level_points:
        level_radius = float(radius_km) * math.sqrt(n0 / float(pts.shape[0]))
        graphs.append(
            SphereGraphGeometry.from_points(
                pts,
                radius_km=level_radius,
                earth_radius_km=earth_radius_km,
                max_neighbors=max_neighbors,
                prefer_scipy=prefer_scipy,
                frame_strategy=frame_strategy,
            )
        )

    poolers = [SpherePooler.nearest(graphs[i], graphs[i + 1]) for i in range(levels - 1)]
    return SphereGraphHierarchy(graphs=tuple(graphs), poolers=tuple(poolers))


def build_sphere_hierarchy(
    points_xyz: Tensor,
    *,
    levels: int,
    radius_km: float,
    max_neighbors: int | None,
    mode: Literal["fps"] = "fps",
    earth_radius_km: float = 6371.0,
    prefer_scipy: bool = True,
    frame_strategy: TangentFrameStrategy = "robust",
) -> SphereGraphHierarchy:
    if mode != "fps":
        raise ValueError("only mode='fps' is currently supported for generic point clouds")
    return build_fps_sphere_hierarchy(
        points_xyz,
        levels=levels,
        radius_km=radius_km,
        max_neighbors=max_neighbors,
        earth_radius_km=earth_radius_km,
        prefer_scipy=prefer_scipy,
        frame_strategy=frame_strategy,
    )
