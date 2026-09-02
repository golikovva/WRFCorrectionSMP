"""SO(2)-steerable intrinsic KPConv-like layers on the sphere."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from collections.abc import Sequence
from typing import Literal
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .radial_basis import default_radial_sigma, radial_centers, triangular_radial_basis
from .profile_ranges import record_region_call
from .triton_irrep_conv import (
    AUTO_INFERENCE_WORK_THRESHOLDS,
    AUTO_TRAINING_WORK_THRESHOLDS,
    AUTO_WORK_THRESHOLD,
    packed_irrep_conv,
    triton_support_reason,
)
from .triton_semi_packed_irrep_conv import (
    semi_packed_irrep_conv,
    semi_packed_support_reason,
    should_use_semi_packed,
)
from .triton_irregular_irrep_conv import (
    irregular_irrep_pair_conv,
    irregular_pair_support_reason,
)
from ...data.spherical.sphere_geometry import rotation_matrices_from_cos_sin
from ...data.spherical.sphere_graph import SphereGraphGeometry


def _reset_weight(weight: Tensor) -> None:
    fan_in = max(1, weight[0].numel())
    bound = math.sqrt(6.0 / fan_in)
    nn.init.uniform_(weight, -bound, bound)


def _graph_tensors(graph: SphereGraphGeometry, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    center_idx = graph.center_idx.to(device=x.device)
    neighbor_idx = graph.neighbor_idx.to(device=x.device)
    r = graph.r.to(device=x.device, dtype=x.dtype)
    q_matrix = rotation_matrices_from_cos_sin(
        graph.cos_phi.to(device=x.device, dtype=x.dtype),
        graph.sin_phi.to(device=x.device, dtype=x.dtype),
    )
    transport = graph.transport.to(device=x.device, dtype=x.dtype)
    neighbor_count = graph.neighbor_count.to(device=x.device, dtype=x.dtype)
    return center_idx, neighbor_idx, r, q_matrix, transport, neighbor_count


def _scatter_to_centers(
    contributions: Tensor,
    center_idx: Tensor,
    n_points: int,
    neighbor_count: Tensor,
    *,
    normalize_by_neighbors: bool,
    edge_weight: Tensor | None = None,
) -> Tensor:
    batch = contributions.shape[0]
    edge_count = contributions.shape[1]
    trailing = contributions.shape[2:]
    out = contributions.new_zeros((batch, n_points, *trailing))
    index_shape = (1, edge_count, *([1] * len(trailing)))
    index = center_idx.view(index_shape).expand(batch, edge_count, *trailing)
    out.scatter_add_(1, index, contributions)
    if normalize_by_neighbors:
        if edge_weight is not None:
            denom = contributions.new_zeros(n_points)
            denom.scatter_add_(0, center_idx, edge_weight.to(device=contributions.device, dtype=contributions.dtype))
            neighbor_count = denom.clamp_min(1.0)
        norm_shape = (1, n_points, *([1] * len(trailing)))
        out = out / neighbor_count.clamp_min(1.0).view(norm_shape)
    return out


def _pack_irrep_field(
    x: Tensor,
    index: Tensor,
    batch: int,
    n_points: int,
    multiplicity: int,
    irrep_dim: int,
) -> Tensor:
    return x.index_select(-1, index).reshape(
        batch, n_points, multiplicity, irrep_dim
    )


def _gather_irrep_neighbors(x: Tensor, neighbor_idx: Tensor) -> Tensor:
    # ``x[:, neighbor_idx, ...]`` goes through advanced-index validation in
    # TorchDynamo.  With a CUDA index tensor that validation reads a scalar
    # from the device (``aten._local_scalar_dense``), which makes the packed
    # and blockwise reference paths incompatible with ``fullgraph=True``.
    # This gather is exactly an index-select along the point dimension and
    # index_select has the same forward/backward semantics without the graph
    # break or a device synchronization.
    return x.index_select(1, neighbor_idx)


def _quadrature_contribution(basis: Tensor, weight: Tensor, x_q: Tensor) -> Tensor:
    kernel = torch.einsum("erabcd,roibc->eoiad", basis, weight)
    return torch.einsum("eoiad,beid->beoa", kernel, x_q)


def _unpack_irrep_field(x: Tensor, index: Tensor) -> Tensor:
    return x.flatten(start_dim=-2).index_select(-1, index)


class _RadialBasisConvBase(nn.Module):
    def __init__(
        self,
        radius_km: float,
        num_radial: int,
        *,
        radial_sigma_km: float | None = None,
        normalize_radial_basis: bool = True,
        normalize_by_neighbors: bool = True,
        origin_eps_km: float = 1e-6,
    ) -> None:
        super().__init__()
        self.radius_km = float(radius_km)
        self.num_radial = int(num_radial)
        self._radial_sigma_is_default = radial_sigma_km is None
        self.radial_sigma_km = (
            float(radial_sigma_km)
            if radial_sigma_km is not None
            else default_radial_sigma(radius_km, num_radial)
        )
        self.normalize_radial_basis = bool(normalize_radial_basis)
        self.normalize_by_neighbors = bool(normalize_by_neighbors)
        self.origin_eps_km = float(origin_eps_km)
        centers = radial_centers(radius_km, num_radial)
        self.register_buffer("centers_km", centers)

    def radial_basis(self, r: Tensor, *, radius_km: float | None = None) -> Tensor:
        if radius_km is None or float(radius_km) == self.radius_km:
            centers = self.centers_km
            sigma = self.radial_sigma_km
        else:
            effective_radius = float(radius_km)
            centers = radial_centers(effective_radius, self.num_radial, dtype=self.centers_km.dtype)
            sigma = (
                default_radial_sigma(effective_radius, self.num_radial)
                if self._radial_sigma_is_default
                else self.radial_sigma_km
            )
        return triangular_radial_basis(
            r,
            centers,
            sigma,
            normalize=self.normalize_radial_basis,
        )

    def active_edge_weight(self, r: Tensor) -> Tensor:
        # The angular direction phi is undefined at r = 0.  Directional
        # steerable kernels therefore skip zero-displacement edges, including
        # self-neighbors and duplicate pole samples on lat-lon grids.
        return (r > self.origin_eps_km).to(dtype=r.dtype)


@dataclass(frozen=True)
class SO2IrrepFieldType:
    """Flat SO(2) field layout with multiplicities for orders rho_0..rho_M."""

    multiplicities: tuple[int, ...]

    def __init__(self, multiplicities: Sequence[int]) -> None:
        values = tuple(int(value) for value in multiplicities)
        if not values:
            raise ValueError("multiplicities must contain at least rho_0")
        if any(value < 0 for value in values):
            raise ValueError("multiplicities must be non-negative")
        if not any(values):
            raise ValueError("at least one multiplicity must be positive")
        object.__setattr__(self, "multiplicities", values)

    @classmethod
    def scalar(cls, channels: int) -> "SO2IrrepFieldType":
        if int(channels) < 1:
            raise ValueError("channels must be positive")
        return cls((int(channels),))

    @classmethod
    def balanced(cls, max_order: int, multiplicity: int) -> "SO2IrrepFieldType":
        if int(max_order) < 0:
            raise ValueError("max_order must be non-negative")
        if int(multiplicity) < 1:
            raise ValueError("multiplicity must be positive")
        return cls(tuple(int(multiplicity) for _ in range(int(max_order) + 1)))

    @staticmethod
    def irrep_dim(order: int) -> int:
        return 1 if int(order) == 0 else 2

    @property
    def max_order(self) -> int:
        return len(self.multiplicities) - 1

    @property
    def orders(self) -> tuple[int, ...]:
        return tuple(order for order, multiplicity in enumerate(self.multiplicities) if multiplicity > 0)

    @property
    def total_dim(self) -> int:
        return sum(multiplicity * self.irrep_dim(order) for order, multiplicity in enumerate(self.multiplicities))

    @property
    def invariant_dim(self) -> int:
        return sum(self.multiplicities)

    def multiplicity(self, order: int) -> int:
        if int(order) < 0:
            raise ValueError("order must be non-negative")
        if int(order) >= len(self.multiplicities):
            return 0
        return self.multiplicities[int(order)]

    def flat_slice(self, order: int) -> slice:
        order = int(order)
        if order < 0 or order >= len(self.multiplicities):
            raise ValueError(f"order {order} is not present")
        start = 0
        for prev_order, multiplicity in enumerate(self.multiplicities[:order]):
            start += multiplicity * self.irrep_dim(prev_order)
        stop = start + self.multiplicities[order] * self.irrep_dim(order)
        return slice(start, stop)

    def block_shape(self, order: int) -> tuple[int, int]:
        return self.multiplicity(order), self.irrep_dim(order)

    def is_regular(self) -> bool:
        first = self.multiplicities[0]
        return all(multiplicity == first and multiplicity > 0 for multiplicity in self.multiplicities)


def _irrep_block(x: Tensor, field_type: SO2IrrepFieldType, order: int) -> Tensor:
    multiplicity, dim = field_type.block_shape(order)
    return x[..., field_type.flat_slice(order)].reshape(*x.shape[:-1], multiplicity, dim)


def _flatten_irrep_block(block: Tensor) -> Tensor:
    return block.reshape(*block.shape[:-2], block.shape[-2] * block.shape[-1])


def _rotate_irrep_block(block: Tensor, angle: Tensor, order: int) -> Tensor:
    if int(order) == 0:
        return block
    theta = angle * float(order)
    return _rotate_irrep_block_from_components(block, torch.cos(theta), torch.sin(theta))


def _rotate_irrep_block_from_components(
    block: Tensor,
    cos_value: Tensor,
    sin_value: Tensor,
) -> Tensor:
    """Rotate an irrep block using graph-static trigonometric components."""

    target_ndim = block.ndim - 1
    if cos_value.ndim == 1 and target_ndim >= 2:
        cos_value = cos_value.unsqueeze(0)
        sin_value = sin_value.unsqueeze(0)
    while cos_value.ndim < target_ndim:
        cos_value = cos_value.unsqueeze(-1)
        sin_value = sin_value.unsqueeze(-1)
    x0 = block[..., 0]
    x1 = block[..., 1]
    return torch.stack((cos_value * x0 - sin_value * x1, sin_value * x0 + cos_value * x1), dim=-1)


def _rotate_packed_irrep_field_from_components(
    field: Tensor,
    cos_values: Tensor,
    sin_values: Tensor,
) -> Tensor:
    """Rotate all non-scalar orders in a regular ``[..., M, 2L+1]`` field."""

    max_order = (int(field.shape[-1]) - 1) // 2
    if max_order == 0:
        return field
    cos_values = cos_values[..., :max_order]
    sin_values = sin_values[..., :max_order]
    vectors = field[..., 1:].reshape(*field.shape[:-1], max_order, 2)
    target_ndim = vectors.ndim - 1
    if cos_values.ndim == 2 and target_ndim >= 4:
        cos_values = cos_values.unsqueeze(0)
        sin_values = sin_values.unsqueeze(0)
    while cos_values.ndim < target_ndim:
        cos_values = cos_values.unsqueeze(-2)
        sin_values = sin_values.unsqueeze(-2)
    x0 = vectors[..., 0]
    x1 = vectors[..., 1]
    rotated = torch.stack(
        (cos_values * x0 - sin_values * x1, sin_values * x0 + cos_values * x1),
        dim=-1,
    )
    return torch.cat((field[..., :1], rotated.flatten(start_dim=-2)), dim=-1)


def _rotate_single_radial_packed_irrep_field(
    field: Tensor,
    scaled_cos_values: Tensor,
    scaled_sin_values: Tensor,
    radial_scale: Tensor,
) -> Tensor:
    """Apply input rotation and a graph-static single-radial scale together."""

    max_order = (int(field.shape[-1]) - 1) // 2
    scale = radial_scale.view(1, -1, *([1] * (field.ndim - 2)))
    if max_order == 0:
        return field.mul_(scale)
    scaled_cos_values = scaled_cos_values[..., :max_order]
    scaled_sin_values = scaled_sin_values[..., :max_order]
    vectors = field[..., 1:].reshape(*field.shape[:-1], max_order, 2)
    cos_values = scaled_cos_values.unsqueeze(0)
    sin_values = scaled_sin_values.unsqueeze(0)
    while cos_values.ndim < vectors.ndim - 1:
        cos_values = cos_values.unsqueeze(-2)
        sin_values = sin_values.unsqueeze(-2)
    x0 = vectors[..., 0]
    x1 = vectors[..., 1]
    rotated = torch.stack(
        (cos_values * x0 - sin_values * x1, sin_values * x0 + cos_values * x1),
        dim=-1,
    )
    return torch.cat(
        (field[..., :1] * scale, rotated.flatten(start_dim=-2)),
        dim=-1,
    )


def _irrep_rotation_matrix(angle: Tensor, order: int) -> Tensor:
    if int(order) == 0:
        return torch.ones(*angle.shape, 1, 1, device=angle.device, dtype=angle.dtype)
    theta = angle * float(order)
    cos_value = torch.cos(theta)
    sin_value = torch.sin(theta)
    row0 = torch.stack((cos_value, -sin_value), dim=-1)
    row1 = torch.stack((sin_value, cos_value), dim=-1)
    return torch.stack((row0, row1), dim=-2)


def _rotate_irrep_field(x: Tensor, field_type: SO2IrrepFieldType, angle: Tensor) -> Tensor:
    blocks = []
    for order in field_type.orders:
        block = _irrep_block(x, field_type, order)
        blocks.append(_flatten_irrep_block(_rotate_irrep_block(block, angle, order)))
    return torch.cat(blocks, dim=-1)


class _IrrepPointwiseSO2(nn.Module):
    """Pointwise linear map that commutes with SO(2) gauge actions."""

    def __init__(
        self,
        in_type: SO2IrrepFieldType,
        out_type: SO2IrrepFieldType,
        *,
        identity_init: bool = False,
    ) -> None:
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.identity_init = bool(identity_init)
        self.scalar_weights = nn.ParameterDict()
        self.vector_i_weights = nn.ParameterDict()
        self.vector_j_weights = nn.ParameterDict()
        for order in out_type.orders:
            in_m = in_type.multiplicity(order)
            out_m = out_type.multiplicity(order)
            if in_m == 0 or out_m == 0:
                continue
            key = str(order)
            if order == 0:
                self.scalar_weights[key] = nn.Parameter(torch.empty(out_m, in_m))
            else:
                self.vector_i_weights[key] = nn.Parameter(torch.empty(out_m, in_m))
                self.vector_j_weights[key] = nn.Parameter(torch.empty(out_m, in_m))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.scalar_weights.values():
            if self.identity_init:
                with torch.no_grad():
                    parameter.zero_()
                    parameter.diagonal().fill_(1.0)
            else:
                nn.init.xavier_uniform_(parameter)
        for key, weight_i in self.vector_i_weights.items():
            weight_j = self.vector_j_weights[key]
            if self.identity_init:
                with torch.no_grad():
                    weight_i.zero_()
                    weight_j.zero_()
                    weight_i.diagonal().fill_(1.0)
            else:
                gain = 1.0 / math.sqrt(2.0)
                nn.init.xavier_uniform_(weight_i, gain=gain)
                nn.init.xavier_uniform_(weight_j, gain=gain)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, N, C_in]")
        if int(x.shape[-1]) != self.in_type.total_dim:
            raise ValueError(f"expected {self.in_type.total_dim} channels, got {int(x.shape[-1])}")
        outputs = []
        for order in self.out_type.orders:
            in_m = self.in_type.multiplicity(order)
            out_m = self.out_type.multiplicity(order)
            dim = self.out_type.irrep_dim(order)
            if in_m == 0:
                outputs.append(x.new_zeros(*x.shape[:-1], out_m * dim))
                continue
            block = _irrep_block(x, self.in_type, order)
            key = str(order)
            if order == 0:
                mixed = torch.einsum(
                    "oi,bnid->bnod",
                    self.scalar_weights[key].to(device=x.device, dtype=x.dtype),
                    block,
                )
            else:
                weight_i = self.vector_i_weights[key].to(device=x.device, dtype=x.dtype)
                weight_j = self.vector_j_weights[key].to(device=x.device, dtype=x.dtype)
                j_block = torch.stack((-block[..., 1], block[..., 0]), dim=-1)
                mixed = torch.einsum("oi,bnid->bnod", weight_i, block)
                mixed = mixed + torch.einsum("oi,bnid->bnod", weight_j, j_block)
            outputs.append(_flatten_irrep_block(mixed))
        return torch.cat(outputs, dim=-1)


@dataclass(frozen=True)
class QuadratureDiskRule:
    """Quadrature points and weights on a tangent disk."""

    radial: int
    angular: int
    radius_km: float
    points: Tensor
    r: Tensor
    theta: Tensor
    weight: Tensor


def _gauss_legendre_nodes_weights(count: int) -> tuple[Tensor, Tensor]:
    if int(count) < 1:
        raise ValueError("quadrature radial count must be positive")
    count = int(count)
    if count == 1:
        return torch.zeros(1, dtype=torch.float64), torch.full((1,), 2.0, dtype=torch.float64)
    i = torch.arange(1, count, dtype=torch.float64)
    beta = i / torch.sqrt(4.0 * i * i - 1.0)
    jacobi = torch.zeros(count, count, dtype=torch.float64)
    jacobi[:-1, 1:] += torch.diag(beta)
    jacobi[1:, :-1] += torch.diag(beta)
    nodes, eigenvectors = torch.linalg.eigh(jacobi)
    weights = 2.0 * eigenvectors[0].square()
    return nodes, weights


def quadrature_disk_rule(
    radial: int,
    angular: int,
    *,
    radius_km: float = 1.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> QuadratureDiskRule:
    """Return the paper's Gauss-radial/uniform-angular disk quadrature."""

    if int(angular) < 1:
        raise ValueError("quadrature angular count must be positive")
    if float(radius_km) <= 0.0:
        raise ValueError("radius_km must be positive")
    device = torch.device(device) if device is not None else torch.device("cpu")
    radial = int(radial)
    angular = int(angular)
    nodes, node_weights = _gauss_legendre_nodes_weights(radial)
    r_unit = torch.sqrt((nodes + 1.0) * 0.5)
    radial_weight = node_weights * 0.5
    theta = -math.pi + (torch.arange(1, angular + 1, dtype=torch.float64) * (2.0 * math.pi / angular))
    r_grid = r_unit.view(radial, 1).expand(radial, angular).reshape(-1)
    theta_grid = theta.view(1, angular).expand(radial, angular).reshape(-1)
    weight = (radial_weight.view(radial, 1).expand(radial, angular) / float(angular)).reshape(-1)
    radius = float(radius_km)
    points = radius * r_grid.unsqueeze(-1) * torch.stack((torch.cos(theta_grid), torch.sin(theta_grid)), dim=-1)
    return QuadratureDiskRule(
        radial=radial,
        angular=angular,
        radius_km=radius,
        points=points.to(device=device, dtype=dtype),
        r=(radius * r_grid).to(device=device, dtype=dtype),
        theta=theta_grid.to(device=device, dtype=dtype),
        weight=weight.to(device=device, dtype=dtype),
    )


@dataclass(frozen=True)
class _IrrepConvGeometry:
    """Immutable tensors shared by convolutions bound to one sphere graph."""

    graph: SphereGraphGeometry
    n_points: int
    center_idx: Tensor
    neighbor_idx: Tensor
    triton_center_idx: Tensor
    triton_neighbor_idx: Tensor
    center_ptr: Tensor
    neighbor_ptr: Tensor
    edges_by_neighbor: Tensor
    degree_bucket: int
    neighbor_count: Tensor
    transport_angle: Tensor
    radial_basis: Tensor | None
    input_rotation_cos: Tensor
    input_rotation_sin: Tensor
    output_rotation_cos: Tensor
    output_rotation_sin: Tensor
    single_radial_input_cos: Tensor
    single_radial_input_sin: Tensor
    quadrature_bases: dict[tuple[int, int], Tensor]


class IrrepSphereConv(_RadialBasisConvBase):
    """Gauge-equivariant sphere graph convolution for SO(2) irrep fields."""

    def __init__(
        self,
        in_type: SO2IrrepFieldType,
        out_type: SO2IrrepFieldType,
        radius_km: float,
        num_radial: int,
        *,
        radial_sigma_km: float | None = None,
        normalize_radial_basis: bool = True,
        normalize_by_neighbors: bool = True,
        include_self: bool = True,
        self_identity_init: bool = False,
        quadrature: bool = False,
        quadrature_radial: int | None = None,
        quadrature_angular: int = 16,
        quadrature_sigma_km: float | None = None,
        backend: Literal["auto", "torch", "triton"] = "auto",
        regular_r1_variant: Literal["auto", "fused", "semi_packed"] = "auto",
        triton_workspace_mib: int = 512,
    ) -> None:
        super().__init__(
            radius_km,
            num_radial,
            radial_sigma_km=radial_sigma_km,
            normalize_radial_basis=normalize_radial_basis,
            normalize_by_neighbors=normalize_by_neighbors,
        )
        self.in_type = in_type
        self.out_type = out_type
        self._in_orders = in_type.orders
        self._out_orders = out_type.orders
        self.quadrature = bool(quadrature)
        self.quadrature_radial = int(quadrature_radial) if quadrature_radial is not None else int(num_radial)
        self.quadrature_angular = int(quadrature_angular)
        self.quadrature_sigma_km = float(quadrature_sigma_km) if quadrature_sigma_km is not None else None
        self.backend = str(backend)
        if self.backend not in ("auto", "torch", "triton"):
            raise ValueError("backend must be 'auto', 'torch', or 'triton'")
        self.regular_r1_variant = str(regular_r1_variant)
        if self.regular_r1_variant not in ("auto", "fused", "semi_packed"):
            raise ValueError(
                "regular_r1_variant must be 'auto', 'fused', or 'semi_packed'"
            )
        if isinstance(triton_workspace_mib, bool) or int(triton_workspace_mib) < 0:
            raise ValueError("triton_workspace_mib must be a non-negative integer")
        self.triton_workspace_mib = int(triton_workspace_mib)
        if self.quadrature_radial < 1:
            raise ValueError("quadrature_radial must be positive")
        if self.quadrature_angular < 1:
            raise ValueError("quadrature_angular must be positive")
        if self.quadrature_sigma_km is not None and self.quadrature_sigma_km <= 0.0:
            raise ValueError("quadrature_sigma_km must be positive")
        if self.quadrature:
            unit_rule = quadrature_disk_rule(
                self.quadrature_radial,
                self.quadrature_angular,
                radius_km=1.0,
                dtype=torch.float32,
            )
            self.register_buffer("quadrature_unit_points", unit_rule.points)
            self.register_buffer("quadrature_unit_r", unit_rule.r)
            self.register_buffer("quadrature_theta", unit_rule.theta)
            self.register_buffer("quadrature_weight", unit_rule.weight)
        self._prepared_geometry: _IrrepConvGeometry | None = None
        # Private benchmark/debug switch. It is deliberately not a parameter or
        # buffer, so checkpoints and the public constructor stay unchanged.
        self._irregular_r1_fast_path = True
        self._weight_keys: dict[tuple[int, int], str] = {}
        self._regular_weights = in_type.is_regular() and out_type.is_regular()
        self._use_packed_fast_path = (
            self._regular_weights
            and not self.quadrature
            and len(self._in_orders) * len(self._out_orders) > 1
        )
        if self._regular_weights:
            in_m = in_type.multiplicity(0)
            out_m = out_type.multiplicity(0)
            self.packed_weight = nn.Parameter(
                torch.empty(
                    num_radial,
                    out_m,
                    2 * out_type.max_order + 1,
                    in_m,
                    2 * in_type.max_order + 1,
                )
            )
            input_pack_index = self._regular_pack_index(in_type)
            output_pack_index = self._regular_pack_index(out_type)
            self.register_buffer("_input_pack_index", input_pack_index, persistent=False)
            self.register_buffer("_triton_input_pack_index", input_pack_index.to(torch.int32), persistent=False)
            self.register_buffer("_triton_output_pack_index", output_pack_index.to(torch.int32), persistent=False)
            self.register_buffer(
                "_output_unpack_index",
                torch.argsort(output_pack_index),
                persistent=False,
            )
        else:
            self.weights = nn.ParameterDict()
            for out_order in self._out_orders:
                out_m, out_dim = out_type.block_shape(out_order)
                for in_order in self._in_orders:
                    in_m, in_dim = in_type.block_shape(in_order)
                    key = self._weight_key(out_order, in_order)
                    self._weight_keys[out_order, in_order] = key
                    self.weights[key] = nn.Parameter(
                        torch.empty(num_radial, out_m, in_m, out_dim, in_dim)
                    )
        self.self_mixing = (
            _IrrepPointwiseSO2(in_type, out_type, identity_init=self_identity_init)
            if include_self
            else None
        )
        self.reset_parameters()

    def __setstate__(self, state: dict) -> None:
        """Load profiler payloads created before regular R1 controls existed."""

        super().__setstate__(state)
        self.__dict__.setdefault("regular_r1_variant", "auto")
        self.__dict__.setdefault("triton_workspace_mib", 512)

    @staticmethod
    def _weight_key(out_order: int, in_order: int) -> str:
        return f"o{int(out_order)}_i{int(in_order)}"

    @staticmethod
    def _packed_order_slice(order: int) -> slice:
        order = int(order)
        if order == 0:
            return slice(0, 1)
        return slice(2 * order - 1, 2 * order + 1)

    @classmethod
    def _regular_pack_index(cls, field_type: SO2IrrepFieldType) -> Tensor:
        multiplicity = field_type.multiplicity(0)
        indices = []
        for channel in range(multiplicity):
            for order in field_type.orders:
                flat_slice = field_type.flat_slice(order)
                dim = field_type.irrep_dim(order)
                indices.extend(
                    flat_slice.start + channel * dim + component
                    for component in range(dim)
                )
        return torch.tensor(indices, dtype=torch.long)

    def _weight_block(self, out_order: int, in_order: int) -> Tensor:
        """Return one canonical irrep block in the legacy blockwise layout."""

        if not self._regular_weights:
            return self.weights[self._weight_keys[out_order, in_order]]
        out_slice = self._packed_order_slice(out_order)
        in_slice = self._packed_order_slice(in_order)
        return self.packed_weight[:, :, out_slice, :, in_slice].permute(0, 1, 3, 2, 4)

    def reset_parameters(self) -> None:
        if self._regular_weights:
            for out_order in self._out_orders:
                for in_order in self._in_orders:
                    _reset_weight(self._weight_block(out_order, in_order))
        else:
            for parameter in self.weights.values():
                _reset_weight(parameter)

    def _apply(self, fn, recurse: bool = True):
        self.clear_prepared_graph()
        return super()._apply(fn, recurse=recurse)

    def _geometry_signature(self) -> tuple[object, ...]:
        """Configuration that must match before two layers share geometry."""

        return (
            self.radius_km,
            self.num_radial,
            self.radial_sigma_km,
            self._radial_sigma_is_default,
            self.normalize_radial_basis,
            self.normalize_by_neighbors,
            self.origin_eps_km,
            self.quadrature,
            self.quadrature_radial,
            self.quadrature_angular,
            self.quadrature_sigma_km,
        )

    def prepare_graph(
        self,
        graph: SphereGraphGeometry,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Precompute immutable graph/layer geometry before the first batch."""

        target_device = torch.device(device) if device is not None else self.centers_km.device
        target_dtype = dtype or self.centers_km.dtype
        self._prepared_geometry = self._build_geometry(
            graph,
            target_device,
            target_dtype,
            in_orders=self._in_orders,
            out_orders=self._out_orders,
        )

    def clear_prepared_graph(self) -> None:
        """Drop the static graph binding without changing learned parameters."""

        self._prepared_geometry = None

    def _bind_prepared_geometry(self, geometry: _IrrepConvGeometry) -> None:
        """Bind geometry built by a compatible layer (used by model geometry banks)."""

        def compact_rotation(tensor: Tensor, max_order: int) -> Tensor:
            narrowed = tensor[:, :max_order]
            if max_order == 0 or narrowed.stride() == (max_order, 1):
                return narrowed
            # Triton kernels use a dense edge/order layout. A view from a wider
            # shared geometry bank retains the bank's larger row stride.
            return narrowed.contiguous()

        max_in_order = self.in_type.max_order
        max_out_order = self.out_type.max_order
        self._prepared_geometry = replace(
            geometry,
            input_rotation_cos=compact_rotation(geometry.input_rotation_cos, max_in_order),
            input_rotation_sin=compact_rotation(geometry.input_rotation_sin, max_in_order),
            output_rotation_cos=compact_rotation(geometry.output_rotation_cos, max_out_order),
            output_rotation_sin=compact_rotation(geometry.output_rotation_sin, max_out_order),
            single_radial_input_cos=compact_rotation(
                geometry.single_radial_input_cos, max_in_order,
            ),
            single_radial_input_sin=compact_rotation(
                geometry.single_radial_input_sin, max_in_order,
            ),
        )

    def _build_geometry(
        self,
        graph: SphereGraphGeometry,
        device: torch.device,
        dtype: torch.dtype,
        *,
        in_orders: Sequence[int],
        out_orders: Sequence[int],
    ) -> _IrrepConvGeometry:
        def move_float(tensor: Tensor) -> Tensor:
            if tensor.device == device and tensor.dtype == dtype:
                return tensor
            return tensor.to(device=device, dtype=dtype)

        def move_index(tensor: Tensor) -> Tensor:
            if tensor.device == device and tensor.dtype == torch.long:
                return tensor
            return tensor.to(device=device, dtype=torch.long)

        with torch.no_grad():
            center_idx = move_index(graph.center_idx)
            neighbor_idx = move_index(graph.neighbor_idx)
            assert graph.transport_angle is not None
            transport_angle = move_float(graph.transport_angle)
            edge_count = int(center_idx.shape[0])
            input_rotation_cos = torch.empty((edge_count, 0), device=device, dtype=dtype)
            input_rotation_sin = torch.empty((edge_count, 0), device=device, dtype=dtype)
            output_rotation_cos = torch.empty((edge_count, 0), device=device, dtype=dtype)
            output_rotation_sin = torch.empty((edge_count, 0), device=device, dtype=dtype)
            single_radial_input_cos = torch.empty((edge_count, 0), device=device, dtype=dtype)
            single_radial_input_sin = torch.empty((edge_count, 0), device=device, dtype=dtype)
            quadrature_bases: dict[tuple[int, int], Tensor] = {}
            radial_basis = None

            if self.quadrature:
                neighbor_count = torch.empty(0, device=device, dtype=dtype)
                interpolation = self._quadrature_interpolation(
                    graph,
                    device=device,
                    dtype=dtype,
                )
                for out_order in out_orders:
                    for in_order in in_orders:
                        basis = self._quadrature_basis(
                            transport_angle,
                            out_order,
                            in_order,
                            interpolation=interpolation,
                        )
                        quadrature_bases[out_order, in_order] = basis.to(dtype=dtype)
                del interpolation
                empty_index = torch.empty(0, device=device, dtype=torch.int32)
                triton_center_idx = empty_index
                triton_neighbor_idx = empty_index
                center_ptr = empty_index
                neighbor_ptr = empty_index
                edges_by_neighbor = empty_index
                degree_bucket = 0
            else:
                r = move_float(graph.r)
                phi = move_float(graph.phi)
                active = self.active_edge_weight(r)
                radial_basis = self.radial_basis(r, radius_km=graph.radius_km)
                radial_basis = radial_basis * active.unsqueeze(-1)
                if self.normalize_by_neighbors:
                    neighbor_count = torch.zeros(graph.n_points, device=device, dtype=dtype)
                    neighbor_count.scatter_add_(0, center_idx, active)
                    neighbor_count.clamp_min_(1.0)
                else:
                    neighbor_count = torch.empty(0, device=device, dtype=dtype)

                input_angle = transport_angle - phi
                max_in_order = max(in_orders, default=0)
                if max_in_order > 0:
                    orders = torch.arange(1, max_in_order + 1, device=device, dtype=dtype)
                    theta = input_angle.unsqueeze(-1) * orders
                    input_rotation_cos = torch.cos(theta)
                    input_rotation_sin = torch.sin(theta)
                    if self.num_radial == 1:
                        radial_scale = radial_basis[:, :1]
                        single_radial_input_cos = input_rotation_cos * radial_scale
                        single_radial_input_sin = input_rotation_sin * radial_scale
                max_out_order = max(out_orders, default=0)
                if max_out_order > 0:
                    orders = torch.arange(1, max_out_order + 1, device=device, dtype=dtype)
                    theta = phi.unsqueeze(-1) * orders
                    output_rotation_cos = torch.cos(theta)
                    output_rotation_sin = torch.sin(theta)

                edge_order = torch.argsort(center_idx, stable=True)
                center_idx = center_idx.index_select(0, edge_order)
                neighbor_idx = neighbor_idx.index_select(0, edge_order)
                transport_angle = transport_angle.index_select(0, edge_order)
                radial_basis = radial_basis.index_select(0, edge_order)
                input_rotation_cos = input_rotation_cos.index_select(0, edge_order)
                input_rotation_sin = input_rotation_sin.index_select(0, edge_order)
                output_rotation_cos = output_rotation_cos.index_select(0, edge_order)
                output_rotation_sin = output_rotation_sin.index_select(0, edge_order)
                single_radial_input_cos = single_radial_input_cos.index_select(0, edge_order)
                single_radial_input_sin = single_radial_input_sin.index_select(0, edge_order)

                center_counts = torch.bincount(center_idx, minlength=graph.n_points).to(torch.int32)
                center_ptr = torch.zeros(graph.n_points + 1, device=device, dtype=torch.int32)
                center_ptr[1:] = torch.cumsum(center_counts, dim=0)
                neighbor_order = torch.argsort(neighbor_idx, stable=True)
                neighbor_counts = torch.bincount(neighbor_idx, minlength=graph.n_points).to(torch.int32)
                max_degree = max(
                    int(center_counts.max().item()),
                    int(neighbor_counts.max().item()),
                )
                degree_bucket = max(8, 1 << max(0, max_degree - 1).bit_length())
                neighbor_ptr = torch.zeros(graph.n_points + 1, device=device, dtype=torch.int32)
                neighbor_ptr[1:] = torch.cumsum(neighbor_counts, dim=0)
                triton_center_idx = center_idx.to(torch.int32)
                triton_neighbor_idx = neighbor_idx.to(torch.int32)
                edges_by_neighbor = neighbor_order.to(torch.int32)

        prepared = _IrrepConvGeometry(
            graph=graph,
            n_points=graph.n_points,
            center_idx=center_idx,
            neighbor_idx=neighbor_idx,
            triton_center_idx=triton_center_idx,
            triton_neighbor_idx=triton_neighbor_idx,
            center_ptr=center_ptr,
            neighbor_ptr=neighbor_ptr,
            edges_by_neighbor=edges_by_neighbor,
            degree_bucket=degree_bucket,
            neighbor_count=neighbor_count,
            transport_angle=transport_angle,
            radial_basis=radial_basis,
            input_rotation_cos=input_rotation_cos,
            input_rotation_sin=input_rotation_sin,
            output_rotation_cos=output_rotation_cos,
            output_rotation_sin=output_rotation_sin,
            single_radial_input_cos=single_radial_input_cos,
            single_radial_input_sin=single_radial_input_sin,
            quadrature_bases=quadrature_bases,
        )
        return prepared

    def _quadrature_sigma_for(self, graph: SphereGraphGeometry) -> float:
        if self.quadrature_sigma_km is not None:
            return self.quadrature_sigma_km
        if self._radial_sigma_is_default:
            return default_radial_sigma(float(graph.radius_km), self.num_radial)
        return self.radial_sigma_km

    def _quadrature_interpolation(
        self,
        graph: SphereGraphGeometry,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not self.quadrature:
            raise RuntimeError("quadrature interpolation requested for a non-quadrature layer")
        sigma = self._quadrature_sigma_for(graph)

        def move_float(tensor: Tensor) -> Tensor:
            if tensor.device == device and tensor.dtype == dtype:
                return tensor
            return tensor.to(device=device, dtype=dtype)

        q_points = move_float(self.quadrature_unit_points) * float(graph.radius_km)
        q_r = move_float(self.quadrature_unit_r) * float(graph.radius_km)
        q_theta = move_float(self.quadrature_theta)
        q_weight = move_float(self.quadrature_weight)
        u = move_float(graph.u)
        r = move_float(graph.r)
        center_idx = (
            graph.center_idx
            if graph.center_idx.device == device
            else graph.center_idx.to(device=device)
        )
        dist2 = (u.unsqueeze(1) - q_points.unsqueeze(0)).square().sum(dim=-1)
        sigma_t = torch.as_tensor(sigma, device=device, dtype=dtype).clamp_min(1e-12)
        interp = torch.exp(-dist2 / (sigma_t * sigma_t))
        interp = interp * self.active_edge_weight(r).unsqueeze(-1)
        normalizer = interp.new_zeros((graph.n_points, int(q_points.shape[0])))
        normalizer.scatter_add_(0, center_idx.view(-1, 1).expand_as(interp), interp)
        coeff = q_weight.view(1, -1) * interp / normalizer[center_idx].clamp_min(1e-12)
        radial_basis = self.radial_basis(q_r, radius_km=graph.radius_km)
        return coeff, radial_basis, q_theta

    def _quadrature_basis(
        self,
        transport_angle: Tensor,
        out_order: int,
        in_order: int,
        *,
        interpolation: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        coeff, radial_basis, q_theta = interpolation
        out_rotation = _irrep_rotation_matrix(q_theta, out_order)
        in_rotation = _irrep_rotation_matrix(transport_angle.unsqueeze(-1) - q_theta.view(1, -1), in_order)
        return torch.einsum("eq,qr,qab,eqcd->erabcd", coeff, radial_basis, out_rotation, in_rotation)

    def forward(self, x: Tensor, graph: SphereGraphGeometry) -> Tensor:
        geometry = self._prepared_geometry
        if geometry is None:
            raise RuntimeError("graph geometry is not prepared; call prepare_graph() before forward()")
        if graph is not geometry.graph:
            raise RuntimeError("the layer is bound to a different graph; call prepare_graph() to replace it")
        return self.forward_prepared(x)

    def forward_prepared(self, x: Tensor) -> Tensor:
        """Run the tensor-only convolution using the statically bound geometry."""

        if x.ndim != 3:
            raise ValueError("x must have shape [B, N, C_in]")
        if int(x.shape[-1]) != self.in_type.total_dim:
            raise ValueError(f"expected {self.in_type.total_dim} channels, got {int(x.shape[-1])}")
        geometry = self._prepared_geometry
        if geometry is None:
            raise RuntimeError("graph geometry is not prepared; call prepare_graph() before forward()")
        if int(x.shape[1]) != geometry.n_points:
            raise ValueError(f"x has {int(x.shape[1])} points, expected {geometry.n_points}")
        if geometry.center_idx.device != x.device or geometry.transport_angle.dtype != x.dtype:
            raise RuntimeError(
                "prepared geometry device/dtype does not match x; call prepare_graph() again"
            )

        capture = getattr(self, "_irrep_profile_capture", None)
        if capture is not None:
            capture(self, x, geometry)
        use_triton = self._should_use_triton(x, geometry)
        profile_name = getattr(self, "_irrep_profile_name", "IrrepSphereConv")
        return record_region_call(
            f"module/{profile_name}",
            self._forward_prepared_backend,
            x,
            geometry,
            use_triton,
        )

    def _forward_prepared_backend(
        self, x: Tensor, geometry: _IrrepConvGeometry, use_triton: bool
    ) -> Tensor:
        if use_triton:
            out = self._forward_spatial_triton(x, geometry)
        elif self._use_packed_fast_path:
            out = self._forward_spatial_packed(x, geometry)
        else:
            out = self._forward_spatial_blockwise(x, geometry)
        if self.self_mixing is not None:
            mixed = record_region_call("op/conv.self_mixing", self.self_mixing, x)
            out = out + mixed
        return out

    def _triton_unsupported_reason(self, x: Tensor, geometry: _IrrepConvGeometry) -> str | None:
        if self.quadrature:
            return "Triton backend only supports quadrature=False convolutions"
        if self._regular_weights:
            reason = triton_support_reason(
                x,
                self.packed_weight,
                self.in_type.max_order,
                self.out_type.max_order,
            )
            if reason is not None:
                return reason
            if self.num_radial == 1 and self.regular_r1_variant == "semi_packed":
                return semi_packed_support_reason(
                    x,
                    self.packed_weight,
                    geometry.degree_bucket,
                    self.triton_workspace_mib << 20,
                )
            return None
        for out_order in self._out_orders:
            for in_order in self._in_orders:
                reason = irregular_pair_support_reason(
                    x,
                    self._weight_block(out_order, in_order),
                    in_order,
                    out_order,
                )
                if reason is not None:
                    return reason
        return None
    
    def _should_use_triton(self, x: Tensor, geometry: _IrrepConvGeometry) -> bool:
        if self.backend == "torch":
            return False
        reason = self._triton_unsupported_reason(x, geometry)
        if reason is not None:
            if self.backend == "triton":
                raise RuntimeError(reason)
            return False
        if self.backend == "triton":
            return True
        # The irregular path is memory-first: use it for every supported CUDA
        # invocation instead of materializing blockwise edge activations.
        if not self._regular_weights:
            return True
        workload = int(x.shape[0]) * int(geometry.neighbor_idx.numel()) * self.out_type.total_dim
        major, _minor = torch.cuda.get_device_capability(x.device)
        if major == 7 and x.dtype == torch.float32 and self.num_radial == 1:
            return False
        architecture = 7 if major == 7 else 8
        training = torch.is_grad_enabled() and (x.requires_grad or self.packed_weight.requires_grad)
        thresholds = AUTO_TRAINING_WORK_THRESHOLDS if training else AUTO_INFERENCE_WORK_THRESHOLDS
        threshold = thresholds.get(architecture, AUTO_WORK_THRESHOLD)
        return workload >= threshold

    def _forward_spatial_triton(self, x: Tensor, geometry: _IrrepConvGeometry) -> Tensor:
        if not self._regular_weights:
            return self._forward_spatial_irregular_triton(x, geometry)
        assert geometry.radial_basis is not None
        major, _minor = torch.cuda.get_device_capability(x.device)
        allow_tf32 = bool(major >= 8 and torch.backends.cuda.matmul.allow_tf32)
        variant = self._selected_regular_r1_variant(x, geometry)
        if variant == "semi_packed":
            return record_region_call(
                "op/conv.triton_semi_packed",
                semi_packed_irrep_conv,
                x,
                self.packed_weight,
                geometry.triton_center_idx,
                geometry.triton_neighbor_idx,
                geometry.center_ptr,
                geometry.neighbor_ptr,
                geometry.edges_by_neighbor,
                geometry.radial_basis,
                geometry.input_rotation_cos,
                geometry.input_rotation_sin,
                geometry.output_rotation_cos,
                geometry.output_rotation_sin,
                self._triton_input_pack_index,
                self._triton_output_pack_index,
                geometry.neighbor_count,
                geometry.degree_bucket,
                self.normalize_by_neighbors,
                allow_tf32,
                self.triton_workspace_mib << 20,
            )
        return record_region_call(
            "op/conv.triton_fused",
            packed_irrep_conv,
            x,
            self.packed_weight,
            geometry.triton_center_idx,
            geometry.triton_neighbor_idx,
            geometry.center_ptr,
            geometry.neighbor_ptr,
            geometry.edges_by_neighbor,
            geometry.radial_basis,
            geometry.input_rotation_cos,
            geometry.input_rotation_sin,
            geometry.output_rotation_cos,
            geometry.output_rotation_sin,
            self._triton_input_pack_index,
            self._triton_output_pack_index,
            geometry.neighbor_count,
            geometry.degree_bucket,
            self.normalize_by_neighbors,
            allow_tf32,
        )

    def _selected_regular_r1_variant(
        self,
        x: Tensor,
        geometry: _IrrepConvGeometry,
    ) -> Literal["fused", "semi_packed"]:
        """Resolve the regular R1 implementation without synchronizing CUDA."""

        if self.num_radial != 1 or not self._regular_weights:
            return "fused"
        if self.regular_r1_variant == "fused":
            return "fused"
        workspace_bytes = self.triton_workspace_mib << 20
        reason = semi_packed_support_reason(
            x, self.packed_weight, geometry.degree_bucket, workspace_bytes,
        )
        if self.regular_r1_variant == "semi_packed":
            if reason is not None:
                raise RuntimeError(reason)
            return "semi_packed"
        if reason is None and should_use_semi_packed(
            x, self.packed_weight, geometry.degree_bucket, workspace_bytes,
            training=(
                torch.is_grad_enabled()
                and (x.requires_grad or self.packed_weight.requires_grad)
            ),
        ):
            return "semi_packed"
        return "fused"

    def _forward_spatial_irregular_triton(
        self,
        x: Tensor,
        geometry: _IrrepConvGeometry,
    ) -> Tensor:
        """Run one fused, edge-free Triton op per active order pair."""

        assert geometry.radial_basis is not None
        major, _minor = torch.cuda.get_device_capability(x.device)
        allow_tf32 = bool(major >= 8 and torch.backends.cuda.matmul.allow_tf32)
        outputs_by_order: list[Tensor] = []
        for out_order in self._out_orders:
            out_block: Tensor | None = None
            for in_order in self._in_orders:
                pair = irregular_irrep_pair_conv(
                    _irrep_block(x, self.in_type, in_order),
                    self._weight_block(out_order, in_order),
                    geometry.triton_center_idx,
                    geometry.triton_neighbor_idx,
                    geometry.center_ptr,
                    geometry.neighbor_ptr,
                    geometry.edges_by_neighbor,
                    geometry.radial_basis,
                    geometry.input_rotation_cos,
                    geometry.input_rotation_sin,
                    geometry.single_radial_input_cos,
                    geometry.single_radial_input_sin,
                    geometry.output_rotation_cos,
                    geometry.output_rotation_sin,
                    geometry.neighbor_count,
                    in_order,
                    out_order,
                    geometry.degree_bucket,
                    self.normalize_by_neighbors,
                    self._irregular_r1_fast_path,
                    allow_tf32,
                )
                out_block = pair if out_block is None else out_block + pair
            assert out_block is not None
            outputs_by_order.append(_flatten_irrep_block(out_block))
        return torch.cat(outputs_by_order, dim=-1)

    def _forward_spatial_packed(self, x: Tensor, geometry: _IrrepConvGeometry) -> Tensor:
        """Regular convolution with one packed gather/scatter edge pipeline."""

        batch = int(x.shape[0])
        in_m = self.in_type.multiplicity(0)
        in_dim = 2 * self.in_type.max_order + 1
        packed = record_region_call(
            "op/conv.pack",
            _pack_irrep_field,
            x,
            self._input_pack_index,
            batch,
            geometry.n_points,
            in_m,
            in_dim,
        )
        x_q = record_region_call(
            "op/conv.gather", _gather_irrep_neighbors, packed, geometry.neighbor_idx
        )
        assert geometry.radial_basis is not None
        if self.num_radial == 1:
            x_q = record_region_call(
                "op/conv.input_rotation",
                _rotate_single_radial_packed_irrep_field,
                x_q,
                geometry.single_radial_input_cos,
                geometry.single_radial_input_sin,
                geometry.radial_basis[:, 0],
            )
        else:
            x_q = record_region_call(
                "op/conv.input_rotation",
                _rotate_packed_irrep_field_from_components,
                x_q,
                geometry.input_rotation_cos,
                geometry.input_rotation_sin,
            )
        weight = self.packed_weight
        if weight.device != x.device or weight.dtype != x.dtype:
            weight = weight.to(device=x.device, dtype=x.dtype)
        weighted = record_region_call(
            "op/conv.weight_einsum", torch.einsum, "roaid,beid->beroa", weight, x_q
        )
        if self.num_radial == 1:
            contrib = record_region_call("op/conv.radial_einsum", torch.squeeze, weighted, 2)
        else:
            contrib = record_region_call(
                "op/conv.radial_einsum",
                torch.einsum,
                "er,beroa->beoa",
                geometry.radial_basis,
                weighted,
            )
        contrib = record_region_call(
            "op/conv.output_rotation",
            _rotate_packed_irrep_field_from_components,
            contrib,
            geometry.output_rotation_cos,
            geometry.output_rotation_sin,
        )
        out = record_region_call(
            "op/conv.scatter",
            _scatter_to_centers,
            contrib,
            geometry.center_idx,
            geometry.n_points,
            geometry.neighbor_count,
            normalize_by_neighbors=self.normalize_by_neighbors,
            edge_weight=None,
        )
        return record_region_call(
            "op/conv.unpack", _unpack_irrep_field, out, self._output_unpack_index
        )

    def _forward_spatial_blockwise(self, x: Tensor, geometry: _IrrepConvGeometry) -> Tensor:
        """Reference path for irregular, scalar-only, and quadrature convolutions."""

        inputs_by_order: dict[int, Tensor] = {}
        for in_order in self._in_orders:
            in_block = _irrep_block(x, self.in_type, in_order)
            x_q = record_region_call(
                "op/conv.blockwise_gather",
                _gather_irrep_neighbors,
                in_block,
                geometry.neighbor_idx,
            )
            if not self.quadrature:
                assert geometry.radial_basis is not None
                if self.num_radial == 1:
                    if in_order == 0:
                        x_q.mul_(geometry.radial_basis[:, 0].view(1, -1, 1, 1))
                    else:
                        cos_value = geometry.single_radial_input_cos[:, in_order - 1]
                        sin_value = geometry.single_radial_input_sin[:, in_order - 1]
                        x_q = _rotate_irrep_block_from_components(x_q, cos_value, sin_value)
                elif in_order > 0:
                    cos_value = geometry.input_rotation_cos[:, in_order - 1]
                    sin_value = geometry.input_rotation_sin[:, in_order - 1]
                    x_q = _rotate_irrep_block_from_components(x_q, cos_value, sin_value)
            inputs_by_order[in_order] = x_q

        outputs_by_order: list[Tensor] = []
        for out_order in self._out_orders:
            out_m, out_dim = self.out_type.block_shape(out_order)
            out_block = x.new_zeros((int(x.shape[0]), geometry.n_points, out_m, out_dim))
            for in_order in self._in_orders:
                x_q = inputs_by_order[in_order]
                weight = self._weight_block(out_order, in_order)
                if weight.device != x.device or weight.dtype != x.dtype:
                    weight = weight.to(device=x.device, dtype=x.dtype)
                if self.quadrature:
                    basis = geometry.quadrature_bases[out_order, in_order]
                    contrib = record_region_call(
                        "op/conv.quadrature_einsum",
                        _quadrature_contribution,
                        basis,
                        weight,
                        x_q,
                    )
                    normalize_by_neighbors = False
                else:
                    assert geometry.radial_basis is not None
                    weighted = record_region_call(
                        "op/conv.blockwise_weight_einsum",
                        torch.einsum,
                        "roiad,beid->beroa",
                        weight,
                        x_q,
                    )
                    if self.num_radial == 1:
                        contrib = weighted.squeeze(2)
                    else:
                        contrib = record_region_call(
                            "op/conv.blockwise_radial_einsum",
                            torch.einsum,
                            "er,beroa->beoa",
                            geometry.radial_basis,
                            weighted,
                        )

                    if out_order > 0:
                        cos_value = geometry.output_rotation_cos[:, out_order - 1]
                        sin_value = geometry.output_rotation_sin[:, out_order - 1]
                        contrib = _rotate_irrep_block_from_components(
                            contrib,
                            cos_value,
                            sin_value,
                        )
                    normalize_by_neighbors = self.normalize_by_neighbors
                scattered = record_region_call(
                    "op/conv.blockwise_scatter",
                    _scatter_to_centers,
                    contrib,
                    geometry.center_idx,
                    geometry.n_points,
                    geometry.neighbor_count,
                    normalize_by_neighbors=normalize_by_neighbors,
                    edge_weight=None,
                )
                out_block = out_block + scattered
            outputs_by_order.append(_flatten_irrep_block(out_block))

        out = torch.cat(outputs_by_order, dim=-1)
        return out


class IrrepBatchNorm(nn.Module):
    """Batch/variance normalization that preserves SO(2) field structure."""

    def __init__(
        self,
        field_type: SO2IrrepFieldType,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.field_type = field_type
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        scalar_m = field_type.multiplicity(0)
        self.scalar_bn = nn.BatchNorm1d(scalar_m, eps=eps, momentum=momentum, affine=affine) if scalar_m > 0 else None
        self.vector_orders = tuple(order for order in field_type.orders if order > 0)
        for order in self.vector_orders:
            multiplicity = field_type.multiplicity(order)
            self.register_buffer(f"running_var_{order}", torch.ones(multiplicity))
            if affine:
                self.register_parameter(f"weight_{order}", nn.Parameter(torch.ones(multiplicity)))
            else:
                self.register_parameter(f"weight_{order}", None)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, N, C]")
        if int(x.shape[-1]) != self.field_type.total_dim:
            raise ValueError(f"expected {self.field_type.total_dim} channels, got {int(x.shape[-1])}")
        return record_region_call("op/batch_norm", self._forward_impl, x)

    def _forward_impl(self, x: Tensor) -> Tensor:
        outputs = []
        for order in self.field_type.orders:
            block = _irrep_block(x, self.field_type, order)
            if order == 0:
                if self.scalar_bn is None:
                    outputs.append(_flatten_irrep_block(block))
                else:
                    batch, points, multiplicity, _dim = block.shape
                    normalized = self.scalar_bn(block.reshape(batch * points, multiplicity))
                    outputs.append(normalized.reshape(batch, points, multiplicity))
                continue
            var = block.square().mean(dim=(0, 1, 3))
            running_var = getattr(self, f"running_var_{order}")
            if self.training:
                running_var.mul_(1.0 - self.momentum).add_(self.momentum * var.detach())
                used_var = var
            else:
                used_var = running_var.to(device=x.device, dtype=x.dtype)
            scale = torch.rsqrt(used_var.to(device=x.device, dtype=x.dtype).clamp_min(self.eps))
            weight = getattr(self, f"weight_{order}")
            if weight is not None:
                scale = scale * weight.to(device=x.device, dtype=x.dtype)
            outputs.append(_flatten_irrep_block(block * scale.view(1, 1, -1, 1)))
        return torch.cat(outputs, dim=-1)


class RegularNonlinearity(nn.Module):
    """Approximate irrep nonlinearity via regular representation samples."""

    def __init__(
        self,
        field_type: SO2IrrepFieldType,
        *,
        regular_samples: int | None = None,
    ) -> None:
        super().__init__()
        if not field_type.is_regular():
            raise ValueError("RegularNonlinearity expects equal positive multiplicity for rho_0..rho_M")
        self.field_type = field_type
        self.regular_samples = int(regular_samples) if regular_samples is not None else max(8, 2 * field_type.max_order + 1)
        if self.regular_samples < 2 * field_type.max_order + 1:
            raise ValueError("regular_samples must satisfy Nyquist: >= 2 * max_order + 1")
        theta = torch.arange(self.regular_samples, dtype=torch.float64) * (2.0 * math.pi / self.regular_samples)
        orders = torch.arange(field_type.max_order + 1, dtype=torch.float64).view(-1, 1)
        self.register_buffer("cos_table", torch.cos(orders * theta.view(1, -1)))
        self.register_buffer("sin_table", torch.sin(orders * theta.view(1, -1)))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, N, C]")
        if int(x.shape[-1]) != self.field_type.total_dim:
            raise ValueError(f"expected {self.field_type.total_dim} channels, got {int(x.shape[-1])}")
        multiplicity = self.field_type.multiplicity(0)
        cos_table = self.cos_table.to(device=x.device, dtype=x.dtype)
        sin_table = self.sin_table.to(device=x.device, dtype=x.dtype)
        samples = record_region_call(
            "op/activation.reconstruct", self._reconstruct, x, cos_table, sin_table
        )
        activated = record_region_call("op/activation.relu", torch.relu, samples)
        return record_region_call(
            "op/activation.project",
            self._project,
            x,
            activated,
            cos_table,
            sin_table,
            multiplicity,
        )

    def _reconstruct(self, x: Tensor, cos_table: Tensor, sin_table: Tensor) -> Tensor:
        samples = _irrep_block(x, self.field_type, 0).squeeze(-1).unsqueeze(-1)
        for order in range(1, self.field_type.max_order + 1):
            block = _irrep_block(x, self.field_type, order)
            samples = samples + block[..., 0].unsqueeze(-1) * cos_table[order].view(1, 1, 1, -1)
            samples = samples + block[..., 1].unsqueeze(-1) * sin_table[order].view(1, 1, 1, -1)
        return samples

    def _project(
        self,
        x: Tensor,
        activated: Tensor,
        cos_table: Tensor,
        sin_table: Tensor,
        multiplicity: int,
    ) -> Tensor:
        outputs = [activated.mean(dim=-1)]
        factor = 2.0 / float(self.regular_samples)
        for order in range(1, self.field_type.max_order + 1):
            coeff_cos = factor * torch.sum(
                activated * cos_table[order].view(1, 1, 1, -1), dim=-1
            )
            coeff_sin = factor * torch.sum(
                activated * sin_table[order].view(1, 1, 1, -1), dim=-1
            )
            outputs.append(
                torch.stack((coeff_cos, coeff_sin), dim=-1).reshape(
                    *x.shape[:-1], multiplicity * 2
                )
            )
        return torch.cat(outputs, dim=-1)
