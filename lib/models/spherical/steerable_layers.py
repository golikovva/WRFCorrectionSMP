"""SO(2)-steerable intrinsic KPConv-like layers on the sphere."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .radial_basis import default_radial_sigma, radial_centers, triangular_radial_basis
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


class _RadialSteerableBase(nn.Module):
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


class VectorPointwiseSO2(nn.Module):
    """SO(2)-equivariant pointwise mixing for tangent-vector channels.

    For every input/output channel pair the only linear maps that commute with
    all gauge rotations are ``a I + b J`` where ``J`` is a 90-degree rotation.
    """

    def __init__(
        self,
        in_vector_channels: int,
        out_vector_channels: int,
        *,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.in_vector_channels = int(in_vector_channels)
        self.out_vector_channels = int(out_vector_channels)
        self.identity_init = bool(identity_init)
        self.weight_i = nn.Parameter(torch.empty(out_vector_channels, in_vector_channels))
        self.weight_j = nn.Parameter(torch.empty(out_vector_channels, in_vector_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.identity_init:
            nn.init.zeros_(self.weight_i)
            nn.init.zeros_(self.weight_j)
            diag = min(self.in_vector_channels, self.out_vector_channels)
            with torch.no_grad():
                self.weight_i[:diag, :diag].fill_(1.0)
            return
        bound = math.sqrt(6.0 / max(1, self.in_vector_channels))
        nn.init.uniform_(self.weight_i, -bound, bound)
        nn.init.uniform_(self.weight_j, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[-1] != 2:
            raise ValueError("x must have shape [B, N, C_in, 2]")
        if int(x.shape[2]) != self.in_vector_channels:
            raise ValueError(
                f"expected {self.in_vector_channels} channels, got {int(x.shape[2])}"
            )
        weight_i = self.weight_i.to(device=x.device, dtype=x.dtype)
        weight_j = self.weight_j.to(device=x.device, dtype=x.dtype)
        jx = torch.stack((-x[..., 1], x[..., 0]), dim=-1)
        mixed_i = torch.einsum("oi,bnid->bnod", weight_i, x)
        mixed_j = torch.einsum("oi,bnid->bnod", weight_j, jx)
        return mixed_i + mixed_j


class ScalarToVectorSphereConv(_RadialSteerableBase):
    """Steerable scalar-to-tangent-vector convolution.

    Kernel form:

        K_s2v(r, phi) = Q(phi) a(r)

    where ``a(r)`` is learned in radial/tangential coordinates.
    """

    def __init__(
        self,
        in_channels: int,
        out_vector_channels: int,
        radius_km: float,
        num_radial: int,
        *,
        radial_sigma_km: float | None = None,
        normalize_radial_basis: bool = True,
        normalize_by_neighbors: bool = True,
    ) -> None:
        super().__init__(
            radius_km,
            num_radial,
            radial_sigma_km=radial_sigma_km,
            normalize_radial_basis=normalize_radial_basis,
            normalize_by_neighbors=normalize_by_neighbors,
        )
        self.in_channels = int(in_channels)
        self.out_vector_channels = int(out_vector_channels)
        self.weight = nn.Parameter(torch.empty(num_radial, out_vector_channels, in_channels, 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_weight(self.weight)

    def forward(self, x: Tensor, graph: SphereGraphGeometry) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, N, C_in]")
        center_idx, neighbor_idx, r, q_matrix, _transport, neighbor_count = _graph_tensors(graph, x)
        h = self.radial_basis(r, radius_km=graph.radius_km)
        active = self.active_edge_weight(r)
        h = h * active.unsqueeze(-1)
        kernel_rt = torch.einsum("ej,joid->eoid", h, self.weight.to(dtype=x.dtype))
        kernel_xy = torch.einsum("edk,eoik->eoid", q_matrix, kernel_rt)
        x_q = x[:, neighbor_idx, :]
        contrib = torch.einsum("bei,eoid->beod", x_q, kernel_xy)
        return _scatter_to_centers(
            contrib,
            center_idx,
            graph.n_points,
            neighbor_count,
            normalize_by_neighbors=self.normalize_by_neighbors,
            edge_weight=active,
        )


class VectorToScalarSphereConv(_RadialSteerableBase):
    """Steerable tangent-vector-to-scalar convolution.

    Neighbor vector features are first parallel transported into the center
    frame.  The kernel then computes a learned dot product in the
    radial/tangential frame:

        K_v2s(r, phi) = b(r)^T Q(phi)^T.
    """

    def __init__(
        self,
        in_vector_channels: int,
        out_channels: int,
        radius_km: float,
        num_radial: int,
        *,
        radial_sigma_km: float | None = None,
        normalize_radial_basis: bool = True,
        normalize_by_neighbors: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__(
            radius_km,
            num_radial,
            radial_sigma_km=radial_sigma_km,
            normalize_radial_basis=normalize_radial_basis,
            normalize_by_neighbors=normalize_by_neighbors,
        )
        self.in_vector_channels = int(in_vector_channels)
        self.out_channels = int(out_channels)
        self.weight = nn.Parameter(torch.empty(num_radial, out_channels, in_vector_channels, 2))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_weight(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, graph: SphereGraphGeometry) -> Tensor:
        if x.ndim != 4 or x.shape[-1] != 2:
            raise ValueError("x must have shape [B, N, C_in, 2]")
        center_idx, neighbor_idx, r, q_matrix, transport, neighbor_count = _graph_tensors(graph, x)
        h = self.radial_basis(r, radius_km=graph.radius_km)
        active = self.active_edge_weight(r)
        h = h * active.unsqueeze(-1)
        kernel = torch.einsum("ej,joik->eoik", h, self.weight.to(dtype=x.dtype))
        x_q = x[:, neighbor_idx, :, :]
        transported = torch.einsum("edc,beic->beid", transport, x_q)
        radial_components = torch.einsum("edk,beid->beik", q_matrix, transported)
        contrib = torch.einsum("beik,eoik->beo", radial_components, kernel)
        out = _scatter_to_centers(
            contrib,
            center_idx,
            graph.n_points,
            neighbor_count,
            normalize_by_neighbors=self.normalize_by_neighbors,
            edge_weight=active,
        )
        if self.bias is not None:
            out = out + self.bias.to(device=out.device, dtype=out.dtype)
        return out


class VectorToVectorSphereConv(_RadialSteerableBase):
    """Steerable tangent-vector-to-tangent-vector convolution.

    Kernel form:

        K_v2v(r, phi) = Q(phi) A(r) Q(phi)^T.

    The input vector from each neighbor is parallel transported before the
    kernel is applied.
    """

    def __init__(
        self,
        in_vector_channels: int,
        out_vector_channels: int,
        radius_km: float,
        num_radial: int,
        *,
        radial_sigma_km: float | None = None,
        normalize_radial_basis: bool = True,
        normalize_by_neighbors: bool = True,
        include_self: bool = False,
        self_identity_init: bool = True,
    ) -> None:
        super().__init__(
            radius_km,
            num_radial,
            radial_sigma_km=radial_sigma_km,
            normalize_radial_basis=normalize_radial_basis,
            normalize_by_neighbors=normalize_by_neighbors,
        )
        self.in_vector_channels = int(in_vector_channels)
        self.out_vector_channels = int(out_vector_channels)
        self.weight = nn.Parameter(
            torch.empty(num_radial, out_vector_channels, in_vector_channels, 2, 2)
        )
        self.self_mixing = (
            VectorPointwiseSO2(
                in_vector_channels,
                out_vector_channels,
                identity_init=self_identity_init,
            )
            if include_self
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _reset_weight(self.weight)

    def forward(self, x: Tensor, graph: SphereGraphGeometry) -> Tensor:
        if x.ndim != 4 or x.shape[-1] != 2:
            raise ValueError("x must have shape [B, N, C_in, 2]")
        center_idx, neighbor_idx, r, q_matrix, transport, neighbor_count = _graph_tensors(graph, x)
        h = self.radial_basis(r, radius_km=graph.radius_km)
        active = self.active_edge_weight(r)
        h = h * active.unsqueeze(-1)
        kernel = torch.einsum("ej,joiab->eoiab", h, self.weight.to(dtype=x.dtype))
        x_q = x[:, neighbor_idx, :, :]
        transported = torch.einsum("edc,beic->beid", transport, x_q)
        radial_in = torch.einsum("edk,beid->beik", q_matrix, transported)
        radial_out = torch.einsum("eoiac,beic->beoa", kernel, radial_in)
        contrib = torch.einsum("eda,beoa->beod", q_matrix, radial_out)
        out = _scatter_to_centers(
            contrib,
            center_idx,
            graph.n_points,
            neighbor_count,
            normalize_by_neighbors=self.normalize_by_neighbors,
            edge_weight=active,
        )
        if self.self_mixing is not None:
            out = out + self.self_mixing(x)
        return out


class VectorNormReLU(nn.Module):
    """Equivariant norm nonlinearity for tangent-vector channels."""

    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[-1] != 2:
            raise ValueError("x must have shape [B, N, C, 2]")
        norm = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + self.eps)
        bias = self.bias.to(device=x.device, dtype=x.dtype).view(1, 1, self.channels, 1)
        scale = torch.relu(norm + bias) / norm.clamp_min(self.eps)
        return scale * x


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
        identity_init: bool = True,
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
            nn.init.zeros_(parameter)
            if self.identity_init:
                diag = min(parameter.shape)
                with torch.no_grad():
                    parameter[:diag, :diag].fill_(1.0)
            else:
                nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
        for key, weight_i in self.vector_i_weights.items():
            weight_j = self.vector_j_weights[key]
            nn.init.zeros_(weight_i)
            nn.init.zeros_(weight_j)
            if self.identity_init:
                diag = min(weight_i.shape)
                with torch.no_grad():
                    weight_i[:diag, :diag].fill_(1.0)
            else:
                bound = math.sqrt(6.0 / max(1, weight_i.shape[1]))
                nn.init.uniform_(weight_i, -bound, bound)
                nn.init.uniform_(weight_j, -bound, bound)

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
    neighbor_count: Tensor
    transport_angle: Tensor
    radial_basis: Tensor | None
    input_rotations: dict[int, tuple[Tensor, Tensor]]
    output_rotations: dict[int, tuple[Tensor, Tensor]]
    quadrature_bases: dict[tuple[int, int], Tensor]


class IrrepSphereConv(_RadialSteerableBase):
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
        include_self: bool = False,
        self_identity_init: bool = True,
        quadrature: bool = False,
        quadrature_radial: int | None = None,
        quadrature_angular: int = 16,
        quadrature_sigma_km: float | None = None,
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
        self._weight_keys: dict[tuple[int, int], str] = {}
        self.weights = nn.ParameterDict()
        for out_order in self._out_orders:
            out_m, out_dim = out_type.block_shape(out_order)
            for in_order in self._in_orders:
                in_m, in_dim = in_type.block_shape(in_order)
                key = self._weight_key(out_order, in_order)
                self._weight_keys[out_order, in_order] = key
                self.weights[key] = nn.Parameter(torch.empty(num_radial, out_m, in_m, out_dim, in_dim))
        self.self_mixing = (
            _IrrepPointwiseSO2(in_type, out_type, identity_init=self_identity_init)
            if include_self
            else None
        )
        self.reset_parameters()

    @staticmethod
    def _weight_key(out_order: int, in_order: int) -> str:
        return f"o{int(out_order)}_i{int(in_order)}"

    def reset_parameters(self) -> None:
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

        self._prepared_geometry = geometry

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
            input_rotations: dict[int, tuple[Tensor, Tensor]] = {}
            output_rotations: dict[int, tuple[Tensor, Tensor]] = {}
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
                for in_order in in_orders:
                    if in_order > 0:
                        theta = input_angle * float(in_order)
                        input_rotations[in_order] = (torch.cos(theta), torch.sin(theta))
                for out_order in out_orders:
                    if out_order > 0:
                        theta = phi * float(out_order)
                        output_rotations[out_order] = (torch.cos(theta), torch.sin(theta))

        prepared = _IrrepConvGeometry(
            graph=graph,
            n_points=graph.n_points,
            center_idx=center_idx,
            neighbor_idx=neighbor_idx,
            neighbor_count=neighbor_count,
            transport_angle=transport_angle,
            radial_basis=radial_basis,
            input_rotations=input_rotations,
            output_rotations=output_rotations,
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
        center_idx = (
            graph.center_idx
            if graph.center_idx.device == device
            else graph.center_idx.to(device=device)
        )
        dist2 = (u.unsqueeze(1) - q_points.unsqueeze(0)).square().sum(dim=-1)
        sigma_t = torch.as_tensor(sigma, device=device, dtype=dtype).clamp_min(1e-12)
        interp = torch.exp(-dist2 / (sigma_t * sigma_t))
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

        inputs_by_order: dict[int, Tensor] = {}
        for in_order in self._in_orders:
            in_block = _irrep_block(x, self.in_type, in_order)
            x_q = in_block[:, geometry.neighbor_idx, :, :]
            if not self.quadrature and in_order > 0:
                cos_value, sin_value = geometry.input_rotations[in_order]
                x_q = _rotate_irrep_block_from_components(x_q, cos_value, sin_value)
            inputs_by_order[in_order] = x_q

        outputs_by_order: list[Tensor] = []
        for out_order in self._out_orders:
            out_m, out_dim = self.out_type.block_shape(out_order)
            out_block = x.new_zeros((int(x.shape[0]), geometry.n_points, out_m, out_dim))
            for in_order in self._in_orders:
                x_q = inputs_by_order[in_order]
                weight = self.weights[self._weight_keys[out_order, in_order]]
                if weight.device != x.device or weight.dtype != x.dtype:
                    weight = weight.to(device=x.device, dtype=x.dtype)
                if self.quadrature:
                    basis = geometry.quadrature_bases[out_order, in_order]
                    # kernel = torch.einsum("erabcd,roibc->eoiad", basis, weight)
                    # contrib = torch.einsum("eoiad,beid->beoa", kernel, x_q)
                    weighted = torch.einsum("roiad,beid->beroa",weight, x_q,)
                    contrib = torch.einsum("er,beroa->beoa",basis,weighted,)
                    
                    normalize_by_neighbors = False
                else:
                    assert geometry.radial_basis is not None
                    # kernel = torch.einsum("er,roiab->eoiab", geometry.radial_basis, weight)
                    # contrib = torch.einsum("eoiad,beid->beoa", kernel, x_q)

                    weighted = torch.einsum("roiad,beid->beroa",weight, x_q,)
                    contrib = torch.einsum("er,beroa->beoa",geometry.radial_basis,weighted,)


                    if out_order > 0:
                        cos_value, sin_value = geometry.output_rotations[out_order]
                        contrib = _rotate_irrep_block_from_components(
                            contrib,
                            cos_value,
                            sin_value,
                        )
                    normalize_by_neighbors = self.normalize_by_neighbors
                out_block = out_block + _scatter_to_centers(
                    contrib,
                    geometry.center_idx,
                    geometry.n_points,
                    geometry.neighbor_count,
                    normalize_by_neighbors=normalize_by_neighbors,
                    edge_weight=None,
                )
            outputs_by_order.append(_flatten_irrep_block(out_block))

        out = torch.cat(outputs_by_order, dim=-1)
        if self.self_mixing is not None:
            out = out + self.self_mixing(x)
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
        samples = _irrep_block(x, self.field_type, 0).squeeze(-1).unsqueeze(-1)
        cos_table = self.cos_table.to(device=x.device, dtype=x.dtype)
        sin_table = self.sin_table.to(device=x.device, dtype=x.dtype)
        for order in range(1, self.field_type.max_order + 1):
            block = _irrep_block(x, self.field_type, order)
            samples = samples + block[..., 0].unsqueeze(-1) * cos_table[order].view(1, 1, 1, -1)
            samples = samples + block[..., 1].unsqueeze(-1) * sin_table[order].view(1, 1, 1, -1)
        activated = torch.relu(samples)
        outputs = [activated.mean(dim=-1)]
        factor = 2.0 / float(self.regular_samples)
        for order in range(1, self.field_type.max_order + 1):
            coeff_cos = factor * torch.sum(activated * cos_table[order].view(1, 1, 1, -1), dim=-1)
            coeff_sin = factor * torch.sum(activated * sin_table[order].view(1, 1, 1, -1), dim=-1)
            outputs.append(torch.stack((coeff_cos, coeff_sin), dim=-1).reshape(*x.shape[:-1], multiplicity * 2))
        return torch.cat(outputs, dim=-1)
