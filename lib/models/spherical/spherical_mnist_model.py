"""Point-cloud classifier for SphericalMNIST sphere signals."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ...data.spherical.sphere_graph import SphereGraphGeometry
from ...data.spherical.sphere_hierarchy import SphereGraphHierarchy, SpherePooler, SphereWeightedPooler
from .steerable_layers import (
    IrrepBatchNorm,
    IrrepSphereConv,
    RegularNonlinearity,
    ScalarToVectorSphereConv,
    SO2IrrepFieldType,
    VectorNormReLU,
    VectorToScalarSphereConv,
    VectorToVectorSphereConv,
    _flatten_irrep_block,
    _graph_tensors,
    _irrep_block,
    _rotate_irrep_block,
)

_DEFAULT_VECTOR_CHANNELS = (8, 16, 16, 24, 24, 32, 64)
_DEFAULT_VECTOR_STRIDES = (1, 2, 1, 2, 1, 2, 1)
_DEFAULT_FC_CHANNELS = (64, 32)


def _as_int_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    if any(value < 1 for value in result):
        raise ValueError(f"{name} values must be positive")
    return result


def _make_scalar_self_mixing(
    in_channels: int,
    out_channels: int,
    *,
    identity_init: bool,
) -> nn.Linear:
    layer = nn.Linear(in_channels, out_channels, bias=False)
    if identity_init:
        nn.init.zeros_(layer.weight)
        diag = min(int(in_channels), int(out_channels))
        with torch.no_grad():
            layer.weight[:diag, :diag].fill_(1.0)
    return layer


def _scatter_edge_chunks(
    reference: Tensor,
    center_idx: Tensor,
    n_points: int,
    neighbor_count: Tensor,
    *,
    trailing_shape: tuple[int, ...],
    normalize_by_neighbors: bool,
    edge_weight: Tensor | None,
    edge_chunk_size: int,
    make_contributions,
) -> Tensor:
    batch = reference.shape[0]
    edge_count = int(center_idx.shape[0])
    out = reference.new_zeros((batch, n_points, *trailing_shape))
    chunk_size = max(1, int(edge_chunk_size))
    for start in range(0, edge_count, chunk_size):
        stop = min(start + chunk_size, edge_count)
        edge_slice = slice(start, stop)
        contributions = make_contributions(edge_slice)
        index_shape = (1, stop - start, *([1] * len(trailing_shape)))
        index = center_idx[edge_slice].view(index_shape).expand(batch, stop - start, *trailing_shape)
        out.scatter_add_(1, index, contributions)
    if normalize_by_neighbors:
        if edge_weight is not None:
            denom = reference.new_zeros(n_points)
            denom.scatter_add_(0, center_idx, edge_weight.to(device=reference.device, dtype=reference.dtype))
            neighbor_count = denom.clamp_min(1.0)
        norm_shape = (1, n_points, *([1] * len(trailing_shape)))
        out = out / neighbor_count.clamp_min(1.0).view(norm_shape)
    return out


class SteerableSphereScalarBlock(nn.Module):
    """Scalar -> vector -> vector -> scalar block on a sphere graph."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        vector_channels: int,
        radius_km: float,
        num_radial: int,
        *,
        vector_depth: int = 1,
        normalize_by_neighbors: bool = True,
        residual: bool = True,
        include_self: bool = False,
        include_vector_self: bool = False,
        self_identity_init: bool = True,
    ) -> None:
        super().__init__()
        if vector_depth < 0:
            raise ValueError("vector_depth must be non-negative")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.residual = bool(residual and in_channels == out_channels)
        self.scalar_self_mixing = (
            _make_scalar_self_mixing(
                in_channels,
                out_channels,
                identity_init=self_identity_init,
            )
            if include_self
            else None
        )

        self.scalar_to_vector = ScalarToVectorSphereConv(
            in_channels,
            vector_channels,
            radius_km,
            num_radial,
            normalize_by_neighbors=normalize_by_neighbors,
        )
        self.vector_layers = nn.ModuleList(
            [
                VectorToVectorSphereConv(
                    vector_channels,
                    vector_channels,
                    radius_km,
                    num_radial,
                    normalize_by_neighbors=normalize_by_neighbors,
                    include_self=include_vector_self,
                    self_identity_init=self_identity_init,
                )
                for _ in range(vector_depth)
            ]
        )
        self.vector_activations = nn.ModuleList(
            [VectorNormReLU(vector_channels) for _ in range(vector_depth + 1)]
        )
        self.vector_to_scalar = VectorToScalarSphereConv(
            vector_channels,
            out_channels,
            radius_km,
            num_radial,
            normalize_by_neighbors=normalize_by_neighbors,
            bias=True,
        )

    def forward(self, x: Tensor, graph: SphereGraphGeometry) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, N, C]")
        y = self.scalar_to_vector(x, graph)
        y = self.vector_activations[0](y)
        for layer_i, layer in enumerate(self.vector_layers):
            y = self.vector_activations[layer_i + 1](layer(y, graph))
        y_scalar = self.vector_to_scalar(y, graph)
        if self.scalar_self_mixing is not None:
            y_scalar = y_scalar + self.scalar_self_mixing(x)
        elif self.residual:
            y_scalar = y_scalar + x
        return y_scalar


class VectorSphereBatchNorm(nn.Module):
    """Gauge-equivariant variance normalization for tangent-vector channels."""

    def __init__(
        self,
        channels: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.register_buffer("running_var", torch.ones(self.channels))
        if affine:
            self.weight = nn.Parameter(torch.ones(self.channels))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[-1] != 2:
            raise ValueError("x must have shape [B, N, C, 2]")
        channels = int(x.shape[2])
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {channels}")
        if self.training:
            var = x.square().mean(dim=(0, 1, 3))
            self.running_var.mul_(1.0 - self.momentum).add_(self.momentum * var.detach())
        else:
            var = self.running_var.to(device=x.device, dtype=x.dtype)
        scale = torch.rsqrt(var.to(device=x.device, dtype=x.dtype).clamp_min(self.eps))
        if self.weight is not None:
            scale = scale * self.weight.to(device=x.device, dtype=x.dtype)
        return x * scale.view(1, 1, self.channels, 1)


class VectorSphereGraphPool(nn.Module):
    """Equivariant graph mean pooling used at stride-2 positions."""

    def __init__(self, edge_chunk_size: int = 4096) -> None:
        super().__init__()
        self.edge_chunk_size = int(edge_chunk_size)

    def forward(self, x: Tensor, graph: SphereGraphGeometry) -> Tensor:
        if x.ndim != 4 or x.shape[-1] != 2:
            raise ValueError("x must have shape [B, N, C, 2]")
        center_idx, neighbor_idx, _r, _q_matrix, transport, neighbor_count = _graph_tensors(graph, x)

        def make_contributions(edge_slice: slice) -> Tensor:
            x_q = x[:, neighbor_idx[edge_slice], :, :]
            return torch.einsum("edc,beic->beid", transport[edge_slice], x_q)

        return _scatter_edge_chunks(
            x,
            center_idx,
            graph.n_points,
            neighbor_count,
            trailing_shape=(int(x.shape[2]), 2),
            normalize_by_neighbors=True,
            edge_weight=None,
            edge_chunk_size=self.edge_chunk_size,
            make_contributions=make_contributions,
        )


class IrrepSphereGraphPool(nn.Module):
    """Equivariant graph/pooler mean pooling for flat SO(2) irrep fields."""

    def __init__(self, field_type: SO2IrrepFieldType, edge_chunk_size: int = 4096) -> None:
        super().__init__()
        self.field_type = field_type
        self.edge_chunk_size = int(edge_chunk_size)

    def forward(self, x: Tensor, graph: SphereGraphGeometry) -> Tensor:
        if x.ndim != 3 or int(x.shape[-1]) != self.field_type.total_dim:
            raise ValueError(f"x must have shape [B, N, {self.field_type.total_dim}]")
        center_idx = (
            graph.center_idx
            if graph.center_idx.device == x.device
            else graph.center_idx.to(device=x.device)
        )
        neighbor_idx = (
            graph.neighbor_idx
            if graph.neighbor_idx.device == x.device
            else graph.neighbor_idx.to(device=x.device)
        )
        neighbor_count = (
            graph.neighbor_count
            if graph.neighbor_count.device == x.device and graph.neighbor_count.dtype == x.dtype
            else graph.neighbor_count.to(device=x.device, dtype=x.dtype)
        )
        assert graph.transport_angle is not None
        transport_angle = graph.transport_angle.to(device=x.device, dtype=x.dtype)
        outputs = []
        for order in self.field_type.orders:
            block = _irrep_block(x, self.field_type, order)

            def make_contributions(edge_slice: slice, *, block=block, order=order) -> Tensor:
                x_q = block[:, neighbor_idx[edge_slice], :, :]
                return _rotate_irrep_block(x_q, transport_angle[edge_slice], order)

            pooled = _scatter_edge_chunks(
                x,
                center_idx,
                graph.n_points,
                neighbor_count,
                trailing_shape=self.field_type.block_shape(order),
                normalize_by_neighbors=True,
                edge_weight=None,
                edge_chunk_size=self.edge_chunk_size,
                make_contributions=make_contributions,
            )
            outputs.append(_flatten_irrep_block(pooled))
        return torch.cat(outputs, dim=-1)

    def pool_with(
        self,
        x: Tensor,
        pooler: SpherePooler | SphereWeightedPooler,
    ) -> Tensor:
        if x.ndim != 3 or int(x.shape[-1]) != self.field_type.total_dim:
            raise ValueError(f"x must have shape [B, N_fine, {self.field_type.total_dim}]")
        if isinstance(pooler, SphereWeightedPooler):
            return self._pool_weighted(x, pooler)
        return self._pool_assignment(x, pooler)

    def _pool_assignment(self, x: Tensor, pooler: SpherePooler) -> Tensor:
        if int(x.shape[1]) != pooler.fine_graph.n_points:
            raise ValueError(f"x has {x.shape[1]} points, expected {pooler.fine_graph.n_points}")
        assignment = pooler.assignment.to(device=x.device)
        count = pooler.count.to(device=x.device, dtype=x.dtype)
        assert pooler.transport_angle is not None
        transport_angle = pooler.transport_angle.to(device=x.device, dtype=x.dtype)
        outputs = []
        for order in self.field_type.orders:
            block = _irrep_block(x, self.field_type, order)
            transported = _rotate_irrep_block(block, transport_angle, order)
            out = x.new_zeros((int(x.shape[0]), pooler.coarse_graph.n_points, *self.field_type.block_shape(order)))
            index = assignment.view(1, -1, 1, 1).expand_as(transported)
            out.scatter_add_(1, index, transported)
            out = out / count.view(1, -1, 1, 1).clamp_min(1.0)
            outputs.append(_flatten_irrep_block(out))
        return torch.cat(outputs, dim=-1)

    def _pool_weighted(self, x: Tensor, pooler: SphereWeightedPooler) -> Tensor:
        if int(x.shape[1]) != pooler.fine_graph.n_points:
            raise ValueError(f"x has {x.shape[1]} points, expected {pooler.fine_graph.n_points}")
        fine_idx = pooler.fine_idx.to(device=x.device)
        coarse_idx = pooler.coarse_idx.to(device=x.device)
        weight = pooler.weight.to(device=x.device, dtype=x.dtype)
        count = pooler.count.to(device=x.device, dtype=x.dtype)
        assert pooler.transport_angle is not None
        transport_angle = pooler.transport_angle.to(device=x.device, dtype=x.dtype)
        outputs = []
        for order in self.field_type.orders:
            block = _irrep_block(x, self.field_type, order)
            transported = _rotate_irrep_block(block[:, fine_idx, :, :], transport_angle, order)
            values = transported * weight.view(1, -1, 1, 1)
            out = x.new_zeros((int(x.shape[0]), pooler.coarse_graph.n_points, *self.field_type.block_shape(order)))
            index = coarse_idx.view(1, -1, 1, 1).expand_as(values)
            out.scatter_add_(1, index, values)
            out = out / count.view(1, -1, 1, 1).clamp_min(1.0)
            outputs.append(_flatten_irrep_block(out))
        return torch.cat(outputs, dim=-1)


class IrrepSphereConvBlock(nn.Module):
    """Irrep convolution followed by irrep BN, regular nonlinearity, and optional pooling."""

    def __init__(
        self,
        in_type: SO2IrrepFieldType,
        out_type: SO2IrrepFieldType,
        radius_km: float,
        num_radial: int,
        *,
        stride: int,
        regular_samples: int | None = None,
        normalize_by_neighbors: bool = True,
        edge_chunk_size: int = 2048,
        include_self: bool = False,
        self_identity_init: bool = True,
        quadrature: bool = False,
        quadrature_radial: int | None = None,
        quadrature_angular: int = 16,
        quadrature_sigma_km: float | None = None,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        self.in_type = in_type
        self.out_type = out_type
        self.stride = int(stride)
        self.conv = IrrepSphereConv(
            in_type,
            out_type,
            radius_km,
            num_radial,
            normalize_by_neighbors=normalize_by_neighbors,
            include_self=include_self,
            self_identity_init=self_identity_init,
            quadrature=quadrature,
            quadrature_radial=quadrature_radial,
            quadrature_angular=quadrature_angular,
            quadrature_sigma_km=quadrature_sigma_km,
        )
        self.norm = IrrepBatchNorm(out_type)
        self.activation = RegularNonlinearity(out_type, regular_samples=regular_samples)
        self.fallback_pool = IrrepSphereGraphPool(out_type, edge_chunk_size=edge_chunk_size) if stride == 2 else None

    def prepare_graph(self, graph: SphereGraphGeometry) -> None:
        self.conv.prepare_graph(graph, device=graph.device, dtype=graph.dtype)

    def forward(
        self,
        x: Tensor,
        graph: SphereGraphGeometry,
        pooler: SpherePooler | SphereWeightedPooler | None = None,
    ) -> Tensor:
        y = self.activation(self.norm(self.conv(x, graph)))
        if self.stride == 2 and self.fallback_pool is not None:
            if pooler is not None:
                y = self.fallback_pool.pool_with(y, pooler)
            else:
                y = self.fallback_pool(y, graph)
        return y


class VectorSphereConvBlock(nn.Module):
    """S2V/V2V convolution followed by vector BN, norm-ReLU, and optional pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        radius_km: float,
        num_radial: int,
        *,
        scalar_input: bool,
        stride: int,
        normalize_by_neighbors: bool = True,
        edge_chunk_size: int = 2048,
        include_self: bool = False,
        self_identity_init: bool = True,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        self.stride = int(stride)
        if scalar_input:
            self.conv = ScalarToVectorSphereConv(
                in_channels,
                out_channels,
                radius_km,
                num_radial,
                normalize_by_neighbors=normalize_by_neighbors,
            )
        else:
            self.conv = VectorToVectorSphereConv(
                in_channels,
                out_channels,
                radius_km,
                num_radial,
                normalize_by_neighbors=normalize_by_neighbors,
                include_self=include_self,
                self_identity_init=self_identity_init,
            )
        self.norm = VectorSphereBatchNorm(out_channels)
        self.activation = VectorNormReLU(out_channels)
        self.fallback_pool = VectorSphereGraphPool(edge_chunk_size=edge_chunk_size) if stride == 2 else None

    def forward(
        self,
        x: Tensor,
        graph: SphereGraphGeometry,
        pooler: SpherePooler | None = None,
    ) -> Tensor:
        y = self.conv(x, graph)
        y = self.norm(y)
        y = self.activation(y)
        if self.stride == 2:
            if pooler is not None:
                y = pooler.pool_vector(y)
            elif self.fallback_pool is not None:
                y = self.fallback_pool(y, graph)
        return y


class SteerableSphereMNISTClassifier(nn.Module):
    """S2V + 6 V2V continuous-gauge classifier for SphericalMNIST."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_classes: int = 10,
        channels: Sequence[int] | None = None,
        strides: Sequence[int] | None = None,
        fc_channels: Sequence[int] | None = None,
        radius_km: float = 1300.0,
        num_radial: int = 4,
        dropout: float = 0.0,
        normalize_by_neighbors: bool = True,
        edge_chunk_size: int = 2048,
        scalar_channels: int | None = None,
        vector_channels: int | None = None,
        num_blocks: int | None = None,
        vector_depth: int | None = None,
        mlp_hidden: int | None = None,
        include_self: bool = False,
        self_identity_init: bool = True,
    ) -> None:
        super().__init__()
        if vector_depth is not None:
            # Accepted for compatibility with the earlier scalar/vector block API.
            pass
        if channels is None:
            if scalar_channels is not None or vector_channels is not None or num_blocks is not None:
                block_count = int(num_blocks) if num_blocks is not None else len(_DEFAULT_VECTOR_CHANNELS)
                if block_count < 1:
                    raise ValueError("num_blocks must be positive")
                width = (
                    int(vector_channels)
                    if vector_channels is not None
                    else int(scalar_channels)
                    if scalar_channels is not None
                    else _DEFAULT_VECTOR_CHANNELS[0]
                )
                channels = tuple(width for _ in range(block_count))
            else:
                channels = _DEFAULT_VECTOR_CHANNELS
        channels = _as_int_tuple("channels", channels)
        if strides is None:
            if channels == _DEFAULT_VECTOR_CHANNELS:
                strides = _DEFAULT_VECTOR_STRIDES
            else:
                strides = tuple(1 for _ in channels)
        strides = _as_int_tuple("strides", strides)
        if len(channels) != len(strides):
            raise ValueError("channels and strides must have the same length")
        if any(stride not in (1, 2) for stride in strides):
            raise ValueError("strides values must be 1 or 2")
        if fc_channels is None:
            if mlp_hidden is not None:
                fc_channels = (int(mlp_hidden), int(mlp_hidden))
            else:
                fc_channels = _DEFAULT_FC_CHANNELS
        fc_channels = _as_int_tuple("fc_channels", fc_channels)

        blocks = []
        current_channels = int(in_channels)
        for layer_i, (out_channels, stride) in enumerate(zip(channels, strides)):
            block = VectorSphereConvBlock(
                current_channels,
                out_channels,
                radius_km,
                num_radial,
                scalar_input=layer_i == 0,
                stride=stride,
                normalize_by_neighbors=normalize_by_neighbors,
                edge_chunk_size=edge_chunk_size,
                include_self=include_self,
                self_identity_init=self_identity_init,
            )
            blocks.append(block)
            current_channels = int(out_channels)
        self.channels = channels
        self.strides = strides
        self.fc_channels = fc_channels
        self.blocks = nn.ModuleList(blocks)
        head_layers: list[nn.Module] = []
        hidden = (current_channels, *fc_channels, int(num_classes))
        for layer_i, (input_dim, output_dim) in enumerate(zip(hidden, hidden[1:])):
            head_layers.append(nn.Linear(input_dim, output_dim))
            if layer_i < len(hidden) - 2:
                head_layers.append(nn.ReLU())
                if dropout > 0.0:
                    head_layers.append(nn.Dropout(dropout))
        self.head = nn.Sequential(*head_layers)

    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        graph: SphereGraphGeometry | SphereGraphHierarchy,
    ) -> Tensor:
        if isinstance(x, dict):
            x = x["features"]
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.ndim != 3:
            raise ValueError("x must have shape [B, N, C] or [B, N]")

        if isinstance(graph, SphereGraphHierarchy):
            hierarchy = graph.to(device=x.device, dtype=x.dtype)
            required_levels = 1 + sum(1 for block in self.blocks if block.stride == 2)
            if hierarchy.n_levels < required_levels:
                raise ValueError(
                    f"hierarchy has {hierarchy.n_levels} levels, but model strides require {required_levels}"
                )
            graphs = hierarchy.graphs
            poolers = hierarchy.poolers
        else:
            hierarchy = None
            graphs = (graph.to(device=x.device, dtype=x.dtype),)
            poolers = ()

        y = x
        level_i = 0
        for block in self.blocks:
            current_graph = graphs[level_i]
            pooler = poolers[level_i] if hierarchy is not None and block.stride == 2 else None
            y = block(y, current_graph, pooler)
            if hierarchy is not None and block.stride == 2:
                level_i += 1
        pooled = torch.linalg.vector_norm(y, dim=-1).mean(dim=1)
        return self.head(pooled)

    @staticmethod
    def loss(logits: Tensor, labels: Tensor) -> Tensor:
        return torch.nn.functional.cross_entropy(logits, labels)

    @staticmethod
    def accuracy(logits: Tensor, labels: Tensor) -> float:
        predicted = torch.argmax(logits.detach(), dim=1)
        return float((predicted == labels).to(dtype=torch.float32).mean().item())


class IrrepSphereMNISTClassifier(nn.Module):
    """Paper-style rho_0..rho_M gauge-irrep classifier for SphericalMNIST."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        num_classes: int = 10,
        channels: Sequence[int] | None = None,
        strides: Sequence[int] | None = None,
        fc_channels: Sequence[int] | None = None,
        radius_km: float = 1300.0,
        num_radial: int = 4,
        dropout: float = 0.0,
        normalize_by_neighbors: bool = True,
        edge_chunk_size: int = 2048,
        max_order: int = 2,
        regular_samples: int | None = None,
        num_blocks: int | None = None,
        mlp_hidden: int | None = None,
        include_self: bool = False,
        self_identity_init: bool = True,
        quadrature: bool = False,
        quadrature_radial: int | None = None,
        quadrature_angular: int = 16,
        quadrature_sigma_km: float | None = None,
    ) -> None:
        super().__init__()
        if channels is None:
            if num_blocks is not None:
                block_count = int(num_blocks)
                if block_count < 1:
                    raise ValueError("num_blocks must be positive")
                channels = tuple(_DEFAULT_VECTOR_CHANNELS[0] for _ in range(block_count))
            else:
                channels = _DEFAULT_VECTOR_CHANNELS
        channels = _as_int_tuple("channels", channels)
        if strides is None:
            strides = _DEFAULT_VECTOR_STRIDES if channels == _DEFAULT_VECTOR_CHANNELS else tuple(1 for _ in channels)
        strides = _as_int_tuple("strides", strides)
        if len(channels) != len(strides):
            raise ValueError("channels and strides must have the same length")
        if any(stride not in (1, 2) for stride in strides):
            raise ValueError("strides values must be 1 or 2")
        if fc_channels is None:
            fc_channels = (int(mlp_hidden), int(mlp_hidden)) if mlp_hidden is not None else _DEFAULT_FC_CHANNELS
        fc_channels = _as_int_tuple("fc_channels", fc_channels)

        blocks = []
        current_type = SO2IrrepFieldType.scalar(in_channels)
        for out_multiplicity, stride in zip(channels, strides):
            out_type = SO2IrrepFieldType.balanced(max_order, out_multiplicity)
            blocks.append(
                IrrepSphereConvBlock(
                    current_type,
                    out_type,
                    radius_km,
                    num_radial,
                    stride=stride,
                    regular_samples=regular_samples,
                    normalize_by_neighbors=normalize_by_neighbors,
                    edge_chunk_size=edge_chunk_size,
                    include_self=include_self,
                    self_identity_init=self_identity_init,
                    quadrature=quadrature,
                    quadrature_radial=quadrature_radial,
                    quadrature_angular=quadrature_angular,
                    quadrature_sigma_km=quadrature_sigma_km,
                )
            )
            current_type = out_type
        self.channels = channels
        self.strides = strides
        self.fc_channels = fc_channels
        self.max_order = int(max_order)
        self.regular_samples = regular_samples
        self.quadrature = bool(quadrature)
        self.quadrature_radial = quadrature_radial
        self.quadrature_angular = int(quadrature_angular)
        self.quadrature_sigma_km = quadrature_sigma_km
        self.blocks = nn.ModuleList(blocks)
        self.output_type = current_type

        head_layers: list[nn.Module] = []
        hidden = (current_type.invariant_dim, *fc_channels, int(num_classes))
        for layer_i, (input_dim, output_dim) in enumerate(zip(hidden, hidden[1:])):
            head_layers.append(nn.Linear(input_dim, output_dim))
            if layer_i < len(hidden) - 2:
                head_layers.append(nn.ReLU())
                if dropout > 0.0:
                    head_layers.append(nn.Dropout(dropout))
        self.head = nn.Sequential(*head_layers)

    def _invariant_readout(self, x: Tensor) -> Tensor:
        pooled = []
        for order in self.output_type.orders:
            block = _irrep_block(x, self.output_type, order)
            if order == 0:
                pooled.append(block.squeeze(-1).mean(dim=1))
            else:
                pooled.append(torch.linalg.vector_norm(block, dim=-1).mean(dim=1))
        return torch.cat(pooled, dim=-1)

    def prepare_graph(
        self,
        graph: SphereGraphGeometry | SphereGraphHierarchy,
    ) -> None:
        """Warm all layer-specific geometry caches before processing batches."""

        if isinstance(graph, SphereGraphHierarchy):
            graphs = graph.graphs
        else:
            graphs = (graph,)
        level_i = 0
        for block in self.blocks:
            block.prepare_graph(graphs[level_i])
            if isinstance(graph, SphereGraphHierarchy) and block.stride == 2:
                level_i += 1

    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        graph: SphereGraphGeometry | SphereGraphHierarchy,
    ) -> Tensor:
        if isinstance(x, dict):
            x = x["features"]
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.ndim != 3:
            raise ValueError("x must have shape [B, N, C] or [B, N]")

        if isinstance(graph, SphereGraphHierarchy):
            if all(level.device == x.device and level.dtype == x.dtype for level in graph.graphs):
                hierarchy = graph
            else:
                hierarchy = graph.to(device=x.device, dtype=x.dtype)
            required_levels = 1 + sum(1 for block in self.blocks if block.stride == 2)
            if hierarchy.n_levels < required_levels:
                raise ValueError(
                    f"hierarchy has {hierarchy.n_levels} levels, but model strides require {required_levels}"
                )
            graphs = hierarchy.graphs
            poolers = hierarchy.poolers
        else:
            hierarchy = None
            current_graph = (
                graph
                if graph.device == x.device and graph.dtype == x.dtype
                else graph.to(device=x.device, dtype=x.dtype)
            )
            graphs = (current_graph,)
            poolers = ()

        y = x
        level_i = 0
        for block in self.blocks:
            current_graph = graphs[level_i]
            pooler = poolers[level_i] if hierarchy is not None and block.stride == 2 else None
            y = block(y, current_graph, pooler)
            if hierarchy is not None and block.stride == 2:
                level_i += 1
        return self.head(self._invariant_readout(y))

    @staticmethod
    def loss(logits: Tensor, labels: Tensor) -> Tensor:
        return torch.nn.functional.cross_entropy(logits, labels)

    @staticmethod
    def accuracy(logits: Tensor, labels: Tensor) -> float:
        predicted = torch.argmax(logits.detach(), dim=1)
        return float((predicted == labels).to(dtype=torch.float32).mean().item())
