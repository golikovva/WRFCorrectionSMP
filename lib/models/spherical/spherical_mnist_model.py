"""Point-cloud classifier for SphericalMNIST sphere signals."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from ...data.spherical.sphere_graph import SphereGraphGeometry
from ...data.spherical.sphere_hierarchy import SphereGraphHierarchy, SpherePooler, SphereWeightedPooler
from .irrep_layers import (
    IrrepBatchNorm,
    IrrepSphereConv,
    RegularNonlinearity,
    SO2IrrepFieldType,
    _IrrepConvGeometry,
    _flatten_irrep_block,
    _irrep_block,
    _rotate_irrep_block,
)

_DEFAULT_IRREP_CHANNELS = (8, 16, 16, 24, 24, 32, 64)
_DEFAULT_IRREP_STRIDES = (1, 2, 1, 2, 1, 2, 1)
_DEFAULT_FC_CHANNELS = (64, 32)


def _as_int_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    if any(value < 1 for value in result):
        raise ValueError(f"{name} values must be positive")
    return result


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
        include_self: bool = True,
        self_identity_init: bool = False,
        quadrature: bool = False,
        quadrature_radial: int | None = None,
        quadrature_angular: int = 16,
        quadrature_sigma_km: float | None = None,
        irrep_conv_backend: Literal["auto", "torch", "triton"] = "auto",
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
            backend=irrep_conv_backend,
        )
        self.norm = IrrepBatchNorm(out_type)
        self.activation = RegularNonlinearity(out_type, regular_samples=regular_samples)
        self.fallback_pool = IrrepSphereGraphPool(out_type, edge_chunk_size=edge_chunk_size) if stride == 2 else None

    def prepare_graph(self, graph: SphereGraphGeometry) -> None:
        self.conv.prepare_graph(graph, device=graph.device, dtype=graph.dtype)

    def bind_prepared_geometry(self, geometry: _IrrepConvGeometry) -> None:
        self.conv._bind_prepared_geometry(geometry)

    def clear_prepared_graph(self) -> None:
        self.conv.clear_prepared_graph()

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

    def forward_prepared(
        self,
        x: Tensor,
        *,
        graph: SphereGraphGeometry,
        pooler: SpherePooler | SphereWeightedPooler | None = None,
    ) -> Tensor:
        y = self.activation(self.norm(self.conv.forward_prepared(x)))
        if self.stride == 2 and self.fallback_pool is not None:
            if pooler is not None:
                y = self.fallback_pool.pool_with(y, pooler)
            else:
                y = self.fallback_pool(y, graph)
        return y


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
        include_self: bool = True,
        self_identity_init: bool = False,
        quadrature: bool = False,
        quadrature_radial: int | None = None,
        quadrature_angular: int = 16,
        quadrature_sigma_km: float | None = None,
        irrep_conv_backend: Literal["auto", "torch", "triton"] = "auto",
    ) -> None:
        super().__init__()
        if channels is None:
            channels = _DEFAULT_IRREP_CHANNELS
        channels = _as_int_tuple("channels", channels)
        if strides is None:
            strides = _DEFAULT_IRREP_STRIDES if channels == _DEFAULT_IRREP_CHANNELS else tuple(1 for _ in channels)
        strides = _as_int_tuple("strides", strides)
        if len(channels) != len(strides):
            raise ValueError("channels and strides must have the same length")
        if any(stride not in (1, 2) for stride in strides):
            raise ValueError("strides values must be 1 or 2")
        if fc_channels is None:
            fc_channels = _DEFAULT_FC_CHANNELS
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
                    irrep_conv_backend=irrep_conv_backend,
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
        self.irrep_conv_backend = str(irrep_conv_backend)
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
        self._prepared_graph: SphereGraphGeometry | SphereGraphHierarchy | None = None
        self._prepared_graphs: tuple[SphereGraphGeometry, ...] = ()
        self._prepared_poolers: tuple[SpherePooler | SphereWeightedPooler, ...] = ()
        self._geometry_banks: tuple[_IrrepConvGeometry, ...] = ()

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
        """Replace the classifier's static graph binding and shared geometry banks."""

        self.clear_prepared_graph()
        required_levels = 1 + sum(1 for block in self.blocks if block.stride == 2)
        if isinstance(graph, SphereGraphHierarchy):
            if graph.n_levels < required_levels:
                raise ValueError(
                    f"hierarchy has {graph.n_levels} levels, but model strides require "
                    f"{required_levels}"
                )
            graphs = graph.graphs[:required_levels]
            poolers = graph.poolers[: required_levels - 1]
            blocks_by_level: list[list[IrrepSphereConvBlock]] = [
                [] for _ in range(required_levels)
            ]
            level_i = 0
            for block in self.blocks:
                blocks_by_level[level_i].append(block)
                if block.stride == 2:
                    level_i += 1
        else:
            graphs = (graph,)
            poolers = ()
            blocks_by_level = [list(self.blocks)]

        banks = []
        for level_graph, level_blocks in zip(graphs, blocks_by_level):
            if not level_blocks:
                continue
            convolutions = tuple(block.conv for block in level_blocks)
            representative = convolutions[0]
            signature = representative._geometry_signature()
            if any(conv._geometry_signature() != signature for conv in convolutions[1:]):
                raise RuntimeError("all classifier convolutions on a level must share geometry settings")
            in_orders = tuple(sorted({order for conv in convolutions for order in conv._in_orders}))
            out_orders = tuple(sorted({order for conv in convolutions for order in conv._out_orders}))
            bank = representative._build_geometry(
                level_graph,
                level_graph.device,
                level_graph.dtype,
                in_orders=in_orders,
                out_orders=out_orders,
            )
            for block in level_blocks:
                block.bind_prepared_geometry(bank)
            banks.append(bank)

        self._prepared_graph = graph
        self._prepared_graphs = tuple(graphs)
        self._prepared_poolers = tuple(poolers)
        self._geometry_banks = tuple(banks)

    def prepare_hierarchy(self, hierarchy: SphereGraphHierarchy) -> None:
        """Wrapper-compatible alias for binding a static hierarchy."""

        self.prepare_graph(hierarchy)

    def clear_prepared_graph(self) -> None:
        for block in self.blocks:
            block.clear_prepared_graph()
        self._prepared_graph = None
        self._prepared_graphs = ()
        self._prepared_poolers = ()
        self._geometry_banks = ()

    def clear_prepared_hierarchy(self) -> None:
        self.clear_prepared_graph()

    def _apply(self, fn, recurse: bool = True):
        self.clear_prepared_graph()
        return super()._apply(fn, recurse=recurse)

    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        graph: SphereGraphGeometry | SphereGraphHierarchy,
    ) -> Tensor:
        if self._prepared_graph is None:
            raise RuntimeError("graph geometry is not prepared; call prepare_graph() before forward()")
        if graph is not self._prepared_graph:
            raise RuntimeError(
                "the classifier is bound to a different graph; call prepare_graph() to replace it"
            )
        return self.forward_prepared(x)

    def forward_prepared(self, x: Tensor | dict[str, Tensor]) -> Tensor:
        """Run classification using the statically bound graph geometry."""

        if isinstance(x, dict):
            x = x["features"]
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.ndim != 3:
            raise ValueError("x must have shape [B, N, C] or [B, N]")

        if not self._prepared_graphs:
            raise RuntimeError("graph geometry is not prepared; call prepare_graph() before forward()")
        graphs = self._prepared_graphs
        poolers = self._prepared_poolers
        hierarchy_bound = isinstance(self._prepared_graph, SphereGraphHierarchy)
        if graphs[0].device != x.device or graphs[0].dtype != x.dtype:
            raise RuntimeError(
                "prepared graph device/dtype does not match x; call prepare_graph() again"
            )

        y = x
        level_i = 0
        for block in self.blocks:
            current_graph = graphs[level_i]
            pooler = poolers[level_i] if hierarchy_bound and block.stride == 2 else None
            y = block.forward_prepared(y, graph=current_graph, pooler=pooler)
            if hierarchy_bound and block.stride == 2:
                level_i += 1
        return self.head(self._invariant_readout(y))

    @staticmethod
    def loss(logits: Tensor, labels: Tensor) -> Tensor:
        return torch.nn.functional.cross_entropy(logits, labels)

    @staticmethod
    def accuracy(logits: Tensor, labels: Tensor) -> float:
        predicted = torch.argmax(logits.detach(), dim=1)
        return float((predicted == labels).to(dtype=torch.float32).mean().item())
