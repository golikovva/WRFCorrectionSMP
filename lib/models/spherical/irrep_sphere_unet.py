from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from ...data.spherical.sphere_graph import SphereGraphGeometry
from ...data.spherical.sphere_hierarchy import SphereGraphHierarchy, SpherePooler, SphereWeightedPooler
from .spherical_mnist_model import IrrepSphereGraphPool
from .profile_ranges import record_region
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

def _concatenated_field_type(
    first_type: SO2IrrepFieldType,
    second_type: SO2IrrepFieldType,
) -> SO2IrrepFieldType:
    max_order = max(first_type.max_order, second_type.max_order)
    return SO2IrrepFieldType(
        tuple(
            first_type.multiplicity(order) + second_type.multiplicity(order)
            for order in range(max_order + 1)
        )
    )


def _concatenate_irrep_fields(
    first: Tensor,
    first_type: SO2IrrepFieldType,
    second: Tensor,
    second_type: SO2IrrepFieldType,
) -> Tensor:
    """Concatenate fields inside each irrep order, preserving the flat layout."""

    if first.shape[:-1] != second.shape[:-1]:
        raise ValueError("irrep fields must have matching batch and point dimensions")
    if int(first.shape[-1]) != first_type.total_dim:
        raise ValueError(f"first field must have {first_type.total_dim} channels")
    if int(second.shape[-1]) != second_type.total_dim:
        raise ValueError(f"second field must have {second_type.total_dim} channels")

    out_type = _concatenated_field_type(first_type, second_type)
    blocks = []
    for order in out_type.orders:
        parts = []
        if first_type.multiplicity(order) > 0:
            parts.append(_irrep_block(first, first_type, order))
        if second_type.multiplicity(order) > 0:
            parts.append(_irrep_block(second, second_type, order))
        blocks.append(_flatten_irrep_block(torch.cat(parts, dim=-2)))
    return torch.cat(blocks, dim=-1)


class IrrepSphereUnpool(nn.Module):
    """Equivariant coarse-to-fine transport for flat SO(2) irrep fields."""

    def __init__(self, field_type: SO2IrrepFieldType) -> None:
        super().__init__()
        self.field_type = field_type

    def forward(
        self,
        x: Tensor,
        pooler: SpherePooler | SphereWeightedPooler,
    ) -> Tensor:
        if x.ndim != 3 or int(x.shape[-1]) != self.field_type.total_dim:
            raise ValueError(f"x must have shape [B, N_coarse, {self.field_type.total_dim}]")
        if int(x.shape[1]) != pooler.coarse_graph.n_points:
            raise ValueError(
                f"x has {x.shape[1]} points, expected {pooler.coarse_graph.n_points}"
            )
        if isinstance(pooler, SphereWeightedPooler):
            return self._unpool_weighted(x, pooler)
        return self._unpool_assignment(x, pooler)

    def _unpool_assignment(self, x: Tensor, pooler: SpherePooler) -> Tensor:
        assignment = pooler.assignment.to(device=x.device)
        assert pooler.transport_angle is not None
        angle = pooler.transport_angle.to(device=x.device, dtype=x.dtype)
        outputs = []
        for order in self.field_type.orders:
            coarse = _irrep_block(x, self.field_type, order)
            fine = _rotate_irrep_block(coarse[:, assignment, :, :], -angle, order)
            outputs.append(_flatten_irrep_block(fine))
        return torch.cat(outputs, dim=-1)

    def _unpool_weighted(self, x: Tensor, pooler: SphereWeightedPooler) -> Tensor:
        fine_idx = pooler.fine_idx.to(device=x.device)
        coarse_idx = pooler.coarse_idx.to(device=x.device)
        weight = pooler.weight.to(device=x.device, dtype=x.dtype)
        assert pooler.transport_angle is not None
        angle = pooler.transport_angle.to(device=x.device, dtype=x.dtype)

        fine_weight = x.new_zeros(pooler.fine_graph.n_points)
        fine_weight.scatter_add_(0, fine_idx, weight)
        if torch.any(fine_weight <= 0):
            raise ValueError("weighted pooler must cover every fine point")

        outputs = []
        for order in self.field_type.orders:
            coarse = _irrep_block(x, self.field_type, order)
            values = _rotate_irrep_block(coarse[:, coarse_idx, :, :], -angle, order)
            values = values * weight.view(1, -1, 1, 1)
            fine = x.new_zeros(
                (
                    int(x.shape[0]),
                    pooler.fine_graph.n_points,
                    *self.field_type.block_shape(order),
                )
            )
            index = fine_idx.view(1, -1, 1, 1).expand_as(values)
            fine.scatter_add_(1, index, values)
            fine = fine / fine_weight.view(1, -1, 1, 1)
            outputs.append(_flatten_irrep_block(fine))
        return torch.cat(outputs, dim=-1)


class _IrrepSphereDoubleConv(nn.Module):
    def __init__(
        self,
        in_type: SO2IrrepFieldType,
        out_type: SO2IrrepFieldType,
        *,
        radius_km: float,
        num_radial: int,
        regular_samples: int | None,
        normalize_by_neighbors: bool,
        include_self: bool,
        self_identity_init: bool,
        quadrature: bool,
        quadrature_radial: int | None,
        quadrature_angular: int,
        quadrature_sigma_km: float | None,
        irrep_conv_backend: Literal["auto", "torch", "triton"],
    ) -> None:
        super().__init__()
        conv_kwargs = {
            "normalize_by_neighbors": normalize_by_neighbors,
            "include_self": include_self,
            "self_identity_init": self_identity_init,
            "quadrature": quadrature,
            "quadrature_radial": quadrature_radial,
            "quadrature_angular": quadrature_angular,
            "quadrature_sigma_km": quadrature_sigma_km,
            "backend": irrep_conv_backend,
        }
        self.in_type = in_type
        self.out_type = out_type
        self.conv1 = IrrepSphereConv(
            in_type,
            out_type,
            radius_km,
            num_radial,
            **conv_kwargs,
        )
        self.norm1 = IrrepBatchNorm(out_type)
        self.activation1 = RegularNonlinearity(out_type, regular_samples=regular_samples)
        self.conv2 = IrrepSphereConv(
            out_type,
            out_type,
            radius_km,
            num_radial,
            **conv_kwargs,
        )
        self.norm2 = IrrepBatchNorm(out_type)
        self.activation2 = RegularNonlinearity(out_type, regular_samples=regular_samples)

    def prepare_graph(self, graph: SphereGraphGeometry) -> None:
        self.conv1.prepare_graph(graph, device=graph.device, dtype=graph.dtype)
        self.conv2.prepare_graph(graph, device=graph.device, dtype=graph.dtype)

    def bind_prepared_geometry(self, geometry: _IrrepConvGeometry) -> None:
        self.conv1._bind_prepared_geometry(geometry)
        self.conv2._bind_prepared_geometry(geometry)

    def clear_prepared_graph(self) -> None:
        self.conv1.clear_prepared_graph()
        self.conv2.clear_prepared_graph()

    def forward(self, x: Tensor, graph: SphereGraphGeometry) -> Tensor:
        x = self.activation1(self.norm1(self.conv1(x, graph)))
        return self.activation2(self.norm2(self.conv2(x, graph)))

    def forward_prepared(self, x: Tensor) -> Tensor:
        profile_name = getattr(self, "_irrep_profile_name", "double_conv")
        with record_region(f"module/{profile_name}"):
            with record_region(f"module/{profile_name}.conv1"):
                x = self.conv1.forward_prepared(x)
            with record_region(f"module/{profile_name}.norm1"):
                x = self.norm1(x)
            with record_region(f"module/{profile_name}.activation1"):
                x = self.activation1(x)
            with record_region(f"module/{profile_name}.conv2"):
                x = self.conv2.forward_prepared(x)
            with record_region(f"module/{profile_name}.norm2"):
                x = self.norm2(x)
            with record_region(f"module/{profile_name}.activation2"):
                return self.activation2(x)


class IrrepSphereUNet(nn.Module):
    """Four-level spherical UNet built entirely from equivariant irrep operators."""

    def __init__(
        self,
        in_type: SO2IrrepFieldType,
        *,
        out_type: SO2IrrepFieldType | None = None,
        multiplicities: Sequence[int] = (8, 16, 32, 64),
        max_order: int | None = None,
        radius_km: float = 1300.0,
        num_radial: int = 4,
        regular_samples: int | None = None,
        normalize_by_neighbors: bool = True,
        include_self: bool = True,
        self_identity_init: bool = False,
        quadrature: bool = False,
        quadrature_radial: int | None = None,
        quadrature_angular: int = 16,
        quadrature_sigma_km: float | None = None,
        irrep_conv_backend: Literal["auto", "torch", "triton"] = "auto",
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in multiplicities)
        if len(widths) != 4:
            raise ValueError("multiplicities must contain exactly four levels")
        if any(value < 1 for value in widths):
            raise ValueError("multiplicities values must be positive")
        output_type = in_type if out_type is None else out_type
        if not isinstance(output_type, SO2IrrepFieldType):
            raise TypeError("out_type must be an SO2IrrepFieldType or None")
        hidden_max_order = (
            max(2, in_type.max_order, output_type.max_order)
            if max_order is None
            else int(max_order)
        )
        if hidden_max_order < 0:
            raise ValueError("max_order must be non-negative")

        self.in_type = in_type
        self.output_type = output_type
        self.multiplicities = widths
        self.max_order = hidden_max_order
        self.irrep_conv_backend = str(irrep_conv_backend)
        self.level_types = tuple(
            SO2IrrepFieldType.balanced(hidden_max_order, width) for width in widths
        )

        block_kwargs = {
            "radius_km": radius_km,
            "num_radial": num_radial,
            "regular_samples": regular_samples,
            "normalize_by_neighbors": normalize_by_neighbors,
            "include_self": include_self,
            "self_identity_init": self_identity_init,
            "quadrature": quadrature,
            "quadrature_radial": quadrature_radial,
            "quadrature_angular": quadrature_angular,
            "quadrature_sigma_km": quadrature_sigma_km,
            "irrep_conv_backend": irrep_conv_backend,
        }
        encoder_in_types = (in_type, *self.level_types[:-1])
        self.encoder_blocks = nn.ModuleList(
            _IrrepSphereDoubleConv(current, output, **block_kwargs)
            for current, output in zip(encoder_in_types, self.level_types)
        )
        self.pool_layers = nn.ModuleList(
            IrrepSphereGraphPool(field_type) for field_type in self.level_types[:-1]
        )
        self.unpool_layers = nn.ModuleList(
            IrrepSphereUnpool(field_type)
            for field_type in reversed(self.level_types[1:])
        )

        decoder_blocks = []
        current_type = self.level_types[-1]
        for skip_type in reversed(self.level_types[:-1]):
            merged_type = _concatenated_field_type(current_type, skip_type)
            decoder_blocks.append(
                _IrrepSphereDoubleConv(merged_type, skip_type, **block_kwargs)
            )
            current_type = skip_type
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        final_kwargs = {
            key: value
            for key, value in block_kwargs.items()
            if key not in ("regular_samples", "irrep_conv_backend")
        }
        final_kwargs["backend"] = irrep_conv_backend
        self.output_conv = IrrepSphereConv(
            self.level_types[0],
            output_type,
            final_kwargs.pop("radius_km"),
            final_kwargs.pop("num_radial"),
            **final_kwargs,
        )
        self._prepared_hierarchy: SphereGraphHierarchy | None = None
        self._geometry_banks: tuple[_IrrepConvGeometry, ...] = ()

    def _apply(self, fn, recurse: bool = True):
        self.clear_prepared_hierarchy()
        return super()._apply(fn, recurse=recurse)

    def _convolutions_by_level(self) -> tuple[tuple[IrrepSphereConv, ...], ...]:
        levels: list[list[IrrepSphereConv]] = [[], [], [], []]
        for level, block in enumerate(self.encoder_blocks):
            levels[level].extend((block.conv1, block.conv2))
        for decoder_index, block in enumerate(self.decoder_blocks):
            level = 2 - decoder_index
            levels[level].extend((block.conv1, block.conv2))
        levels[0].append(self.output_conv)
        return tuple(tuple(convs) for convs in levels)

    def prepare_hierarchy(self, hierarchy: SphereGraphHierarchy) -> None:
        self._validate_hierarchy(hierarchy)
        self.clear_prepared_hierarchy()

        banks = []
        for graph, convolutions in zip(hierarchy.graphs[:4], self._convolutions_by_level()):
            compatible_groups: dict[tuple, list[IrrepSphereConv]] = {}
            for conv in convolutions:
                compatible_groups.setdefault(conv._geometry_signature(), []).append(conv)
            for compatible_convolutions in compatible_groups.values():
                representative = compatible_convolutions[0]
                in_orders = tuple(
                    sorted(
                        {
                            order
                            for conv in compatible_convolutions
                            for order in conv._in_orders
                        }
                    )
                )
                out_orders = tuple(
                    sorted(
                        {
                            order
                            for conv in compatible_convolutions
                            for order in conv._out_orders
                        }
                    )
                )
                bank = representative._build_geometry(
                    graph,
                    graph.device,
                    graph.dtype,
                    in_orders=in_orders,
                    out_orders=out_orders,
                )
                for conv in compatible_convolutions:
                    conv._bind_prepared_geometry(bank)
                banks.append(bank)

        self._geometry_banks = tuple(banks)
        self._prepared_hierarchy = hierarchy

    def clear_prepared_hierarchy(self) -> None:
        """Drop all shared geometry banks while preserving learned parameters."""

        for block in self.encoder_blocks:
            block.clear_prepared_graph()
        for block in self.decoder_blocks:
            block.clear_prepared_graph()
        self.output_conv.clear_prepared_graph()
        self._geometry_banks = ()
        self._prepared_hierarchy = None

    def forward(self, x: Tensor, hierarchy: SphereGraphHierarchy) -> Tensor:
        if x.ndim != 3 or int(x.shape[-1]) != self.in_type.total_dim:
            raise ValueError(f"x must have shape [B, N, {self.in_type.total_dim}]")
        self._validate_hierarchy(hierarchy)
        if self._prepared_hierarchy is None:
            raise RuntimeError(
                "hierarchy geometry is not prepared; call prepare_hierarchy() before forward()"
            )
        if hierarchy is not self._prepared_hierarchy:
            raise RuntimeError(
                "the model is bound to a different hierarchy; call prepare_hierarchy() to replace it"
            )
        return self.forward_prepared(x)

    def forward_prepared(self, x: Tensor) -> Tensor:
        """Run the U-Net using the statically bound shared geometry banks."""

        if x.ndim != 3 or int(x.shape[-1]) != self.in_type.total_dim:
            raise ValueError(f"x must have shape [B, N, {self.in_type.total_dim}]")
        hierarchy = self._prepared_hierarchy
        if hierarchy is None:
            raise RuntimeError(
                "hierarchy geometry is not prepared; call prepare_hierarchy() before forward()"
            )
        graphs = hierarchy.graphs[:4]
        poolers = hierarchy.poolers[:3]
        if graphs[0].device != x.device or graphs[0].dtype != x.dtype:
            raise RuntimeError(
                "prepared hierarchy device/dtype does not match x; call prepare_hierarchy() again"
            )
        if int(x.shape[1]) != graphs[0].n_points:
            raise ValueError(f"x has {x.shape[1]} points, expected {graphs[0].n_points}")

        skips = []
        y = x
        for level, block in enumerate(self.encoder_blocks):
            with record_region(f"stage/encoder.{level}"):
                y = block.forward_prepared(y)
            if level < 3:
                skips.append(y)
                with record_region(f"module/pool.{level}"):
                    y = self.pool_layers[level].pool_with(y, poolers[level])

        for decoder_index, (unpool, block) in enumerate(
            zip(self.unpool_layers, self.decoder_blocks)
        ):
            pooler = poolers[2 - decoder_index]
            with record_region(f"module/unpool.{decoder_index}"):
                y = unpool(y, pooler)
            skip = skips[2 - decoder_index]
            with record_region(f"op/skip_concat.{decoder_index}"):
                y = _concatenate_irrep_fields(
                    y,
                    self.level_types[3 - decoder_index],
                    skip,
                    self.level_types[2 - decoder_index],
                )
            with record_region(f"stage/decoder.{decoder_index}"):
                y = block.forward_prepared(y)
        with record_region("module/output_conv"):
            return self.output_conv.forward_prepared(y)

    @staticmethod
    def _validate_hierarchy(hierarchy: SphereGraphHierarchy) -> None:
        if not isinstance(hierarchy, SphereGraphHierarchy):
            raise TypeError("hierarchy must be a SphereGraphHierarchy")
        if hierarchy.n_levels < 4:
            raise ValueError(
                f"hierarchy has {hierarchy.n_levels} levels, but IrrepSphereUNet requires 4"
            )


__all__ = ["IrrepSphereUNet", "IrrepSphereUnpool"]
