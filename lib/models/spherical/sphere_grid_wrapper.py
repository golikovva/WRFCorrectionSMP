"""Adapters between regular latitude/longitude grids and sphere graph models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor, nn

from ...data.spherical.sphere_geometry import TangentFrameStrategy, latlon_to_xyz
from ...data.spherical.sphere_hierarchy import SphereGraphHierarchy, build_fps_sphere_hierarchy
from .irrep_layers import IrrepSphereConv, SO2IrrepFieldType

OutputLayout = Literal["auto", "grid", "flat"]


@dataclass(frozen=True)
class FieldSchema:
    """Immutable mapping between external channels and scalar/vector irreps."""

    _scalars: tuple[tuple[str, int], ...]
    _vectors: tuple[tuple[str, tuple[int, int]], ...]
    irrep_type: SO2IrrepFieldType
    num_channels: int
    canonical_indices: tuple[int, ...]
    inverse_indices: tuple[int, ...]

    def __init__(
        self,
        *,
        scalars: Mapping[str, int] | None = None,
        vectors: Mapping[str, tuple[int, int]] | None = None,
    ) -> None:
        if scalars is not None and not isinstance(scalars, Mapping):
            raise TypeError("scalars must be a mapping from names to channel indices")
        if vectors is not None and not isinstance(vectors, Mapping):
            raise TypeError("vectors must be a mapping from names to channel-index pairs")

        scalar_items = tuple((name, index) for name, index in (scalars or {}).items())
        vector_items = tuple((name, channels) for name, channels in (vectors or {}).items())
        scalar_names = self._validate_names(scalar_items, "scalar")
        vector_names = self._validate_names(vector_items, "vector")
        overlap = scalar_names.intersection(vector_names)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"field names must be unique; repeated name(s): {names}")

        normalized_scalars = tuple(
            (name, self._channel_index(index, field_name=name))
            for name, index in scalar_items
        )
        normalized_vectors = []
        for name, channels in vector_items:
            if not isinstance(channels, (tuple, list)) or len(channels) != 2:
                raise ValueError(f"vector field {name!r} must contain exactly two channels")
            first = self._channel_index(channels[0], field_name=name)
            second = self._channel_index(channels[1], field_name=name)
            if first == second:
                raise ValueError(f"vector field {name!r} must use two distinct channels")
            normalized_vectors.append((name, (first, second)))
        normalized_vectors_tuple = tuple(normalized_vectors)

        canonical_indices = tuple(index for _, index in normalized_scalars) + tuple(
            index
            for _, channels in normalized_vectors_tuple
            for index in channels
        )
        if not canonical_indices:
            raise ValueError("FieldSchema must contain at least one scalar or vector field")
        expected = set(range(len(canonical_indices)))
        actual = set(canonical_indices)
        if len(actual) != len(canonical_indices):
            raise ValueError("each external channel must be used exactly once")
        if actual != expected:
            missing = sorted(expected.difference(actual))
            unexpected = sorted(actual.difference(expected))
            raise ValueError(
                "field channels must exactly cover 0..C-1; "
                f"missing={missing}, unexpected={unexpected}"
            )

        inverse_indices = [0] * len(canonical_indices)
        for canonical_index, external_index in enumerate(canonical_indices):
            inverse_indices[external_index] = canonical_index
        scalar_count = len(normalized_scalars)
        vector_count = len(normalized_vectors_tuple)
        multiplicities = (
            (scalar_count, vector_count) if vector_count > 0 else (scalar_count,)
        )

        object.__setattr__(self, "_scalars", normalized_scalars)
        object.__setattr__(self, "_vectors", normalized_vectors_tuple)
        object.__setattr__(self, "irrep_type", SO2IrrepFieldType(multiplicities))
        object.__setattr__(self, "num_channels", len(canonical_indices))
        object.__setattr__(self, "canonical_indices", canonical_indices)
        object.__setattr__(self, "inverse_indices", tuple(inverse_indices))

    @staticmethod
    def _validate_names(items: tuple[tuple[str, Any], ...], kind: str) -> set[str]:
        names = []
        for name, _ in items:
            if not isinstance(name, str) or not name:
                raise ValueError(f"{kind} field names must be non-empty strings")
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError(f"{kind} field names must be unique")
        return set(names)

    @staticmethod
    def _channel_index(value: Any, *, field_name: str) -> int:
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError(f"channel index for field {field_name!r} must be an integer")
        index = int(value)
        if index < 0:
            raise ValueError(f"channel index for field {field_name!r} must be non-negative")
        return index

    @property
    def scalars(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._scalars))

    @property
    def vectors(self) -> Mapping[str, tuple[int, int]]:
        return MappingProxyType(dict(self._vectors))


class _IrrepChannelAdapter(nn.Module):
    def __init__(self, schema: FieldSchema, indices: tuple[int, ...]) -> None:
        super().__init__()
        if not isinstance(schema, FieldSchema):
            raise TypeError("schema must be a FieldSchema")
        self.schema = schema
        self.register_buffer(
            "indices",
            torch.tensor(indices, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim < 1 or int(x.shape[-1]) != self.schema.num_channels:
            raise ValueError(
                f"x must have {self.schema.num_channels} channels in its last dimension"
            )
        return x.index_select(-1, self.indices)


class IrrepInputAdapter(_IrrepChannelAdapter):
    """Pack external channels into canonical SO(2) irrep order."""

    def __init__(self, schema: FieldSchema) -> None:
        super().__init__(schema, schema.canonical_indices)


class IrrepOutputAdapter(_IrrepChannelAdapter):
    """Unpack canonical SO(2) irreps into the schema's external order."""

    def __init__(self, schema: FieldSchema) -> None:
        super().__init__(schema, schema.inverse_indices)


class SphereGridModelWrapper(nn.Module):
    """Expose a ``model(x, hierarchy)`` sphere model as a grid model.

    Input tensors use the conventional ``[B, C, H, W]`` layout.  The wrapper
    flattens the fixed latitude/longitude grid to ``[B, N, C]``, supplies the
    pre-built graph hierarchy, and converts pointwise outputs back to a grid.

    Example::

        unet = IrrepSphereUNet(SO2IrrepFieldType.scalar(in_channels))
        model = SphereGridModelWrapper(unet, grid)
        output = model(data)  # [B, C, H, W] -> [B, C, H, W]

    ``grid`` must contain ``"latitude"`` and ``"longitude"`` values.  They may
    be matching two-dimensional arrays or a pair of one-dimensional coordinate
    arrays.  ``frame_strategy="east_north"`` interprets vector components in
    the conventional east/north basis; no channel conversion is performed.
    Pass ``from_file`` to reuse a hierarchy saved by an earlier run, or
    ``to_file`` to persist the resulting hierarchy for future runs.  Cached
    hierarchies are accepted only when their finest level matches ``grid``.
    """

    def __init__(
        self,
        model: nn.Module,
        grid: Mapping[str, Any],
        *,
        levels: int = 4,
        radius_km: float = 1300.0,
        max_neighbors: int | None = None,
        earth_radius_km: float = 6371.0,
        degrees: bool = True,
        prefer_scipy: bool = True,
        pool_ratio: float = 0.25,
        min_points: int = 16,
        output_layout: OutputLayout = "auto",
        frame_strategy: TangentFrameStrategy = "robust",
        input_schema: FieldSchema | None = None,
        output_schema: FieldSchema | None = None,
        irrep_conv_backend: Literal["auto", "torch", "triton"] | None = None,
        from_file: str | PathLike[str] | None = None,
        to_file: str | PathLike[str] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(grid, Mapping):
            raise TypeError("grid must be a mapping with latitude and longitude")
        missing = {"latitude", "longitude"}.difference(grid)
        if missing:
            names = ", ".join(sorted(missing))
            raise KeyError(f"grid is missing required coordinate(s): {names}")
        if output_layout not in ("auto", "grid", "flat"):
            raise ValueError("output_layout must be 'auto', 'grid', or 'flat'")
        self._validate_model_schema(model, input_schema, "in_type", "input_schema")
        self._validate_model_schema(model, output_schema, "output_type", "output_schema")
        if irrep_conv_backend is not None:
            if irrep_conv_backend not in ("auto", "torch", "triton"):
                raise ValueError("irrep_conv_backend must be 'auto', 'torch', 'triton', or None")
            for module in model.modules():
                if isinstance(module, IrrepSphereConv):
                    module.backend = irrep_conv_backend

        self.model = model
        self.irrep_conv_backend = irrep_conv_backend
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.input_adapter = (
            IrrepInputAdapter(input_schema) if input_schema is not None else None
        )
        self.output_adapter = (
            IrrepOutputAdapter(output_schema) if output_schema is not None else None
        )
        self.output_layout: OutputLayout = output_layout
        self._hierarchy_options = {
            "levels": int(levels),
            "radius_km": float(radius_km),
            "max_neighbors": max_neighbors,
            "earth_radius_km": float(earth_radius_km),
            "degrees": bool(degrees),
            "prefer_scipy": bool(prefer_scipy),
            "pool_ratio": float(pool_ratio),
            "min_points": int(min_points),
            "frame_strategy": frame_strategy,
        }
        target_device, target_dtype = self._model_device_dtype()
        self.register_buffer(
            "_geometry_anchor",
            torch.empty(0, device=target_device, dtype=target_dtype),
            persistent=False,
        )
        self._grid_shape = (0, 0)
        self._hierarchy: SphereGraphHierarchy | None = None
        self.set_grid(grid, from_file=from_file, to_file=to_file)

    @staticmethod
    def _validate_model_schema(
        model: nn.Module,
        schema: FieldSchema | None,
        type_attribute: str,
        argument_name: str,
    ) -> None:
        if schema is None:
            return
        if not isinstance(schema, FieldSchema):
            raise TypeError(f"{argument_name} must be a FieldSchema or None")
        model_type = getattr(model, type_attribute, None)
        if not isinstance(model_type, SO2IrrepFieldType):
            raise TypeError(
                f"{argument_name} requires the wrapped model to expose "
                f"{type_attribute}: SO2IrrepFieldType"
            )
        if model_type != schema.irrep_type:
            raise ValueError(
                f"{argument_name}.irrep_type {schema.irrep_type} does not match "
                f"model.{type_attribute} {model_type}"
            )

    def _model_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        for tensor in self.model.parameters():
            if tensor.is_floating_point():
                return tensor.device, tensor.dtype
        for tensor in self.model.buffers():
            if tensor.is_floating_point():
                return tensor.device, tensor.dtype
        return torch.device("cpu"), torch.float32

    def _build_hierarchy(
        self,
        grid: Mapping[str, Any],
        *,
        from_file: str | PathLike[str] | None,
        to_file: str | PathLike[str] | None,
    ) -> tuple[SphereGraphHierarchy, tuple[int, int]]:
        if not isinstance(grid, Mapping):
            raise TypeError("grid must be a mapping with latitude and longitude")
        missing = {"latitude", "longitude"}.difference(grid)
        if missing:
            names = ", ".join(sorted(missing))
            raise KeyError(f"grid is missing required coordinate(s): {names}")

        options = self._hierarchy_options
        points_xyz, grid_shape = latlon_to_xyz(
            grid["latitude"],
            grid["longitude"],
            degrees=options["degrees"],
            dtype=torch.float32,
        )
        if from_file is None:
            hierarchy = build_fps_sphere_hierarchy(
                points_xyz,
                levels=options["levels"],
                radius_km=options["radius_km"],
                max_neighbors=options["max_neighbors"],
                earth_radius_km=options["earth_radius_km"],
                prefer_scipy=options["prefer_scipy"],
                pool_ratio=options["pool_ratio"],
                min_points=options["min_points"],
                frame_strategy=options["frame_strategy"],
            )
        else:
            load_kwargs = {"map_location": "cpu"}
            try:
                hierarchy = torch.load(from_file, weights_only=False, **load_kwargs)
            except TypeError:
                hierarchy = torch.load(from_file, **load_kwargs)
            if not isinstance(hierarchy, SphereGraphHierarchy):
                raise TypeError(
                    "from_file must contain a SphereGraphHierarchy, "
                    f"got {type(hierarchy).__name__}"
                )
            cached_points = hierarchy.graphs[0].points_xyz
            expected_points = torch.nn.functional.normalize(points_xyz.detach().cpu(), dim=-1)
            if cached_points.shape != expected_points.shape or not torch.allclose(
                cached_points,
                expected_points.to(dtype=cached_points.dtype),
            ):
                raise ValueError("the hierarchy in from_file does not match the supplied grid")

        if to_file is not None:
            destination = Path(to_file)
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(hierarchy, destination)
        return hierarchy, grid_shape

    def _bind_hierarchy(self, hierarchy: SphereGraphHierarchy) -> None:
        clear = getattr(self.model, "clear_prepared_hierarchy", None)
        if callable(clear):
            clear()
        self._hierarchy = hierarchy
        prepare = getattr(self.model, "prepare_hierarchy", None)
        if callable(prepare):
            prepare(hierarchy)
            return
        prepare_graph = getattr(self.model, "prepare_graph", None)
        if callable(prepare_graph):
            prepare_graph(hierarchy)

    def set_grid(
        self,
        grid: Mapping[str, Any],
        *,
        from_file: str | PathLike[str] | None = None,
        to_file: str | PathLike[str] | None = None,
    ) -> None:
        """Replace the static grid and rebuild the wrapped model's geometry."""

        hierarchy, grid_shape = self._build_hierarchy(
            grid,
            from_file=from_file,
            to_file=to_file,
        )
        hierarchy = hierarchy.to(
            device=self._geometry_anchor.device,
            dtype=self._geometry_anchor.dtype,
        )
        self._grid_shape = grid_shape
        self._bind_hierarchy(hierarchy)

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        hierarchy = self._hierarchy
        if hierarchy is not None:
            hierarchy = hierarchy.to(
                device=self._geometry_anchor.device,
                dtype=self._geometry_anchor.dtype,
            )
            self._bind_hierarchy(hierarchy)
        return result

    @property
    def grid_shape(self) -> tuple[int, int]:
        """The fixed ``(H, W)`` shape represented by the hierarchy."""

        return self._grid_shape

    @property
    def hierarchy(self) -> SphereGraphHierarchy:
        """The statically bound hierarchy."""

        hierarchy = self._hierarchy
        if hierarchy is None:
            raise RuntimeError("grid hierarchy is not prepared; call set_grid()")
        return hierarchy

    def _hierarchy_for(self, x: Tensor) -> SphereGraphHierarchy:
        hierarchy = self._hierarchy
        if hierarchy is None:
            raise RuntimeError("grid hierarchy is not prepared; call set_grid()")
        if any(
            graph.device != x.device or graph.dtype != x.dtype
            for graph in hierarchy.graphs
        ):
            raise RuntimeError(
                "prepared hierarchy device/dtype does not match x; move the wrapper "
                "with .to() before calling forward()"
            )
        return hierarchy

    def _to_grid(self, output: Tensor, *, batch_size: int) -> Tensor:
        height, width = self._grid_shape
        n_points = height * width
        if (
            output.ndim != 3
            or int(output.shape[0]) != batch_size
            or int(output.shape[1]) != n_points
        ):
            raise ValueError(
                "grid output requires a tensor with shape "
                f"[{batch_size}, {n_points}, C_out], got {tuple(output.shape)}"
            )
        channels = int(output.shape[2])
        return output.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

    def forward(self, x: Tensor) -> Any:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 4:
            raise ValueError("x must have shape [B, C, H, W]")
        if not x.is_floating_point():
            raise TypeError("x must have a floating-point dtype")

        batch, channels, height, width = (int(value) for value in x.shape)
        expected_height, expected_width = self._grid_shape
        if (height, width) != self._grid_shape:
            raise ValueError(
                f"input grid has shape [{height}, {width}], expected "
                f"[{expected_height}, {expected_width}]"
            )

        x_flat = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        if self.input_adapter is not None:
            x_flat = self.input_adapter(x_flat)
        hierarchy = self._hierarchy_for(x)
        forward_prepared = getattr(self.model, "forward_prepared", None)
        if callable(forward_prepared):
            output = forward_prepared(x_flat)
        else:
            output = self.model(x_flat, hierarchy)

        if self.output_adapter is not None:
            if not isinstance(output, Tensor):
                raise TypeError(
                    "output_schema requires the wrapped model to return a [B, N, C] tensor"
                )
            if (
                output.ndim != 3
                or int(output.shape[0]) != batch
                or int(output.shape[1]) != height * width
            ):
                raise ValueError(
                    "output_schema requires the wrapped model to return a tensor with "
                    f"shape [{batch}, {height * width}, C], got {tuple(output.shape)}"
                )
            output = self.output_adapter(output)

        if self.output_layout == "flat":
            return output
        if self.output_layout == "grid":
            if not isinstance(output, Tensor):
                raise TypeError(
                    "output_layout='grid' requires the wrapped model to return a tensor"
                )
            return self._to_grid(output, batch_size=batch)
        if (
            isinstance(output, Tensor)
            and output.ndim == 3
            and int(output.shape[0]) == batch
            and int(output.shape[1]) == height * width
        ):
            return self._to_grid(output, batch_size=batch)
        return output


__all__ = ["SphereGridModelWrapper"]
