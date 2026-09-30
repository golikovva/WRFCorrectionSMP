"""Bounded exact-edge streaming forward for regular R1 convolutions.

The production inference path divides center-sorted CSR into complete-center
chunks, materializes only real edge rows, and reduces every center in one
Triton program.  The operator is deterministic, uses no atomics, and applies a
hard 128 MiB auxiliary-workspace cap even when the public workspace cap is
larger.  Extra weight-layout and matmul-family switches remain available only
for the benchmark A/B harness; production dispatch always uses the proven
transpose-workspace plus dense-SM8 route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import Tensor

from ._triton_layout import fix_triton_strides

from . import triton_semi_packed_irrep_conv as _semi


EXACT_EDGE_HARD_CAP_BYTES = 128 << 20
# Train-derived threshold on the larger packed matrix dimension. Keeping both
# dimensions below it preserves the fused scalar/vector calibration regime.
EXACT_EDGE_AUTO_MIN_MATRIX_DIM = 80
EXACT_EDGE_WEIGHT_LAYOUTS = ("transpose_workspace", "direct")
EXACT_EDGE_MATMUL_FAMILIES = ("production", "wide_sm8", "dense_sm8")
PRODUCTION_EXACT_EDGE_WEIGHT_LAYOUT = "transpose_workspace"
PRODUCTION_EXACT_EDGE_MATMUL_FAMILY = "dense_sm8"


@dataclass(frozen=True)
class ExactEdgeMatmulConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


EXACT_EDGE_PRODUCTION_SM8_CONFIGS = (
    ExactEdgeMatmulConfig(16, 32, 32, 4, 2),
    ExactEdgeMatmulConfig(32, 32, 32, 4, 3),
    ExactEdgeMatmulConfig(32, 64, 32, 4, 3),
    ExactEdgeMatmulConfig(64, 32, 64, 8, 3),
)
EXACT_EDGE_WIDE_SM8_ADDITIONAL_CONFIGS = (
    ExactEdgeMatmulConfig(64, 64, 32, 8, 3),
    ExactEdgeMatmulConfig(64, 64, 64, 8, 3),
    ExactEdgeMatmulConfig(128, 32, 32, 8, 3),
    ExactEdgeMatmulConfig(32, 128, 32, 8, 3),
    ExactEdgeMatmulConfig(128, 64, 32, 8, 3),
)
EXACT_EDGE_WIDE_SM8_CONFIGS = (
    EXACT_EDGE_PRODUCTION_SM8_CONFIGS
    + EXACT_EDGE_WIDE_SM8_ADDITIONAL_CONFIGS
)
EXACT_EDGE_DENSE_SM8_ADDITIONAL_CONFIGS = (
    ExactEdgeMatmulConfig(64, 64, 32, 4, 3),
    ExactEdgeMatmulConfig(64, 64, 64, 4, 3),
    ExactEdgeMatmulConfig(64, 128, 32, 8, 3),
    ExactEdgeMatmulConfig(128, 64, 64, 8, 3),
    ExactEdgeMatmulConfig(32, 64, 64, 4, 3),
)
EXACT_EDGE_DENSE_SM8_CONFIGS = (
    EXACT_EDGE_WIDE_SM8_CONFIGS
    + EXACT_EDGE_DENSE_SM8_ADDITIONAL_CONFIGS
)

try:
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton
except (ImportError, AttributeError):  # pragma: no cover - CPU-only installs
    triton = None
    tl = None
    triton_op = None
    wrap_triton = None


EXACT_EDGE_AVAILABLE = (
    triton is not None
    and triton_op is not None
    and _semi.TRITON_SEMI_PACKED_AVAILABLE
)
EXPERIMENTAL_EXACT_EDGE_AVAILABLE = EXACT_EDGE_AVAILABLE


@dataclass(frozen=True)
class ExactEdgeChunk:
    point_start: int
    point_count: int
    edge_start: int
    edge_count: int


@dataclass(frozen=True)
class ExactEdgeWorkspacePlan:
    chunks: tuple[ExactEdgeChunk, ...]
    edge_capacity: int
    point_capacity: int
    max_center_degree: int
    matrix_bytes: int
    weight_bytes: int
    allocated_bytes: int
    requested_limit_bytes: int
    effective_limit_bytes: int
    hard_cap_bytes: int
    weight_layout: str
    matmul_family: str


def _as_int_ptr(center_ptr: Tensor | Sequence[int]) -> list[int]:
    if isinstance(center_ptr, Tensor):
        if center_ptr.ndim != 1:
            raise ValueError("center_ptr must be one-dimensional")
        values = center_ptr.detach().to(device="cpu").tolist()
    else:
        values = list(center_ptr)
    result = [int(value) for value in values]
    if not result:
        raise ValueError("center_ptr must contain at least the initial zero")
    if result[0] != 0:
        raise ValueError("center_ptr must start at zero")
    if any(right < left for left, right in zip(result, result[1:])):
        raise ValueError("center_ptr must be monotonic")
    return result


def _bisect_right(ptr: Sequence[int], value: int, lo: int, hi: int) -> int:
    """Right insertion point using host operations traceable by fullgraph."""
    while lo < hi:
        mid = (lo + hi) // 2
        if value < ptr[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _bisect_left(ptr: Sequence[int], value: int, lo: int, hi: int) -> int:
    """Left insertion point using host operations traceable by fullgraph."""
    while lo < hi:
        mid = (lo + hi) // 2
        if ptr[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _greedy_boundaries(ptr: Sequence[int], edge_capacity: int) -> list[int]:
    """Return the minimum-count complete-center partition under ``capacity``."""

    n_points = len(ptr) - 1
    boundaries = [0]
    point_start = 0
    while point_start < n_points:
        edge_stop = ptr[point_start] + edge_capacity
        point_stop = _bisect_right(
            ptr, edge_stop, point_start + 1, n_points + 1,
        ) - 1
        if point_stop <= point_start:
            raise ValueError(
                "one center has more edges than the bounded row capacity"
            )
        boundaries.append(point_stop)
        point_start = point_stop
    return boundaries


def _balanced_boundaries(
    ptr: Sequence[int], edge_capacity: int, chunk_count: int,
) -> list[int] | None:
    """Balance a fixed number of complete-center chunks without exceeding cap."""

    n_points = len(ptr) - 1
    total_edges = ptr[-1]
    boundaries = [0]
    point_start = 0
    for chunk_index in range(chunk_count - 1):
        remaining_chunks = chunk_count - chunk_index - 1
        max_stop = min(
            n_points - remaining_chunks,
            _bisect_right(
                ptr,
                ptr[point_start] + edge_capacity,
                point_start + 1,
                n_points + 1,
            )
            - 1,
        )
        minimum_edge_stop = total_edges - remaining_chunks * edge_capacity
        min_stop = max(
            point_start + 1,
            _bisect_left(
                ptr,
                minimum_edge_stop,
                point_start + 1,
                max_stop + 1,
            ),
        )
        if min_stop > max_stop:
            return None

        remaining_edges = total_edges - ptr[point_start]
        target = ptr[point_start] + round(
            remaining_edges / (remaining_chunks + 1)
        )
        insertion = _bisect_left(
            ptr, target, min_stop, max_stop + 1,
        )
        point_stop = max(min_stop, min(max_stop, insertion))
        previous_stop = max(min_stop, min(max_stop, insertion - 1))
        if (abs(ptr[previous_stop] - target), previous_stop) < (
            abs(ptr[point_stop] - target), point_stop,
        ):
            point_stop = previous_stop
        boundaries.append(point_stop)
        point_start = point_stop

    if ptr[-1] - ptr[point_start] > edge_capacity:
        return None
    boundaries.append(n_points)
    return boundaries


def exact_edge_workspace_plan(
    center_ptr: Tensor | Sequence[int],
    x: Tensor,
    weight: Tensor,
    workspace_bytes: int,
    *,
    hard_cap_bytes: int = EXACT_EDGE_HARD_CAP_BYTES,
    weight_layout: str = "transpose_workspace",
    matmul_family: str = "production",
) -> ExactEdgeWorkspacePlan | None:
    """Plan bounded real-edge rows while keeping every center in one chunk."""

    if workspace_bytes <= 0 or hard_cap_bytes <= 0:
        return None
    if weight_layout not in EXACT_EDGE_WEIGHT_LAYOUTS:
        raise ValueError(f"unknown exact-edge weight layout {weight_layout!r}")
    if matmul_family not in EXACT_EDGE_MATMUL_FAMILIES:
        raise ValueError(f"unknown exact-edge matmul family {matmul_family!r}")
    if x.ndim != 3 or weight.ndim != 5:
        raise ValueError("expected x[B,N,C] and weight[R,Mo,Do,Mi,Di]")
    radial, out_m, out_dim, in_m, in_dim = map(int, weight.shape)
    if radial != 1:
        raise ValueError("exact-edge streaming only supports regular R1")
    batch, n_points = map(int, x.shape[:2])
    ptr = _as_int_ptr(center_ptr)
    if len(ptr) != n_points + 1:
        raise ValueError("center_ptr length does not match x.shape[1]")

    in_total = in_m * in_dim
    out_total = out_m * out_dim
    element_size = int(x.element_size())
    weight_bytes = (
        in_total * out_total * int(torch.float32.itemsize)
        if weight_layout == "transpose_workspace" and ptr[-1] > 0
        else 0
    )
    bytes_per_edge = batch * (in_total + out_total) * element_size
    effective_limit = min(int(workspace_bytes), int(hard_cap_bytes))
    available = effective_limit - weight_bytes
    if available < 0 or bytes_per_edge <= 0:
        return None
    raw_edge_capacity = available // bytes_per_edge

    degrees = [right - left for left, right in zip(ptr, ptr[1:])]
    max_center_degree = max(degrees, default=0)
    if raw_edge_capacity < max(1, max_center_degree):
        return None

    if n_points == 0:
        boundaries = [0]
    elif ptr[-1] == 0:
        boundaries = [0, n_points]
    else:
        greedy = _greedy_boundaries(ptr, raw_edge_capacity)
        boundaries = (
            _balanced_boundaries(ptr, raw_edge_capacity, len(greedy) - 1)
            or greedy
        )

    chunks = tuple(
        ExactEdgeChunk(
            point_start=point_start,
            point_count=point_stop - point_start,
            edge_start=ptr[point_start],
            edge_count=ptr[point_stop] - ptr[point_start],
        )
        for point_start, point_stop in zip(boundaries, boundaries[1:])
    )
    edge_capacity = max((chunk.edge_count for chunk in chunks), default=0)
    point_capacity = max((chunk.point_count for chunk in chunks), default=0)
    matrix_bytes = edge_capacity * bytes_per_edge
    allocated_bytes = matrix_bytes + weight_bytes
    if allocated_bytes > effective_limit:
        raise AssertionError("exact-edge planner exceeded its effective cap")
    return ExactEdgeWorkspacePlan(
        chunks=chunks,
        edge_capacity=edge_capacity,
        point_capacity=point_capacity,
        max_center_degree=max_center_degree,
        matrix_bytes=matrix_bytes,
        weight_bytes=weight_bytes,
        allocated_bytes=allocated_bytes,
        requested_limit_bytes=int(workspace_bytes),
        effective_limit_bytes=effective_limit,
        hard_cap_bytes=int(hard_cap_bytes),
        weight_layout=weight_layout,
        matmul_family=matmul_family,
    )


def exact_edge_workspace_report(
    plan: ExactEdgeWorkspacePlan,
    x: Tensor,
    weight: Tensor,
) -> dict[str, object]:
    """Return benchmark metadata with byte-exact persistent buffer shapes."""

    _radial, out_m, out_dim, in_m, in_dim = map(int, weight.shape)
    batch = int(x.shape[0])
    in_total = in_m * in_dim
    out_total = out_m * out_dim
    rows = batch * plan.edge_capacity
    element_size = int(x.element_size())

    def buffer(name: str, shape: list[int], dtype: str, size: int) -> dict[str, object]:
        numel = 1
        for extent in shape:
            numel *= extent
        return {
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "element_size_bytes": size,
            "numel": numel,
            "allocated_bytes": numel * size,
        }

    buffers = [
        buffer("input_workspace", [rows, in_total], str(x.dtype), element_size),
        buffer("output_workspace", [rows, out_total], str(x.dtype), element_size),
    ]
    if plan.weight_bytes:
        buffers.append(
            buffer(
                "weight_transpose",
                [out_total, in_total],
                str(torch.float32),
                int(torch.float32.itemsize),
            )
        )
    descriptors = [
        {
            "point_start": chunk.point_start,
            "point_count": chunk.point_count,
            "edge_start": chunk.edge_start,
            "edge_count": chunk.edge_count,
            "launch_point_count": plan.point_capacity,
            "launch_edge_count": plan.edge_capacity,
            "inactive_point_rows": plan.point_capacity - chunk.point_count,
            "inactive_edge_rows": plan.edge_capacity - chunk.edge_count,
        }
        for chunk in plan.chunks
    ]
    variant = (
        "experimental_exact_edge"
        if plan.weight_layout == "transpose_workspace"
        else "experimental_exact_edge_direct_weight"
    )
    if plan.matmul_family != "production":
        variant += f"_{plan.matmul_family}"
    if plan.matmul_family == "dense_sm8":
        config_specs = EXACT_EDGE_DENSE_SM8_CONFIGS
    elif plan.matmul_family == "wide_sm8":
        config_specs = EXACT_EDGE_WIDE_SM8_CONFIGS
    else:
        config_specs = EXACT_EDGE_PRODUCTION_SM8_CONFIGS
    serialized_configs = [
        {
            "BLOCK_M": config.block_m,
            "BLOCK_N": config.block_n,
            "BLOCK_K": config.block_k,
            "num_warps": config.num_warps,
            "num_stages": config.num_stages,
        }
        for config in config_specs
    ]
    phase = {
        "phase": "forward",
        "variant": variant,
        "weight_layout": plan.weight_layout,
        "weight_scratch_bytes": plan.weight_bytes,
        "matmul_family": plan.matmul_family,
        "sm8_matmul_config_search_space": serialized_configs,
        "wide_sm8_retains_production_configs": (
            plan.matmul_family in ("wide_sm8", "dense_sm8")
        ),
        "dense_sm8_retains_wide_configs": plan.matmul_family == "dense_sm8",
        "point_aligned": True,
        "gathers_only_real_edges": True,
        "inactive_gather_rows_zero_filled": True,
        "fixed_shape_across_chunks": True,
        "deterministic_complete_center_reduction": True,
        "uses_atomics": False,
        "full_edge_materialization": False,
        "descriptor_count": len(descriptors),
        "descriptors": descriptors,
        "edge_capacity": plan.edge_capacity,
        "point_capacity": plan.point_capacity,
        "max_center_degree": plan.max_center_degree,
        "buffers": buffers,
        "allocated_bytes": plan.allocated_bytes,
        "planned_peak_bytes": plan.allocated_bytes,
        "cap_bytes": plan.effective_limit_bytes,
    }
    workspace_plan = {
        "variant": variant,
        "cap_applies": True,
        "weight_layout": plan.weight_layout,
        "matmul_family": plan.matmul_family,
        "peak_phase": "forward",
        "planned_peak_phase": "forward",
        "allocated_bytes": plan.allocated_bytes,
        "planned_peak_bytes": plan.allocated_bytes,
        "cap_bytes": plan.effective_limit_bytes,
        "requested_limit_bytes": plan.requested_limit_bytes,
        "hard_cap_bytes": plan.hard_cap_bytes,
        "phases": [phase],
    }
    return {
        "limit_bytes": plan.requested_limit_bytes,
        "cap_bytes": plan.effective_limit_bytes,
        "requested_variant": variant,
        "selected_variant": variant,
        "selection_source": "benchmark_cli",
        "selection_error": None,
        "semi_packed_plan": None,
        "semi_packed_candidate": None,
        "workspace_plan": workspace_plan,
        "actual_bytes": plan.allocated_bytes,
        "actual_derivation": "exact-edge descriptor plan",
        "fused_grad_weight_partials": None,
    }


def production_exact_edge_workspace_report(
    plan: ExactEdgeWorkspacePlan,
    x: Tensor,
    weight: Tensor,
) -> dict[str, object]:
    """Return the exact report with production dispatch labeling."""

    report = exact_edge_workspace_report(plan, x, weight)
    report["requested_variant"] = "exact_edge"
    report["selected_variant"] = "exact_edge"
    report["selection_source"] = "production_policy"
    workspace_plan = report["workspace_plan"]
    assert isinstance(workspace_plan, dict)
    workspace_plan["variant"] = "exact_edge"
    phases = workspace_plan["phases"]
    assert isinstance(phases, list)
    if phases:
        phases[0]["variant"] = "exact_edge"
    report["actual_derivation"] = (
        "production exact-edge fixed-shape descriptor plan"
    )
    return report


def exact_edge_support_reason(
    center_ptr: Tensor | Sequence[int],
    x: Tensor,
    weight: Tensor,
    workspace_bytes: int,
) -> str | None:
    """Return why the fixed production exact-edge inference route cannot run."""

    if not EXACT_EDGE_AVAILABLE:
        return "Triton exact-edge backend is unavailable"
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        return "exact-edge backend currently supports FP32 tensors only"
    if int(weight.shape[0]) != 1:
        return "exact-edge backend only supports one radial basis (R1)"
    plan = exact_edge_workspace_plan(
        center_ptr,
        x,
        weight,
        workspace_bytes,
        weight_layout=PRODUCTION_EXACT_EDGE_WEIGHT_LAYOUT,
        matmul_family=PRODUCTION_EXACT_EDGE_MATMUL_FAMILY,
    )
    if plan is None:
        return (
            "triton_workspace_mib is too small for one bounded exact-edge "
            "center chunk under the 128 MiB hard cap"
        )
    return None


def should_use_exact_edge(
    center_ptr: Tensor | Sequence[int],
    x: Tensor,
    weight: Tensor,
    workspace_bytes: int,
    *,
    training: bool,
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Conservative train-derived production inference policy.

    Automatic selection is intentionally limited to the calibrated CC 8.9
    architecture family. Shape selection uses only the generic larger-matrix-
    width gate derived from training proxies; it contains no held-out
    signatures.
    """

    if (
        training
        or not EXACT_EDGE_AVAILABLE
        or x.dtype != torch.float32
        or weight.dtype != torch.float32
        or int(weight.shape[0]) != 1
    ):
        return False
    plan = exact_edge_workspace_plan(
        center_ptr,
        x,
        weight,
        workspace_bytes,
        weight_layout=PRODUCTION_EXACT_EDGE_WEIGHT_LAYOUT,
        matmul_family=PRODUCTION_EXACT_EDGE_MATMUL_FAMILY,
    )
    return should_use_exact_edge_plan(
        x,
        weight,
        plan,
        training=training,
        compute_capability=compute_capability,
    )


def should_use_exact_edge_plan(
    x: Tensor,
    weight: Tensor,
    plan: ExactEdgeWorkspacePlan | None,
    *,
    training: bool,
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Apply the auto gate to an already prepared immutable descriptor plan."""

    if (
        training
        or not EXACT_EDGE_AVAILABLE
        or plan is None
        or x.dtype != torch.float32
        or weight.dtype != torch.float32
        or int(weight.shape[0]) != 1
    ):
        return False
    _radial, out_m, out_dim, in_m, in_dim = map(int, weight.shape)
    if max(in_m * in_dim, out_m * out_dim) < EXACT_EDGE_AUTO_MIN_MATRIX_DIM:
        return False
    capability = (
        torch.cuda.get_device_capability(x.device)
        if compute_capability is None
        else compute_capability
    )
    return tuple(map(int, capability)) == (8, 9)


if EXACT_EDGE_AVAILABLE:

    @triton.jit(do_not_specialize=["EDGE_START", "EDGE_COUNT"])
    def _gather_exact_edge_r1_kernel(
        x,
        neighbor_idx,
        radial_basis,
        input_cos,
        input_sin,
        input_pack,
        workspace,
        stride_xb: tl.constexpr,
        stride_xn: tl.constexpr,
        stride_xc: tl.constexpr,
        EDGE_START,
        EDGE_COUNT,
        EDGE_CAPACITY: tl.constexpr,
        IN_M: tl.constexpr,
        IN_DIM: tl.constexpr,
        MAX_IN_ORDER: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // EDGE_CAPACITY
        local_edge = row - batch * EDGE_CAPACITY
        edge_mask = local_edge < EDGE_COUNT
        edge = EDGE_START + local_edge
        neighbor = tl.load(neighbor_idx + edge, mask=edge_mask, other=0)

        packed = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        in_total: tl.constexpr = IN_M * IN_DIM
        packed_mask = packed < in_total
        in_channel = packed // IN_DIM
        component = packed - in_channel * IN_DIM
        scalar_mask = packed_mask & (component == 0)
        external = tl.load(input_pack + packed, mask=scalar_mask, other=0)
        direct = tl.load(
            x + batch * stride_xb + neighbor * stride_xn + external * stride_xc,
            mask=edge_mask & scalar_mask,
            other=0.0,
        )

        vector_mask = packed_mask & (component > 0)
        first_component = tl.where((component & 1) == 1, component, component - 1)
        first_packed = in_channel * IN_DIM + first_component
        external0 = tl.load(input_pack + first_packed, mask=vector_mask, other=0)
        external1 = tl.load(input_pack + first_packed + 1, mask=vector_mask, other=0)
        x0 = tl.load(
            x + batch * stride_xb + neighbor * stride_xn + external0 * stride_xc,
            mask=edge_mask & vector_mask,
            other=0.0,
        )
        x1 = tl.load(
            x + batch * stride_xb + neighbor * stride_xn + external1 * stride_xc,
            mask=edge_mask & vector_mask,
            other=0.0,
        )
        order_index = (first_component - 1) // 2
        c = tl.load(
            input_cos + edge * MAX_IN_ORDER + order_index,
            mask=edge_mask & vector_mask,
            other=1.0,
        )
        s = tl.load(
            input_sin + edge * MAX_IN_ORDER + order_index,
            mask=edge_mask & vector_mask,
            other=0.0,
        )
        first = c * x0 - s * x1
        second = s * x0 + c * x1
        value = tl.where(
            component == 0,
            direct,
            tl.where((component & 1) == 1, first, second),
        )
        radial = tl.load(radial_basis + edge, mask=edge_mask, other=0.0)
        gathered = tl.where(edge_mask, value * radial, 0.0)
        # Every inactive row is explicitly initialized because the fixed-row
        # GEMM below consumes the full capacity for every descriptor.
        tl.store(workspace + row * in_total + packed, gathered, mask=packed_mask)


    @triton.jit
    def _wide_transposed_matmul_r1_kernel(
        a,
        transposed_weight,
        c,
        ROWS: tl.constexpr,
        IN_M: tl.constexpr,
        OUT_M: tl.constexpr,
        IN_DIM: tl.constexpr,
        OUT_DIM: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        input_total: tl.constexpr = IN_M * IN_DIM
        output_total: tl.constexpr = OUT_M * OUT_DIM
        offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        k_base = 0
        while k_base < input_total:
            offs_k = k_base + tl.arange(0, BLOCK_K)
            a_values = tl.load(
                a + offs_m[:, None] * input_total + offs_k[None, :],
                mask=(offs_m[:, None] < ROWS) & (offs_k[None, :] < input_total),
                other=0.0,
            )
            weight_values = tl.load(
                transposed_weight
                + offs_k[:, None] * output_total
                + offs_n[None, :],
                mask=(offs_k[:, None] < input_total)
                & (offs_n[None, :] < output_total),
                other=0.0,
            )
            if ALLOW_TF32:
                acc += tl.dot(a_values, weight_values, input_precision="tf32")
            else:
                acc += tl.dot(a_values, weight_values, input_precision="ieee")
            k_base += BLOCK_K
        tl.store(
            c + offs_m[:, None] * output_total + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < ROWS) & (offs_n[None, :] < output_total),
        )


    @triton.jit
    def _direct_weight_matmul_r1_kernel(
        a,
        weight,
        c,
        ROWS: tl.constexpr,
        IN_M: tl.constexpr,
        OUT_M: tl.constexpr,
        IN_DIM: tl.constexpr,
        OUT_DIM: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        input_total: tl.constexpr = IN_M * IN_DIM
        output_total: tl.constexpr = OUT_M * OUT_DIM
        offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        k_base = 0
        while k_base < input_total:
            offs_k = k_base + tl.arange(0, BLOCK_K)
            a_values = tl.load(
                a + offs_m[:, None] * input_total + offs_k[None, :],
                mask=(offs_m[:, None] < ROWS) & (offs_k[None, :] < input_total),
                other=0.0,
            )
            # The original weight is contiguous [OUT, IN].  Loads therefore
            # run along K, then tl.trans performs only a register-tile reorder
            # to produce the [K, N] right operand required by tl.dot.
            weight_nk = tl.load(
                weight + offs_n[:, None] * input_total + offs_k[None, :],
                mask=(offs_n[:, None] < output_total)
                & (offs_k[None, :] < input_total),
                other=0.0,
            )
            weight_kn = tl.trans(weight_nk)
            if ALLOW_TF32:
                acc += tl.dot(a_values, weight_kn, input_precision="tf32")
            else:
                acc += tl.dot(a_values, weight_kn, input_precision="ieee")
            k_base += BLOCK_K
        tl.store(
            c + offs_m[:, None] * output_total + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < ROWS) & (offs_n[None, :] < output_total),
        )


    _EXACT_MATMUL_KEY = [
        "ROWS", "IN_M", "OUT_M", "IN_DIM", "OUT_DIM", "ALLOW_TF32",
    ]


    def _sm8_triton_configs(configs: Sequence[ExactEdgeMatmulConfig]) -> list:
        return [
            triton.Config(
                {
                    "BLOCK_M": config.block_m,
                    "BLOCK_N": config.block_n,
                    "BLOCK_K": config.block_k,
                },
                num_warps=config.num_warps,
                num_stages=config.num_stages,
            )
            for config in configs
        ]


    _direct_weight_matmul_kernels = {
        architecture: triton.autotune(
            configs=_semi._matmul_configs(architecture),
            key=_EXACT_MATMUL_KEY,
        )(_direct_weight_matmul_r1_kernel)
        for architecture in (7, 8, 9)
    }
    _wide_sm8_transposed_matmul_kernel = triton.autotune(
        configs=_sm8_triton_configs(EXACT_EDGE_WIDE_SM8_CONFIGS),
        key=_EXACT_MATMUL_KEY,
    )(_wide_transposed_matmul_r1_kernel)
    _wide_sm8_direct_weight_matmul_kernel = triton.autotune(
        configs=_sm8_triton_configs(EXACT_EDGE_WIDE_SM8_CONFIGS),
        key=_EXACT_MATMUL_KEY,
    )(_direct_weight_matmul_r1_kernel)
    _dense_sm8_transposed_matmul_kernel = triton.autotune(
        configs=_sm8_triton_configs(EXACT_EDGE_DENSE_SM8_CONFIGS),
        key=_EXACT_MATMUL_KEY,
    )(_wide_transposed_matmul_r1_kernel)
    _dense_sm8_direct_weight_matmul_kernel = triton.autotune(
        configs=_sm8_triton_configs(EXACT_EDGE_DENSE_SM8_CONFIGS),
        key=_EXACT_MATMUL_KEY,
    )(_direct_weight_matmul_r1_kernel)


    def _matmul_grid(rows: int, output_total: int):
        return lambda meta: (
            triton.cdiv(rows, meta["BLOCK_M"]),
            triton.cdiv(output_total, meta["BLOCK_N"]),
        )


    def _launch_wide_transposed_matmul(
        a: Tensor,
        transposed_weight: Tensor,
        c: Tensor,
        *,
        rows: int,
        in_m: int,
        out_m: int,
        in_dim: int,
        out_dim: int,
        allow_tf32: bool,
        dense_sm8: bool,
    ) -> None:
        if _semi._architecture(a) != 8:
            _semi._launch_matmul(
                a,
                transposed_weight,
                c,
                rows=rows,
                in_m=in_m,
                out_m=out_m,
                in_dim=in_dim,
                out_dim=out_dim,
                transpose_weight=False,
                allow_tf32=allow_tf32,
            )
            return
        output_total = out_m * out_dim
        kernel = (
            _dense_sm8_transposed_matmul_kernel
            if dense_sm8
            else _wide_sm8_transposed_matmul_kernel
        )
        wrap_triton(kernel)[
            _matmul_grid(rows, output_total)
        ](
            a,
            transposed_weight,
            c,
            ROWS=rows,
            IN_M=in_m,
            OUT_M=out_m,
            IN_DIM=in_dim,
            OUT_DIM=out_dim,
            ALLOW_TF32=allow_tf32,
        )


    def _launch_direct_weight_matmul(
        a: Tensor,
        weight: Tensor,
        c: Tensor,
        *,
        rows: int,
        in_m: int,
        out_m: int,
        in_dim: int,
        out_dim: int,
        allow_tf32: bool,
        wide_sm8: bool,
        dense_sm8: bool,
    ) -> None:
        architecture = _semi._architecture(a)
        kernel = (
            _dense_sm8_direct_weight_matmul_kernel
            if dense_sm8 and architecture == 8
            else _wide_sm8_direct_weight_matmul_kernel
            if wide_sm8 and architecture == 8
            else _direct_weight_matmul_kernels[architecture]
        )
        output_total = out_m * out_dim
        wrap_triton(kernel)[_matmul_grid(rows, output_total)](
            a,
            weight,
            c,
            ROWS=rows,
            IN_M=in_m,
            OUT_M=out_m,
            IN_DIM=in_dim,
            OUT_DIM=out_dim,
            ALLOW_TF32=allow_tf32,
        )


    @triton.jit(
        do_not_specialize=["POINT_START", "POINT_COUNT", "EDGE_START", "EDGE_COUNT"]
    )
    def _reduce_exact_edge_r1_kernel(
        workspace,
        center_ptr,
        output_cos,
        output_sin,
        output_pack,
        neighbor_count,
        out,
        POINT_START,
        POINT_COUNT,
        EDGE_START,
        EDGE_COUNT,
        N_POINTS: tl.constexpr,
        POINT_CAPACITY: tl.constexpr,
        EDGE_CAPACITY: tl.constexpr,
        MAX_CENTER_DEGREE: tl.constexpr,
        OUT_M: tl.constexpr,
        OUT_DIM: tl.constexpr,
        MAX_OUT_ORDER: tl.constexpr,
        NORMALIZE: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        channel_blocks: tl.constexpr = tl.cdiv(OUT_M, BLOCK_C)
        pid = tl.program_id(0)
        out_order = pid % (MAX_OUT_ORDER + 1)
        pid //= MAX_OUT_ORDER + 1
        channel_block = pid % channel_blocks
        row = pid // channel_blocks
        batch = row // POINT_CAPACITY
        local_point = row - batch * POINT_CAPACITY
        point = POINT_START + local_point
        point_mask = (local_point < POINT_COUNT) & (point < N_POINTS)
        channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
        channel_mask = channels < OUT_M
        component0 = tl.where(out_order == 0, 0, 2 * out_order - 1)
        component1 = component0 + 1
        total0 = tl.zeros((BLOCK_C,), tl.float32)
        total1 = tl.zeros((BLOCK_C,), tl.float32)
        edge_start = tl.load(center_ptr + point, mask=point_mask, other=0)
        edge_stop = tl.load(center_ptr + point + 1, mask=point_mask, other=0)
        edge_base = 0
        out_total: tl.constexpr = OUT_M * OUT_DIM
        while edge_base < MAX_CENTER_DEGREE:
            degrees = edge_base + tl.arange(0, BLOCK_E)
            edge = edge_start + degrees
            local_edge = edge - EDGE_START
            edge_mask = (
                point_mask
                & (edge < edge_stop)
                & (local_edge >= 0)
                & (local_edge < EDGE_COUNT)
            )
            workspace_row = (batch * EDGE_CAPACITY + local_edge) * out_total
            value0 = tl.load(
                workspace + workspace_row[:, None] + channels[None, :] * OUT_DIM + component0,
                mask=edge_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            if out_order == 0:
                total0 += tl.sum(value0, axis=0)
            else:
                value1 = tl.load(
                    workspace + workspace_row[:, None] + channels[None, :] * OUT_DIM + component1,
                    mask=edge_mask[:, None] & channel_mask[None, :],
                    other=0.0,
                )
                c = tl.load(
                    output_cos + edge * MAX_OUT_ORDER + out_order - 1,
                    mask=edge_mask,
                    other=1.0,
                )
                s = tl.load(
                    output_sin + edge * MAX_OUT_ORDER + out_order - 1,
                    mask=edge_mask,
                    other=0.0,
                )
                total0 += tl.sum(value0 * c[:, None] - value1 * s[:, None], axis=0)
                total1 += tl.sum(value0 * s[:, None] + value1 * c[:, None], axis=0)
            edge_base += BLOCK_E
        if NORMALIZE:
            denom = tl.maximum(
                tl.load(neighbor_count + point, mask=point_mask, other=1.0),
                1.0,
            )
            total0 /= denom
            total1 /= denom
        packed0 = channels * OUT_DIM + component0
        external0 = tl.load(output_pack + packed0, mask=channel_mask, other=0)
        out_base = (batch * N_POINTS + point) * out_total
        tl.store(out + out_base + external0, total0, mask=point_mask & channel_mask)
        if out_order > 0:
            external1 = tl.load(output_pack + packed0 + 1, mask=channel_mask, other=0)
            tl.store(out + out_base + external1, total1, mask=point_mask & channel_mask)


    @triton_op("kpconv_intrinsic::exact_edge_irrep_conv", mutates_args={})
    def exact_edge_irrep_conv(
        x: Tensor,
        weight: Tensor,
        neighbor_idx: Tensor,
        center_ptr: Tensor,
        radial_basis: Tensor,
        input_cos: Tensor,
        input_sin: Tensor,
        output_cos: Tensor,
        output_sin: Tensor,
        input_pack: Tensor,
        output_pack: Tensor,
        neighbor_count: Tensor,
        point_starts: List[int],
        point_counts: List[int],
        edge_starts: List[int],
        edge_counts: List[int],
        edge_capacity: int,
        point_capacity: int,
        max_center_degree: int,
        direct_weight: bool,
        wide_sm8_matmul: bool,
        dense_sm8_matmul: bool,
        normalize: bool,
        allow_tf32: bool,
        workspace_bytes: int,
    ) -> Tensor:
        x = fix_triton_strides(x)
        weight = fix_triton_strides(weight)
        radial, out_m, out_dim, in_m, in_dim = map(int, weight.shape)
        if radial != 1:
            raise RuntimeError("exact-edge streaming only supports regular R1")
        if not (
            len(point_starts)
            == len(point_counts)
            == len(edge_starts)
            == len(edge_counts)
        ):
            raise RuntimeError("exact-edge descriptor lists must have equal lengths")
        batch, n_points = map(int, x.shape[:2])
        in_total = in_m * in_dim
        out_total = out_m * out_dim
        if int(edge_capacity) == 0:
            return torch.zeros(
                (batch, n_points, out_total), device=x.device, dtype=x.dtype,
            )
        fixed_bytes = (
            0
            if direct_weight
            else in_total * out_total * int(torch.float32.itemsize)
        )
        matrix_bytes = (
            batch
            * int(edge_capacity)
            * (in_total + out_total)
            * int(x.element_size())
        )
        effective_limit = min(int(workspace_bytes), EXACT_EDGE_HARD_CAP_BYTES)
        if fixed_bytes + matrix_bytes > effective_limit:
            raise RuntimeError("exact-edge workspaces exceed the effective 128 MiB cap")

        input_workspace = torch.empty(
            (batch * edge_capacity, in_total), device=x.device, dtype=x.dtype,
        )
        output_workspace = torch.empty(
            (batch * edge_capacity, out_total), device=x.device, dtype=x.dtype,
        )
        out = torch.empty(
            (batch, n_points, out_total), device=x.device, dtype=x.dtype,
        )
        if not direct_weight:
            weight_workspace = torch.empty(
                (out_total, in_total), device=x.device, dtype=torch.float32,
            )
            _semi._launch_transpose_weight(weight, weight_workspace)

        for point_start, point_count, edge_start, edge_count in zip(
            point_starts, point_counts, edge_starts, edge_counts,
        ):
            rows = batch * edge_capacity
            grid = (rows, triton.cdiv(in_total, 64))
            wrap_triton(_gather_exact_edge_r1_kernel)[grid](
                x,
                neighbor_idx,
                radial_basis,
                input_cos,
                input_sin,
                input_pack,
                input_workspace,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                EDGE_START=edge_start,
                EDGE_COUNT=edge_count,
                EDGE_CAPACITY=edge_capacity,
                IN_M=in_m,
                IN_DIM=in_dim,
                MAX_IN_ORDER=(in_dim - 1) // 2,
                BLOCK_K=64,
                num_warps=4,
            )
            if direct_weight:
                _launch_direct_weight_matmul(
                    input_workspace,
                    weight,
                    output_workspace,
                    rows=rows,
                    in_m=in_m,
                    out_m=out_m,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    allow_tf32=allow_tf32,
                    wide_sm8=wide_sm8_matmul,
                    dense_sm8=dense_sm8_matmul,
                )
            elif wide_sm8_matmul or dense_sm8_matmul:
                _launch_wide_transposed_matmul(
                    input_workspace,
                    weight_workspace,
                    output_workspace,
                    rows=rows,
                    in_m=in_m,
                    out_m=out_m,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    allow_tf32=allow_tf32,
                    dense_sm8=dense_sm8_matmul,
                )
            else:
                _semi._launch_matmul(
                    input_workspace,
                    weight_workspace,
                    output_workspace,
                    rows=rows,
                    in_m=in_m,
                    out_m=out_m,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    transpose_weight=False,
                    allow_tf32=allow_tf32,
                )
            grid = (
                batch
                * point_capacity
                * triton.cdiv(out_m, 8)
                * ((out_dim - 1) // 2 + 1),
            )
            wrap_triton(_reduce_exact_edge_r1_kernel)[grid](
                output_workspace,
                center_ptr,
                output_cos,
                output_sin,
                output_pack,
                neighbor_count,
                out,
                POINT_START=point_start,
                POINT_COUNT=point_count,
                EDGE_START=edge_start,
                EDGE_COUNT=edge_count,
                N_POINTS=n_points,
                POINT_CAPACITY=point_capacity,
                EDGE_CAPACITY=edge_capacity,
                MAX_CENTER_DEGREE=max_center_degree,
                OUT_M=out_m,
                OUT_DIM=out_dim,
                MAX_OUT_ORDER=(out_dim - 1) // 2,
                NORMALIZE=normalize,
                BLOCK_C=8,
                BLOCK_E=16,
                num_warps=4,
            )
        return out

else:

    def exact_edge_irrep_conv(*_args, **_kwargs) -> Tensor:
        raise RuntimeError("the exact-edge Triton forward is unavailable")


# Backward-compatible benchmark name.  Production code calls the non-
# experimental spelling above.
experimental_exact_edge_irrep_conv = exact_edge_irrep_conv


def plan_argument_lists(
    plan: ExactEdgeWorkspacePlan,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Convert immutable descriptors to custom-op list arguments."""

    return (
        [chunk.point_start for chunk in plan.chunks],
        [chunk.point_count for chunk in plan.chunks],
        [chunk.edge_start for chunk in plan.chunks],
        [chunk.edge_count for chunk in plan.chunks],
    )


__all__ = [
    "EXACT_EDGE_AUTO_MIN_MATRIX_DIM",
    "EXACT_EDGE_AVAILABLE",
    "EXACT_EDGE_DENSE_SM8_ADDITIONAL_CONFIGS",
    "EXACT_EDGE_DENSE_SM8_CONFIGS",
    "EXACT_EDGE_HARD_CAP_BYTES",
    "EXACT_EDGE_MATMUL_FAMILIES",
    "EXACT_EDGE_PRODUCTION_SM8_CONFIGS",
    "EXACT_EDGE_WEIGHT_LAYOUTS",
    "EXACT_EDGE_WIDE_SM8_ADDITIONAL_CONFIGS",
    "EXACT_EDGE_WIDE_SM8_CONFIGS",
    "EXPERIMENTAL_EXACT_EDGE_AVAILABLE",
    "PRODUCTION_EXACT_EDGE_MATMUL_FAMILY",
    "PRODUCTION_EXACT_EDGE_WEIGHT_LAYOUT",
    "ExactEdgeChunk",
    "ExactEdgeMatmulConfig",
    "ExactEdgeWorkspacePlan",
    "exact_edge_irrep_conv",
    "exact_edge_support_reason",
    "exact_edge_workspace_plan",
    "exact_edge_workspace_report",
    "experimental_exact_edge_irrep_conv",
    "plan_argument_lists",
    "production_exact_edge_workspace_report",
    "should_use_exact_edge",
    "should_use_exact_edge_plan",
]
