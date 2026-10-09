"""Bounded-workspace Triton backend for regular packed R1 convolutions.

The implementation deliberately keeps the gathered edge matrices private to a
small point chunk.  It therefore gets the matrix-shaped work needed by
``tl.dot`` without materialising the full ``[batch, edges, channels]`` tensors
used by the Torch packed reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._triton_layout import fix_triton_strides

try:
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton
except (ImportError, AttributeError):  # pragma: no cover - CPU-only installs
    triton = None
    tl = None
    triton_op = None
    wrap_triton = None


TRITON_SEMI_PACKED_AVAILABLE = triton is not None and triton_op is not None
MIN_AUTO_EDGE_CAPACITY = 4096
MIN_AUTO_MATRIX_DIM = 64
SPLIT_K_SPLITS = 32
SPLIT_K_CARRY_LEVELS = 32
BACKWARD_WORKSPACE_HEADROOM_BYTES = 4 << 20
@dataclass(frozen=True)
class SemiPackedWorkspacePlan:
    chunk_points: int
    padded_degree: int
    edge_capacity: int
    allocated_bytes: int
    limit_bytes: int


@dataclass(frozen=True)
class SemiPackedBackwardPhasePlan:
    chunk_points: int
    padded_degree: int
    edge_capacity: int
    matrix_bytes: int
    partial_bytes: int
    scratch_bytes: int
    carry_bytes: int
    allocated_bytes: int
    headroom_bytes: int
    limit_bytes: int
    split_k: bool
    splits: int


@dataclass(frozen=True)
class SemiPackedBackwardWorkspacePlan:
    grad_input: SemiPackedBackwardPhasePlan
    grad_weight: SemiPackedBackwardPhasePlan
    use_split_k: bool
    allocated_bytes: int
    planned_peak_bytes: int
    headroom_bytes: int
    limit_bytes: int


def semi_packed_workspace_plan(
    x: Tensor,
    weight: Tensor,
    padded_degree: int,
    workspace_bytes: int,
) -> SemiPackedWorkspacePlan | None:
    """Plan two matrices plus a weight tile for one exact padded row stride."""

    if workspace_bytes <= 0 or padded_degree <= 0:
        return None
    _radial, out_m, out_dim, in_m, in_dim = map(int, weight.shape)
    batch, n_points = map(int, x.shape[:2])
    in_total = in_m * in_dim
    out_total = out_m * out_dim
    element_size = int(x.element_size())
    # X/G and Y/Z are live together.  Grad-weight additionally needs one FP32
    # output tile covering the complete packed weight matrix.
    fixed_bytes = out_total * in_total * torch.float32.itemsize
    padded_degree = int(padded_degree)
    bytes_per_point = batch * padded_degree * (in_total + out_total) * element_size
    available = int(workspace_bytes) - fixed_bytes
    if available < bytes_per_point:
        return None
    max_chunk_points = min(n_points, available // bytes_per_point)
    if max_chunk_points <= 0:
        return None
    # Preserve the minimum number of launches allowed by the cap, then spread
    # points evenly across them.  Taking the largest aligned chunk leaves a
    # potentially tiny final remainder, while every kernel still launches the
    # full chunk shape.  Balanced chunks keep that masked tail below one point
    # per launch and also avoid manufacturing a second near-full launch when
    # all points fit but ``n_points`` is not a multiple of 32.
    num_chunks = (n_points + max_chunk_points - 1) // max_chunk_points
    chunk_points = (n_points + num_chunks - 1) // num_chunks
    allocated = chunk_points * bytes_per_point + fixed_bytes
    if allocated > int(workspace_bytes):
        return None
    return SemiPackedWorkspacePlan(
        chunk_points=chunk_points,
        padded_degree=padded_degree,
        edge_capacity=chunk_points * padded_degree,
        allocated_bytes=allocated,
        limit_bytes=int(workspace_bytes),
    )


def _backward_phase_workspace_plan(
    x: Tensor,
    weight: Tensor,
    padded_degree: int,
    workspace_bytes: int,
    *,
    partial_bytes: int,
    scratch_bytes: int = 0,
    carry_bytes: int = 0,
    split_k: bool = False,
    splits: int = 1,
    headroom_bytes: int = BACKWARD_WORKSPACE_HEADROOM_BYTES,
) -> SemiPackedBackwardPhasePlan | None:
    """Plan one backward phase while retaining allocator safety headroom."""

    padded_degree = int(padded_degree)
    workspace_bytes = int(workspace_bytes)
    headroom_bytes = int(headroom_bytes)
    fixed_bytes = int(partial_bytes) + int(scratch_bytes) + int(carry_bytes)
    if (
        workspace_bytes <= headroom_bytes
        or padded_degree <= 0
        or fixed_bytes < 0
        or headroom_bytes < 0
    ):
        return None
    _radial, out_m, out_dim, in_m, in_dim = map(int, weight.shape)
    batch, n_points = map(int, x.shape[:2])
    in_total = in_m * in_dim
    out_total = out_m * out_dim
    bytes_per_point = (
        batch
        * padded_degree
        * (in_total + out_total)
        * int(x.element_size())
    )
    available = workspace_bytes - headroom_bytes - fixed_bytes
    if n_points <= 0 or available < bytes_per_point:
        return None
    max_chunk_points = min(n_points, available // bytes_per_point)
    num_chunks = (n_points + max_chunk_points - 1) // max_chunk_points
    chunk_points = (n_points + num_chunks - 1) // num_chunks
    matrix_bytes = chunk_points * bytes_per_point
    allocated_bytes = matrix_bytes + fixed_bytes
    if (
        chunk_points <= 0
        or allocated_bytes + headroom_bytes > workspace_bytes
    ):
        return None
    return SemiPackedBackwardPhasePlan(
        chunk_points=chunk_points,
        padded_degree=padded_degree,
        edge_capacity=chunk_points * padded_degree,
        matrix_bytes=matrix_bytes,
        partial_bytes=int(partial_bytes),
        scratch_bytes=int(scratch_bytes),
        carry_bytes=int(carry_bytes),
        allocated_bytes=allocated_bytes,
        headroom_bytes=headroom_bytes,
        limit_bytes=workspace_bytes,
        split_k=bool(split_k),
        splits=int(splits),
    )


def semi_packed_backward_workspace_plan(
    x: Tensor,
    weight: Tensor,
    max_center_degree: int,
    max_neighbor_degree: int,
    workspace_bytes: int,
    *,
    prefer_split_k: bool,
) -> SemiPackedBackwardWorkspacePlan | None:
    """Plan disjoint dX and dW phases under one hard workspace limit."""

    _radial, out_m, out_dim, in_m, in_dim = map(int, weight.shape)
    weight_bytes = (
        out_m * out_dim * in_m * in_dim * torch.float32.itemsize
    )
    grad_input = None
    grad_weight = None
    if prefer_split_k:
        grad_input = _backward_phase_workspace_plan(
            x,
            weight,
            max_neighbor_degree,
            workspace_bytes,
            partial_bytes=0,
        )
        grad_weight = _backward_phase_workspace_plan(
            x,
            weight,
            max_center_degree,
            workspace_bytes,
            partial_bytes=SPLIT_K_SPLITS * weight_bytes,
            scratch_bytes=((SPLIT_K_SPLITS + 1) // 2) * weight_bytes,
            carry_bytes=SPLIT_K_CARRY_LEVELS * weight_bytes,
            split_k=True,
            splits=SPLIT_K_SPLITS,
        )
        if grad_input is not None and grad_weight is not None:
            chunks = (
                int(x.shape[1]) + grad_weight.chunk_points - 1
            ) // grad_weight.chunk_points
            if chunks >= 1 << SPLIT_K_CARRY_LEVELS:
                grad_weight = None
        if grad_input is not None and grad_weight is not None:
            allocated_bytes = max(
                grad_input.allocated_bytes, grad_weight.allocated_bytes,
            )
            planned_peak_bytes = (
                allocated_bytes + BACKWARD_WORKSPACE_HEADROOM_BYTES
            )
            return SemiPackedBackwardWorkspacePlan(
                grad_input=grad_input,
                grad_weight=grad_weight,
                use_split_k=True,
                allocated_bytes=allocated_bytes,
                planned_peak_bytes=planned_peak_bytes,
                headroom_bytes=BACKWARD_WORKSPACE_HEADROOM_BYTES,
                limit_bytes=int(workspace_bytes),
            )

    # Preserve the pre-existing deterministic dW implementation whenever the
    # fixed split buffers cannot fit.  Normal production caps retain the same
    # 4 MiB reserve; only legacy sub-4-MiB forced-semi calls fall back to their
    # historical zero-headroom behavior.
    for headroom_bytes in (BACKWARD_WORKSPACE_HEADROOM_BYTES, 0):
        if headroom_bytes and int(workspace_bytes) <= headroom_bytes:
            continue
        grad_input = _backward_phase_workspace_plan(
            x,
            weight,
            max_neighbor_degree,
            workspace_bytes,
            partial_bytes=0,
            headroom_bytes=headroom_bytes,
        )
        grad_weight = _backward_phase_workspace_plan(
            x,
            weight,
            max_center_degree,
            workspace_bytes,
            partial_bytes=weight_bytes,
            headroom_bytes=headroom_bytes,
        )
        if grad_input is None or grad_weight is None:
            continue
        allocated_bytes = max(
            grad_input.allocated_bytes, grad_weight.allocated_bytes,
        )
        planned_peak_bytes = allocated_bytes + headroom_bytes
        if planned_peak_bytes > int(workspace_bytes):
            continue
        return SemiPackedBackwardWorkspacePlan(
            grad_input=grad_input,
            grad_weight=grad_weight,
            use_split_k=False,
            allocated_bytes=allocated_bytes,
            planned_peak_bytes=planned_peak_bytes,
            headroom_bytes=headroom_bytes,
            limit_bytes=int(workspace_bytes),
        )
    return None


def _online_binary_carry_schedule(
    num_chunks: int,
    carry_levels: int = SPLIT_K_CARRY_LEVELS,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return insertion levels and final occupied levels for an online tree."""

    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if carry_levels <= 0 or num_chunks >= 1 << carry_levels:
        raise ValueError("num_chunks exceeds the fixed binary-carry capacity")
    occupied = 0
    insertion_levels: list[int] = []
    for _ in range(num_chunks):
        level = 0
        while occupied & (1 << level):
            occupied &= ~(1 << level)
            level += 1
        occupied |= 1 << level
        insertion_levels.append(level)
    final_levels = tuple(
        level for level in range(carry_levels) if occupied & (1 << level)
    )
    return tuple(insertion_levels), final_levels


def semi_packed_support_reason(
    x: Tensor,
    weight: Tensor,
    max_center_degree: int,
    workspace_bytes: int,
) -> str | None:
    if not TRITON_SEMI_PACKED_AVAILABLE:
        return "Triton semi-packed backend is unavailable"
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        return "semi-packed backend currently supports FP32 tensors only"
    if int(weight.shape[0]) != 1:
        return "semi-packed backend only supports one radial basis (R1)"
    plan = semi_packed_workspace_plan(x, weight, max_center_degree, workspace_bytes)
    if plan is None:
        return (
            "triton_workspace_mib is too small for one bounded semi-packed "
            "point chunk"
        )
    return None


def should_use_semi_packed(
    x: Tensor,
    weight: Tensor,
    max_center_degree: int,
    degree_bucket: int,
    workspace_bytes: int,
    *,
    training: bool,
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Return the conservative automatic policy for the legacy semi route.

    No measured held-out shape signatures are embedded in production.  The
    implementation remains available when explicitly selected, while auto
    inference is handled by the bounded exact-edge route.
    """

    exact_plan = semi_packed_workspace_plan(
        x, weight, max_center_degree, workspace_bytes,
    )
    classification_plan = semi_packed_workspace_plan(
        x, weight, degree_bucket, workspace_bytes,
    )
    if (
        exact_plan is None
        or classification_plan is None
        or classification_plan.edge_capacity < MIN_AUTO_EDGE_CAPACITY
    ):
        return False
    # The current recompute-based backward has no calibrated wins yet.
    if training:
        return False
    _radial, out_m, out_dim, in_m, in_dim = map(int, weight.shape)
    in_total = in_m * in_dim
    out_total = out_m * out_dim
    samples = int(x.shape[0]) * int(x.shape[1]) * int(degree_bucket)
    if min(in_total, out_total) < MIN_AUTO_MATRIX_DIM or samples < MIN_AUTO_EDGE_CAPACITY:
        return False
    del compute_capability
    return False


if TRITON_SEMI_PACKED_AVAILABLE:

    @triton.jit(do_not_specialize=["POINT_START"])
    def _gather_input_r1_kernel(
        x,
        neighbor_idx,
        point_ptr,
        radial_basis,
        input_cos,
        input_sin,
        input_pack,
        workspace,
        stride_xb: tl.constexpr,
        stride_xn: tl.constexpr,
        stride_xc: tl.constexpr,
        POINT_START,
        N_POINTS: tl.constexpr,
        CHUNK_POINTS: tl.constexpr,
        PADDED_DEGREE: tl.constexpr,
        IN_M: tl.constexpr,
        IN_DIM: tl.constexpr,
        MAX_IN_ORDER: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // (CHUNK_POINTS * PADDED_DEGREE)
        point_edge = row - batch * CHUNK_POINTS * PADDED_DEGREE
        local_point = point_edge // PADDED_DEGREE
        degree_offset = point_edge - local_point * PADDED_DEGREE
        point = POINT_START + local_point
        point_mask = point < N_POINTS
        edge_start = tl.load(point_ptr + point, mask=point_mask, other=0)
        edge_stop = tl.load(point_ptr + point + 1, mask=point_mask, other=0)
        edge = edge_start + degree_offset
        edge_mask = point_mask & (edge < edge_stop)
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
        tl.store(
            workspace + row * in_total + packed,
            value * radial,
            mask=packed_mask,
        )


    @triton.jit(do_not_specialize=["POINT_START"])
    def _gather_grad_output_r1_kernel(
        grad_out,
        center_idx,
        point_ptr,
        edges_by_neighbor,
        output_cos,
        output_sin,
        output_pack,
        neighbor_count,
        workspace,
        POINT_START,
        N_POINTS: tl.constexpr,
        CHUNK_POINTS: tl.constexpr,
        PADDED_DEGREE: tl.constexpr,
        OUT_M: tl.constexpr,
        OUT_DIM: tl.constexpr,
        MAX_OUT_ORDER: tl.constexpr,
        BATCH: tl.constexpr,
        BY_NEIGHBOR: tl.constexpr,
        NORMALIZE: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // (CHUNK_POINTS * PADDED_DEGREE)
        point_edge = row - batch * CHUNK_POINTS * PADDED_DEGREE
        local_point = point_edge // PADDED_DEGREE
        degree_offset = point_edge - local_point * PADDED_DEGREE
        point = POINT_START + local_point
        point_mask = (batch < BATCH) & (point < N_POINTS)
        pos_start = tl.load(point_ptr + point, mask=point_mask, other=0)
        pos_stop = tl.load(point_ptr + point + 1, mask=point_mask, other=0)
        pos = pos_start + degree_offset
        edge_mask = point_mask & (pos < pos_stop)
        if BY_NEIGHBOR:
            edge = tl.load(edges_by_neighbor + pos, mask=edge_mask, other=0)
            center = tl.load(center_idx + edge, mask=edge_mask, other=0)
        else:
            edge = pos
            center = point

        packed = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        out_total: tl.constexpr = OUT_M * OUT_DIM
        packed_mask = packed < out_total
        out_channel = packed // OUT_DIM
        component = packed - out_channel * OUT_DIM
        scalar_mask = packed_mask & (component == 0)
        external = tl.load(output_pack + packed, mask=scalar_mask, other=0)
        direct = tl.load(
            grad_out + (batch * N_POINTS + center) * out_total + external,
            mask=edge_mask & scalar_mask,
            other=0.0,
        )

        vector_mask = packed_mask & (component > 0)
        first_component = tl.where((component & 1) == 1, component, component - 1)
        first_packed = out_channel * OUT_DIM + first_component
        external0 = tl.load(output_pack + first_packed, mask=vector_mask, other=0)
        external1 = tl.load(output_pack + first_packed + 1, mask=vector_mask, other=0)
        g0 = tl.load(
            grad_out + (batch * N_POINTS + center) * out_total + external0,
            mask=edge_mask & vector_mask,
            other=0.0,
        )
        g1 = tl.load(
            grad_out + (batch * N_POINTS + center) * out_total + external1,
            mask=edge_mask & vector_mask,
            other=0.0,
        )
        order_index = (first_component - 1) // 2
        c = tl.load(
            output_cos + edge * MAX_OUT_ORDER + order_index,
            mask=edge_mask & vector_mask,
            other=1.0,
        )
        s = tl.load(
            output_sin + edge * MAX_OUT_ORDER + order_index,
            mask=edge_mask & vector_mask,
            other=0.0,
        )
        first = c * g0 + s * g1
        second = -s * g0 + c * g1
        value = tl.where(
            component == 0,
            direct,
            tl.where((component & 1) == 1, first, second),
        )
        if NORMALIZE:
            denom = tl.maximum(tl.load(neighbor_count + center, mask=edge_mask, other=1.0), 1.0)
            value /= denom
        tl.store(workspace + row * out_total + packed, value, mask=packed_mask)


    @triton.jit
    def _semi_matmul_r1_kernel(
        a,
        weight_matrix,
        c,
        ROWS: tl.constexpr,
        IN_M: tl.constexpr,
        OUT_M: tl.constexpr,
        IN_DIM: tl.constexpr,
        OUT_DIM: tl.constexpr,
        TRANSPOSE_WEIGHT: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        input_total: tl.constexpr = OUT_M * OUT_DIM if TRANSPOSE_WEIGHT else IN_M * IN_DIM
        output_total: tl.constexpr = IN_M * IN_DIM if TRANSPOSE_WEIGHT else OUT_M * OUT_DIM
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
            w = tl.load(
                weight_matrix
                + offs_k[:, None] * output_total
                + offs_n[None, :],
                mask=(offs_k[:, None] < input_total) & (offs_n[None, :] < output_total),
                other=0.0,
            )
            if ALLOW_TF32:
                acc += tl.dot(a_values, w, input_precision="tf32")
            else:
                acc += tl.dot(a_values, w, input_precision="ieee")
            k_base += BLOCK_K
        tl.store(
            c + offs_m[:, None] * output_total + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < ROWS) & (offs_n[None, :] < output_total),
        )


    @triton.jit
    def _transpose_weight_r1_kernel(
        weight,
        transposed,
        IN_TOTAL: tl.constexpr,
        OUT_TOTAL: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_O: tl.constexpr,
    ):
        in_offsets = tl.program_id(0) * BLOCK_I + tl.arange(0, BLOCK_I)
        out_offsets = tl.program_id(1) * BLOCK_O + tl.arange(0, BLOCK_O)
        mask = (out_offsets[:, None] < OUT_TOTAL) & (in_offsets[None, :] < IN_TOTAL)
        values = tl.load(
            weight + out_offsets[:, None] * IN_TOTAL + in_offsets[None, :],
            mask=mask,
            other=0.0,
        )
        tl.store(
            transposed + in_offsets[None, :] * OUT_TOTAL + out_offsets[:, None],
            values,
            mask=mask,
        )


    @triton.jit(do_not_specialize=["POINT_START"])
    def _reduce_output_r1_kernel(
        workspace,
        center_ptr,
        output_cos,
        output_sin,
        output_pack,
        neighbor_count,
        out,
        POINT_START,
        N_POINTS: tl.constexpr,
        CHUNK_POINTS: tl.constexpr,
        PADDED_DEGREE: tl.constexpr,
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
        batch = row // CHUNK_POINTS
        local_point = row - batch * CHUNK_POINTS
        point = POINT_START + local_point
        point_mask = point < N_POINTS
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
        while edge_base < PADDED_DEGREE:
            degrees = edge_base + tl.arange(0, BLOCK_E)
            edge = edge_start + degrees
            edge_mask = point_mask & (edge < edge_stop) & (degrees < PADDED_DEGREE)
            workspace_row = (row * PADDED_DEGREE + degrees) * out_total
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
            denom = tl.maximum(tl.load(neighbor_count + point, mask=point_mask, other=1.0), 1.0)
            total0 /= denom
            total1 /= denom
        packed0 = channels * OUT_DIM + component0
        external0 = tl.load(output_pack + packed0, mask=channel_mask, other=0)
        out_base = (batch * N_POINTS + point) * out_total
        tl.store(out + out_base + external0, total0, mask=point_mask & channel_mask)
        if out_order > 0:
            external1 = tl.load(output_pack + packed0 + 1, mask=channel_mask, other=0)
            tl.store(out + out_base + external1, total1, mask=point_mask & channel_mask)


    @triton.jit(do_not_specialize=["POINT_START"])
    def _reduce_grad_input_r1_kernel(
        workspace,
        neighbor_ptr,
        edges_by_neighbor,
        radial_basis,
        input_cos,
        input_sin,
        input_pack,
        grad_x,
        POINT_START,
        N_POINTS: tl.constexpr,
        CHUNK_POINTS: tl.constexpr,
        PADDED_DEGREE: tl.constexpr,
        IN_M: tl.constexpr,
        IN_DIM: tl.constexpr,
        MAX_IN_ORDER: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        channel_blocks: tl.constexpr = tl.cdiv(IN_M, BLOCK_C)
        pid = tl.program_id(0)
        in_order = pid % (MAX_IN_ORDER + 1)
        pid //= MAX_IN_ORDER + 1
        channel_block = pid % channel_blocks
        row = pid // channel_blocks
        batch = row // CHUNK_POINTS
        local_point = row - batch * CHUNK_POINTS
        point = POINT_START + local_point
        point_mask = point < N_POINTS
        channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
        channel_mask = channels < IN_M
        component0 = tl.where(in_order == 0, 0, 2 * in_order - 1)
        component1 = component0 + 1
        total0 = tl.zeros((BLOCK_C,), tl.float32)
        total1 = tl.zeros((BLOCK_C,), tl.float32)
        pos_start = tl.load(neighbor_ptr + point, mask=point_mask, other=0)
        pos_stop = tl.load(neighbor_ptr + point + 1, mask=point_mask, other=0)
        edge_base = 0
        in_total: tl.constexpr = IN_M * IN_DIM
        while edge_base < PADDED_DEGREE:
            degrees = edge_base + tl.arange(0, BLOCK_E)
            pos = pos_start + degrees
            edge_mask = point_mask & (pos < pos_stop) & (degrees < PADDED_DEGREE)
            edge = tl.load(edges_by_neighbor + pos, mask=edge_mask, other=0)
            workspace_row = (row * PADDED_DEGREE + degrees) * in_total
            value0 = tl.load(
                workspace + workspace_row[:, None] + channels[None, :] * IN_DIM + component0,
                mask=edge_mask[:, None] & channel_mask[None, :],
                other=0.0,
            )
            radial = tl.load(radial_basis + edge, mask=edge_mask, other=0.0)
            value0 *= radial[:, None]
            if in_order == 0:
                total0 += tl.sum(value0, axis=0)
            else:
                value1 = tl.load(
                    workspace + workspace_row[:, None] + channels[None, :] * IN_DIM + component1,
                    mask=edge_mask[:, None] & channel_mask[None, :],
                    other=0.0,
                ) * radial[:, None]
                c = tl.load(
                    input_cos + edge * MAX_IN_ORDER + in_order - 1,
                    mask=edge_mask,
                    other=1.0,
                )
                s = tl.load(
                    input_sin + edge * MAX_IN_ORDER + in_order - 1,
                    mask=edge_mask,
                    other=0.0,
                )
                total0 += tl.sum(c[:, None] * value0 + s[:, None] * value1, axis=0)
                total1 += tl.sum(-s[:, None] * value0 + c[:, None] * value1, axis=0)
            edge_base += BLOCK_E
        packed0 = channels * IN_DIM + component0
        external0 = tl.load(input_pack + packed0, mask=channel_mask, other=0)
        base = (batch * N_POINTS + point) * in_total
        tl.store(grad_x + base + external0, total0, mask=point_mask & channel_mask)
        if in_order > 0:
            external1 = tl.load(input_pack + packed0 + 1, mask=channel_mask, other=0)
            tl.store(grad_x + base + external1, total1, mask=point_mask & channel_mask)


    @triton.jit
    def _grad_weight_matmul_r1_kernel(
        grad_workspace,
        input_workspace,
        partial,
        ROWS: tl.constexpr,
        IN_TOTAL: tl.constexpr,
        OUT_TOTAL: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_O: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        offs_o = tl.program_id(0) * BLOCK_O + tl.arange(0, BLOCK_O)
        offs_i = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
        if ALLOW_TF32:
            acc = tl.zeros((BLOCK_O, BLOCK_I), tl.float32)
        elif grad_workspace.dtype.element_ty == tl.float32:
            acc = tl.zeros((BLOCK_O, BLOCK_I), tl.float64)
        else:
            acc = tl.zeros((BLOCK_O, BLOCK_I), tl.float32)
        sample_base = 0
        while sample_base < ROWS:
            samples = sample_base + tl.arange(0, BLOCK_S)
            g = tl.load(
                grad_workspace + samples[:, None] * OUT_TOTAL + offs_o[None, :],
                mask=(samples[:, None] < ROWS) & (offs_o[None, :] < OUT_TOTAL),
                other=0.0,
            )
            x = tl.load(
                input_workspace + samples[:, None] * IN_TOTAL + offs_i[None, :],
                mask=(samples[:, None] < ROWS) & (offs_i[None, :] < IN_TOTAL),
                other=0.0,
            )
            if ALLOW_TF32:
                acc += tl.dot(tl.trans(g), x, input_precision="tf32")
            else:
                acc += tl.dot(tl.trans(g), x, input_precision="ieee")
            sample_base += BLOCK_S
        tl.store(
            partial + offs_o[:, None] * IN_TOTAL + offs_i[None, :],
            acc,
            mask=(offs_o[:, None] < OUT_TOTAL) & (offs_i[None, :] < IN_TOTAL),
        )


    @triton.jit
    def _split_k_grad_weight_r1_kernel(
        grad_workspace,
        input_workspace,
        partials,
        ROWS: tl.constexpr,
        SPLITS: tl.constexpr,
        IN_TOTAL: tl.constexpr,
        OUT_TOTAL: tl.constexpr,
        BATCH: tl.constexpr,
        N_POINTS: tl.constexpr,
        CHUNK_POINTS: tl.constexpr,
        IN_M: tl.constexpr,
        OUT_M: tl.constexpr,
        IN_DIM: tl.constexpr,
        OUT_DIM: tl.constexpr,
        PADDED_DEGREE: tl.constexpr,
        DEGREE_BUCKET: tl.constexpr,
        NORMALIZE: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_O: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        offs_o = tl.program_id(0) * BLOCK_O + tl.arange(0, BLOCK_O)
        offs_i = tl.program_id(1) * BLOCK_I + tl.arange(0, BLOCK_I)
        split = tl.program_id(2)
        split_rows: tl.constexpr = tl.cdiv(ROWS, SPLITS)
        sample_base = split * split_rows
        sample_stop = tl.minimum(sample_base + split_rows, ROWS)
        acc = tl.zeros((BLOCK_O, BLOCK_I), tl.float32)
        while sample_base < sample_stop:
            samples = sample_base + tl.arange(0, BLOCK_S)
            sample_mask = samples < sample_stop
            g = tl.load(
                grad_workspace
                + samples[:, None] * OUT_TOTAL
                + offs_o[None, :],
                mask=sample_mask[:, None] & (offs_o[None, :] < OUT_TOTAL),
                other=0.0,
            )
            x = tl.load(
                input_workspace
                + samples[:, None] * IN_TOTAL
                + offs_i[None, :],
                mask=sample_mask[:, None] & (offs_i[None, :] < IN_TOTAL),
                other=0.0,
            )
            if ALLOW_TF32:
                acc += tl.dot(tl.trans(g), x, input_precision="tf32")
            else:
                acc += tl.dot(tl.trans(g), x, input_precision="ieee")
            sample_base += BLOCK_S
        offsets = (
            (split * OUT_TOTAL + offs_o[:, None]) * IN_TOTAL
            + offs_i[None, :]
        )
        tl.store(
            partials + offsets,
            acc,
            mask=(offs_o[:, None] < OUT_TOTAL) & (offs_i[None, :] < IN_TOTAL),
        )


    @triton.jit
    def _pairwise_reduce_weight_tiles_kernel(
        source,
        destination,
        NUM_PARTIALS: tl.constexpr,
        NUMEL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pair = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < NUMEL
        left = tl.load(
            source + (2 * pair) * NUMEL + offsets,
            mask=mask,
            other=0.0,
        )
        has_right = 2 * pair + 1 < NUM_PARTIALS
        right = tl.load(
            source + (2 * pair + 1) * NUMEL + offsets,
            mask=mask & has_right,
            other=0.0,
        )
        tl.store(destination + pair * NUMEL + offsets, left + right, mask=mask)


    @triton.jit
    def _add_weight_tiles_kernel(
        left,
        right,
        destination,
        NUMEL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < NUMEL
        lhs = tl.load(left + offsets, mask=mask, other=0.0)
        rhs = tl.load(right + offsets, mask=mask, other=0.0)
        tl.store(destination + offsets, lhs + rhs, mask=mask)


    @triton.jit
    def _copy_weight_tile_kernel(
        source,
        destination,
        NUMEL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < NUMEL
        value = tl.load(source + offsets, mask=mask, other=0.0)
        tl.store(destination + offsets, value, mask=mask)


    @triton.jit
    def _accumulate_grad_weight_kernel(
        partial,
        grad_weight,
        NUMEL: tl.constexpr,
        FIRST: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < NUMEL
        value = tl.load(partial + offsets, mask=mask, other=0.0)
        if not FIRST:
            value += tl.load(grad_weight + offsets, mask=mask, other=0.0)
        tl.store(grad_weight + offsets, value, mask=mask)


    _MATMUL_KEY = ["ROWS", "IN_M", "OUT_M", "IN_DIM", "OUT_DIM", "TRANSPOSE_WEIGHT", "ALLOW_TF32"]
    _GRAD_WEIGHT_KEY = ["ROWS", "IN_TOTAL", "OUT_TOTAL", "ALLOW_TF32"]
    _SPLIT_K_GRAD_WEIGHT_KEY = [
        "ROWS",
        "SPLITS",
        "IN_TOTAL",
        "OUT_TOTAL",
        "BATCH",
        "N_POINTS",
        "CHUNK_POINTS",
        "IN_M",
        "OUT_M",
        "IN_DIM",
        "OUT_DIM",
        "PADDED_DEGREE",
        "DEGREE_BUCKET",
        "NORMALIZE",
        "ALLOW_TF32",
    ]

    def _matmul_configs(architecture: int) -> list:
        if architecture >= 9:
            return [
                triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
                triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=4),
                triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3),
                triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=4),
            ]
        if architecture >= 8:
            return [
                triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
                triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3),
                triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
                triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=8, num_stages=3),
            ]
        return [triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 16}, num_warps=4, num_stages=2)]

    def _grad_weight_configs(architecture: int) -> list:
        configs = [
            triton.Config({"BLOCK_O": 16, "BLOCK_I": 16, "BLOCK_S": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_O": 32, "BLOCK_I": 32, "BLOCK_S": 32}, num_warps=4, num_stages=3),
        ]
        if architecture >= 8:
            configs.append(triton.Config({"BLOCK_O": 32, "BLOCK_I": 64, "BLOCK_S": 64}, num_warps=8, num_stages=3))
        if architecture >= 9:
            configs.append(triton.Config({"BLOCK_O": 64, "BLOCK_I": 64, "BLOCK_S": 64}, num_warps=8, num_stages=4))
        return configs

    def _split_k_grad_weight_configs(architecture: int) -> list:
        if architecture == 8:
            # Calibrated on RTX 4060 and pinned for reproducible FP32 order.
            return [
                triton.Config(
                    {"BLOCK_O": 64, "BLOCK_I": 64, "BLOCK_S": 64},
                    num_warps=8,
                    num_stages=3,
                )
            ]
        if architecture >= 9:
            # Bounded compile-safe search family; execution remains unclaimed
            # until it is calibrated on an SM90 device.
            return [
                triton.Config(
                    {"BLOCK_O": 32, "BLOCK_I": 32, "BLOCK_S": 32},
                    num_warps=4,
                    num_stages=2,
                ),
                triton.Config(
                    {"BLOCK_O": 32, "BLOCK_I": 64, "BLOCK_S": 64},
                    num_warps=8,
                    num_stages=2,
                ),
                triton.Config(
                    {"BLOCK_O": 64, "BLOCK_I": 32, "BLOCK_S": 64},
                    num_warps=8,
                    num_stages=2,
                ),
                triton.Config(
                    {"BLOCK_O": 64, "BLOCK_I": 64, "BLOCK_S": 64},
                    num_warps=8,
                    num_stages=3,
                ),
            ]
        return []

    _semi_matmul_kernels = {
        architecture: triton.autotune(configs=_matmul_configs(architecture), key=_MATMUL_KEY)(
            _semi_matmul_r1_kernel
        )
        for architecture in (7, 8, 9)
    }
    _grad_weight_kernels = {
        architecture: triton.autotune(configs=_grad_weight_configs(architecture), key=_GRAD_WEIGHT_KEY)(
            _grad_weight_matmul_r1_kernel
        )
        for architecture in (7, 8, 9)
    }
    _split_k_grad_weight_kernels = {
        architecture: triton.autotune(
            configs=_split_k_grad_weight_configs(architecture),
            key=_SPLIT_K_GRAD_WEIGHT_KEY,
        )(_split_k_grad_weight_r1_kernel)
        for architecture in (8, 9)
    }


    def _architecture(tensor: Tensor) -> int:
        major = torch.cuda.get_device_capability(tensor.device)[0]
        return 7 if major < 8 else 8 if major < 9 else 9


    def _split_k_grad_weight_grid(
        *, splits: int, out_total: int, in_total: int,
    ):
        return lambda meta: (
            triton.cdiv(out_total, meta["BLOCK_O"]),
            triton.cdiv(in_total, meta["BLOCK_I"]),
            splits,
        )


    def _allocate_workspaces(
        x: Tensor,
        weight: Tensor,
        padded_degree: int,
        workspace_bytes: int,
    ) -> tuple[SemiPackedWorkspacePlan, Tensor, Tensor, Tensor]:
        plan = semi_packed_workspace_plan(x, weight, padded_degree, workspace_bytes)
        if plan is None:
            raise RuntimeError(
                "triton_workspace_mib is too small for one bounded semi-packed point chunk"
            )
        _radial, out_m, out_dim, in_m, in_dim = map(int, weight.shape)
        rows = int(x.shape[0]) * plan.chunk_points * plan.padded_degree
        input_workspace = torch.empty((rows, in_m * in_dim), device=x.device, dtype=x.dtype)
        output_workspace = torch.empty((rows, out_m * out_dim), device=x.device, dtype=x.dtype)
        partial = torch.empty((out_m * out_dim, in_m * in_dim), device=x.device, dtype=torch.float32)
        return plan, input_workspace, output_workspace, partial


    def _allocate_backward_phase_matrices(
        x: Tensor,
        weight: Tensor,
        plan: SemiPackedBackwardPhasePlan,
    ) -> tuple[Tensor, Tensor]:
        _radial, out_m, out_dim, in_m, in_dim = map(int, weight.shape)
        rows = int(x.shape[0]) * plan.chunk_points * plan.padded_degree
        input_workspace = torch.empty(
            (rows, in_m * in_dim), device=x.device, dtype=x.dtype,
        )
        output_workspace = torch.empty(
            (rows, out_m * out_dim), device=x.device, dtype=x.dtype,
        )
        allocated = (
            input_workspace.numel() * input_workspace.element_size()
            + output_workspace.numel() * output_workspace.element_size()
        )
        if allocated != plan.matrix_bytes:
            raise RuntimeError("semi-packed phase matrix allocation disagrees with plan")
        return input_workspace, output_workspace


    def _copy_weight_tile(source: Tensor, destination: Tensor, numel: int) -> None:
        grid = (triton.cdiv(numel, 256),)
        wrap_triton(_copy_weight_tile_kernel)[grid](
            source, destination, NUMEL=numel, BLOCK=256, num_warps=4,
        )


    def _add_weight_tiles(
        left: Tensor,
        right: Tensor,
        destination: Tensor,
        numel: int,
    ) -> None:
        grid = (triton.cdiv(numel, 256),)
        wrap_triton(_add_weight_tiles_kernel)[grid](
            left, right, destination, NUMEL=numel, BLOCK=256, num_warps=4,
        )


    def _reduce_split_partials(
        partials: Tensor,
        scratch: Tensor,
        numel: int,
    ) -> tuple[Tensor, bool]:
        current = partials
        destination = scratch
        count = SPLIT_K_SPLITS
        root_is_scratch = False
        while count > 1:
            next_count = (count + 1) // 2
            grid = (next_count, triton.cdiv(numel, 256))
            wrap_triton(_pairwise_reduce_weight_tiles_kernel)[grid](
                current,
                destination,
                NUM_PARTIALS=count,
                NUMEL=numel,
                BLOCK=256,
                num_warps=4,
            )
            current, destination = destination, current
            root_is_scratch = not root_is_scratch
            count = next_count
        return current[0], root_is_scratch


    def _launch_gather_input(
        x: Tensor, neighbor_idx: Tensor, center_ptr: Tensor, radial_basis: Tensor,
        input_cos: Tensor, input_sin: Tensor, input_pack: Tensor, workspace: Tensor,
        *, point_start: int, chunk_points: int, padded_degree: int, in_m: int,
        in_dim: int, n_points: int,
    ) -> None:
        rows = int(x.shape[0]) * chunk_points * padded_degree
        grid = (rows, triton.cdiv(in_m * in_dim, 64))
        wrap_triton(_gather_input_r1_kernel)[grid](
            x, neighbor_idx, center_ptr, radial_basis, input_cos, input_sin,
            input_pack, workspace, x.stride(0), x.stride(1), x.stride(2),
            POINT_START=point_start, N_POINTS=n_points, CHUNK_POINTS=chunk_points,
            PADDED_DEGREE=padded_degree, IN_M=in_m, IN_DIM=in_dim,
            MAX_IN_ORDER=(in_dim - 1) // 2, BLOCK_K=64, num_warps=4,
        )


    def _launch_gather_grad(
        grad_out: Tensor, center_idx: Tensor, point_ptr: Tensor,
        edges_by_neighbor: Tensor, output_cos: Tensor, output_sin: Tensor,
        output_pack: Tensor, neighbor_count: Tensor, workspace: Tensor, *,
        point_start: int, chunk_points: int, padded_degree: int, out_m: int,
        out_dim: int, n_points: int, by_neighbor: bool, normalize: bool,
    ) -> None:
        batch = int(grad_out.shape[0])
        rows = batch * chunk_points * padded_degree
        grid = (rows, triton.cdiv(out_m * out_dim, 64))
        wrap_triton(_gather_grad_output_r1_kernel)[grid](
            grad_out, center_idx, point_ptr, edges_by_neighbor, output_cos,
            output_sin, output_pack, neighbor_count, workspace,
            POINT_START=point_start, N_POINTS=n_points, CHUNK_POINTS=chunk_points,
            PADDED_DEGREE=padded_degree, OUT_M=out_m, OUT_DIM=out_dim,
            MAX_OUT_ORDER=(out_dim - 1) // 2, BATCH=batch,
            BY_NEIGHBOR=by_neighbor, NORMALIZE=normalize, BLOCK_K=64, num_warps=4,
        )


    def _launch_matmul(
        a: Tensor, weight_matrix: Tensor, c: Tensor, *, rows: int,
        in_m: int, out_m: int, in_dim: int, out_dim: int,
        transpose_weight: bool, allow_tf32: bool,
    ) -> None:
        architecture = _architecture(a)
        kernel = _semi_matmul_kernels[architecture]
        output_total = in_m * in_dim if transpose_weight else out_m * out_dim
        grid = lambda meta: (
            triton.cdiv(rows, meta["BLOCK_M"]),
            triton.cdiv(output_total, meta["BLOCK_N"]),
        )
        wrap_triton(kernel)[grid](
            a, weight_matrix, c, ROWS=rows, IN_M=in_m, OUT_M=out_m,
            IN_DIM=in_dim, OUT_DIM=out_dim, TRANSPOSE_WEIGHT=transpose_weight,
            ALLOW_TF32=allow_tf32,
        )


    def _launch_transpose_weight(weight: Tensor, workspace: Tensor) -> None:
        _radial, out_m, out_dim, in_m, in_dim = map(int, weight.shape)
        in_total = in_m * in_dim
        out_total = out_m * out_dim
        grid = (triton.cdiv(in_total, 32), triton.cdiv(out_total, 32))
        wrap_triton(_transpose_weight_r1_kernel)[grid](
            weight, workspace, IN_TOTAL=in_total, OUT_TOTAL=out_total,
            BLOCK_I=32, BLOCK_O=32, num_warps=4,
        )


    @triton_op("kpconv_intrinsic::semi_packed_irrep_conv", mutates_args={})
    def semi_packed_irrep_conv(
        x: Tensor,
        weight: Tensor,
        center_idx: Tensor,
        neighbor_idx: Tensor,
        center_ptr: Tensor,
        neighbor_ptr: Tensor,
        edges_by_neighbor: Tensor,
        radial_basis: Tensor,
        input_cos: Tensor,
        input_sin: Tensor,
        output_cos: Tensor,
        output_sin: Tensor,
        input_pack: Tensor,
        output_pack: Tensor,
        neighbor_count: Tensor,
        max_center_degree: int,
        max_neighbor_degree: int,
        degree_bucket: int,
        normalize: bool,
        allow_tf32: bool,
        workspace_bytes: int,
    ) -> Tensor:
        x = fix_triton_strides(x)
        weight = fix_triton_strides(weight)
        del center_idx, neighbor_ptr, edges_by_neighbor, max_neighbor_degree, degree_bucket
        radial, out_m, out_dim, in_m, in_dim = map(int, weight.shape)
        if radial != 1:
            raise RuntimeError("semi-packed Triton convolution only supports R1")
        batch, n_points = map(int, x.shape[:2])
        plan, input_workspace, output_workspace, weight_workspace = _allocate_workspaces(
            x, weight, max_center_degree, workspace_bytes,
        )
        out = torch.empty((batch, n_points, out_m * out_dim), device=x.device, dtype=x.dtype)
        rows = batch * plan.chunk_points * max_center_degree
        _launch_transpose_weight(weight, weight_workspace)
        for point_start in range(0, n_points, plan.chunk_points):
            _launch_gather_input(
                x, neighbor_idx, center_ptr, radial_basis, input_cos, input_sin,
                input_pack, input_workspace, point_start=point_start,
                chunk_points=plan.chunk_points, padded_degree=max_center_degree,
                in_m=in_m, in_dim=in_dim, n_points=n_points,
            )
            _launch_matmul(
                input_workspace, weight_workspace, output_workspace, rows=rows, in_m=in_m,
                out_m=out_m, in_dim=in_dim, out_dim=out_dim,
                transpose_weight=False, allow_tf32=allow_tf32,
            )
            grid = (
                batch * plan.chunk_points * triton.cdiv(out_m, 8) * ((out_dim - 1) // 2 + 1),
            )
            wrap_triton(_reduce_output_r1_kernel)[grid](
                output_workspace, center_ptr, output_cos, output_sin, output_pack,
                neighbor_count, out, POINT_START=point_start, N_POINTS=n_points,
                CHUNK_POINTS=plan.chunk_points, PADDED_DEGREE=max_center_degree,
                OUT_M=out_m, OUT_DIM=out_dim, MAX_OUT_ORDER=(out_dim - 1) // 2,
                NORMALIZE=normalize, BLOCK_C=8, BLOCK_E=16, num_warps=4,
            )
        return out


    @triton_op("kpconv_intrinsic::semi_packed_irrep_conv_backward", mutates_args={})
    def _semi_packed_irrep_conv_backward(
        grad_out: Tensor,
        x: Tensor,
        weight: Tensor,
        center_idx: Tensor,
        neighbor_idx: Tensor,
        center_ptr: Tensor,
        neighbor_ptr: Tensor,
        edges_by_neighbor: Tensor,
        radial_basis: Tensor,
        input_cos: Tensor,
        input_sin: Tensor,
        output_cos: Tensor,
        output_sin: Tensor,
        input_pack: Tensor,
        output_pack: Tensor,
        neighbor_count: Tensor,
        max_center_degree: int,
        max_neighbor_degree: int,
        degree_bucket: int,
        normalize: bool,
        allow_tf32: bool,
        workspace_bytes: int,
    ) -> tuple[Tensor, Tensor]:
        grad_out = fix_triton_strides(grad_out)
        x = fix_triton_strides(x)
        weight = fix_triton_strides(weight)
        _radial, out_m, out_dim, in_m, in_dim = map(int, weight.shape)
        batch, n_points = map(int, x.shape[:2])
        architecture = _architecture(x)
        plan = semi_packed_backward_workspace_plan(
            x,
            weight,
            max_center_degree,
            max_neighbor_degree,
            workspace_bytes,
            prefer_split_k=architecture >= 8,
        )
        if plan is None:
            raise RuntimeError(
                "triton_workspace_mib is too small for one bounded "
                "semi-packed backward point chunk"
            )
        grad_x = torch.empty_like(x, memory_format=torch.contiguous_format)
        grad_weight = torch.empty_like(weight, memory_format=torch.contiguous_format)

        # Neighbor-aligned chunks give every program exclusive ownership of a
        # grad-input point, so no atomics are required.
        dx_plan = plan.grad_input
        input_workspace, output_workspace = _allocate_backward_phase_matrices(
            x, weight, dx_plan,
        )
        rows_neighbor = batch * dx_plan.chunk_points * max_neighbor_degree
        for point_start in range(0, n_points, dx_plan.chunk_points):
            _launch_gather_grad(
                grad_out, center_idx, neighbor_ptr, edges_by_neighbor, output_cos,
                output_sin, output_pack, neighbor_count, output_workspace,
                point_start=point_start, chunk_points=dx_plan.chunk_points,
                padded_degree=max_neighbor_degree, out_m=out_m, out_dim=out_dim,
                n_points=n_points, by_neighbor=True, normalize=normalize,
            )
            _launch_matmul(
                output_workspace, weight, input_workspace, rows=rows_neighbor, in_m=in_m,
                out_m=out_m, in_dim=in_dim, out_dim=out_dim,
                transpose_weight=True, allow_tf32=allow_tf32,
            )
            grid_x = (
                batch * dx_plan.chunk_points * triton.cdiv(in_m, 8)
                * ((in_dim - 1) // 2 + 1),
            )
            wrap_triton(_reduce_grad_input_r1_kernel)[grid_x](
                input_workspace, neighbor_ptr, edges_by_neighbor, radial_basis,
                input_cos, input_sin, input_pack, grad_x, POINT_START=point_start,
                N_POINTS=n_points, CHUNK_POINTS=dx_plan.chunk_points,
                PADDED_DEGREE=max_neighbor_degree, IN_M=in_m, IN_DIM=in_dim,
                MAX_IN_ORDER=(in_dim - 1) // 2, BLOCK_C=8, BLOCK_E=16, num_warps=4,
            )
        del input_workspace, output_workspace

        # Center-aligned chunks form dense sample matrices for dW = G^T X.
        dw_plan = plan.grad_weight
        input_workspace, output_workspace = _allocate_backward_phase_matrices(
            x, weight, dw_plan,
        )
        rows_center = batch * dw_plan.chunk_points * max_center_degree
        in_total = in_m * in_dim
        out_total = out_m * out_dim
        weight_numel = in_total * out_total

        if plan.use_split_k:
            partials = torch.empty(
                (SPLIT_K_SPLITS, out_total, in_total),
                device=x.device,
                dtype=torch.float32,
            )
            scratch = torch.empty(
                ((SPLIT_K_SPLITS + 1) // 2, out_total, in_total),
                device=x.device,
                dtype=torch.float32,
            )
            carries = torch.empty(
                (SPLIT_K_CARRY_LEVELS, out_total, in_total),
                device=x.device,
                dtype=torch.float32,
            )
            fixed_allocated = sum(
                workspace.numel() * workspace.element_size()
                for workspace in (partials, scratch, carries)
            )
            if dw_plan.matrix_bytes + fixed_allocated != dw_plan.allocated_bytes:
                raise RuntimeError("split-K workspace allocation disagrees with plan")

            num_chunks = (
                n_points + dw_plan.chunk_points - 1
            ) // dw_plan.chunk_points
            insertion_levels, final_levels = _online_binary_carry_schedule(
                num_chunks,
            )
            split_kernel = _split_k_grad_weight_kernels[architecture]
            grid_w = _split_k_grad_weight_grid(
                splits=SPLIT_K_SPLITS,
                out_total=out_total,
                in_total=in_total,
            )
            for chunk_index, point_start in enumerate(
                range(0, n_points, dw_plan.chunk_points)
            ):
                _launch_gather_input(
                    x, neighbor_idx, center_ptr, radial_basis, input_cos,
                    input_sin, input_pack, input_workspace,
                    point_start=point_start,
                    chunk_points=dw_plan.chunk_points,
                    padded_degree=max_center_degree,
                    in_m=in_m, in_dim=in_dim, n_points=n_points,
                )
                _launch_gather_grad(
                    grad_out, center_idx, center_ptr, edges_by_neighbor,
                    output_cos, output_sin, output_pack, neighbor_count,
                    output_workspace, point_start=point_start,
                    chunk_points=dw_plan.chunk_points,
                    padded_degree=max_center_degree,
                    out_m=out_m, out_dim=out_dim, n_points=n_points,
                    by_neighbor=False, normalize=normalize,
                )
                wrap_triton(split_kernel)[grid_w](
                    output_workspace,
                    input_workspace,
                    partials,
                    ROWS=rows_center,
                    SPLITS=SPLIT_K_SPLITS,
                    IN_TOTAL=in_total,
                    OUT_TOTAL=out_total,
                    BATCH=batch,
                    N_POINTS=n_points,
                    CHUNK_POINTS=dw_plan.chunk_points,
                    IN_M=in_m,
                    OUT_M=out_m,
                    IN_DIM=in_dim,
                    OUT_DIM=out_dim,
                    PADDED_DEGREE=max_center_degree,
                    DEGREE_BUCKET=degree_bucket,
                    NORMALIZE=normalize,
                    ALLOW_TF32=allow_tf32,
                )
                root, root_is_scratch = _reduce_split_partials(
                    partials, scratch, weight_numel,
                )
                store_level = insertion_levels[chunk_index]
                for level in range(store_level):
                    destination = (
                        partials[0] if root_is_scratch else scratch[0]
                    )
                    _add_weight_tiles(
                        carries[level], root, destination, weight_numel,
                    )
                    root = destination
                    root_is_scratch = not root_is_scratch
                _copy_weight_tile(root, carries[store_level], weight_numel)

            root = carries[final_levels[0]]
            use_partial_destination = True
            for level in final_levels[1:]:
                destination = (
                    partials[0] if use_partial_destination else scratch[0]
                )
                _add_weight_tiles(
                    carries[level], root, destination, weight_numel,
                )
                root = destination
                use_partial_destination = not use_partial_destination
            _copy_weight_tile(root, grad_weight, weight_numel)
        else:
            partial = torch.empty(
                (out_total, in_total), device=x.device, dtype=torch.float32,
            )
            fixed_allocated = partial.numel() * partial.element_size()
            if dw_plan.matrix_bytes + fixed_allocated != dw_plan.allocated_bytes:
                raise RuntimeError("fallback dW workspace allocation disagrees with plan")
            first = True
            weight_kernel = _grad_weight_kernels[architecture]
            grid_w = lambda meta: (
                triton.cdiv(out_total, meta["BLOCK_O"]),
                triton.cdiv(in_total, meta["BLOCK_I"]),
            )
            for point_start in range(0, n_points, dw_plan.chunk_points):
                _launch_gather_input(
                    x, neighbor_idx, center_ptr, radial_basis, input_cos,
                    input_sin, input_pack, input_workspace,
                    point_start=point_start,
                    chunk_points=dw_plan.chunk_points,
                    padded_degree=max_center_degree,
                    in_m=in_m, in_dim=in_dim, n_points=n_points,
                )
                _launch_gather_grad(
                    grad_out, center_idx, center_ptr, edges_by_neighbor,
                    output_cos, output_sin, output_pack, neighbor_count,
                    output_workspace, point_start=point_start,
                    chunk_points=dw_plan.chunk_points,
                    padded_degree=max_center_degree,
                    out_m=out_m, out_dim=out_dim, n_points=n_points,
                    by_neighbor=False, normalize=normalize,
                )
                wrap_triton(weight_kernel)[grid_w](
                    output_workspace,
                    input_workspace,
                    partial,
                    ROWS=rows_center,
                    IN_TOTAL=in_total,
                    OUT_TOTAL=out_total,
                    ALLOW_TF32=allow_tf32,
                )
                grid_accumulate = (triton.cdiv(weight_numel, 256),)
                wrap_triton(_accumulate_grad_weight_kernel)[grid_accumulate](
                    partial,
                    grad_weight,
                    NUMEL=weight_numel,
                    FIRST=first,
                    BLOCK=256,
                    num_warps=4,
                )
                first = False
        return grad_x, grad_weight


    def _setup_context(ctx, inputs, output) -> None:
        del output
        ctx.save_for_backward(*inputs[:-6])
        ctx.max_center_degree = int(inputs[-6])
        ctx.max_neighbor_degree = int(inputs[-5])
        ctx.degree_bucket = int(inputs[-4])
        ctx.normalize = bool(inputs[-3])
        ctx.allow_tf32 = bool(inputs[-2])
        ctx.workspace_bytes = int(inputs[-1])


    def _backward(ctx, grad_out: Tensor):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "higher-order gradients are unsupported by the Triton irrep convolution; "
                "use backend='torch'"
            )
        grad_x, grad_weight = _semi_packed_irrep_conv_backward(
            grad_out.contiguous(), *ctx.saved_tensors, ctx.max_center_degree,
            ctx.max_neighbor_degree, ctx.degree_bucket, ctx.normalize,
            ctx.allow_tf32, ctx.workspace_bytes,
        )
        return grad_x, grad_weight, *(None for _ in range(19))


    semi_packed_irrep_conv.register_autograd(_backward, setup_context=_setup_context)

else:

    def semi_packed_irrep_conv(*args, **kwargs):  # type: ignore[no-redef]
        del args, kwargs
        raise RuntimeError("Triton semi-packed backend is unavailable")


__all__ = [
    "MIN_AUTO_EDGE_CAPACITY",
    "MIN_AUTO_MATRIX_DIM",
    "SemiPackedWorkspacePlan",
    "TRITON_SEMI_PACKED_AVAILABLE",
    "semi_packed_irrep_conv",
    "semi_packed_support_reason",
    "semi_packed_workspace_plan",
    "should_use_semi_packed",
]
