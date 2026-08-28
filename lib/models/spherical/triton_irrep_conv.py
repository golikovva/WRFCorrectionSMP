"""Fused Triton backend for regular packed intrinsic irrep convolutions."""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton
except (ImportError, AttributeError):  # pragma: no cover - exercised by CPU-only installs
    triton = None
    tl = None
    triton_op = None
    wrap_triton = None


TRITON_AVAILABLE = triton is not None and triton_op is not None
# SM70 keeps the initial threshold until the production V100 benchmark is run.
# The SM80+ inference threshold is calibrated on the local Ada GPU. Training
# remains on the reference path by default because recomputing edge activations
# makes the current memory-first backward slower on the local benchmark.
AUTO_WORK_THRESHOLD = 1 << 20
AUTO_INFERENCE_WORK_THRESHOLDS = {7: AUTO_WORK_THRESHOLD, 8: 1 << 24}
AUTO_TRAINING_WORK_THRESHOLDS = {7: 1 << 29, 8: 1 << 30}
MAX_ORDER = 8
MAX_MULTIPLICITY = 512
MAX_RADIAL = 4
R1_GRAD_WEIGHT_WORKSPACE_BYTES = 64 << 20


def triton_support_reason(x: Tensor, weight: Tensor, max_in_order: int, max_out_order: int) -> str | None:
    """Return why the fused backend cannot handle this invocation, or ``None``."""

    if not TRITON_AVAILABLE:
        return "Triton or torch.library.triton_op is unavailable"
    if not x.is_cuda:
        return "Triton backend requires CUDA tensors"
    if weight.device != x.device:
        return "input and packed weights must be on the same CUDA device"
    if weight.dtype != x.dtype:
        return "input and packed weights must have the same dtype"
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return f"unsupported input dtype {x.dtype}; expected float16, bfloat16, or float32"
    major, _minor = torch.cuda.get_device_capability(x.device)
    if major < 7:
        return "Triton backend requires compute capability 7.0 or newer"
    if x.dtype == torch.bfloat16 and major < 8:
        return "bfloat16 Triton path requires compute capability 8.0 or newer"
    if int(max_in_order) > MAX_ORDER or int(max_out_order) > MAX_ORDER:
        return f"maximum supported irrep order is {MAX_ORDER}"
    radial, out_m, _out_dim, in_m, _in_dim = map(int, weight.shape)
    if radial > MAX_RADIAL:
        return f"maximum supported radial count is {MAX_RADIAL}"
    if in_m > MAX_MULTIPLICITY or out_m > MAX_MULTIPLICITY:
        return f"maximum supported multiplicity is {MAX_MULTIPLICITY}"
    return None


def _r1_grad_weight_partial_count(
    weight: Tensor,
    *,
    batch: int,
    n_edges: int,
) -> int:
    """Choose bounded sample parallelism for the two-stage R=1 weight gradient."""

    _radial, out_m, out_dim, in_m, in_dim = map(int, weight.shape)
    out_tiles = (out_m + 15) // 16
    in_tiles = (in_m + 15) // 16
    weight_tiles = out_dim * in_dim * out_tiles * in_tiles
    target_programs = torch.cuda.get_device_properties(weight.device).multi_processor_count * 8
    desired_partials = max(1, (target_programs + weight_tiles - 1) // weight_tiles)
    max_workspace_partials = max(
        1,
        R1_GRAD_WEIGHT_WORKSPACE_BYTES // (int(weight.numel()) * torch.float32.itemsize),
    )
    sample_blocks = max(1, (batch * n_edges + 31) // 32)
    return min(desired_partials, max_workspace_partials, sample_blocks)


if TRITON_AVAILABLE:

    @triton.jit
    def _load_rotated_input(
        x,
        neighbor,
        packed_component,
        edge,
        input_cos,
        input_sin,
        input_pack,
        stride_xb: tl.constexpr,
        stride_xn: tl.constexpr,
        stride_xc: tl.constexpr,
        batch,
        IN_DIM: tl.constexpr,
        MAX_IN_ORDER: tl.constexpr,
        component_mask,
        edge_mask,
    ):
        in_channel = packed_component // IN_DIM
        component = packed_component - in_channel * IN_DIM
        direct_external = tl.load(input_pack + packed_component, mask=component_mask, other=0)
        direct = tl.load(
            x + batch * stride_xb + neighbor[None, :] * stride_xn + direct_external[:, None] * stride_xc,
            mask=component_mask[:, None] & edge_mask[None, :],
            other=0.0,
        )
        is_vector = component > 0
        first_component = tl.where((component & 1) == 1, component, component - 1)
        first_packed = in_channel * IN_DIM + first_component
        first_external = tl.load(input_pack + first_packed, mask=component_mask & is_vector, other=0)
        second_external = tl.load(input_pack + first_packed + 1, mask=component_mask & is_vector, other=0)
        pair_mask = component_mask[:, None] & is_vector[:, None] & edge_mask[None, :]
        x0 = tl.load(
            x + batch * stride_xb + neighbor[None, :] * stride_xn + first_external[:, None] * stride_xc,
            mask=pair_mask,
            other=0.0,
        )
        x1 = tl.load(
            x + batch * stride_xb + neighbor[None, :] * stride_xn + second_external[:, None] * stride_xc,
            mask=pair_mask,
            other=0.0,
        )
        order_index = (first_component - 1) // 2
        cos_value = tl.load(
            input_cos + edge[None, :] * MAX_IN_ORDER + order_index[:, None],
            mask=pair_mask,
            other=1.0,
        )
        sin_value = tl.load(
            input_sin + edge[None, :] * MAX_IN_ORDER + order_index[:, None],
            mask=pair_mask,
            other=0.0,
        )
        first = cos_value * x0 - sin_value * x1
        second = sin_value * x0 + cos_value * x1
        rotated = tl.where((component & 1)[:, None] == 1, first, second)
        return tl.where(is_vector[:, None], rotated, direct)


    @triton.jit
    def _packed_forward_kernel(
        x,
        weight,
        neighbor_idx,
        center_ptr,
        radial_basis,
        input_cos,
        input_sin,
        output_cos,
        output_sin,
        input_pack,
        output_pack,
        neighbor_count,
        out,
        stride_xb: tl.constexpr,
        stride_xn: tl.constexpr,
        stride_xc: tl.constexpr,
        stride_wr: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wa: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_wd: tl.constexpr,
        N_POINTS: tl.constexpr,
        IN_M: tl.constexpr,
        OUT_M: tl.constexpr,
        IN_DIM: tl.constexpr,
        OUT_DIM: tl.constexpr,
        NUM_RADIAL: tl.constexpr,
        MAX_IN_ORDER: tl.constexpr,
        MAX_OUT_ORDER: tl.constexpr,
        NORMALIZE: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_O: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        out_blocks = tl.cdiv(OUT_M, BLOCK_O)
        pid = tl.program_id(0)
        out_block = pid % out_blocks
        pid //= out_blocks
        out_order = pid % (MAX_OUT_ORDER + 1)
        pid //= MAX_OUT_ORDER + 1
        center = pid % N_POINTS
        batch = pid // N_POINTS

        out_channels = out_block * BLOCK_O + tl.arange(0, BLOCK_O)
        out_mask = out_channels < OUT_M
        first_out_component = tl.where(out_order == 0, 0, 2 * out_order - 1)
        second_out_component = first_out_component + 1
        total0 = tl.zeros((BLOCK_O,), tl.float32)
        total1 = tl.zeros((BLOCK_O,), tl.float32)
        edge_start = tl.load(center_ptr + center)
        edge_stop = tl.load(center_ptr + center + 1)
        edge_base = edge_start
        k_total: tl.constexpr = IN_M * IN_DIM

        while edge_base < edge_stop:
            edge_offsets = tl.arange(0, BLOCK_E)
            edge = edge_base + edge_offsets
            edge_mask = edge < edge_stop
            neighbor = tl.load(neighbor_idx + edge, mask=edge_mask, other=0)
            for radial in range(NUM_RADIAL):
                raw0 = tl.zeros((BLOCK_O, BLOCK_E), tl.float32)
                raw1 = tl.zeros((BLOCK_O, BLOCK_E), tl.float32)
                k_base = 0
                while k_base < k_total:
                    k_offsets = tl.arange(0, BLOCK_K)
                    packed_component = k_base + k_offsets
                    k_mask = packed_component < k_total
                    rotated_input = _load_rotated_input(
                        x,
                        neighbor,
                        packed_component,
                        edge,
                        input_cos,
                        input_sin,
                        input_pack,
                        stride_xb,
                        stride_xn,
                        stride_xc,
                        batch,
                        IN_DIM,
                        MAX_IN_ORDER,
                        k_mask,
                        edge_mask,
                    )
                    radial_value = tl.load(
                        radial_basis + edge[None, :] * NUM_RADIAL + radial,
                        mask=edge_mask[None, :],
                        other=0.0,
                    )
                    z = rotated_input * radial_value
                    in_channel = packed_component // IN_DIM
                    in_component = packed_component - in_channel * IN_DIM
                    w0_ptr = (
                        weight
                        + radial * stride_wr
                        + out_channels[:, None] * stride_wo
                        + first_out_component * stride_wa
                        + in_channel[None, :] * stride_wi
                        + in_component[None, :] * stride_wd
                    )
                    weight_mask = out_mask[:, None] & k_mask[None, :]
                    w0 = tl.load(w0_ptr, mask=weight_mask, other=0.0).to(z.dtype)
                    if ALLOW_TF32:
                        raw0 += tl.dot(w0, z, input_precision="tf32")
                    else:
                        raw0 += tl.dot(w0, z, input_precision="ieee")
                    if out_order > 0:
                        w1 = tl.load(w0_ptr + stride_wa, mask=weight_mask, other=0.0).to(z.dtype)
                        if ALLOW_TF32:
                            raw1 += tl.dot(w1, z, input_precision="tf32")
                        else:
                            raw1 += tl.dot(w1, z, input_precision="ieee")
                    k_base += BLOCK_K
                if out_order == 0:
                    total0 += tl.sum(raw0, axis=1)
                else:
                    cos_value = tl.load(
                        output_cos + edge * MAX_OUT_ORDER + out_order - 1,
                        mask=edge_mask,
                        other=1.0,
                    )
                    sin_value = tl.load(
                        output_sin + edge * MAX_OUT_ORDER + out_order - 1,
                        mask=edge_mask,
                        other=0.0,
                    )
                    total0 += tl.sum(raw0 * cos_value[None, :] - raw1 * sin_value[None, :], axis=1)
                    total1 += tl.sum(raw0 * sin_value[None, :] + raw1 * cos_value[None, :], axis=1)
            edge_base += BLOCK_E

        if NORMALIZE:
            scale = tl.maximum(tl.load(neighbor_count + center), 1.0)
            total0 /= scale
            total1 /= scale
        packed0 = out_channels * OUT_DIM + first_out_component
        external0 = tl.load(output_pack + packed0, mask=out_mask, other=0)
        out_base = (batch * N_POINTS + center) * (OUT_M * OUT_DIM)
        tl.store(out + out_base + external0, total0, mask=out_mask)
        if out_order > 0:
            external1 = tl.load(output_pack + packed0 + 1, mask=out_mask, other=0)
            tl.store(out + out_base + external1, total1, mask=out_mask)


    @triton.jit
    def _packed_forward_r1_kernel(
        x,
        weight,
        neighbor_idx,
        center_ptr,
        radial_basis,
        input_cos,
        input_sin,
        output_cos,
        output_sin,
        input_pack,
        output_pack,
        neighbor_count,
        out,
        stride_xb: tl.constexpr,
        stride_xn: tl.constexpr,
        stride_xc: tl.constexpr,
        stride_wr: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wa: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_wd: tl.constexpr,
        N_POINTS: tl.constexpr,
        IN_M: tl.constexpr,
        OUT_M: tl.constexpr,
        IN_DIM: tl.constexpr,
        OUT_DIM: tl.constexpr,
        MAX_IN_ORDER: tl.constexpr,
        MAX_OUT_ORDER: tl.constexpr,
        BLOCK_ORDERS: tl.constexpr,
        DEGREE_BUCKET: tl.constexpr,
        NORMALIZE: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_O: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        out_blocks = tl.cdiv(OUT_M, BLOCK_O)
        pid = tl.program_id(0)
        out_block = pid % out_blocks
        pid //= out_blocks
        center = pid % N_POINTS
        batch = pid // N_POINTS

        pair_count: tl.constexpr = BLOCK_O * BLOCK_ORDERS
        row_count: tl.constexpr = pair_count * 2
        row = tl.arange(0, row_count)
        pair = row // 2
        pair_component = row - pair * 2
        local_out_channel = pair // BLOCK_ORDERS
        out_order = pair - local_out_channel * BLOCK_ORDERS
        out_channel = out_block * BLOCK_O + local_out_channel
        out_component = tl.where(out_order == 0, pair_component, 2 * out_order - 1 + pair_component)
        row_mask = (
            (out_channel < OUT_M)
            & (out_order <= MAX_OUT_ORDER)
            & ((out_order > 0) | (pair_component == 0))
        )
        totals = tl.zeros((row_count,), tl.float32)
        edge_start = tl.load(center_ptr + center)
        edge_stop = tl.load(center_ptr + center + 1)
        edge_base = edge_start
        k_total: tl.constexpr = IN_M * IN_DIM

        while edge_base < edge_stop:
            edge_offsets = tl.arange(0, BLOCK_E)
            edge = edge_base + edge_offsets
            edge_mask = edge < edge_stop
            neighbor = tl.load(neighbor_idx + edge, mask=edge_mask, other=0)
            raw = tl.zeros((BLOCK_E, row_count), tl.float32)
            k_base = 0
            while k_base < k_total:
                k_offsets = tl.arange(0, BLOCK_K)
                packed_component = k_base + k_offsets
                k_mask = packed_component < k_total
                rotated_input = _load_rotated_input(
                    x,
                    neighbor,
                    packed_component,
                    edge,
                    input_cos,
                    input_sin,
                    input_pack,
                    stride_xb,
                    stride_xn,
                    stride_xc,
                    batch,
                    IN_DIM,
                    MAX_IN_ORDER,
                    k_mask,
                    edge_mask,
                )
                radial_value = tl.load(radial_basis + edge, mask=edge_mask, other=0.0)
                z = rotated_input * radial_value[None, :]
                in_channel = packed_component // IN_DIM
                in_component = packed_component - in_channel * IN_DIM
                weight_ptr = (
                    weight
                    + out_channel[None, :] * stride_wo
                    + out_component[None, :] * stride_wa
                    + in_channel[:, None] * stride_wi
                    + in_component[:, None] * stride_wd
                )
                w = tl.load(
                    weight_ptr,
                    mask=k_mask[:, None] & row_mask[None, :],
                    other=0.0,
                ).to(z.dtype)
                if ALLOW_TF32:
                    raw += tl.dot(tl.trans(z), w, input_precision="tf32")
                else:
                    raw += tl.dot(tl.trans(z), w, input_precision="ieee")
                k_base += BLOCK_K

            paired = tl.reshape(raw, (BLOCK_E, pair_count, 2))
            raw0, raw1 = tl.split(paired)
            pair_offsets = tl.arange(0, pair_count)
            pair_local_out = pair_offsets // BLOCK_ORDERS
            pair_order = pair_offsets - pair_local_out * BLOCK_ORDERS
            rotation_mask = edge_mask[:, None] & (pair_order[None, :] > 0) & (
                pair_order[None, :] <= MAX_OUT_ORDER
            )
            cos_value = tl.load(
                output_cos + edge[:, None] * MAX_OUT_ORDER + pair_order[None, :] - 1,
                mask=rotation_mask,
                other=1.0,
            )
            sin_value = tl.load(
                output_sin + edge[:, None] * MAX_OUT_ORDER + pair_order[None, :] - 1,
                mask=rotation_mask,
                other=0.0,
            )
            rotated0 = raw0 * cos_value - raw1 * sin_value
            rotated1 = raw0 * sin_value + raw1 * cos_value
            rotated = tl.join(rotated0, rotated1)
            totals += tl.sum(tl.reshape(rotated, (BLOCK_E, row_count)), axis=0)
            edge_base += BLOCK_E

        if NORMALIZE:
            scale = tl.maximum(tl.load(neighbor_count + center), 1.0)
            totals /= scale
        pair_totals = tl.reshape(totals, (pair_count, 2))
        total0, total1 = tl.split(pair_totals)
        pair_offsets = tl.arange(0, pair_count)
        pair_local_out = pair_offsets // BLOCK_ORDERS
        pair_order = pair_offsets - pair_local_out * BLOCK_ORDERS
        pair_out_channel = out_block * BLOCK_O + pair_local_out
        pair_mask = (pair_out_channel < OUT_M) & (pair_order <= MAX_OUT_ORDER)
        component0 = tl.where(pair_order == 0, 0, 2 * pair_order - 1)
        packed0 = pair_out_channel * OUT_DIM + component0
        external0 = tl.load(output_pack + packed0, mask=pair_mask, other=0)
        out_base = (batch * N_POINTS + center) * (OUT_M * OUT_DIM)
        tl.store(out + out_base + external0, total0, mask=pair_mask)
        vector_mask = pair_mask & (pair_order > 0)
        external1 = tl.load(output_pack + packed0 + 1, mask=vector_mask, other=0)
        tl.store(out + out_base + external1, total1, mask=vector_mask)


    @triton.jit
    def _packed_grad_input_kernel(
        grad_out,
        weight,
        center_idx,
        neighbor_ptr,
        edges_by_neighbor,
        radial_basis,
        input_cos,
        input_sin,
        output_cos,
        output_sin,
        input_pack,
        output_pack,
        neighbor_count,
        grad_x,
        stride_wr: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wa: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_wd: tl.constexpr,
        N_POINTS: tl.constexpr,
        IN_M: tl.constexpr,
        OUT_M: tl.constexpr,
        IN_DIM: tl.constexpr,
        OUT_DIM: tl.constexpr,
        NUM_RADIAL: tl.constexpr,
        MAX_IN_ORDER: tl.constexpr,
        MAX_OUT_ORDER: tl.constexpr,
        NORMALIZE: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        in_blocks = tl.cdiv(IN_M, BLOCK_I)
        pid = tl.program_id(0)
        in_block = pid % in_blocks
        pid //= in_blocks
        in_order = pid % (MAX_IN_ORDER + 1)
        pid //= MAX_IN_ORDER + 1
        point = pid % N_POINTS
        batch = pid // N_POINTS
        in_channels = in_block * BLOCK_I + tl.arange(0, BLOCK_I)
        in_mask = in_channels < IN_M
        in_component0 = tl.where(in_order == 0, 0, 2 * in_order - 1)
        in_component1 = in_component0 + 1
        total0 = tl.zeros((BLOCK_I,), tl.float32)
        total1 = tl.zeros((BLOCK_I,), tl.float32)
        pos = tl.load(neighbor_ptr + point)
        pos_stop = tl.load(neighbor_ptr + point + 1)
        out_total: tl.constexpr = OUT_M * OUT_DIM
        while pos < pos_stop:
            e_offsets = tl.arange(0, BLOCK_E)
            positions = pos + e_offsets
            e_mask = positions < pos_stop
            edge = tl.load(edges_by_neighbor + positions, mask=e_mask, other=0)
            center = tl.load(center_idx + edge, mask=e_mask, other=0)
            edge_grad0 = tl.zeros((BLOCK_I, BLOCK_E), tl.float32)
            edge_grad1 = tl.zeros((BLOCK_I, BLOCK_E), tl.float32)
            for radial in range(NUM_RADIAL):
                radial_value = tl.load(radial_basis + edge * NUM_RADIAL + radial, mask=e_mask, other=0.0)
                raw0 = tl.zeros((BLOCK_I, BLOCK_E), tl.float32)
                raw1 = tl.zeros((BLOCK_I, BLOCK_E), tl.float32)
                k_base = 0
                while k_base < out_total:
                    k_offsets = tl.arange(0, BLOCK_K)
                    packed_out = k_base + k_offsets
                    k_mask = packed_out < out_total
                    out_channel = packed_out // OUT_DIM
                    out_component = packed_out - out_channel * OUT_DIM
                    external = tl.load(output_pack + packed_out, mask=k_mask, other=0)
                    g_direct = tl.load(
                        grad_out + (batch * N_POINTS + center[None, :]) * out_total + external[:, None],
                        mask=k_mask[:, None] & e_mask[None, :],
                        other=0.0,
                    )
                    is_vector = out_component > 0
                    pair0 = tl.where((out_component & 1) == 1, out_component, out_component - 1)
                    pair_packed = out_channel * OUT_DIM + pair0
                    ext0 = tl.load(output_pack + pair_packed, mask=k_mask & is_vector, other=0)
                    ext1 = tl.load(output_pack + pair_packed + 1, mask=k_mask & is_vector, other=0)
                    pair_mask = k_mask[:, None] & is_vector[:, None] & e_mask[None, :]
                    g0 = tl.load(
                        grad_out + (batch * N_POINTS + center[None, :]) * out_total + ext0[:, None],
                        mask=pair_mask,
                        other=0.0,
                    )
                    g1 = tl.load(
                        grad_out + (batch * N_POINTS + center[None, :]) * out_total + ext1[:, None],
                        mask=pair_mask,
                        other=0.0,
                    )
                    out_order_index = (pair0 - 1) // 2
                    c = tl.load(
                        output_cos + edge[None, :] * MAX_OUT_ORDER + out_order_index[:, None],
                        mask=pair_mask,
                        other=1.0,
                    )
                    s = tl.load(
                        output_sin + edge[None, :] * MAX_OUT_ORDER + out_order_index[:, None],
                        mask=pair_mask,
                        other=0.0,
                    )
                    first = c * g0 + s * g1
                    second = -s * g0 + c * g1
                    g = tl.where(is_vector[:, None], tl.where((out_component & 1)[:, None] == 1, first, second), g_direct)
                    if NORMALIZE:
                        denom = tl.maximum(tl.load(neighbor_count + center, mask=e_mask, other=1.0), 1.0)
                        g /= denom[None, :]
                    w0 = tl.load(
                        weight
                        + radial * stride_wr
                        + out_channel[None, :] * stride_wo
                        + out_component[None, :] * stride_wa
                        + in_channels[:, None] * stride_wi
                        + in_component0 * stride_wd,
                        mask=in_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    ).to(g.dtype)
                    if ALLOW_TF32:
                        raw0 += tl.dot(w0, g, input_precision="tf32")
                    else:
                        raw0 += tl.dot(w0, g, input_precision="ieee")
                    if in_order > 0:
                        w1 = tl.load(
                            weight
                            + radial * stride_wr
                            + out_channel[None, :] * stride_wo
                            + out_component[None, :] * stride_wa
                            + in_channels[:, None] * stride_wi
                            + in_component1 * stride_wd,
                            mask=in_mask[:, None] & k_mask[None, :],
                            other=0.0,
                        ).to(g.dtype)
                        if ALLOW_TF32:
                            raw1 += tl.dot(w1, g, input_precision="tf32")
                        else:
                            raw1 += tl.dot(w1, g, input_precision="ieee")
                    k_base += BLOCK_K
                edge_grad0 += raw0 * radial_value[None, :]
                edge_grad1 += raw1 * radial_value[None, :]
            if in_order == 0:
                total0 += tl.sum(edge_grad0, axis=1)
            else:
                c = tl.load(input_cos + edge * MAX_IN_ORDER + in_order - 1, mask=e_mask, other=1.0)
                s = tl.load(input_sin + edge * MAX_IN_ORDER + in_order - 1, mask=e_mask, other=0.0)
                total0 += tl.sum(c[None, :] * edge_grad0 + s[None, :] * edge_grad1, axis=1)
                total1 += tl.sum(-s[None, :] * edge_grad0 + c[None, :] * edge_grad1, axis=1)
            pos += BLOCK_E
        packed0 = in_channels * IN_DIM + in_component0
        ext0 = tl.load(input_pack + packed0, mask=in_mask, other=0)
        x_total: tl.constexpr = IN_M * IN_DIM
        base = (batch * N_POINTS + point) * x_total
        tl.store(grad_x + base + ext0, total0, mask=in_mask)
        if in_order > 0:
            ext1 = tl.load(input_pack + packed0 + 1, mask=in_mask, other=0)
            tl.store(grad_x + base + ext1, total1, mask=in_mask)


    @triton.jit
    def _packed_grad_input_r1_kernel(
        grad_out,
        weight,
        center_idx,
        neighbor_ptr,
        edges_by_neighbor,
        radial_basis,
        input_cos,
        input_sin,
        output_cos,
        output_sin,
        input_pack,
        output_pack,
        neighbor_count,
        grad_x,
        stride_wr: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wa: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_wd: tl.constexpr,
        N_POINTS: tl.constexpr,
        IN_M: tl.constexpr,
        OUT_M: tl.constexpr,
        IN_DIM: tl.constexpr,
        OUT_DIM: tl.constexpr,
        MAX_IN_ORDER: tl.constexpr,
        MAX_OUT_ORDER: tl.constexpr,
        BLOCK_ORDERS: tl.constexpr,
        DEGREE_BUCKET: tl.constexpr,
        NORMALIZE: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        in_blocks = tl.cdiv(IN_M, BLOCK_I)
        pid = tl.program_id(0)
        in_block = pid % in_blocks
        pid //= in_blocks
        point = pid % N_POINTS
        batch = pid // N_POINTS

        pair_count: tl.constexpr = BLOCK_I * BLOCK_ORDERS
        row_count: tl.constexpr = pair_count * 2
        row = tl.arange(0, row_count)
        pair = row // 2
        pair_component = row - pair * 2
        local_in_channel = pair // BLOCK_ORDERS
        in_order = pair - local_in_channel * BLOCK_ORDERS
        in_channel = in_block * BLOCK_I + local_in_channel
        in_component = tl.where(in_order == 0, pair_component, 2 * in_order - 1 + pair_component)
        row_mask = (
            (in_channel < IN_M)
            & (in_order <= MAX_IN_ORDER)
            & ((in_order > 0) | (pair_component == 0))
        )
        totals = tl.zeros((row_count,), tl.float32)
        pos = tl.load(neighbor_ptr + point)
        pos_stop = tl.load(neighbor_ptr + point + 1)
        out_total: tl.constexpr = OUT_M * OUT_DIM

        while pos < pos_stop:
            edge_offsets = tl.arange(0, BLOCK_E)
            positions = pos + edge_offsets
            edge_mask = positions < pos_stop
            edge = tl.load(edges_by_neighbor + positions, mask=edge_mask, other=0)
            center = tl.load(center_idx + edge, mask=edge_mask, other=0)
            radial_value = tl.load(radial_basis + edge, mask=edge_mask, other=0.0)
            if NORMALIZE:
                denom = tl.maximum(
                    tl.load(neighbor_count + center, mask=edge_mask, other=1.0),
                    1.0,
                )
            raw = tl.zeros((BLOCK_E, row_count), tl.float32)
            k_base = 0
            while k_base < out_total:
                k_offsets = tl.arange(0, BLOCK_K)
                packed_out = k_base + k_offsets
                k_mask = packed_out < out_total
                out_channel = packed_out // OUT_DIM
                out_component = packed_out - out_channel * OUT_DIM
                external = tl.load(output_pack + packed_out, mask=k_mask, other=0)
                g_direct = tl.load(
                    grad_out
                    + (batch * N_POINTS + center[None, :]) * out_total
                    + external[:, None],
                    mask=k_mask[:, None] & edge_mask[None, :],
                    other=0.0,
                )
                is_vector = out_component > 0
                pair0 = tl.where((out_component & 1) == 1, out_component, out_component - 1)
                pair_packed = out_channel * OUT_DIM + pair0
                ext0 = tl.load(output_pack + pair_packed, mask=k_mask & is_vector, other=0)
                ext1 = tl.load(output_pack + pair_packed + 1, mask=k_mask & is_vector, other=0)
                vector_mask = k_mask[:, None] & is_vector[:, None] & edge_mask[None, :]
                g0 = tl.load(
                    grad_out
                    + (batch * N_POINTS + center[None, :]) * out_total
                    + ext0[:, None],
                    mask=vector_mask,
                    other=0.0,
                )
                g1 = tl.load(
                    grad_out
                    + (batch * N_POINTS + center[None, :]) * out_total
                    + ext1[:, None],
                    mask=vector_mask,
                    other=0.0,
                )
                out_order_index = (pair0 - 1) // 2
                output_cos_value = tl.load(
                    output_cos + edge[None, :] * MAX_OUT_ORDER + out_order_index[:, None],
                    mask=vector_mask,
                    other=1.0,
                )
                output_sin_value = tl.load(
                    output_sin + edge[None, :] * MAX_OUT_ORDER + out_order_index[:, None],
                    mask=vector_mask,
                    other=0.0,
                )
                first = output_cos_value * g0 + output_sin_value * g1
                second = -output_sin_value * g0 + output_cos_value * g1
                g = tl.where(
                    is_vector[:, None],
                    tl.where((out_component & 1)[:, None] == 1, first, second),
                    g_direct,
                )
                if NORMALIZE:
                    g /= denom[None, :]
                g *= radial_value[None, :]
                weight_ptr = (
                    weight
                    + out_channel[:, None] * stride_wo
                    + out_component[:, None] * stride_wa
                    + in_channel[None, :] * stride_wi
                    + in_component[None, :] * stride_wd
                )
                w = tl.load(
                    weight_ptr,
                    mask=k_mask[:, None] & row_mask[None, :],
                    other=0.0,
                ).to(g.dtype)
                if ALLOW_TF32:
                    raw += tl.dot(tl.trans(g), w, input_precision="tf32")
                else:
                    raw += tl.dot(tl.trans(g), w, input_precision="ieee")
                k_base += BLOCK_K

            paired = tl.reshape(raw, (BLOCK_E, pair_count, 2))
            raw0, raw1 = tl.split(paired)
            pair_offsets = tl.arange(0, pair_count)
            pair_local_in = pair_offsets // BLOCK_ORDERS
            pair_order = pair_offsets - pair_local_in * BLOCK_ORDERS
            rotation_mask = edge_mask[:, None] & (pair_order[None, :] > 0) & (
                pair_order[None, :] <= MAX_IN_ORDER
            )
            input_cos_value = tl.load(
                input_cos + edge[:, None] * MAX_IN_ORDER + pair_order[None, :] - 1,
                mask=rotation_mask,
                other=1.0,
            )
            input_sin_value = tl.load(
                input_sin + edge[:, None] * MAX_IN_ORDER + pair_order[None, :] - 1,
                mask=rotation_mask,
                other=0.0,
            )
            rotated0 = input_cos_value * raw0 + input_sin_value * raw1
            rotated1 = -input_sin_value * raw0 + input_cos_value * raw1
            rotated = tl.join(rotated0, rotated1)
            totals += tl.sum(tl.reshape(rotated, (BLOCK_E, row_count)), axis=0)
            pos += BLOCK_E

        pair_totals = tl.reshape(totals, (pair_count, 2))
        total0, total1 = tl.split(pair_totals)
        pair_offsets = tl.arange(0, pair_count)
        pair_local_in = pair_offsets // BLOCK_ORDERS
        pair_order = pair_offsets - pair_local_in * BLOCK_ORDERS
        pair_in_channel = in_block * BLOCK_I + pair_local_in
        pair_mask = (pair_in_channel < IN_M) & (pair_order <= MAX_IN_ORDER)
        component0 = tl.where(pair_order == 0, 0, 2 * pair_order - 1)
        packed0 = pair_in_channel * IN_DIM + component0
        external0 = tl.load(input_pack + packed0, mask=pair_mask, other=0)
        x_total: tl.constexpr = IN_M * IN_DIM
        base = (batch * N_POINTS + point) * x_total
        tl.store(grad_x + base + external0, total0, mask=pair_mask)
        vector_mask = pair_mask & (pair_order > 0)
        external1 = tl.load(input_pack + packed0 + 1, mask=vector_mask, other=0)
        tl.store(grad_x + base + external1, total1, mask=vector_mask)


    @triton.jit
    def _packed_grad_weight_kernel(
        x,
        grad_out,
        center_idx,
        neighbor_idx,
        radial_basis,
        input_cos,
        input_sin,
        output_cos,
        output_sin,
        input_pack,
        output_pack,
        neighbor_count,
        grad_weight,
        stride_xb: tl.constexpr,
        stride_xn: tl.constexpr,
        stride_xc: tl.constexpr,
        BATCH: tl.constexpr,
        N_POINTS: tl.constexpr,
        N_EDGES: tl.constexpr,
        IN_M: tl.constexpr,
        OUT_M: tl.constexpr,
        IN_DIM: tl.constexpr,
        OUT_DIM: tl.constexpr,
        NUM_RADIAL: tl.constexpr,
        MAX_IN_ORDER: tl.constexpr,
        MAX_OUT_ORDER: tl.constexpr,
        NORMALIZE: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_O: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        out_blocks = tl.cdiv(OUT_M, BLOCK_O)
        in_blocks = tl.cdiv(IN_M, BLOCK_I)
        pid = tl.program_id(0)
        in_block = pid % in_blocks
        pid //= in_blocks
        out_block = pid % out_blocks
        pid //= out_blocks
        in_component = pid % IN_DIM
        pid //= IN_DIM
        out_component = pid % OUT_DIM
        radial = pid // OUT_DIM
        in_channels = in_block * BLOCK_I + tl.arange(0, BLOCK_I)
        out_channels = out_block * BLOCK_O + tl.arange(0, BLOCK_O)
        in_mask = in_channels < IN_M
        out_mask = out_channels < OUT_M
        acc = tl.zeros((BLOCK_O, BLOCK_I), tl.float32)
        sample_total: tl.constexpr = BATCH * N_EDGES
        sample_base = 0
        while sample_base < sample_total:
            sample = sample_base + tl.arange(0, BLOCK_S)
            sample_mask = sample < sample_total
            batch = sample // N_EDGES
            edge = sample - batch * N_EDGES
            center = tl.load(center_idx + edge, mask=sample_mask, other=0)
            neighbor = tl.load(neighbor_idx + edge, mask=sample_mask, other=0)

            packed_in = in_channels * IN_DIM + in_component
            ext_in = tl.load(input_pack + packed_in, mask=in_mask, other=0)
            x_direct = tl.load(
                x + batch[:, None] * stride_xb + neighbor[:, None] * stride_xn + ext_in[None, :] * stride_xc,
                mask=sample_mask[:, None] & in_mask[None, :],
                other=0.0,
            )
            input_is_vector = in_component > 0
            pair0 = tl.where((in_component & 1) == 1, in_component, in_component - 1)
            pair_input_mask = in_mask & input_is_vector
            ext0 = tl.load(input_pack + in_channels * IN_DIM + pair0, mask=pair_input_mask, other=0)
            ext1 = tl.load(input_pack + in_channels * IN_DIM + pair0 + 1, mask=pair_input_mask, other=0)
            x0 = tl.load(x + batch[:, None] * stride_xb + neighbor[:, None] * stride_xn + ext0[None, :] * stride_xc, mask=sample_mask[:, None] & pair_input_mask[None, :], other=0.0)
            x1 = tl.load(x + batch[:, None] * stride_xb + neighbor[:, None] * stride_xn + ext1[None, :] * stride_xc, mask=sample_mask[:, None] & pair_input_mask[None, :], other=0.0)
            order = (pair0 + 1) // 2
            c = tl.load(input_cos + edge * MAX_IN_ORDER + order - 1, mask=sample_mask & input_is_vector, other=1.0)
            s = tl.load(input_sin + edge * MAX_IN_ORDER + order - 1, mask=sample_mask & input_is_vector, other=0.0)
            first_x = c[:, None] * x0 - s[:, None] * x1
            second_x = s[:, None] * x0 + c[:, None] * x1
            vector_x = tl.where((in_component & 1) == 1, first_x, second_x)
            x_value = tl.where(input_is_vector, vector_x, x_direct)

            packed_out = out_channels * OUT_DIM + out_component
            ext_out = tl.load(output_pack + packed_out, mask=out_mask, other=0)
            g_direct = tl.load(
                grad_out + (batch[:, None] * N_POINTS + center[:, None]) * (OUT_M * OUT_DIM) + ext_out[None, :],
                mask=sample_mask[:, None] & out_mask[None, :],
                other=0.0,
            )
            output_is_vector = out_component > 0
            pair0_out = tl.where((out_component & 1) == 1, out_component, out_component - 1)
            pair_output_mask = out_mask & output_is_vector
            ext0_out = tl.load(output_pack + out_channels * OUT_DIM + pair0_out, mask=pair_output_mask, other=0)
            ext1_out = tl.load(output_pack + out_channels * OUT_DIM + pair0_out + 1, mask=pair_output_mask, other=0)
            g0 = tl.load(grad_out + (batch[:, None] * N_POINTS + center[:, None]) * (OUT_M * OUT_DIM) + ext0_out[None, :], mask=sample_mask[:, None] & pair_output_mask[None, :], other=0.0)
            g1 = tl.load(grad_out + (batch[:, None] * N_POINTS + center[:, None]) * (OUT_M * OUT_DIM) + ext1_out[None, :], mask=sample_mask[:, None] & pair_output_mask[None, :], other=0.0)
            out_order = (pair0_out + 1) // 2
            c = tl.load(output_cos + edge * MAX_OUT_ORDER + out_order - 1, mask=sample_mask & output_is_vector, other=1.0)
            s = tl.load(output_sin + edge * MAX_OUT_ORDER + out_order - 1, mask=sample_mask & output_is_vector, other=0.0)
            first_g = c[:, None] * g0 + s[:, None] * g1
            second_g = -s[:, None] * g0 + c[:, None] * g1
            vector_g = tl.where((out_component & 1) == 1, first_g, second_g)
            g_value = tl.where(output_is_vector, vector_g, g_direct)
            if NORMALIZE:
                denom = tl.maximum(tl.load(neighbor_count + center, mask=sample_mask, other=1.0), 1.0)
                g_value /= denom[:, None]
            beta = tl.load(radial_basis + edge * NUM_RADIAL + radial, mask=sample_mask, other=0.0)
            left = (g_value * beta[:, None]).to(x_value.dtype)
            if ALLOW_TF32:
                acc += tl.dot(tl.trans(left), x_value, input_precision="tf32")
            else:
                acc += tl.dot(tl.trans(left), x_value, input_precision="ieee")
            sample_base += BLOCK_S
        offset = (((radial * OUT_M + out_channels[:, None]) * OUT_DIM + out_component) * IN_M + in_channels[None, :]) * IN_DIM + in_component
        tl.store(grad_weight + offset, acc, mask=out_mask[:, None] & in_mask[None, :])


    @triton.jit
    def _packed_grad_weight_r1_partial_kernel(
        x,
        grad_out,
        center_idx,
        neighbor_idx,
        radial_basis,
        input_cos,
        input_sin,
        output_cos,
        output_sin,
        input_pack,
        output_pack,
        neighbor_count,
        partial_weight,
        stride_xb: tl.constexpr,
        stride_xn: tl.constexpr,
        stride_xc: tl.constexpr,
        BATCH: tl.constexpr,
        N_POINTS: tl.constexpr,
        N_EDGES: tl.constexpr,
        IN_M: tl.constexpr,
        OUT_M: tl.constexpr,
        IN_DIM: tl.constexpr,
        OUT_DIM: tl.constexpr,
        MAX_IN_ORDER: tl.constexpr,
        MAX_OUT_ORDER: tl.constexpr,
        PARTIALS: tl.constexpr,
        DEGREE_BUCKET: tl.constexpr,
        NORMALIZE: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        BLOCK_O: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        out_blocks = tl.cdiv(OUT_M, BLOCK_O)
        in_blocks = tl.cdiv(IN_M, BLOCK_I)
        pid = tl.program_id(0)
        partial = pid % PARTIALS
        pid //= PARTIALS
        in_block = pid % in_blocks
        pid //= in_blocks
        out_block = pid % out_blocks
        pid //= out_blocks
        in_component = pid % IN_DIM
        out_component = pid // IN_DIM
        in_channels = in_block * BLOCK_I + tl.arange(0, BLOCK_I)
        out_channels = out_block * BLOCK_O + tl.arange(0, BLOCK_O)
        in_mask = in_channels < IN_M
        out_mask = out_channels < OUT_M
        acc = tl.zeros((BLOCK_O, BLOCK_I), tl.float32)
        sample_total: tl.constexpr = BATCH * N_EDGES
        chunk_size: tl.constexpr = tl.cdiv(sample_total, PARTIALS)
        sample_base = partial * chunk_size
        sample_stop = tl.minimum(sample_base + chunk_size, sample_total)
        while sample_base < sample_stop:
            sample = sample_base + tl.arange(0, BLOCK_S)
            sample_mask = sample < sample_stop
            batch = sample // N_EDGES
            edge = sample - batch * N_EDGES
            center = tl.load(center_idx + edge, mask=sample_mask, other=0)
            neighbor = tl.load(neighbor_idx + edge, mask=sample_mask, other=0)

            packed_in = in_channels * IN_DIM + in_component
            ext_in = tl.load(input_pack + packed_in, mask=in_mask, other=0)
            x_direct = tl.load(
                x
                + batch[:, None] * stride_xb
                + neighbor[:, None] * stride_xn
                + ext_in[None, :] * stride_xc,
                mask=sample_mask[:, None] & in_mask[None, :],
                other=0.0,
            )
            input_is_vector = in_component > 0
            pair0 = tl.where((in_component & 1) == 1, in_component, in_component - 1)
            pair_input_mask = in_mask & input_is_vector
            ext0 = tl.load(
                input_pack + in_channels * IN_DIM + pair0,
                mask=pair_input_mask,
                other=0,
            )
            ext1 = tl.load(
                input_pack + in_channels * IN_DIM + pair0 + 1,
                mask=pair_input_mask,
                other=0,
            )
            x0 = tl.load(
                x
                + batch[:, None] * stride_xb
                + neighbor[:, None] * stride_xn
                + ext0[None, :] * stride_xc,
                mask=sample_mask[:, None] & pair_input_mask[None, :],
                other=0.0,
            )
            x1 = tl.load(
                x
                + batch[:, None] * stride_xb
                + neighbor[:, None] * stride_xn
                + ext1[None, :] * stride_xc,
                mask=sample_mask[:, None] & pair_input_mask[None, :],
                other=0.0,
            )
            order = (pair0 + 1) // 2
            input_cos_value = tl.load(
                input_cos + edge * MAX_IN_ORDER + order - 1,
                mask=sample_mask & input_is_vector,
                other=1.0,
            )
            input_sin_value = tl.load(
                input_sin + edge * MAX_IN_ORDER + order - 1,
                mask=sample_mask & input_is_vector,
                other=0.0,
            )
            first_x = input_cos_value[:, None] * x0 - input_sin_value[:, None] * x1
            second_x = input_sin_value[:, None] * x0 + input_cos_value[:, None] * x1
            vector_x = tl.where((in_component & 1) == 1, first_x, second_x)
            x_value = tl.where(input_is_vector, vector_x, x_direct)

            packed_out = out_channels * OUT_DIM + out_component
            ext_out = tl.load(output_pack + packed_out, mask=out_mask, other=0)
            g_direct = tl.load(
                grad_out
                + (batch[:, None] * N_POINTS + center[:, None]) * (OUT_M * OUT_DIM)
                + ext_out[None, :],
                mask=sample_mask[:, None] & out_mask[None, :],
                other=0.0,
            )
            output_is_vector = out_component > 0
            pair0_out = tl.where((out_component & 1) == 1, out_component, out_component - 1)
            pair_output_mask = out_mask & output_is_vector
            ext0_out = tl.load(
                output_pack + out_channels * OUT_DIM + pair0_out,
                mask=pair_output_mask,
                other=0,
            )
            ext1_out = tl.load(
                output_pack + out_channels * OUT_DIM + pair0_out + 1,
                mask=pair_output_mask,
                other=0,
            )
            g0 = tl.load(
                grad_out
                + (batch[:, None] * N_POINTS + center[:, None]) * (OUT_M * OUT_DIM)
                + ext0_out[None, :],
                mask=sample_mask[:, None] & pair_output_mask[None, :],
                other=0.0,
            )
            g1 = tl.load(
                grad_out
                + (batch[:, None] * N_POINTS + center[:, None]) * (OUT_M * OUT_DIM)
                + ext1_out[None, :],
                mask=sample_mask[:, None] & pair_output_mask[None, :],
                other=0.0,
            )
            out_order = (pair0_out + 1) // 2
            output_cos_value = tl.load(
                output_cos + edge * MAX_OUT_ORDER + out_order - 1,
                mask=sample_mask & output_is_vector,
                other=1.0,
            )
            output_sin_value = tl.load(
                output_sin + edge * MAX_OUT_ORDER + out_order - 1,
                mask=sample_mask & output_is_vector,
                other=0.0,
            )
            first_g = output_cos_value[:, None] * g0 + output_sin_value[:, None] * g1
            second_g = -output_sin_value[:, None] * g0 + output_cos_value[:, None] * g1
            vector_g = tl.where((out_component & 1) == 1, first_g, second_g)
            g_value = tl.where(output_is_vector, vector_g, g_direct)
            if NORMALIZE:
                denom = tl.maximum(
                    tl.load(neighbor_count + center, mask=sample_mask, other=1.0),
                    1.0,
                )
                g_value /= denom[:, None]
            beta = tl.load(radial_basis + edge, mask=sample_mask, other=0.0)
            left = (g_value * beta[:, None]).to(x_value.dtype)
            if ALLOW_TF32:
                acc += tl.dot(tl.trans(left), x_value, input_precision="tf32")
            else:
                acc += tl.dot(tl.trans(left), x_value, input_precision="ieee")
            sample_base += BLOCK_S

        weight_numel: tl.constexpr = OUT_M * OUT_DIM * IN_M * IN_DIM
        offset = (
            ((out_channels[:, None] * OUT_DIM + out_component) * IN_M + in_channels[None, :])
            * IN_DIM
            + in_component
        )
        tl.store(
            partial_weight + partial * weight_numel + offset,
            acc,
            mask=out_mask[:, None] & in_mask[None, :],
        )


    @triton.jit
    def _reduce_grad_weight_partials_kernel(
        partial_weight,
        grad_weight,
        WEIGHT_NUMEL: tl.constexpr,
        PARTIALS: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        weight_offset = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
        weight_mask = weight_offset < WEIGHT_NUMEL
        acc = tl.zeros((BLOCK_W,), tl.float32)
        partial_base = 0
        while partial_base < PARTIALS:
            partial = partial_base + tl.arange(0, BLOCK_P)
            partial_mask = partial < PARTIALS
            values = tl.load(
                partial_weight + partial[:, None] * WEIGHT_NUMEL + weight_offset[None, :],
                mask=partial_mask[:, None] & weight_mask[None, :],
                other=0.0,
            )
            acc += tl.sum(values, axis=0)
            partial_base += BLOCK_P
        tl.store(grad_weight + weight_offset, acc, mask=weight_mask)


    _FORWARD_AUTOTUNE_KEY = ["IN_M", "OUT_M", "IN_DIM", "OUT_DIM", "NUM_RADIAL"]
    _BACKWARD_AUTOTUNE_KEY = ["IN_M", "OUT_M", "IN_DIM", "OUT_DIM", "NUM_RADIAL"]
    _GRAD_WEIGHT_AUTOTUNE_KEY = [
        "BATCH",
        "N_EDGES",
        "IN_M",
        "OUT_M",
        "IN_DIM",
        "OUT_DIM",
        "NUM_RADIAL",
    ]
    _R1_AUTOTUNE_KEY = ["IN_M", "OUT_M", "IN_DIM", "OUT_DIM", "DEGREE_BUCKET"]
    _R1_GRAD_WEIGHT_AUTOTUNE_KEY = [
        "BATCH",
        "N_EDGES",
        "IN_M",
        "OUT_M",
        "IN_DIM",
        "OUT_DIM",
        "PARTIALS",
        "DEGREE_BUCKET",
    ]

    def _r1_configs(tile_name: str, tiles: tuple[int, ...], *, sm80: bool) -> list:
        configs = []
        edge_tiles = (16, 32) if sm80 else (16,)
        for tile in tiles:
            for block_e in edge_tiles:
                configs.append(
                    triton.Config(
                        {tile_name: tile, "BLOCK_E": block_e, "BLOCK_K": 32},
                        num_warps=4,
                    )
                )
            configs.append(
                triton.Config(
                    {tile_name: tile, "BLOCK_E": edge_tiles[-1], "BLOCK_K": 64},
                    num_warps=4,
                )
            )
        return configs

    _R1_TILE_FAMILIES = {
        "scalar": (8,),
        "low": (4, 8),
        "medium": (2, 4),
        "high": (1, 2),
    }
    _packed_forward_r1_kernels = {
        (architecture, family): triton.autotune(
            configs=_r1_configs("BLOCK_O", tiles, sm80=architecture == 8),
            key=_R1_AUTOTUNE_KEY,
        )(_packed_forward_r1_kernel)
        for architecture in (7, 8)
        for family, tiles in _R1_TILE_FAMILIES.items()
    }
    _packed_grad_input_r1_kernels = {
        (architecture, family): triton.autotune(
            configs=_r1_configs("BLOCK_I", tiles, sm80=architecture == 8),
            key=_R1_AUTOTUNE_KEY,
        )(_packed_grad_input_r1_kernel)
        for architecture in (7, 8)
        for family, tiles in _R1_TILE_FAMILIES.items()
    }
    _packed_grad_weight_r1_partial_sm70 = triton.autotune(
        configs=[
            triton.Config({"BLOCK_O": 16, "BLOCK_I": 16, "BLOCK_S": 16}, num_warps=4),
            triton.Config({"BLOCK_O": 16, "BLOCK_I": 16, "BLOCK_S": 32}, num_warps=4),
        ],
        key=_R1_GRAD_WEIGHT_AUTOTUNE_KEY,
    )(_packed_grad_weight_r1_partial_kernel)
    _packed_grad_weight_r1_partial_sm80 = triton.autotune(
        configs=[
            triton.Config({"BLOCK_O": 16, "BLOCK_I": 16, "BLOCK_S": 16}, num_warps=4),
            triton.Config({"BLOCK_O": 16, "BLOCK_I": 16, "BLOCK_S": 32}, num_warps=4),
            triton.Config({"BLOCK_O": 16, "BLOCK_I": 16, "BLOCK_S": 64}, num_warps=4),
        ],
        key=_R1_GRAD_WEIGHT_AUTOTUNE_KEY,
    )(_packed_grad_weight_r1_partial_kernel)
    _packed_forward_kernel_sm70 = triton.autotune(
        configs=[
            # tl.dot on SM70 requires every matrix dimension to be at least 16.
            triton.Config({"BLOCK_O": 16, "BLOCK_E": 16, "BLOCK_K": 32}, num_warps=4),
        ],
        key=_FORWARD_AUTOTUNE_KEY,
    )(_packed_forward_kernel)
    _packed_forward_kernel_sm80 = triton.autotune(
        configs=[
            triton.Config({"BLOCK_O": 16, "BLOCK_E": 16, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_O": 16, "BLOCK_E": 32, "BLOCK_K": 32}, num_warps=4),
        ],
        key=_FORWARD_AUTOTUNE_KEY,
    )(_packed_forward_kernel)
    _packed_grad_input_kernel_sm70 = triton.autotune(
        configs=[
            # BLOCK_E is the N dimension of tl.dot and must be at least 16 on SM70.
            triton.Config({"BLOCK_I": 16, "BLOCK_E": 16, "BLOCK_K": 32}, num_warps=4),
        ],
        key=_BACKWARD_AUTOTUNE_KEY,
    )(_packed_grad_input_kernel)
    _packed_grad_input_kernel_sm80 = triton.autotune(
        configs=[
            triton.Config({"BLOCK_I": 16, "BLOCK_E": 16, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_I": 16, "BLOCK_E": 32, "BLOCK_K": 32}, num_warps=4),
        ],
        key=_BACKWARD_AUTOTUNE_KEY,
    )(_packed_grad_input_kernel)
    _packed_grad_weight_kernel_sm70 = triton.autotune(
        configs=[
            triton.Config({"BLOCK_O": 16, "BLOCK_I": 16, "BLOCK_S": 16}, num_warps=4),
            triton.Config({"BLOCK_O": 16, "BLOCK_I": 16, "BLOCK_S": 32}, num_warps=4),
        ],
        key=_GRAD_WEIGHT_AUTOTUNE_KEY,
    )(_packed_grad_weight_kernel)
    _packed_grad_weight_kernel_sm80 = triton.autotune(
        configs=[
            triton.Config({"BLOCK_O": 16, "BLOCK_I": 16, "BLOCK_S": 16}, num_warps=4),
            triton.Config({"BLOCK_O": 16, "BLOCK_I": 16, "BLOCK_S": 32}, num_warps=4),
            triton.Config({"BLOCK_O": 16, "BLOCK_I": 16, "BLOCK_S": 64}, num_warps=4),
        ],
        key=_GRAD_WEIGHT_AUTOTUNE_KEY,
    )(_packed_grad_weight_kernel)


    def _launch_meta(x: Tensor, weight: Tensor) -> tuple[int, ...]:
        radial, out_m, out_dim, in_m, in_dim = map(int, weight.shape)
        return radial, out_m, out_dim, in_m, in_dim, int(x.shape[0]), int(x.shape[1])


    def _r1_tile_family(component_dim: int) -> str:
        if component_dim == 1:
            return "scalar"
        if component_dim <= 5:
            return "low"
        if component_dim <= 9:
            return "medium"
        return "high"


    @triton_op("kpconv_intrinsic::packed_irrep_conv", mutates_args={})
    def packed_irrep_conv(
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
        degree_bucket: int,
        normalize: bool,
        allow_tf32: bool,
    ) -> Tensor:
        radial, out_m, out_dim, in_m, in_dim, batch, n_points = _launch_meta(x, weight)
        out = torch.empty((batch, n_points, out_m * out_dim), device=x.device, dtype=x.dtype)
        max_in_order = (in_dim - 1) // 2
        max_out_order = (out_dim - 1) // 2
        major = torch.cuda.get_device_capability(x.device)[0]
        if radial == 1:
            block_orders = triton.next_power_of_2(max_out_order + 1)
            architecture = 7 if major < 8 else 8
            kernel = _packed_forward_r1_kernels[(architecture, _r1_tile_family(out_dim))]
            grid = lambda meta: (
                batch * n_points * triton.cdiv(out_m, meta["BLOCK_O"]),
            )
            wrap_triton(kernel)[grid](
                x, weight, neighbor_idx, center_ptr, radial_basis, input_cos, input_sin,
                output_cos, output_sin, input_pack, output_pack, neighbor_count, out,
                x.stride(0), x.stride(1), x.stride(2), *weight.stride(),
                N_POINTS=n_points, IN_M=in_m, OUT_M=out_m, IN_DIM=in_dim, OUT_DIM=out_dim,
                MAX_IN_ORDER=max_in_order, MAX_OUT_ORDER=max_out_order,
                BLOCK_ORDERS=block_orders, DEGREE_BUCKET=degree_bucket,
                NORMALIZE=normalize, ALLOW_TF32=allow_tf32,
            )
            return out
        kernel = _packed_forward_kernel_sm70 if major < 8 else _packed_forward_kernel_sm80
        grid = (batch * n_points * (max_out_order + 1) * triton.cdiv(out_m, 16),)
        wrap_triton(kernel)[grid](
            x, weight, neighbor_idx, center_ptr, radial_basis, input_cos, input_sin,
            output_cos, output_sin, input_pack, output_pack, neighbor_count, out,
            x.stride(0), x.stride(1), x.stride(2), *weight.stride(),
            N_POINTS=n_points, IN_M=in_m, OUT_M=out_m, IN_DIM=in_dim, OUT_DIM=out_dim,
            NUM_RADIAL=radial, MAX_IN_ORDER=max_in_order, MAX_OUT_ORDER=max_out_order,
            NORMALIZE=normalize, ALLOW_TF32=allow_tf32,
        )
        return out


    @triton_op("kpconv_intrinsic::packed_irrep_conv_backward", mutates_args={})
    def _packed_irrep_conv_backward(
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
        degree_bucket: int,
        normalize: bool,
        allow_tf32: bool,
    ) -> tuple[Tensor, Tensor]:
        del center_ptr
        radial, out_m, out_dim, in_m, in_dim, batch, n_points = _launch_meta(x, weight)
        max_in_order = (in_dim - 1) // 2
        max_out_order = (out_dim - 1) // 2
        major = torch.cuda.get_device_capability(x.device)[0]
        grad_input_kernel = (
            _packed_grad_input_kernel_sm70 if major < 8 else _packed_grad_input_kernel_sm80
        )
        grad_weight_kernel = (
            _packed_grad_weight_kernel_sm70 if major < 8 else _packed_grad_weight_kernel_sm80
        )
        grad_x = torch.empty_like(x, memory_format=torch.contiguous_format)
        grad_weight = torch.empty_like(weight, memory_format=torch.contiguous_format)
        if radial == 1:
            block_orders = triton.next_power_of_2(max_in_order + 1)
            architecture = 7 if major < 8 else 8
            kernel = _packed_grad_input_r1_kernels[(architecture, _r1_tile_family(in_dim))]
            grid_x = lambda meta: (
                batch * n_points * triton.cdiv(in_m, meta["BLOCK_I"]),
            )
            wrap_triton(kernel)[grid_x](
                grad_out, weight, center_idx, neighbor_ptr, edges_by_neighbor, radial_basis,
                input_cos, input_sin, output_cos, output_sin, input_pack, output_pack,
                neighbor_count, grad_x, *weight.stride(), N_POINTS=n_points, IN_M=in_m,
                OUT_M=out_m, IN_DIM=in_dim, OUT_DIM=out_dim,
                MAX_IN_ORDER=max_in_order, MAX_OUT_ORDER=max_out_order,
                BLOCK_ORDERS=block_orders, DEGREE_BUCKET=degree_bucket,
                NORMALIZE=normalize, ALLOW_TF32=allow_tf32,
            )
        else:
            grid_x = (batch * n_points * (max_in_order + 1) * triton.cdiv(in_m, 16),)
            wrap_triton(grad_input_kernel)[grid_x](
                grad_out, weight, center_idx, neighbor_ptr, edges_by_neighbor, radial_basis,
                input_cos, input_sin, output_cos, output_sin, input_pack, output_pack,
                neighbor_count, grad_x, *weight.stride(), N_POINTS=n_points, IN_M=in_m,
                OUT_M=out_m, IN_DIM=in_dim, OUT_DIM=out_dim, NUM_RADIAL=radial,
                MAX_IN_ORDER=max_in_order, MAX_OUT_ORDER=max_out_order, NORMALIZE=normalize,
                ALLOW_TF32=allow_tf32,
            )
        n_edges = int(neighbor_idx.numel())
        if radial == 1:
            weight_numel = int(weight.numel())
            weight_tiles = out_dim * in_dim * triton.cdiv(out_m, 16) * triton.cdiv(in_m, 16)
            partials = _r1_grad_weight_partial_count(weight, batch=batch, n_edges=n_edges)
            partial_weight = (
                grad_weight
                if partials == 1
                else torch.empty(
                    (partials, *weight.shape),
                    device=weight.device,
                    dtype=torch.float32,
                )
            )
            grid_w = (partials * weight_tiles,)
            partial_kernel = (
                _packed_grad_weight_r1_partial_sm70
                if major < 8
                else _packed_grad_weight_r1_partial_sm80
            )
            wrap_triton(partial_kernel)[grid_w](
                x, grad_out, center_idx, neighbor_idx, radial_basis, input_cos, input_sin,
                output_cos, output_sin, input_pack, output_pack, neighbor_count,
                partial_weight, x.stride(0), x.stride(1), x.stride(2), BATCH=batch,
                N_POINTS=n_points, N_EDGES=n_edges, IN_M=in_m, OUT_M=out_m,
                IN_DIM=in_dim, OUT_DIM=out_dim, MAX_IN_ORDER=max_in_order,
                MAX_OUT_ORDER=max_out_order, PARTIALS=partials,
                DEGREE_BUCKET=degree_bucket, NORMALIZE=normalize,
                ALLOW_TF32=allow_tf32,
            )
            if partials > 1:
                grid_reduce = (triton.cdiv(weight_numel, 128),)
                wrap_triton(_reduce_grad_weight_partials_kernel)[grid_reduce](
                    partial_weight,
                    grad_weight,
                    WEIGHT_NUMEL=weight_numel,
                    PARTIALS=partials,
                    BLOCK_W=128,
                    BLOCK_P=16,
                    num_warps=4,
                )
        else:
            grid_w = (
                radial * out_dim * in_dim * triton.cdiv(out_m, 16) * triton.cdiv(in_m, 16),
            )
            wrap_triton(grad_weight_kernel)[grid_w](
                x, grad_out, center_idx, neighbor_idx, radial_basis, input_cos, input_sin,
                output_cos, output_sin, input_pack, output_pack, neighbor_count, grad_weight,
                x.stride(0), x.stride(1), x.stride(2), BATCH=batch, N_POINTS=n_points,
                N_EDGES=n_edges, IN_M=in_m, OUT_M=out_m, IN_DIM=in_dim, OUT_DIM=out_dim,
                NUM_RADIAL=radial, MAX_IN_ORDER=max_in_order, MAX_OUT_ORDER=max_out_order,
                NORMALIZE=normalize, ALLOW_TF32=allow_tf32,
            )
        return grad_x, grad_weight


    def _setup_context(ctx, inputs, output) -> None:
        del output
        tensors = inputs[:-3]
        ctx.save_for_backward(*tensors)
        ctx.degree_bucket = int(inputs[-3])
        ctx.normalize = bool(inputs[-2])
        ctx.allow_tf32 = bool(inputs[-1])


    def _backward(ctx, grad_out: Tensor):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "higher-order gradients are unsupported by the Triton irrep convolution; "
                "use backend='torch'"
            )
        grad_x, grad_weight = _packed_irrep_conv_backward(
            grad_out.contiguous(), *ctx.saved_tensors, ctx.degree_bucket,
            ctx.normalize, ctx.allow_tf32
        )
        return grad_x, grad_weight, *(None for _ in range(16))


    packed_irrep_conv.register_autograd(_backward, setup_context=_setup_context)

else:

    def packed_irrep_conv(*args, **kwargs):  # type: ignore[no-redef]
        del args, kwargs
        raise RuntimeError("Triton or torch.library.triton_op is unavailable")


__all__ = [
    "AUTO_INFERENCE_WORK_THRESHOLDS",
    "AUTO_TRAINING_WORK_THRESHOLDS",
    "AUTO_WORK_THRESHOLD",
    "MAX_MULTIPLICITY",
    "MAX_ORDER",
    "MAX_RADIAL",
    "R1_GRAD_WEIGHT_WORKSPACE_BYTES",
    "TRITON_AVAILABLE",
    "packed_irrep_conv",
    "triton_support_reason",
]
