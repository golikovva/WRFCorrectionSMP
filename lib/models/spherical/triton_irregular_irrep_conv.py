"""Memory-efficient Triton backend for one irregular irrep-order pair."""

from __future__ import annotations

import torch
from torch import Tensor

from .triton_irrep_conv import (
    MAX_MULTIPLICITY,
    MAX_ORDER,
    MAX_RADIAL,
    R1_GRAD_WEIGHT_WORKSPACE_BYTES,
    TRITON_AVAILABLE,
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


def irregular_pair_support_reason(
    x: Tensor,
    weight: Tensor,
    in_order: int,
    out_order: int,
) -> str | None:
    """Return why an irregular pair cannot use Triton, or ``None``."""

    if not TRITON_AVAILABLE or triton_op is None:
        return "Triton or torch.library.triton_op is unavailable"
    if not x.is_cuda:
        return "Triton backend requires CUDA tensors"
    if weight.device != x.device:
        return "input and irregular weights must be on the same CUDA device"
    if weight.dtype != x.dtype:
        return "input and irregular weights must have the same dtype"
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return f"unsupported input dtype {x.dtype}; expected float16, bfloat16, or float32"
    major, _minor = torch.cuda.get_device_capability(x.device)
    if major < 7:
        return "Triton backend requires compute capability 7.0 or newer"
    if x.dtype == torch.bfloat16 and major < 8:
        return "bfloat16 Triton path requires compute capability 8.0 or newer"
    if int(in_order) > MAX_ORDER or int(out_order) > MAX_ORDER:
        return f"maximum supported irrep order is {MAX_ORDER}"
    radial, out_m, in_m, out_dim, in_dim = map(int, weight.shape)
    if radial > MAX_RADIAL:
        return f"maximum supported radial count is {MAX_RADIAL}"
    if in_m > MAX_MULTIPLICITY or out_m > MAX_MULTIPLICITY:
        return f"maximum supported multiplicity is {MAX_MULTIPLICITY}"
    expected_in_dim = 1 if int(in_order) == 0 else 2
    expected_out_dim = 1 if int(out_order) == 0 else 2
    if in_dim != expected_in_dim or out_dim != expected_out_dim:
        return "weight component dimensions do not match the irrep orders"
    return None


def _grad_weight_partial_count(weight: Tensor, *, batch: int, n_edges: int) -> int:
    """Bound sample parallelism by the same 64 MiB budget as the packed path."""

    radial, out_m, in_m, out_dim, in_dim = map(int, weight.shape)
    tiles = radial * out_dim * in_dim * ((out_m + 15) // 16) * ((in_m + 15) // 16)
    target = torch.cuda.get_device_properties(weight.device).multi_processor_count * 8
    desired = max(1, (target + tiles - 1) // tiles)
    max_workspace = max(
        1,
        R1_GRAD_WEIGHT_WORKSPACE_BYTES // (int(weight.numel()) * torch.float32.itemsize),
    )
    sample_blocks = max(1, (batch * n_edges + 31) // 32)
    return min(desired, max_workspace, sample_blocks)


if TRITON_AVAILABLE and triton_op is not None:

    @triton.jit
    def _pair_forward_kernel(
        x,
        weight,
        neighbor_idx,
        center_ptr,
        radial_basis,
        input_cos,
        input_sin,
        output_cos,
        output_sin,
        neighbor_count,
        out,
        stride_xb: tl.constexpr,
        stride_xn: tl.constexpr,
        stride_xi: tl.constexpr,
        stride_xd: tl.constexpr,
        stride_wr: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_wa: tl.constexpr,
        stride_wd: tl.constexpr,
        N_POINTS: tl.constexpr,
        IN_M: tl.constexpr,
        OUT_M: tl.constexpr,
        NUM_RADIAL: tl.constexpr,
        MAX_IN_ORDER: tl.constexpr,
        MAX_OUT_ORDER: tl.constexpr,
        IN_ORDER: tl.constexpr,
        OUT_ORDER: tl.constexpr,
        NORMALIZE: tl.constexpr,
        BLOCK_O: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        out_blocks = tl.cdiv(OUT_M, BLOCK_O)
        pid = tl.program_id(0)
        out_block = pid % out_blocks
        pid //= out_blocks
        center = pid % N_POINTS
        batch = pid // N_POINTS
        out_channels = out_block * BLOCK_O + tl.arange(0, BLOCK_O)
        out_mask = out_channels < OUT_M
        total0 = tl.zeros((BLOCK_O,), tl.float32)
        total1 = tl.zeros((BLOCK_O,), tl.float32)
        edge_start = tl.load(center_ptr + center)
        edge_stop = tl.load(center_ptr + center + 1)
        edge_base = edge_start

        while edge_base < edge_stop:
            edge = edge_base + tl.arange(0, BLOCK_E)
            edge_mask = edge < edge_stop
            neighbor = tl.load(neighbor_idx + edge, mask=edge_mask, other=0)
            edge_total0 = tl.zeros((BLOCK_O, BLOCK_E), tl.float32)
            edge_total1 = tl.zeros((BLOCK_O, BLOCK_E), tl.float32)
            for radial in range(NUM_RADIAL):
                raw0 = tl.zeros((BLOCK_O, BLOCK_E), tl.float32)
                raw1 = tl.zeros((BLOCK_O, BLOCK_E), tl.float32)
                in_base = 0
                while in_base < IN_M:
                    in_channels = in_base + tl.arange(0, BLOCK_I)
                    in_mask = in_channels < IN_M
                    x0 = tl.load(
                        x
                        + batch * stride_xb
                        + neighbor[:, None] * stride_xn
                        + in_channels[None, :] * stride_xi,
                        mask=edge_mask[:, None] & in_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    if IN_ORDER == 0:
                        z0 = x0
                        z1 = x0
                    else:
                        x1 = tl.load(
                            x
                            + batch * stride_xb
                            + neighbor[:, None] * stride_xn
                            + in_channels[None, :] * stride_xi
                            + stride_xd,
                            mask=edge_mask[:, None] & in_mask[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        c_in = tl.load(
                            input_cos + edge * MAX_IN_ORDER + IN_ORDER - 1,
                            mask=edge_mask,
                            other=1.0,
                        )
                        s_in = tl.load(
                            input_sin + edge * MAX_IN_ORDER + IN_ORDER - 1,
                            mask=edge_mask,
                            other=0.0,
                        )
                        z0 = c_in[:, None] * x0 - s_in[:, None] * x1
                        z1 = s_in[:, None] * x0 + c_in[:, None] * x1
                    beta = tl.load(
                        radial_basis + edge * NUM_RADIAL + radial,
                        mask=edge_mask,
                        other=0.0,
                    ).to(tl.float32)
                    w00 = tl.load(
                        weight
                        + radial * stride_wr
                        + out_channels[:, None] * stride_wo
                        + in_channels[None, :] * stride_wi,
                        mask=out_mask[:, None] & in_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    raw0 += tl.sum(
                        w00[:, None, :] * z0[None, :, :], axis=2
                    ) * beta[None, :]
                    if IN_ORDER > 0:
                        w01 = tl.load(
                            weight
                            + radial * stride_wr
                            + out_channels[:, None] * stride_wo
                            + in_channels[None, :] * stride_wi
                            + stride_wd,
                            mask=out_mask[:, None] & in_mask[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        raw0 += tl.sum(
                            w01[:, None, :] * z1[None, :, :], axis=2
                        ) * beta[None, :]
                    if OUT_ORDER > 0:
                        w10 = tl.load(
                            weight
                            + radial * stride_wr
                            + out_channels[:, None] * stride_wo
                            + in_channels[None, :] * stride_wi
                            + stride_wa,
                            mask=out_mask[:, None] & in_mask[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        raw1 += tl.sum(
                            w10[:, None, :] * z0[None, :, :], axis=2
                        ) * beta[None, :]
                        if IN_ORDER > 0:
                            w11 = tl.load(
                                weight
                                + radial * stride_wr
                                + out_channels[:, None] * stride_wo
                                + in_channels[None, :] * stride_wi
                                + stride_wa
                                + stride_wd,
                                mask=out_mask[:, None] & in_mask[None, :],
                                other=0.0,
                            ).to(tl.float32)
                            raw1 += tl.sum(
                                w11[:, None, :] * z1[None, :, :], axis=2
                            ) * beta[None, :]
                    in_base += BLOCK_I
                edge_total0 += raw0
                edge_total1 += raw1
            if OUT_ORDER == 0:
                total0 += tl.sum(edge_total0, axis=1)
            else:
                c_out = tl.load(
                    output_cos + edge * MAX_OUT_ORDER + OUT_ORDER - 1,
                    mask=edge_mask,
                    other=1.0,
                )
                s_out = tl.load(
                    output_sin + edge * MAX_OUT_ORDER + OUT_ORDER - 1,
                    mask=edge_mask,
                    other=0.0,
                )
                total0 += tl.sum(edge_total0 * c_out[None, :] - edge_total1 * s_out[None, :], axis=1)
                total1 += tl.sum(edge_total0 * s_out[None, :] + edge_total1 * c_out[None, :], axis=1)
            edge_base += BLOCK_E

        if NORMALIZE:
            scale = tl.maximum(tl.load(neighbor_count + center), 1.0)
            total0 /= scale
            total1 /= scale
        out_dim: tl.constexpr = 1 if OUT_ORDER == 0 else 2
        out_base = (batch * N_POINTS + center) * (OUT_M * out_dim)
        tl.store(out + out_base + out_channels * out_dim, total0, mask=out_mask)
        if OUT_ORDER > 0:
            tl.store(out + out_base + out_channels * out_dim + 1, total1, mask=out_mask)


    @triton.jit
    def _pair_grad_input_kernel(
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
        neighbor_count,
        grad_x,
        stride_wr: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_wa: tl.constexpr,
        stride_wd: tl.constexpr,
        N_POINTS: tl.constexpr,
        IN_M: tl.constexpr,
        OUT_M: tl.constexpr,
        NUM_RADIAL: tl.constexpr,
        MAX_IN_ORDER: tl.constexpr,
        MAX_OUT_ORDER: tl.constexpr,
        IN_ORDER: tl.constexpr,
        OUT_ORDER: tl.constexpr,
        NORMALIZE: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_O: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        in_blocks = tl.cdiv(IN_M, BLOCK_I)
        pid = tl.program_id(0)
        in_block = pid % in_blocks
        pid //= in_blocks
        point = pid % N_POINTS
        batch = pid // N_POINTS
        in_channels = in_block * BLOCK_I + tl.arange(0, BLOCK_I)
        in_mask = in_channels < IN_M
        total0 = tl.zeros((BLOCK_I,), tl.float32)
        total1 = tl.zeros((BLOCK_I,), tl.float32)
        pos = tl.load(neighbor_ptr + point)
        pos_stop = tl.load(neighbor_ptr + point + 1)
        out_dim: tl.constexpr = 1 if OUT_ORDER == 0 else 2

        while pos < pos_stop:
            positions = pos + tl.arange(0, BLOCK_E)
            edge_mask = positions < pos_stop
            edge = tl.load(edges_by_neighbor + positions, mask=edge_mask, other=0)
            center = tl.load(center_idx + edge, mask=edge_mask, other=0)
            edge_grad0 = tl.zeros((BLOCK_I, BLOCK_E), tl.float32)
            edge_grad1 = tl.zeros((BLOCK_I, BLOCK_E), tl.float32)
            for radial in range(NUM_RADIAL):
                raw0 = tl.zeros((BLOCK_I, BLOCK_E), tl.float32)
                raw1 = tl.zeros((BLOCK_I, BLOCK_E), tl.float32)
                out_base = 0
                while out_base < OUT_M:
                    out_channels = out_base + tl.arange(0, BLOCK_O)
                    out_mask = out_channels < OUT_M
                    g0 = tl.load(
                        grad_out
                        + (batch * N_POINTS + center[:, None]) * (OUT_M * out_dim)
                        + out_channels[None, :] * out_dim,
                        mask=edge_mask[:, None] & out_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    if OUT_ORDER > 0:
                        g1 = tl.load(
                            grad_out
                            + (batch * N_POINTS + center[:, None]) * (OUT_M * out_dim)
                            + out_channels[None, :] * out_dim
                            + 1,
                            mask=edge_mask[:, None] & out_mask[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        c_out = tl.load(
                            output_cos + edge * MAX_OUT_ORDER + OUT_ORDER - 1,
                            mask=edge_mask,
                            other=1.0,
                        )
                        s_out = tl.load(
                            output_sin + edge * MAX_OUT_ORDER + OUT_ORDER - 1,
                            mask=edge_mask,
                            other=0.0,
                        )
                        q0 = c_out[:, None] * g0 + s_out[:, None] * g1
                        q1 = -s_out[:, None] * g0 + c_out[:, None] * g1
                    else:
                        q0 = g0
                        q1 = g0
                    if NORMALIZE:
                        denom = tl.maximum(
                            tl.load(neighbor_count + center, mask=edge_mask, other=1.0), 1.0
                        )
                        q0 /= denom[:, None]
                        q1 /= denom[:, None]
                    w00 = tl.load(
                        weight
                        + radial * stride_wr
                        + out_channels[None, :] * stride_wo
                        + in_channels[:, None] * stride_wi,
                        mask=in_mask[:, None] & out_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    raw0 += tl.sum(
                        w00[:, None, :] * q0[None, :, :], axis=2
                    )
                    if OUT_ORDER > 0:
                        w10 = tl.load(
                            weight
                            + radial * stride_wr
                            + out_channels[None, :] * stride_wo
                            + in_channels[:, None] * stride_wi
                            + stride_wa,
                            mask=in_mask[:, None] & out_mask[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        raw0 += tl.sum(
                            w10[:, None, :] * q1[None, :, :], axis=2
                        )
                    if IN_ORDER > 0:
                        w01 = tl.load(
                            weight
                            + radial * stride_wr
                            + out_channels[None, :] * stride_wo
                            + in_channels[:, None] * stride_wi
                            + stride_wd,
                            mask=in_mask[:, None] & out_mask[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        raw1 += tl.sum(
                            w01[:, None, :] * q0[None, :, :], axis=2
                        )
                        if OUT_ORDER > 0:
                            w11 = tl.load(
                                weight
                                + radial * stride_wr
                                + out_channels[None, :] * stride_wo
                                + in_channels[:, None] * stride_wi
                                + stride_wa
                                + stride_wd,
                                mask=in_mask[:, None] & out_mask[None, :],
                                other=0.0,
                            ).to(tl.float32)
                            raw1 += tl.sum(
                                w11[:, None, :] * q1[None, :, :], axis=2
                            )
                    out_base += BLOCK_O
                beta = tl.load(
                    radial_basis + edge * NUM_RADIAL + radial,
                    mask=edge_mask,
                    other=0.0,
                )
                edge_grad0 += raw0 * beta[None, :]
                edge_grad1 += raw1 * beta[None, :]
            if IN_ORDER == 0:
                total0 += tl.sum(edge_grad0, axis=1)
            else:
                c_in = tl.load(
                    input_cos + edge * MAX_IN_ORDER + IN_ORDER - 1,
                    mask=edge_mask,
                    other=1.0,
                )
                s_in = tl.load(
                    input_sin + edge * MAX_IN_ORDER + IN_ORDER - 1,
                    mask=edge_mask,
                    other=0.0,
                )
                total0 += tl.sum(c_in[None, :] * edge_grad0 + s_in[None, :] * edge_grad1, axis=1)
                total1 += tl.sum(-s_in[None, :] * edge_grad0 + c_in[None, :] * edge_grad1, axis=1)
            pos += BLOCK_E

        in_dim: tl.constexpr = 1 if IN_ORDER == 0 else 2
        base = (batch * N_POINTS + point) * (IN_M * in_dim)
        tl.store(grad_x + base + in_channels * in_dim, total0, mask=in_mask)
        if IN_ORDER > 0:
            tl.store(grad_x + base + in_channels * in_dim + 1, total1, mask=in_mask)


    @triton.jit
    def _pair_grad_weight_partial_kernel(
        x,
        grad_out,
        center_idx,
        neighbor_idx,
        radial_basis,
        input_cos,
        input_sin,
        output_cos,
        output_sin,
        neighbor_count,
        partial_weight,
        stride_xb: tl.constexpr,
        stride_xn: tl.constexpr,
        stride_xi: tl.constexpr,
        stride_xd: tl.constexpr,
        BATCH: tl.constexpr,
        N_POINTS: tl.constexpr,
        N_EDGES: tl.constexpr,
        IN_M: tl.constexpr,
        OUT_M: tl.constexpr,
        NUM_RADIAL: tl.constexpr,
        IN_ORDER: tl.constexpr,
        OUT_ORDER: tl.constexpr,
        MAX_IN_ORDER: tl.constexpr,
        MAX_OUT_ORDER: tl.constexpr,
        PARTIALS: tl.constexpr,
        NORMALIZE: tl.constexpr,
        BLOCK_O: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        out_blocks = tl.cdiv(OUT_M, BLOCK_O)
        in_blocks = tl.cdiv(IN_M, BLOCK_I)
        in_dim: tl.constexpr = 1 if IN_ORDER == 0 else 2
        out_dim: tl.constexpr = 1 if OUT_ORDER == 0 else 2
        pid = tl.program_id(0)
        in_block = pid % in_blocks
        pid //= in_blocks
        out_block = pid % out_blocks
        pid //= out_blocks
        in_component = pid % in_dim
        pid //= in_dim
        out_component = pid % out_dim
        pid //= out_dim
        radial = pid % NUM_RADIAL
        partial = pid // NUM_RADIAL
        in_channels = in_block * BLOCK_I + tl.arange(0, BLOCK_I)
        out_channels = out_block * BLOCK_O + tl.arange(0, BLOCK_O)
        in_mask = in_channels < IN_M
        out_mask = out_channels < OUT_M
        acc = tl.zeros((BLOCK_O, BLOCK_I), tl.float32)
        sample_total: tl.constexpr = BATCH * N_EDGES
        samples_per_partial: tl.constexpr = tl.cdiv(sample_total, PARTIALS)
        sample_base = partial * samples_per_partial
        sample_stop = tl.minimum(sample_base + samples_per_partial, sample_total)

        while sample_base < sample_stop:
            sample = sample_base + tl.arange(0, BLOCK_S)
            sample_mask = sample < sample_stop
            batch = sample // N_EDGES
            edge = sample - batch * N_EDGES
            center = tl.load(center_idx + edge, mask=sample_mask, other=0)
            neighbor = tl.load(neighbor_idx + edge, mask=sample_mask, other=0)
            x0 = tl.load(
                x
                + batch[:, None] * stride_xb
                + neighbor[:, None] * stride_xn
                + in_channels[None, :] * stride_xi,
                mask=sample_mask[:, None] & in_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if IN_ORDER > 0:
                x1 = tl.load(
                    x
                    + batch[:, None] * stride_xb
                    + neighbor[:, None] * stride_xn
                    + in_channels[None, :] * stride_xi
                    + stride_xd,
                    mask=sample_mask[:, None] & in_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                c_in = tl.load(
                    input_cos + edge * MAX_IN_ORDER + IN_ORDER - 1,
                    mask=sample_mask,
                    other=1.0,
                )
                s_in = tl.load(
                    input_sin + edge * MAX_IN_ORDER + IN_ORDER - 1,
                    mask=sample_mask,
                    other=0.0,
                )
                z0 = c_in[:, None] * x0 - s_in[:, None] * x1
                z1 = s_in[:, None] * x0 + c_in[:, None] * x1
                z = tl.where(in_component == 0, z0, z1)
            else:
                z = x0
            g0 = tl.load(
                grad_out
                + (batch[:, None] * N_POINTS + center[:, None]) * (OUT_M * out_dim)
                + out_channels[None, :] * out_dim,
                mask=sample_mask[:, None] & out_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if OUT_ORDER > 0:
                g1 = tl.load(
                    grad_out
                    + (batch[:, None] * N_POINTS + center[:, None]) * (OUT_M * out_dim)
                    + out_channels[None, :] * out_dim
                    + 1,
                    mask=sample_mask[:, None] & out_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                c_out = tl.load(
                    output_cos + edge * MAX_OUT_ORDER + OUT_ORDER - 1,
                    mask=sample_mask,
                    other=1.0,
                )
                s_out = tl.load(
                    output_sin + edge * MAX_OUT_ORDER + OUT_ORDER - 1,
                    mask=sample_mask,
                    other=0.0,
                )
                q0 = c_out[:, None] * g0 + s_out[:, None] * g1
                q1 = -s_out[:, None] * g0 + c_out[:, None] * g1
                q = tl.where(out_component == 0, q0, q1)
            else:
                q = g0
            if NORMALIZE:
                denom = tl.maximum(
                    tl.load(neighbor_count + center, mask=sample_mask, other=1.0), 1.0
                )
                q /= denom[:, None]
            beta = tl.load(
                radial_basis + edge * NUM_RADIAL + radial,
                mask=sample_mask,
                other=0.0,
            ).to(tl.float32)
            left = q * beta[:, None]
            acc += tl.sum(
                left[:, :, None] * z[:, None, :], axis=0
            )
            sample_base += BLOCK_S

        weight_numel: tl.constexpr = NUM_RADIAL * OUT_M * IN_M * out_dim * in_dim
        offset = (
            (((radial * OUT_M + out_channels[:, None]) * IN_M + in_channels[None, :]) * out_dim + out_component)
            * in_dim
            + in_component
        )
        tl.store(
            partial_weight + partial * weight_numel + offset,
            acc,
            mask=out_mask[:, None] & in_mask[None, :],
        )


    @triton.jit
    def _pair_forward_r1_kernel(
        x,
        weight,
        neighbor_idx,
        center_ptr,
        radial_basis,
        single_input_cos,
        single_input_sin,
        output_cos,
        output_sin,
        neighbor_count,
        out,
        stride_xb: tl.constexpr,
        stride_xn: tl.constexpr,
        stride_xi: tl.constexpr,
        stride_xd: tl.constexpr,
        stride_wr: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_wa: tl.constexpr,
        stride_wd: tl.constexpr,
        N_POINTS: tl.constexpr,
        IN_M: tl.constexpr,
        OUT_M: tl.constexpr,
        MAX_IN_ORDER: tl.constexpr,
        MAX_OUT_ORDER: tl.constexpr,
        IN_ORDER: tl.constexpr,
        OUT_ORDER: tl.constexpr,
        DEGREE_BUCKET: tl.constexpr,
        NORMALIZE: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        STRICT_FP32: tl.constexpr,
        USE_DOT: tl.constexpr,
        BLOCK_O: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        out_dim: tl.constexpr = 1 if OUT_ORDER == 0 else 2
        in_dim: tl.constexpr = 1 if IN_ORDER == 0 else 2
        out_blocks = tl.cdiv(OUT_M, BLOCK_O)
        pid = tl.program_id(0)
        out_block = pid % out_blocks
        pid //= out_blocks
        center = pid % N_POINTS
        batch = pid // N_POINTS
        out_width: tl.constexpr = BLOCK_O * out_dim
        out_packed = tl.arange(0, out_width)
        out_channels = out_block * BLOCK_O + out_packed // out_dim
        out_components = out_packed % out_dim
        out_mask = out_channels < OUT_M
        total0 = tl.zeros((BLOCK_O,), tl.float32)
        total1 = tl.zeros((BLOCK_O,), tl.float32)
        edge_start = tl.load(center_ptr + center)
        edge_stop = tl.load(center_ptr + center + 1)
        edge_base = edge_start
        in_width: tl.constexpr = IN_M * in_dim

        while edge_base < edge_stop:
            edge = edge_base + tl.arange(0, BLOCK_E)
            edge_mask = edge < edge_stop
            neighbor = tl.load(neighbor_idx + edge, mask=edge_mask, other=0)
            raw = tl.zeros((BLOCK_E, out_width), tl.float32)
            k_base = 0
            while k_base < in_width:
                packed_in = k_base + tl.arange(0, BLOCK_K)
                k_mask = packed_in < in_width
                in_channels = packed_in // in_dim
                in_components = packed_in % in_dim
                x0 = tl.load(
                    x
                    + batch * stride_xb
                    + neighbor[:, None] * stride_xn
                    + in_channels[None, :] * stride_xi,
                    mask=edge_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                if IN_ORDER == 0:
                    beta = tl.load(radial_basis + edge, mask=edge_mask, other=0.0)
                    z = x0 * beta[:, None]
                else:
                    x1 = tl.load(
                        x
                        + batch * stride_xb
                        + neighbor[:, None] * stride_xn
                        + in_channels[None, :] * stride_xi
                        + stride_xd,
                        mask=edge_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )
                    c = tl.load(
                        single_input_cos + edge * MAX_IN_ORDER + IN_ORDER - 1,
                        mask=edge_mask,
                        other=0.0,
                    )
                    s = tl.load(
                        single_input_sin + edge * MAX_IN_ORDER + IN_ORDER - 1,
                        mask=edge_mask,
                        other=0.0,
                    )
                    z0 = c[:, None] * x0 - s[:, None] * x1
                    z1 = s[:, None] * x0 + c[:, None] * x1
                    z = tl.where(in_components[None, :] == 0, z0, z1)
                w = tl.load(
                    weight
                    + out_channels[None, :] * stride_wo
                    + in_channels[:, None] * stride_wi
                    + out_components[None, :] * stride_wa
                    + in_components[:, None] * stride_wd,
                    mask=k_mask[:, None] & out_mask[None, :],
                    other=0.0,
                ).to(z.dtype)
                # An explicit multiply/reduce can itself be recognized as a
                # matmul and lowered to TF32.  Make strict FP32 an explicit
                # IEEE dot whenever the tile dimensions allow it.
                if STRICT_FP32 and out_width >= 16:
                    raw += tl.dot(z, w, input_precision="ieee")
                elif USE_DOT and not STRICT_FP32:
                    if ALLOW_TF32:
                        raw += tl.dot(z, w, input_precision="tf32")
                    else:
                        raw += tl.dot(z, w, input_precision="ieee")
                else:
                    raw += tl.sum(
                        z.to(tl.float32)[:, :, None]
                        * w.to(tl.float32)[None, :, :],
                        axis=1,
                    )
                k_base += BLOCK_K

            if OUT_ORDER == 0:
                total0 += tl.sum(raw, axis=0)
            else:
                paired = tl.reshape(raw, (BLOCK_E, BLOCK_O, 2))
                raw0, raw1 = tl.split(paired)
                c_out = tl.load(
                    output_cos + edge * MAX_OUT_ORDER + OUT_ORDER - 1,
                    mask=edge_mask,
                    other=1.0,
                )
                s_out = tl.load(
                    output_sin + edge * MAX_OUT_ORDER + OUT_ORDER - 1,
                    mask=edge_mask,
                    other=0.0,
                )
                total0 += tl.sum(raw0 * c_out[:, None] - raw1 * s_out[:, None], axis=0)
                total1 += tl.sum(raw0 * s_out[:, None] + raw1 * c_out[:, None], axis=0)
            edge_base += BLOCK_E

        if NORMALIZE:
            scale = tl.maximum(tl.load(neighbor_count + center), 1.0)
            total0 /= scale
            total1 /= scale
        out_channels_store = out_block * BLOCK_O + tl.arange(0, BLOCK_O)
        store_mask = out_channels_store < OUT_M
        out_base = (batch * N_POINTS + center) * (OUT_M * out_dim)
        tl.store(out + out_base + out_channels_store * out_dim, total0, mask=store_mask)
        if OUT_ORDER > 0:
            tl.store(out + out_base + out_channels_store * out_dim + 1, total1, mask=store_mask)


    @triton.jit
    def _pair_grad_input_r1_kernel(
        grad_out,
        weight,
        center_idx,
        neighbor_ptr,
        edges_by_neighbor,
        radial_basis,
        single_input_cos,
        single_input_sin,
        output_cos,
        output_sin,
        neighbor_count,
        grad_x,
        stride_wr: tl.constexpr,
        stride_wo: tl.constexpr,
        stride_wi: tl.constexpr,
        stride_wa: tl.constexpr,
        stride_wd: tl.constexpr,
        N_POINTS: tl.constexpr,
        IN_M: tl.constexpr,
        OUT_M: tl.constexpr,
        MAX_IN_ORDER: tl.constexpr,
        MAX_OUT_ORDER: tl.constexpr,
        IN_ORDER: tl.constexpr,
        OUT_ORDER: tl.constexpr,
        DEGREE_BUCKET: tl.constexpr,
        NORMALIZE: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        STRICT_FP32: tl.constexpr,
        USE_DOT: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        in_dim: tl.constexpr = 1 if IN_ORDER == 0 else 2
        out_dim: tl.constexpr = 1 if OUT_ORDER == 0 else 2
        in_blocks = tl.cdiv(IN_M, BLOCK_I)
        pid = tl.program_id(0)
        in_block = pid % in_blocks
        pid //= in_blocks
        point = pid % N_POINTS
        batch = pid // N_POINTS
        in_width: tl.constexpr = BLOCK_I * in_dim
        packed_in = tl.arange(0, in_width)
        in_channels = in_block * BLOCK_I + packed_in // in_dim
        in_components = packed_in % in_dim
        in_mask = in_channels < IN_M
        total0 = tl.zeros((BLOCK_I,), tl.float32)
        total1 = tl.zeros((BLOCK_I,), tl.float32)
        pos = tl.load(neighbor_ptr + point)
        pos_stop = tl.load(neighbor_ptr + point + 1)
        out_width: tl.constexpr = OUT_M * out_dim

        while pos < pos_stop:
            positions = pos + tl.arange(0, BLOCK_E)
            edge_mask = positions < pos_stop
            edge = tl.load(edges_by_neighbor + positions, mask=edge_mask, other=0)
            center = tl.load(center_idx + edge, mask=edge_mask, other=0)
            raw = tl.zeros((BLOCK_E, in_width), tl.float32)
            k_base = 0
            while k_base < out_width:
                packed_out = k_base + tl.arange(0, BLOCK_K)
                k_mask = packed_out < out_width
                out_channels = packed_out // out_dim
                out_components = packed_out % out_dim
                g0 = tl.load(
                    grad_out
                    + (batch * N_POINTS + center[:, None]) * (OUT_M * out_dim)
                    + out_channels[None, :] * out_dim,
                    mask=edge_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                if OUT_ORDER == 0:
                    q = g0
                else:
                    g1 = tl.load(
                        grad_out
                        + (batch * N_POINTS + center[:, None]) * (OUT_M * out_dim)
                        + out_channels[None, :] * out_dim
                        + 1,
                        mask=edge_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )
                    c_out = tl.load(
                        output_cos + edge * MAX_OUT_ORDER + OUT_ORDER - 1,
                        mask=edge_mask,
                        other=1.0,
                    )
                    s_out = tl.load(
                        output_sin + edge * MAX_OUT_ORDER + OUT_ORDER - 1,
                        mask=edge_mask,
                        other=0.0,
                    )
                    q0 = c_out[:, None] * g0 + s_out[:, None] * g1
                    q1 = -s_out[:, None] * g0 + c_out[:, None] * g1
                    q = tl.where(out_components[None, :] == 0, q0, q1)
                if NORMALIZE:
                    denom = tl.maximum(
                        tl.load(neighbor_count + center, mask=edge_mask, other=1.0), 1.0
                    )
                    q /= denom[:, None]
                w = tl.load(
                    weight
                    + out_channels[:, None] * stride_wo
                    + in_channels[None, :] * stride_wi
                    + out_components[:, None] * stride_wa
                    + in_components[None, :] * stride_wd,
                    mask=k_mask[:, None] & in_mask[None, :],
                    other=0.0,
                ).to(q.dtype)
                if STRICT_FP32 and in_width >= 16:
                    raw += tl.dot(q, w, input_precision="ieee")
                elif USE_DOT and not STRICT_FP32:
                    if ALLOW_TF32:
                        raw += tl.dot(q, w, input_precision="tf32")
                    else:
                        raw += tl.dot(q, w, input_precision="ieee")
                else:
                    raw += tl.sum(
                        q.to(tl.float32)[:, :, None]
                        * w.to(tl.float32)[None, :, :],
                        axis=1,
                    )
                k_base += BLOCK_K

            if IN_ORDER == 0:
                beta = tl.load(radial_basis + edge, mask=edge_mask, other=0.0)
                total0 += tl.sum(raw * beta[:, None], axis=0)
            else:
                paired = tl.reshape(raw, (BLOCK_E, BLOCK_I, 2))
                raw0, raw1 = tl.split(paired)
                c = tl.load(
                    single_input_cos + edge * MAX_IN_ORDER + IN_ORDER - 1,
                    mask=edge_mask,
                    other=0.0,
                )
                s = tl.load(
                    single_input_sin + edge * MAX_IN_ORDER + IN_ORDER - 1,
                    mask=edge_mask,
                    other=0.0,
                )
                total0 += tl.sum(
                    c[:, None] * raw0 + s[:, None] * raw1, axis=0
                )
                total1 += tl.sum(
                    -s[:, None] * raw0 + c[:, None] * raw1, axis=0
                )
            pos += BLOCK_E

        channels = in_block * BLOCK_I + tl.arange(0, BLOCK_I)
        store_mask = channels < IN_M
        base = (batch * N_POINTS + point) * (IN_M * in_dim)
        tl.store(grad_x + base + channels * in_dim, total0, mask=store_mask)
        if IN_ORDER > 0:
            tl.store(grad_x + base + channels * in_dim + 1, total1, mask=store_mask)


    @triton.jit
    def _pair_grad_weight_r1_partial_kernel(
        x,
        grad_out,
        center_idx,
        neighbor_idx,
        radial_basis,
        single_input_cos,
        single_input_sin,
        output_cos,
        output_sin,
        neighbor_count,
        partial_weight,
        stride_xb: tl.constexpr,
        stride_xn: tl.constexpr,
        stride_xi: tl.constexpr,
        stride_xd: tl.constexpr,
        BATCH: tl.constexpr,
        N_POINTS: tl.constexpr,
        N_EDGES: tl.constexpr,
        IN_M: tl.constexpr,
        OUT_M: tl.constexpr,
        MAX_IN_ORDER: tl.constexpr,
        MAX_OUT_ORDER: tl.constexpr,
        IN_ORDER: tl.constexpr,
        OUT_ORDER: tl.constexpr,
        PARTIALS: tl.constexpr,
        DEGREE_BUCKET: tl.constexpr,
        NORMALIZE: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        STRICT_FP32: tl.constexpr,
        USE_DOT: tl.constexpr,
        BLOCK_O: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        in_dim: tl.constexpr = 1 if IN_ORDER == 0 else 2
        out_dim: tl.constexpr = 1 if OUT_ORDER == 0 else 2
        out_blocks = tl.cdiv(OUT_M, BLOCK_O)
        in_blocks = tl.cdiv(IN_M, BLOCK_I)
        pid = tl.program_id(0)
        in_block = pid % in_blocks
        pid //= in_blocks
        out_block = pid % out_blocks
        partial = pid // out_blocks
        out_width: tl.constexpr = BLOCK_O * out_dim
        in_width: tl.constexpr = BLOCK_I * in_dim
        packed_out = tl.arange(0, out_width)
        packed_in = tl.arange(0, in_width)
        out_channels = out_block * BLOCK_O + packed_out // out_dim
        out_components = packed_out % out_dim
        in_channels = in_block * BLOCK_I + packed_in // in_dim
        in_components = packed_in % in_dim
        out_mask = out_channels < OUT_M
        in_mask = in_channels < IN_M
        acc = tl.zeros((out_width, in_width), tl.float32)
        sample_total: tl.constexpr = BATCH * N_EDGES
        samples_per_partial: tl.constexpr = tl.cdiv(sample_total, PARTIALS)
        sample_base = partial * samples_per_partial
        sample_stop = tl.minimum(sample_base + samples_per_partial, sample_total)

        while sample_base < sample_stop:
            sample = sample_base + tl.arange(0, BLOCK_S)
            sample_mask = sample < sample_stop
            batch = sample // N_EDGES
            edge = sample - batch * N_EDGES
            center = tl.load(center_idx + edge, mask=sample_mask, other=0)
            neighbor = tl.load(neighbor_idx + edge, mask=sample_mask, other=0)
            x0 = tl.load(
                x
                + batch[:, None] * stride_xb
                + neighbor[:, None] * stride_xn
                + in_channels[None, :] * stride_xi,
                mask=sample_mask[:, None] & in_mask[None, :],
                other=0.0,
            )
            if IN_ORDER == 0:
                beta = tl.load(radial_basis + edge, mask=sample_mask, other=0.0)
                z = x0 * beta[:, None]
            else:
                x1 = tl.load(
                    x
                    + batch[:, None] * stride_xb
                    + neighbor[:, None] * stride_xn
                    + in_channels[None, :] * stride_xi
                    + stride_xd,
                    mask=sample_mask[:, None] & in_mask[None, :],
                    other=0.0,
                )
                c = tl.load(
                    single_input_cos + edge * MAX_IN_ORDER + IN_ORDER - 1,
                    mask=sample_mask,
                    other=0.0,
                )
                s = tl.load(
                    single_input_sin + edge * MAX_IN_ORDER + IN_ORDER - 1,
                    mask=sample_mask,
                    other=0.0,
                )
                z0 = c[:, None] * x0 - s[:, None] * x1
                z1 = s[:, None] * x0 + c[:, None] * x1
                z = tl.where(in_components[None, :] == 0, z0, z1)
            g0 = tl.load(
                grad_out
                + (batch[:, None] * N_POINTS + center[:, None]) * (OUT_M * out_dim)
                + out_channels[None, :] * out_dim,
                mask=sample_mask[:, None] & out_mask[None, :],
                other=0.0,
            )
            if OUT_ORDER == 0:
                q = g0
            else:
                g1 = tl.load(
                    grad_out
                    + (batch[:, None] * N_POINTS + center[:, None]) * (OUT_M * out_dim)
                    + out_channels[None, :] * out_dim
                    + 1,
                    mask=sample_mask[:, None] & out_mask[None, :],
                    other=0.0,
                )
                c_out = tl.load(
                    output_cos + edge * MAX_OUT_ORDER + OUT_ORDER - 1,
                    mask=sample_mask,
                    other=1.0,
                )
                s_out = tl.load(
                    output_sin + edge * MAX_OUT_ORDER + OUT_ORDER - 1,
                    mask=sample_mask,
                    other=0.0,
                )
                q0 = c_out[:, None] * g0 + s_out[:, None] * g1
                q1 = -s_out[:, None] * g0 + c_out[:, None] * g1
                q = tl.where(out_components[None, :] == 0, q0, q1)
            if NORMALIZE:
                denom = tl.maximum(
                    tl.load(neighbor_count + center, mask=sample_mask, other=1.0), 1.0
                )
                q /= denom[:, None]
            if STRICT_FP32:
                if out_width >= 16:
                    if in_width >= 16:
                        acc += tl.dot(
                            tl.trans(q.to(z.dtype)), z, input_precision="ieee"
                        )
                    else:
                        acc += tl.sum(
                            q.to(tl.float32)[:, :, None]
                            * z.to(tl.float32)[:, None, :],
                            axis=0,
                        )
                else:
                    acc += tl.sum(
                        q.to(tl.float32)[:, :, None]
                        * z.to(tl.float32)[:, None, :],
                        axis=0,
                    )
            elif USE_DOT and not STRICT_FP32:
                left = q.to(z.dtype)
                if ALLOW_TF32:
                    acc += tl.dot(tl.trans(left), z, input_precision="tf32")
                else:
                    acc += tl.dot(tl.trans(left), z, input_precision="ieee")
            else:
                acc += tl.sum(
                    q.to(tl.float32)[:, :, None]
                    * z.to(tl.float32)[:, None, :],
                    axis=0,
                )
            sample_base += BLOCK_S

        weight_numel: tl.constexpr = OUT_M * IN_M * out_dim * in_dim
        offset = (
            ((out_channels[:, None] * IN_M + in_channels[None, :]) * out_dim + out_components[:, None])
            * in_dim
            + in_components[None, :]
        )
        tl.store(
            partial_weight + partial * weight_numel + offset,
            acc,
            mask=out_mask[:, None] & in_mask[None, :],
        )


    @triton.jit
    def _reduce_partials_kernel(
        partial_weight,
        grad_weight,
        WEIGHT_NUMEL: tl.constexpr,
        PARTIALS: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_W + tl.arange(0, BLOCK_W)
        mask = offsets < WEIGHT_NUMEL
        total = tl.zeros((BLOCK_W,), tl.float32)
        for partial in range(PARTIALS):
            total += tl.load(
                partial_weight + partial * WEIGHT_NUMEL + offsets,
                mask=mask,
                other=0.0,
            )
        tl.store(grad_weight + offsets, total, mask=mask)


    _R1_PAIR_KEY = [
        "IN_M",
        "OUT_M",
        "IN_ORDER",
        "OUT_ORDER",
        "DEGREE_BUCKET",
        "NORMALIZE",
        "ALLOW_TF32",
        "STRICT_FP32",
    ]
    _R1_GRAD_WEIGHT_KEY = [
        "BATCH",
        "N_EDGES",
        "IN_M",
        "OUT_M",
        "IN_ORDER",
        "OUT_ORDER",
        "PARTIALS",
        "DEGREE_BUCKET",
        "NORMALIZE",
        "ALLOW_TF32",
        "STRICT_FP32",
    ]

    def _r1_forward_configs(architecture: int) -> list:
        configs = [
            triton.Config(
                {"USE_DOT": False, "BLOCK_O": 8, "BLOCK_E": 16, "BLOCK_K": 16},
                num_warps=4,
            ),
            triton.Config(
                {"USE_DOT": True, "BLOCK_O": 16, "BLOCK_E": 16, "BLOCK_K": 16},
                num_warps=4,
            ),
        ]
        if architecture >= 9:
            configs.extend(
                [
                    triton.Config(
                        {"USE_DOT": True, "BLOCK_O": 16, "BLOCK_E": 32, "BLOCK_K": 32},
                        num_warps=4,
                    ),
                    triton.Config(
                        {"USE_DOT": True, "BLOCK_O": 16, "BLOCK_E": 64, "BLOCK_K": 32},
                        num_warps=4,
                    ),
                    triton.Config(
                        {"USE_DOT": True, "BLOCK_O": 16, "BLOCK_E": 64, "BLOCK_K": 64},
                        num_warps=8,
                    ),
                ]
            )
        return configs

    def _r1_grad_input_configs(architecture: int) -> list:
        configs = [
            triton.Config(
                {"USE_DOT": False, "BLOCK_I": 8, "BLOCK_E": 16, "BLOCK_K": 16},
                num_warps=4,
            ),
            triton.Config(
                {"USE_DOT": True, "BLOCK_I": 16, "BLOCK_E": 16, "BLOCK_K": 16},
                num_warps=4,
            ),
        ]
        if architecture >= 8:
            configs.append(
                triton.Config(
                    {"USE_DOT": True, "BLOCK_I": 16, "BLOCK_E": 32, "BLOCK_K": 32},
                    num_warps=4,
                )
            )
        if architecture >= 9:
            configs.extend(
                [
                    triton.Config(
                        {"USE_DOT": True, "BLOCK_I": 16, "BLOCK_E": 64, "BLOCK_K": 32},
                        num_warps=4,
                    ),
                    triton.Config(
                        {"USE_DOT": True, "BLOCK_I": 16, "BLOCK_E": 64, "BLOCK_K": 64},
                        num_warps=8,
                    ),
                ]
            )
        return configs

    def _r1_grad_weight_configs(architecture: int) -> list:
        configs = [
            triton.Config(
                {"USE_DOT": False, "BLOCK_O": 8, "BLOCK_I": 8, "BLOCK_S": 16},
                num_warps=4,
            ),
            triton.Config(
                {"USE_DOT": True, "BLOCK_O": 16, "BLOCK_I": 16, "BLOCK_S": 16},
                num_warps=4,
            ),
        ]
        if architecture >= 8:
            configs.append(
                triton.Config(
                    {"USE_DOT": True, "BLOCK_O": 16, "BLOCK_I": 16, "BLOCK_S": 32},
                    num_warps=4,
                )
            )
        if architecture >= 9:
            configs.extend(
                [
                    triton.Config(
                        {"USE_DOT": True, "BLOCK_O": 16, "BLOCK_I": 16, "BLOCK_S": 64},
                        num_warps=4,
                    ),
                    triton.Config(
                        {"USE_DOT": True, "BLOCK_O": 16, "BLOCK_I": 16, "BLOCK_S": 64},
                        num_warps=8,
                    ),
                ]
            )
        return configs

    _pair_forward_r1_kernels = {
        architecture: triton.autotune(
            configs=_r1_forward_configs(architecture), key=_R1_PAIR_KEY,
        )(_pair_forward_r1_kernel)
        for architecture in (7, 8, 9)
    }
    _pair_grad_input_r1_kernels = {
        architecture: triton.autotune(
            configs=_r1_grad_input_configs(architecture), key=_R1_PAIR_KEY,
        )(_pair_grad_input_r1_kernel)
        for architecture in (7, 8, 9)
    }
    _pair_grad_weight_r1_kernels = {
        architecture: triton.autotune(
            configs=_r1_grad_weight_configs(architecture), key=_R1_GRAD_WEIGHT_KEY,
        )(_pair_grad_weight_r1_partial_kernel)
        for architecture in (7, 8, 9)
    }


    @triton_op("kpconv_intrinsic::irregular_irrep_pair_conv", mutates_args={})
    def irregular_irrep_pair_conv(
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
        single_input_cos: Tensor,
        single_input_sin: Tensor,
        output_cos: Tensor,
        output_sin: Tensor,
        neighbor_count: Tensor,
        in_order: int,
        out_order: int,
        degree_bucket: int,
        normalize: bool,
        use_r1_fast_path: bool,
        allow_tf32: bool,
    ) -> Tensor:
        del center_idx, neighbor_ptr, edges_by_neighbor
        radial, out_m, _in_m, out_dim, _in_dim = map(int, weight.shape)
        batch, n_points = int(x.shape[0]), int(x.shape[1])
        out = torch.empty((batch, n_points, out_m, out_dim), device=x.device, dtype=x.dtype)
        if radial == 1 and use_r1_fast_path:
            major = torch.cuda.get_device_capability(x.device)[0]
            architecture = 9 if major >= 9 else (8 if major >= 8 else 7)
            strict_fp32 = bool(x.dtype == torch.float32 and not allow_tf32)
            kernel = _pair_forward_r1_kernels[architecture]
            grid = lambda meta: (
                batch * n_points * triton.cdiv(out_m, meta["BLOCK_O"]),
            )
            wrap_triton(kernel)[grid](
                x,
                weight,
                neighbor_idx,
                center_ptr,
                radial_basis,
                single_input_cos,
                single_input_sin,
                output_cos,
                output_sin,
                neighbor_count,
                out,
                *x.stride(),
                *weight.stride(),
                N_POINTS=n_points,
                IN_M=int(weight.shape[2]),
                OUT_M=out_m,
                MAX_IN_ORDER=int(single_input_cos.shape[1]),
                MAX_OUT_ORDER=int(output_cos.shape[1]),
                IN_ORDER=int(in_order),
                OUT_ORDER=int(out_order),
                DEGREE_BUCKET=int(degree_bucket),
                NORMALIZE=normalize,
                ALLOW_TF32=allow_tf32,
                STRICT_FP32=strict_fp32,
            )
            return out
        grid = (batch * n_points * triton.cdiv(out_m, 16),)
        wrap_triton(_pair_forward_kernel)[grid](
            x,
            weight,
            neighbor_idx,
            center_ptr,
            radial_basis,
            input_cos,
            input_sin,
            output_cos,
            output_sin,
            neighbor_count,
            out,
            *x.stride(),
            *weight.stride(),
            N_POINTS=n_points,
            IN_M=int(weight.shape[2]),
            OUT_M=out_m,
            NUM_RADIAL=radial,
            MAX_IN_ORDER=int(input_cos.shape[1]),
            MAX_OUT_ORDER=int(output_cos.shape[1]),
            IN_ORDER=int(in_order),
            OUT_ORDER=int(out_order),
            NORMALIZE=normalize,
            BLOCK_O=16,
            BLOCK_I=16,
            BLOCK_E=16,
            num_warps=4,
        )
        return out


    @triton_op("kpconv_intrinsic::irregular_irrep_pair_conv_backward", mutates_args={})
    def _irregular_irrep_pair_conv_backward(
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
        single_input_cos: Tensor,
        single_input_sin: Tensor,
        output_cos: Tensor,
        output_sin: Tensor,
        neighbor_count: Tensor,
        in_order: int,
        out_order: int,
        degree_bucket: int,
        normalize: bool,
        use_r1_fast_path: bool,
        allow_tf32: bool,
    ) -> tuple[Tensor, Tensor]:
        del center_ptr
        radial, out_m, in_m, out_dim, in_dim = map(int, weight.shape)
        batch, n_points = int(x.shape[0]), int(x.shape[1])
        n_edges = int(neighbor_idx.numel())
        grad_x = torch.empty_like(x, memory_format=torch.contiguous_format)
        grad_weight = torch.empty_like(weight, memory_format=torch.contiguous_format)
        if radial == 1 and use_r1_fast_path:
            major = torch.cuda.get_device_capability(x.device)[0]
            architecture = 9 if major >= 9 else (8 if major >= 8 else 7)
            strict_fp32 = bool(x.dtype == torch.float32 and not allow_tf32)
            grad_input_kernel = _pair_grad_input_r1_kernels[architecture]
            grid_x = lambda meta: (
                batch * n_points * triton.cdiv(in_m, meta["BLOCK_I"]),
            )
            wrap_triton(grad_input_kernel)[grid_x](
                grad_out,
                weight,
                center_idx,
                neighbor_ptr,
                edges_by_neighbor,
                radial_basis,
                single_input_cos,
                single_input_sin,
                output_cos,
                output_sin,
                neighbor_count,
                grad_x,
                *weight.stride(),
                N_POINTS=n_points,
                IN_M=in_m,
                OUT_M=out_m,
                MAX_IN_ORDER=int(single_input_cos.shape[1]),
                MAX_OUT_ORDER=int(output_cos.shape[1]),
                IN_ORDER=int(in_order),
                OUT_ORDER=int(out_order),
                DEGREE_BUCKET=int(degree_bucket),
                NORMALIZE=normalize,
                ALLOW_TF32=allow_tf32,
                STRICT_FP32=strict_fp32,
            )
        else:
            grid_x = (batch * n_points * triton.cdiv(in_m, 16),)
            wrap_triton(_pair_grad_input_kernel)[grid_x](
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
                neighbor_count,
                grad_x,
                *weight.stride(),
                N_POINTS=n_points,
                IN_M=in_m,
                OUT_M=out_m,
                NUM_RADIAL=radial,
                MAX_IN_ORDER=int(input_cos.shape[1]),
                MAX_OUT_ORDER=int(output_cos.shape[1]),
                IN_ORDER=int(in_order),
                OUT_ORDER=int(out_order),
                NORMALIZE=normalize,
                BLOCK_I=16,
                BLOCK_O=16,
                BLOCK_E=16,
                num_warps=4,
            )
        partials = _grad_weight_partial_count(weight, batch=batch, n_edges=n_edges)
        partial_weight = (
            grad_weight
            if partials == 1
            else torch.empty((partials, *weight.shape), device=weight.device, dtype=torch.float32)
        )
        if radial == 1 and use_r1_fast_path:
            grad_weight_kernel = _pair_grad_weight_r1_kernels[architecture]
            grid_w = lambda meta: (
                partials
                * triton.cdiv(out_m, meta["BLOCK_O"])
                * triton.cdiv(in_m, meta["BLOCK_I"]),
            )
            wrap_triton(grad_weight_kernel)[grid_w](
                x,
                grad_out,
                center_idx,
                neighbor_idx,
                radial_basis,
                single_input_cos,
                single_input_sin,
                output_cos,
                output_sin,
                neighbor_count,
                partial_weight,
                *x.stride(),
                BATCH=batch,
                N_POINTS=n_points,
                N_EDGES=n_edges,
                IN_M=in_m,
                OUT_M=out_m,
                MAX_IN_ORDER=int(single_input_cos.shape[1]),
                MAX_OUT_ORDER=int(output_cos.shape[1]),
                IN_ORDER=int(in_order),
                OUT_ORDER=int(out_order),
                PARTIALS=partials,
                DEGREE_BUCKET=int(degree_bucket),
                NORMALIZE=normalize,
                ALLOW_TF32=allow_tf32,
                STRICT_FP32=strict_fp32,
            )
        else:
            weight_tiles = (
                radial
                * out_dim
                * in_dim
                * triton.cdiv(out_m, 16)
                * triton.cdiv(in_m, 16)
            )
            grid_w = (partials * weight_tiles,)
            wrap_triton(_pair_grad_weight_partial_kernel)[grid_w](
                x,
                grad_out,
                center_idx,
                neighbor_idx,
                radial_basis,
                input_cos,
                input_sin,
                output_cos,
                output_sin,
                neighbor_count,
                partial_weight,
                *x.stride(),
                BATCH=batch,
                N_POINTS=n_points,
                N_EDGES=n_edges,
                IN_M=in_m,
                OUT_M=out_m,
                NUM_RADIAL=radial,
                IN_ORDER=int(in_order),
                OUT_ORDER=int(out_order),
                MAX_IN_ORDER=int(input_cos.shape[1]),
                MAX_OUT_ORDER=int(output_cos.shape[1]),
                PARTIALS=partials,
                NORMALIZE=normalize,
                BLOCK_O=16,
                BLOCK_I=16,
                BLOCK_S=32,
                num_warps=4,
            )
        if partials > 1:
            weight_numel = int(weight.numel())
            grid_reduce = (triton.cdiv(weight_numel, 128),)
            wrap_triton(_reduce_partials_kernel)[grid_reduce](
                partial_weight,
                grad_weight,
                WEIGHT_NUMEL=weight_numel,
                PARTIALS=partials,
                BLOCK_W=128,
                num_warps=4,
            )
        return grad_x, grad_weight


    def _setup_context(ctx, inputs, output) -> None:
        del output
        ctx.save_for_backward(*inputs[:-6])
        ctx.in_order = int(inputs[-6])
        ctx.out_order = int(inputs[-5])
        ctx.degree_bucket = int(inputs[-4])
        ctx.normalize = bool(inputs[-3])
        ctx.use_r1_fast_path = bool(inputs[-2])
        ctx.allow_tf32 = bool(inputs[-1])


    def _backward(ctx, grad_out: Tensor):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "higher-order gradients are unsupported by the Triton irrep convolution; "
                "use backend='torch'"
            )
        grad_x, grad_weight = _irregular_irrep_pair_conv_backward(
            grad_out.contiguous(),
            *ctx.saved_tensors,
            ctx.in_order,
            ctx.out_order,
            ctx.degree_bucket,
            ctx.normalize,
            ctx.use_r1_fast_path,
            ctx.allow_tf32,
        )
        return grad_x, grad_weight, *(None for _ in range(19))


    irregular_irrep_pair_conv.register_autograd(_backward, setup_context=_setup_context)

else:

    def irregular_irrep_pair_conv(*args, **kwargs):  # type: ignore[no-redef]
        del args, kwargs
        raise RuntimeError("Triton or torch.library.triton_op is unavailable")


__all__ = [
    "irregular_irrep_pair_conv",
    "irregular_pair_support_reason",
]
