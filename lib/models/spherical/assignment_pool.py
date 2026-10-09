"""Prepared assignment reductions with a fixed child order and exact transposes.

Plans are ephemeral geometry caches, shared by prepared U-Net pool/unpool pairs.
They are not buffers or learned state, and must be rebuilt after changing the
assignment geometry or device. Weighted and oversized assignment pooling retain
the legacy path; fixed-order reduction is guaranteed only for admitted plans.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ...data.spherical.sphere_hierarchy import SpherePooler
from .irrep_layers import _irrep_block, _flatten_irrep_block, _rotate_irrep_block


_MAX_ASSIGNMENT_PLAN_BYTES = 16 << 20
_MAX_ASSIGNMENT_PADDING_FACTOR = 4


@dataclass(frozen=True)
class AssignmentPlan:
    pooler: SpherePooler
    children: Tensor
    valid: Tensor
    assignment: Tensor
    fine_points: int
    coarse_points: int
    maximum_degree: int
    degree_histogram: dict[int, int]

    def metadata(self) -> dict[str, Any]:
        slots = self.children.numel()
        return {
            "fine_points": self.fine_points,
            "coarse_points": self.coarse_points,
            "maximum_degree": self.maximum_degree,
            "degree_histogram": self.degree_histogram,
            "padded_child_slots": slots,
            "padding_slots": slots - self.fine_points,
            "additional_index_mask_bytes": (
                self.children.numel() * self.children.element_size()
                + self.valid.numel() * self.valid.element_size()
            ),
            "gathered_values_per_batch_channel": slots,
            "reduction_order": "stable increasing fine index per coarse point; fixed child-axis reduction",
            "weighted_fallback": False,
        }


def prepare_assignment_plan(pooler: SpherePooler) -> AssignmentPlan | None:
    """CPU metadata work and device transfer, strictly before compilation/timing."""
    if torch.compiler.is_compiling():
        raise RuntimeError("prepare assignment pooling before compilation")
    if not isinstance(pooler, SpherePooler):
        raise TypeError("assignment plan requires SpherePooler")
    assignment = pooler.assignment.detach().to(device="cpu", dtype=torch.long)
    fine_points = int(pooler.fine_graph.n_points)
    coarse_points = int(pooler.coarse_graph.n_points)
    if assignment.shape != (fine_points,):
        raise ValueError("assignment shape does not match fine graph")
    if fine_points and (
        int(assignment.min()) < 0 or int(assignment.max()) >= coarse_points
    ):
        raise ValueError("assignment index out of bounds")
    counts = torch.bincount(assignment, minlength=coarse_points)
    maximum = int(counts.max()) if coarse_points else 0
    width = max(maximum, 1)
    slots = coarse_points * width
    # int64 child indexes and boolean masks consume 9 bytes per slot. Bound
    # both persistent tables and padding amplification before dense allocation.
    # Returning None preserves the existing standalone/legacy reduction path.
    if (
        slots * 9 > _MAX_ASSIGNMENT_PLAN_BYTES
        or slots > _MAX_ASSIGNMENT_PADDING_FACTOR * max(fine_points, coarse_points, 1)
    ):
        return None
    children = torch.zeros((coarse_points, width), dtype=torch.long)
    valid = torch.arange(width).view(1, -1) < counts.view(-1, 1)
    if fine_points:
        order = torch.argsort(assignment, stable=True)
        offsets = counts.cumsum(0) - counts
        ranks = torch.arange(fine_points) - torch.repeat_interleave(offsets, counts)
        children[assignment[order], ranks] = order
    values, frequencies = torch.unique(counts, return_counts=True)
    device = pooler.assignment.device
    return AssignmentPlan(
        pooler,
        children.to(device),
        valid.to(device),
        pooler.assignment,
        fine_points,
        coarse_points,
        maximum,
        dict(zip(values.tolist(), frequencies.tolist())),
    )


def _fixed_child_sum(values: Tensor, children: Tensor, valid: Tensor) -> Tensor:
    # Every output element has one reduction owner. There are no atomic writes.
    if values.shape[1] == 0:
        return values.new_zeros(
            (values.shape[0], children.shape[0], *values.shape[2:])
        )
    gathered = values[:, children]
    mask = valid.view(1, *valid.shape, *([1] * (values.ndim - 2)))
    return torch.where(mask, gathered, 0.).sum(dim=2)


class _AssignmentMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, children, valid, assignment, count):
        ctx.save_for_backward(assignment, count)
        summed = _fixed_child_sum(values, children, valid)
        divisor = count.clamp_min(1.).view(1, -1, *([1] * (values.ndim - 2)))
        return summed / divisor

    @staticmethod
    def backward(ctx, gradient):
        assignment, count = ctx.saved_tensors
        divisor = count[assignment].clamp_min(1.).view(
            1, -1, *([1] * (gradient.ndim - 2))
        )
        # Each fine point has exactly one owner: a gather, not scatter-add.
        return gradient[:, assignment] / divisor, None, None, None, None


class _AssignmentExpand(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, children, valid, assignment):
        ctx.save_for_backward(children, valid)
        return values[:, assignment]

    @staticmethod
    def backward(ctx, gradient):
        children, valid = ctx.saved_tensors
        return _fixed_child_sum(gradient, children, valid), None, None, None


def _prepared_plan(module, pooler, x):
    plan = getattr(module, "_deterministic_assignment_plan", None)
    if plan is None or plan.pooler is not pooler or plan.children.device != x.device:
        raise RuntimeError(
            "deterministic assignment plan is missing or stale; call model.prepare_hierarchy() after device/geometry changes"
        )
    return plan


def deterministic_pool_assignment(self, x, pooler):
    if int(x.shape[1]) != pooler.fine_graph.n_points:
        raise ValueError("fine point count differs")
    plan = _prepared_plan(self, pooler, x)
    count = pooler.count.to(device=x.device, dtype=x.dtype)
    assert pooler.transport_angle is not None
    angle = pooler.transport_angle.to(device=x.device, dtype=x.dtype)
    outputs = []
    for order in self.field_type.orders:
        block = _irrep_block(x, self.field_type, order)
        transported = _rotate_irrep_block(block, angle, order)
        pooled = _AssignmentMean.apply(
            transported, plan.children, plan.valid, plan.assignment, count
        )
        outputs.append(_flatten_irrep_block(pooled))
    return torch.cat(outputs, dim=-1)


def deterministic_unpool_assignment(self, x, pooler):
    plan = _prepared_plan(self, pooler, x)
    assert pooler.transport_angle is not None
    angle = pooler.transport_angle.to(device=x.device, dtype=x.dtype)
    outputs = []
    for order in self.field_type.orders:
        coarse = _irrep_block(x, self.field_type, order)
        gathered = _AssignmentExpand.apply(
            coarse, plan.children, plan.valid, plan.assignment
        )
        outputs.append(_flatten_irrep_block(_rotate_irrep_block(gathered, -angle, order)))
    return torch.cat(outputs, dim=-1)
