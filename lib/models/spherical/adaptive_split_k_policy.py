"""Pure shape policy for bounded semi-packed split-K weight gradients.

This module intentionally has no Torch, Triton, or CUDA dependency.  Runtime
code owns device discovery and passes the architecture major and SM count in.
The policy then has two independent responsibilities:

* choose a power-of-two split count that fills a bounded number of CTA waves;
* fit the selected reduction storage and point-chunk matrices below a hard
  workspace limit.

The split-K tree stores ``S`` partial weight tiles, ``S / 2`` pairwise-reduce
scratch tiles, and one weight tile per online carry level.  The serial
``S == 1`` fallback stores exactly one partial tile and no scratch or carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MIN_SPLIT_K_SPLITS = 1
MAX_SPLIT_K_SPLITS = 32
DEFAULT_SPLIT_K_BLOCK_OUT = 64
DEFAULT_SPLIT_K_BLOCK_IN = 64
DEFAULT_CTA_WAVES_PER_SM = 4
SM8_DEFAULT_SM_COUNT = 24
SM9_DEFAULT_SM_COUNT = 132
SM8_DEFAULT_CTA_TARGET = DEFAULT_CTA_WAVES_PER_SM * SM8_DEFAULT_SM_COUNT
# Bound future-SM9 and non-H100 callers to the calibrated H100-sized search
# envelope unless integration code deliberately supplies another cap.
SM9_DEFAULT_CTA_TARGET_CAP = DEFAULT_CTA_WAVES_PER_SM * SM9_DEFAULT_SM_COUNT
DEFAULT_POINT_CHUNK_QUANTUM = 32
DEFAULT_TREE_CARRY_LEVEL_LIMIT = 32
DEFAULT_BACKWARD_WORKSPACE_HEADROOM_BYTES = 4 << 20


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _ceil_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _floor_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("a positive value is required")
    return 1 << (value.bit_length() - 1)


def _require_positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_non_negative(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class ArchitectureCtaTarget:
    """Bounded CTA-wave target selected from caller-supplied device facts."""

    architecture: int
    sm_count: int
    waves_per_sm: int
    unbounded_target_ctas: int
    target_cap_ctas: int | None
    target_ctas: int

    def __post_init__(self) -> None:
        if self.architecture not in (8, 9):
            raise ValueError("architecture must be 8 or 9")
        if self.sm_count <= 0 or self.waves_per_sm <= 0:
            raise ValueError("SM count and waves per SM must be positive")
        if self.unbounded_target_ctas != self.sm_count * self.waves_per_sm:
            raise ValueError("unbounded CTA target disagrees with SM wave count")
        expected = self.unbounded_target_ctas
        if self.target_cap_ctas is not None:
            if self.target_cap_ctas <= 0:
                raise ValueError("CTA target cap must be positive")
            expected = min(expected, self.target_cap_ctas)
        if self.target_ctas != expected:
            raise ValueError("bounded CTA target is inconsistent")


@dataclass(frozen=True)
class SplitKSelection:
    """Shape-only launch selection and its ordered memory fallbacks."""

    target: ArchitectureCtaTarget
    reduction_rows: int
    out_total: int
    in_total: int
    block_out: int
    block_in: int
    output_tiles: int
    input_tiles: int
    base_ctas: int
    row_split_cap: int
    ideal_splits: int
    candidates: tuple[int, ...]

    def __post_init__(self) -> None:
        if min(
            self.reduction_rows,
            self.out_total,
            self.in_total,
            self.block_out,
            self.block_in,
            self.output_tiles,
            self.input_tiles,
            self.base_ctas,
            self.row_split_cap,
            self.ideal_splits,
        ) <= 0:
            raise ValueError("split-K selection dimensions must be positive")
        if self.output_tiles != _ceil_div(self.out_total, self.block_out):
            raise ValueError("output tile count is inconsistent")
        if self.input_tiles != _ceil_div(self.in_total, self.block_in):
            raise ValueError("input tile count is inconsistent")
        if self.base_ctas != self.output_tiles * self.input_tiles:
            raise ValueError("base CTA count is inconsistent")
        if not _is_power_of_two(self.row_split_cap):
            raise ValueError("row split cap must be a power of two")
        if self.row_split_cap > min(MAX_SPLIT_K_SPLITS, self.reduction_rows):
            raise ValueError("row split cap exceeds the available rows")
        if not _is_power_of_two(self.ideal_splits):
            raise ValueError("ideal split count must be a power of two")
        if not 1 <= self.ideal_splits <= self.row_split_cap:
            raise ValueError("ideal split count exceeds its cap")
        expected_candidates = tuple(
            self.ideal_splits >> shift
            for shift in range(self.ideal_splits.bit_length())
        )
        if self.candidates != expected_candidates:
            raise ValueError("candidate fallback sequence is inconsistent")


@dataclass(frozen=True)
class SplitKStorageAccounting:
    """Exact auxiliary weight-tile storage for one reduction strategy."""

    strategy: Literal["serial", "tree"]
    splits: int
    weight_bytes: int
    partial_tiles: int
    scratch_tiles: int
    carry_levels: int
    carry_tiles: int
    partial_bytes: int
    scratch_bytes: int
    carry_bytes: int
    total_bytes: int

    def __post_init__(self) -> None:
        if not _is_power_of_two(self.splits):
            raise ValueError("split count must be a power of two")
        if not MIN_SPLIT_K_SPLITS <= self.splits <= MAX_SPLIT_K_SPLITS:
            raise ValueError("split count is outside the supported range")
        if self.weight_bytes <= 0:
            raise ValueError("weight tile size must be positive")
        if self.strategy == "serial":
            if (
                self.partial_tiles != self.splits
                or self.scratch_tiles != (0 if self.splits == 1 else self.splits // 2)
                or self.carry_levels != 0
                or self.carry_tiles != 0
            ):
                raise ValueError(
                    "serial cross-chunk storage has invalid partial/scratch tiles"
                )
        elif self.strategy == "tree":
            if self.partial_tiles != self.splits:
                raise ValueError("tree storage needs one partial tile per split")
            if self.scratch_tiles != (0 if self.splits == 1 else self.splits // 2):
                raise ValueError("tree scratch must hold the first reduce level")
            if self.carry_levels <= 0 or self.carry_tiles != self.carry_levels:
                raise ValueError("tree storage needs one tile per carry level")
        else:
            raise ValueError("unknown split-K storage strategy")
        if self.partial_bytes != self.partial_tiles * self.weight_bytes:
            raise ValueError("partial byte accounting is inconsistent")
        if self.scratch_bytes != self.scratch_tiles * self.weight_bytes:
            raise ValueError("scratch byte accounting is inconsistent")
        if self.carry_bytes != self.carry_tiles * self.weight_bytes:
            raise ValueError("carry byte accounting is inconsistent")
        if self.total_bytes != (
            self.partial_bytes + self.scratch_bytes + self.carry_bytes
        ):
            raise ValueError("total storage byte accounting is inconsistent")


@dataclass(frozen=True)
class AdaptiveSplitKWorkspacePlan:
    """A selected launch/storage layout proven to fit its hard byte limit."""

    selection: SplitKSelection
    selected_splits: int
    candidate_index: int
    storage: SplitKStorageAccounting
    batch: int
    n_points: int
    padded_degree: int
    point_quantum: int
    chunk_points: int
    reduction_rows_per_chunk: int
    num_chunks: int
    bytes_per_point: int
    matrix_bytes: int
    allocated_bytes: int
    headroom_bytes: int
    planned_peak_bytes: int
    limit_bytes: int

    def __post_init__(self) -> None:
        if self.candidate_index < 0:
            raise ValueError("candidate index must be non-negative")
        if self.candidate_index >= len(self.selection.candidates):
            raise ValueError("candidate index is outside the fallback sequence")
        if self.selected_splits != self.selection.candidates[self.candidate_index]:
            raise ValueError("selected split count disagrees with candidate index")
        if self.storage.splits != self.selected_splits:
            raise ValueError("storage split count disagrees with launch selection")
        if min(
            self.batch,
            self.n_points,
            self.padded_degree,
            self.point_quantum,
            self.chunk_points,
            self.reduction_rows_per_chunk,
            self.num_chunks,
            self.bytes_per_point,
        ) <= 0:
            raise ValueError("workspace plan dimensions must be positive")
        if self.chunk_points > self.n_points:
            raise ValueError("point chunk exceeds the input")
        if (
            self.chunk_points >= self.point_quantum
            and self.chunk_points % self.point_quantum
        ):
            raise ValueError("large point chunks must be quantum-aligned")
        if self.reduction_rows_per_chunk != (
            self.batch * self.chunk_points * self.padded_degree
        ):
            raise ValueError("reduction row count is inconsistent")
        if self.selected_splits > self.reduction_rows_per_chunk:
            raise ValueError("split count exceeds rows in a workspace chunk")
        if self.num_chunks != _ceil_div(self.n_points, self.chunk_points):
            raise ValueError("point chunk count is inconsistent")
        if self.storage.strategy == "tree":
            if self.storage.carry_levels != self.num_chunks.bit_length():
                raise ValueError("tree carry storage is not self-consistent")
        elif self.storage.carry_levels != 0:
            raise ValueError("serial fallback must not allocate carry storage")
        if self.matrix_bytes != self.chunk_points * self.bytes_per_point:
            raise ValueError("matrix byte accounting is inconsistent")
        if self.allocated_bytes != self.matrix_bytes + self.storage.total_bytes:
            raise ValueError("allocated byte accounting is inconsistent")
        if self.planned_peak_bytes != self.allocated_bytes + self.headroom_bytes:
            raise ValueError("planned peak does not include exact headroom")
        if self.headroom_bytes < 0 or self.planned_peak_bytes > self.limit_bytes:
            raise ValueError("workspace plan exceeds its hard limit")

    @property
    def uses_tree(self) -> bool:
        return self.storage.strategy == "tree"

    @property
    def used_fallback(self) -> bool:
        return self.candidate_index != 0


def architecture_cta_target(
    architecture: int,
    *,
    sm_count: int | None = None,
    waves_per_sm: int = DEFAULT_CTA_WAVES_PER_SM,
    sm9_target_cap_ctas: int = SM9_DEFAULT_CTA_TARGET_CAP,
) -> ArchitectureCtaTarget:
    """Return an SM8/SM9 CTA target without querying a runtime device."""

    architecture = int(architecture)
    if architecture not in (8, 9):
        raise ValueError("architecture must be 8 or 9")
    if sm_count is None:
        sm_count = (
            SM8_DEFAULT_SM_COUNT if architecture == 8 else SM9_DEFAULT_SM_COUNT
        )
    sm_count = _require_positive("sm_count", sm_count)
    waves_per_sm = _require_positive("waves_per_sm", waves_per_sm)
    unbounded_target = sm_count * waves_per_sm
    target_cap = None
    target = unbounded_target
    if architecture == 9:
        target_cap = _require_positive(
            "sm9_target_cap_ctas", sm9_target_cap_ctas,
        )
        target = min(unbounded_target, target_cap)
    return ArchitectureCtaTarget(
        architecture=architecture,
        sm_count=sm_count,
        waves_per_sm=waves_per_sm,
        unbounded_target_ctas=unbounded_target,
        target_cap_ctas=target_cap,
        target_ctas=target,
    )


def select_split_k_candidates(
    *,
    architecture: int,
    reduction_rows: int,
    out_total: int,
    in_total: int,
    sm_count: int | None = None,
    block_out: int = DEFAULT_SPLIT_K_BLOCK_OUT,
    block_in: int = DEFAULT_SPLIT_K_BLOCK_IN,
    waves_per_sm: int = DEFAULT_CTA_WAVES_PER_SM,
    sm9_target_cap_ctas: int = SM9_DEFAULT_CTA_TARGET_CAP,
    max_splits: int = MAX_SPLIT_K_SPLITS,
    min_rows_per_split: int = 64,
) -> SplitKSelection:
    """Choose a CTA-wave split and every smaller power-of-two fallback.

    ``reduction_rows`` caps the split count so the launch never contains an
    empty split.  The workspace planner applies the same cap again after its
    chunk size is known.
    """

    reduction_rows = _require_positive("reduction_rows", reduction_rows)
    out_total = _require_positive("out_total", out_total)
    in_total = _require_positive("in_total", in_total)
    block_out = _require_positive("block_out", block_out)
    block_in = _require_positive("block_in", block_in)
    max_splits = _require_positive("max_splits", max_splits)
    min_rows_per_split = _require_positive(
        "min_rows_per_split", min_rows_per_split,
    )
    if (
        not _is_power_of_two(max_splits)
        or max_splits > MAX_SPLIT_K_SPLITS
    ):
        raise ValueError("max_splits must be a power of two no greater than 32")

    target = architecture_cta_target(
        architecture,
        sm_count=sm_count,
        waves_per_sm=waves_per_sm,
        sm9_target_cap_ctas=sm9_target_cap_ctas,
    )
    output_tiles = _ceil_div(out_total, block_out)
    input_tiles = _ceil_div(in_total, block_in)
    base_ctas = output_tiles * input_tiles
    requested_splits = _ceil_power_of_two(
        _ceil_div(target.target_ctas, base_ctas),
    )
    requested_splits = min(requested_splits, max_splits)
    # Keep at least one useful reduction tile in every split.  A sub-tile
    # problem remains valid with S=1, rather than launching mostly empty CTAs.
    useful_row_tiles = max(1, reduction_rows // min_rows_per_split)
    row_split_cap = _floor_power_of_two(min(useful_row_tiles, max_splits))
    ideal_splits = min(requested_splits, row_split_cap)
    candidates = tuple(
        ideal_splits >> shift
        for shift in range(ideal_splits.bit_length())
    )
    return SplitKSelection(
        target=target,
        reduction_rows=reduction_rows,
        out_total=out_total,
        in_total=in_total,
        block_out=block_out,
        block_in=block_in,
        output_tiles=output_tiles,
        input_tiles=input_tiles,
        base_ctas=base_ctas,
        row_split_cap=row_split_cap,
        ideal_splits=ideal_splits,
        candidates=candidates,
    )


def split_k_storage_accounting(
    *,
    splits: int,
    weight_bytes: int,
    cross_chunk_strategy: Literal["serial", "tree"] = "serial",
    carry_levels: int = 0,
) -> SplitKStorageAccounting:
    """Account within-chunk split reduction and cross-chunk accumulation.

    ``serial`` still permits ``S > 1``: the S partials are reduced inside a
    chunk, then the root is accumulated into ``grad_weight`` in point order.
    Only the cross-chunk tree needs full-size carry tiles.
    """

    splits = _require_positive("splits", splits)
    weight_bytes = _require_positive("weight_bytes", weight_bytes)
    carry_levels = _require_non_negative("carry_levels", carry_levels)
    if not _is_power_of_two(splits) or splits > MAX_SPLIT_K_SPLITS:
        raise ValueError("splits must be a power of two no greater than 32")
    if cross_chunk_strategy not in ("serial", "tree"):
        raise ValueError("cross_chunk_strategy must be 'serial' or 'tree'")
    strategy: Literal["serial", "tree"] = cross_chunk_strategy
    partial_tiles = splits
    scratch_tiles = 0 if splits == 1 else splits // 2
    if strategy == "serial":
        if carry_levels:
            raise ValueError("serial cross-chunk storage cannot have carry levels")
        carry_tiles = 0
    else:
        if carry_levels <= 0:
            raise ValueError("tree storage requires at least one carry level")
        carry_tiles = carry_levels
    partial_bytes = partial_tiles * weight_bytes
    scratch_bytes = scratch_tiles * weight_bytes
    carry_bytes = carry_tiles * weight_bytes
    return SplitKStorageAccounting(
        strategy=strategy,
        splits=splits,
        weight_bytes=weight_bytes,
        partial_tiles=partial_tiles,
        scratch_tiles=scratch_tiles,
        carry_levels=carry_levels,
        carry_tiles=carry_tiles,
        partial_bytes=partial_bytes,
        scratch_bytes=scratch_bytes,
        carry_bytes=carry_bytes,
        total_bytes=partial_bytes + scratch_bytes + carry_bytes,
    )


def _rounded_chunk_points(
    *,
    n_points: int,
    bytes_per_point: int,
    storage_bytes: int,
    workspace_bytes: int,
    headroom_bytes: int,
    point_quantum: int,
) -> int | None:
    available = workspace_bytes - headroom_bytes - storage_bytes
    if available < bytes_per_point:
        return None
    chunk_points = min(n_points, available // bytes_per_point)
    # Match the semi-packed launch contract: align useful chunks down, while
    # retaining a sub-quantum escape hatch under very small caps.
    if chunk_points >= point_quantum:
        chunk_points = (chunk_points // point_quantum) * point_quantum
    return chunk_points if chunk_points > 0 else None


def _build_workspace_plan(
    *,
    selection: SplitKSelection,
    selected_splits: int,
    candidate_index: int,
    storage: SplitKStorageAccounting,
    batch: int,
    n_points: int,
    padded_degree: int,
    point_quantum: int,
    chunk_points: int,
    bytes_per_point: int,
    min_rows_per_split: int,
    headroom_bytes: int,
    workspace_bytes: int,
) -> AdaptiveSplitKWorkspacePlan | None:
    reduction_rows_per_chunk = batch * chunk_points * padded_degree
    if selected_splits > reduction_rows_per_chunk:
        return None
    # Chunk shrinking must preserve the same useful-work floor used for the
    # full-shape split selection. Otherwise a tight workspace can retain a
    # large S while leaving almost every split with a tiny reduction range.
    # S == 1 remains the scratch-free last-resort fallback.
    if (
        selected_splits > 1
        and reduction_rows_per_chunk < selected_splits * min_rows_per_split
    ):
        return None
    num_chunks = _ceil_div(n_points, chunk_points)
    matrix_bytes = chunk_points * bytes_per_point
    allocated_bytes = matrix_bytes + storage.total_bytes
    planned_peak_bytes = allocated_bytes + headroom_bytes
    if planned_peak_bytes > workspace_bytes:
        return None
    return AdaptiveSplitKWorkspacePlan(
        selection=selection,
        selected_splits=selected_splits,
        candidate_index=candidate_index,
        storage=storage,
        batch=batch,
        n_points=n_points,
        padded_degree=padded_degree,
        point_quantum=point_quantum,
        chunk_points=chunk_points,
        reduction_rows_per_chunk=reduction_rows_per_chunk,
        num_chunks=num_chunks,
        bytes_per_point=bytes_per_point,
        matrix_bytes=matrix_bytes,
        allocated_bytes=allocated_bytes,
        headroom_bytes=headroom_bytes,
        planned_peak_bytes=planned_peak_bytes,
        limit_bytes=workspace_bytes,
    )


def _fit_tree_candidate(
    *,
    selection: SplitKSelection,
    selected_splits: int,
    candidate_index: int,
    weight_bytes: int,
    batch: int,
    n_points: int,
    padded_degree: int,
    point_quantum: int,
    bytes_per_point: int,
    min_rows_per_split: int,
    headroom_bytes: int,
    workspace_bytes: int,
    carry_level_limit: int,
) -> AdaptiveSplitKWorkspacePlan | None:
    # Carry bytes influence chunk size; chunk size influences chunk count and
    # therefore carry levels.  Starting at one and only increasing reaches the
    # least fixed point, or proves that no tree fits the hard cap.
    carry_levels = 1
    while carry_levels <= carry_level_limit:
        storage = split_k_storage_accounting(
            splits=selected_splits,
            weight_bytes=weight_bytes,
            cross_chunk_strategy="tree",
            carry_levels=carry_levels,
        )
        chunk_points = _rounded_chunk_points(
            n_points=n_points,
            bytes_per_point=bytes_per_point,
            storage_bytes=storage.total_bytes,
            workspace_bytes=workspace_bytes,
            headroom_bytes=headroom_bytes,
            point_quantum=point_quantum,
        )
        if chunk_points is None:
            return None
        num_chunks = _ceil_div(n_points, chunk_points)
        required_levels = num_chunks.bit_length()
        if required_levels > carry_level_limit:
            return None
        if required_levels == carry_levels:
            return _build_workspace_plan(
                selection=selection,
                selected_splits=selected_splits,
                candidate_index=candidate_index,
                storage=storage,
                batch=batch,
                n_points=n_points,
                padded_degree=padded_degree,
                point_quantum=point_quantum,
                chunk_points=chunk_points,
                bytes_per_point=bytes_per_point,
                min_rows_per_split=min_rows_per_split,
                headroom_bytes=headroom_bytes,
                workspace_bytes=workspace_bytes,
            )
        # Increasing carry storage cannot increase chunk size, so required
        # levels are monotonic and this update cannot oscillate.
        carry_levels = required_levels
    return None


def adaptive_split_k_workspace_plan(
    *,
    architecture: int,
    batch: int,
    n_points: int,
    padded_degree: int,
    out_total: int,
    in_total: int,
    workspace_bytes: int,
    sm_count: int | None = None,
    element_size: int = 4,
    accumulator_element_size: int = 4,
    block_out: int = DEFAULT_SPLIT_K_BLOCK_OUT,
    block_in: int = DEFAULT_SPLIT_K_BLOCK_IN,
    waves_per_sm: int = DEFAULT_CTA_WAVES_PER_SM,
    sm9_target_cap_ctas: int = SM9_DEFAULT_CTA_TARGET_CAP,
    max_splits: int = MAX_SPLIT_K_SPLITS,
    min_rows_per_split: int = 64,
    point_quantum: int = DEFAULT_POINT_CHUNK_QUANTUM,
    carry_level_limit: int = DEFAULT_TREE_CARRY_LEVEL_LIMIT,
    headroom_bytes: int = DEFAULT_BACKWARD_WORKSPACE_HEADROOM_BYTES,
    cross_chunk_strategy: Literal["serial", "tree"] = "serial",
) -> AdaptiveSplitKWorkspacePlan | None:
    """Select and fit adaptive split-K below a strict workspace byte cap.

    Candidates are attempted from the shape-optimal split count down through
    smaller powers of two. After workspace-driven chunk shrinking, every
    ``S > 1`` candidate is checked again against ``min_rows_per_split``; a
    candidate that became too fine-grained falls back to the next smaller
    power of two. Serial cross-chunk accumulation keeps adaptive within-chunk
    split-K but allocates no carry matrices. Tree mode uses the least
    self-consistent carry count and finally tries scratch-free serial
    ``S == 1`` if no tree fits. Every returned plan includes allocator
    headroom in ``planned_peak_bytes``.
    """

    batch = _require_positive("batch", batch)
    n_points = _require_positive("n_points", n_points)
    padded_degree = _require_positive("padded_degree", padded_degree)
    out_total = _require_positive("out_total", out_total)
    in_total = _require_positive("in_total", in_total)
    workspace_bytes = _require_positive("workspace_bytes", workspace_bytes)
    element_size = _require_positive("element_size", element_size)
    accumulator_element_size = _require_positive(
        "accumulator_element_size", accumulator_element_size,
    )
    point_quantum = _require_positive("point_quantum", point_quantum)
    carry_level_limit = _require_positive(
        "carry_level_limit", carry_level_limit,
    )
    headroom_bytes = _require_non_negative("headroom_bytes", headroom_bytes)
    if cross_chunk_strategy not in ("serial", "tree"):
        raise ValueError("cross_chunk_strategy must be 'serial' or 'tree'")

    full_reduction_rows = batch * n_points * padded_degree
    selection = select_split_k_candidates(
        architecture=architecture,
        reduction_rows=full_reduction_rows,
        out_total=out_total,
        in_total=in_total,
        sm_count=sm_count,
        block_out=block_out,
        block_in=block_in,
        waves_per_sm=waves_per_sm,
        sm9_target_cap_ctas=sm9_target_cap_ctas,
        max_splits=max_splits,
        min_rows_per_split=min_rows_per_split,
    )
    weight_bytes = out_total * in_total * accumulator_element_size
    bytes_per_point = (
        batch * padded_degree * (out_total + in_total) * element_size
    )

    for candidate_index, selected_splits in enumerate(selection.candidates):
        if cross_chunk_strategy == "tree":
            plan = _fit_tree_candidate(
                selection=selection,
                selected_splits=selected_splits,
                candidate_index=candidate_index,
                weight_bytes=weight_bytes,
                batch=batch,
                n_points=n_points,
                padded_degree=padded_degree,
                point_quantum=point_quantum,
                bytes_per_point=bytes_per_point,
                min_rows_per_split=min_rows_per_split,
                headroom_bytes=headroom_bytes,
                workspace_bytes=workspace_bytes,
                carry_level_limit=carry_level_limit,
            )
            if plan is None and selected_splits == 1:
                storage = split_k_storage_accounting(
                    splits=1,
                    weight_bytes=weight_bytes,
                    cross_chunk_strategy="serial",
                )
                chunk_points = _rounded_chunk_points(
                    n_points=n_points,
                    bytes_per_point=bytes_per_point,
                    storage_bytes=storage.total_bytes,
                    workspace_bytes=workspace_bytes,
                    headroom_bytes=headroom_bytes,
                    point_quantum=point_quantum,
                )
                if chunk_points is not None:
                    plan = _build_workspace_plan(
                        selection=selection,
                        selected_splits=selected_splits,
                        candidate_index=candidate_index,
                        storage=storage,
                        batch=batch,
                        n_points=n_points,
                        padded_degree=padded_degree,
                        point_quantum=point_quantum,
                        chunk_points=chunk_points,
                        bytes_per_point=bytes_per_point,
                        min_rows_per_split=min_rows_per_split,
                        headroom_bytes=headroom_bytes,
                        workspace_bytes=workspace_bytes,
                    )
        else:
            storage = split_k_storage_accounting(
                splits=selected_splits,
                weight_bytes=weight_bytes,
                cross_chunk_strategy="serial",
            )
            chunk_points = _rounded_chunk_points(
                n_points=n_points,
                bytes_per_point=bytes_per_point,
                storage_bytes=storage.total_bytes,
                workspace_bytes=workspace_bytes,
                headroom_bytes=headroom_bytes,
                point_quantum=point_quantum,
            )
            plan = None
            if chunk_points is not None:
                plan = _build_workspace_plan(
                    selection=selection,
                    selected_splits=selected_splits,
                    candidate_index=candidate_index,
                    storage=storage,
                    batch=batch,
                    n_points=n_points,
                    padded_degree=padded_degree,
                    point_quantum=point_quantum,
                    chunk_points=chunk_points,
                    bytes_per_point=bytes_per_point,
                    min_rows_per_split=min_rows_per_split,
                    headroom_bytes=headroom_bytes,
                    workspace_bytes=workspace_bytes,
                )
        if plan is not None:
            return plan
    return None


__all__ = [
    "AdaptiveSplitKWorkspacePlan",
    "ArchitectureCtaTarget",
    "DEFAULT_BACKWARD_WORKSPACE_HEADROOM_BYTES",
    "DEFAULT_CTA_WAVES_PER_SM",
    "DEFAULT_POINT_CHUNK_QUANTUM",
    "DEFAULT_SPLIT_K_BLOCK_IN",
    "DEFAULT_SPLIT_K_BLOCK_OUT",
    "DEFAULT_TREE_CARRY_LEVEL_LIMIT",
    "MAX_SPLIT_K_SPLITS",
    "MIN_SPLIT_K_SPLITS",
    "SM8_DEFAULT_CTA_TARGET",
    "SM8_DEFAULT_SM_COUNT",
    "SM9_DEFAULT_CTA_TARGET_CAP",
    "SM9_DEFAULT_SM_COUNT",
    "SplitKSelection",
    "SplitKStorageAccounting",
    "adaptive_split_k_workspace_plan",
    "architecture_cta_target",
    "select_split_k_candidates",
    "split_k_storage_accounting",
]
