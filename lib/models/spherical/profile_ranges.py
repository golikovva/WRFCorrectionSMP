"""Low-overhead named profiler ranges for spherical Irrep models."""

from __future__ import annotations

from contextlib import nullcontext
from contextvars import ContextVar
from collections import defaultdict
from typing import ContextManager, Any

import torch
from torch.profiler import record_function


_RANGES_ENABLED: ContextVar[bool] = ContextVar("irrep_profile_ranges_enabled", default=False)
_CUDA_EVENTS: ContextVar[list[tuple[str, Any, Any]] | None] = ContextVar(
    "irrep_profile_cuda_events", default=None,
)


class profile_ranges:
    """Enable Irrep-specific ``record_function`` ranges in the current context."""

    def __init__(self, enabled: bool = True, *, collect_cuda: bool = True) -> None:
        self.enabled = bool(enabled)
        self.collect_cuda = bool(collect_cuda and torch.cuda.is_available())
        self._token = None
        self._events_token = None
        self.cuda_events: list[tuple[str, Any, Any]] = []

    def __enter__(self) -> "profile_ranges":
        self._token = _RANGES_ENABLED.set(self.enabled)
        self._events_token = _CUDA_EVENTS.set(self.cuda_events if self.collect_cuda else None)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self._token is not None
        assert self._events_token is not None
        _CUDA_EVENTS.reset(self._events_token)
        _RANGES_ENABLED.reset(self._token)

    def cuda_rows(self, mode: str) -> list[dict[str, Any]]:
        """Aggregate recorded CUDA events. Call after synchronizing the device."""

        totals: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "cuda_total_ms": 0.0}
        )
        for name, start, end in self.cuda_events:
            row = totals[name]
            row["calls"] += 1
            row["cuda_total_ms"] += float(start.elapsed_time(end))
        return [
            {
                "mode": mode,
                "name": name,
                "calls": values["calls"],
                "cuda_total_ms": values["cuda_total_ms"],
                "cuda_average_ms": values["cuda_total_ms"] / values["calls"],
            }
            for name, values in totals.items()
        ]


class _RecordedRegion:
    def __init__(self, name: str, events: list[tuple[str, Any, Any]] | None) -> None:
        self.name = name
        self.events = events
        self.profiler_region = record_function(name)
        self.start = None
        self.end = None

    def __enter__(self) -> "_RecordedRegion":
        self.profiler_region.__enter__()
        if self.events is not None:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.events is not None:
            assert self.start is not None and self.end is not None
            self.end.record()
            self.events.append((self.name, self.start, self.end))
        self.profiler_region.__exit__(exc_type, exc_value, traceback)


def record_region(name: str) -> ContextManager[object]:
    """Return a named profiler range, or a no-op when profiling is disabled."""

    if not _RANGES_ENABLED.get():
        return nullcontext()
    full_name = f"irrep::{name}"
    return _RecordedRegion(full_name, _CUDA_EVENTS.get())


__all__ = ["profile_ranges", "record_region"]
