import json
import os
import time
import faulthandler
from collections import defaultdict
from typing import Any

import torch


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _process_rss_mb():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass

    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except (ImportError, OSError):
        return None


def count_temporal_entries(results, aggregator_name):
    total = 0
    for metric_results in results.values():
        accumulator = metric_results.get(aggregator_name)
        if not isinstance(accumulator, dict):
            continue
        for region_name, date_data in accumulator.items():
            if region_name == "__meta__" or not isinstance(date_data, dict):
                continue
            total += len(date_data)
    return total


class TestLoopProfiler:
    __test__ = False

    def __init__(self, cfg, save_dir):
        profile_cfg = _cfg_get(_cfg_get(cfg, "test_config"), "profiling", {})
        self.enabled = bool(_cfg_get(profile_cfg, "enabled", False))
        self.log_every = max(1, int(_cfg_get(profile_cfg, "log_every", 10)))
        self.warmup_batches = max(0, int(_cfg_get(profile_cfg, "warmup_batches", 1)))
        self.top_k = max(1, int(_cfg_get(profile_cfg, "top_k", 12)))
        self.sync_cuda = bool(_cfg_get(profile_cfg, "sync_cuda", True))
        self.console_output = bool(_cfg_get(profile_cfg, "console_output", False))
        self.slow_batch_seconds = float(_cfg_get(profile_cfg, "slow_batch_seconds", 60))
        self.stack_dump_seconds = float(_cfg_get(profile_cfg, "stack_dump_seconds", 0))
        self.output_path = os.path.join(
            save_dir,
            str(_cfg_get(profile_cfg, "output_file", "test_profile.jsonl")),
        )

        device = torch.device(_cfg_get(cfg, "device", "cpu"))
        self.cuda_enabled = (
            self.enabled
            and self.sync_cuda
            and device.type == "cuda"
            and torch.cuda.is_available()
        )
        self.cuda_device = device if device.type == "cuda" else None

        self.batch_index = 0
        self.window_batches = 0
        self.window_durations = defaultdict(float)
        self.window_counts = defaultdict(int)
        self.wait_started_at = None
        self.initial_rss_mb = _process_rss_mb() if self.enabled else None

        if self.enabled:
            os.makedirs(save_dir, exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8"):
                pass
            if self.cuda_enabled:
                torch.cuda.reset_peak_memory_stats(self.cuda_device)
            if self.stack_dump_seconds > 0:
                faulthandler.dump_traceback_later(
                    self.stack_dump_seconds,
                    repeat=True,
                )
            if self.console_output:
                print(
                    "[test-profile] enabled "
                    f"log_every={self.log_every}, sync_cuda={self.cuda_enabled}, "
                    f"output={self.output_path}"
                )

    def _sync(self):
        if self.cuda_enabled:
            torch.cuda.synchronize(self.cuda_device)

    def start(self):
        if not self.enabled:
            return None
        self._sync()
        return time.perf_counter()

    def stop(self, name, started_at):
        if not self.enabled or started_at is None:
            return
        self._sync()
        self.record(name, time.perf_counter() - started_at)

    def record(self, name, duration_seconds):
        if not self.enabled or self.batch_index < self.warmup_batches:
            return
        if (
            self.console_output
            and name == "batch_processing"
            and duration_seconds >= self.slow_batch_seconds
        ):
            print(
                f"[test-profile] slow batch {self.batch_index + 1}: "
                f"{duration_seconds:.1f}s"
            )
        self.window_durations[name] += float(duration_seconds)
        self.window_counts[name] += 1

    def start_data_wait(self):
        if self.enabled:
            self.wait_started_at = time.perf_counter()

    def batch_received(self):
        if not self.enabled:
            return
        now = time.perf_counter()
        if self.wait_started_at is not None:
            self.record("data_wait", now - self.wait_started_at)
        self.wait_started_at = None

    def finish_batch(self, *, regional_date_entries=None):
        if not self.enabled:
            return

        self.batch_index += 1
        if self.batch_index > self.warmup_batches:
            self.window_batches += 1

        if self.window_batches >= self.log_every:
            self._report(regional_date_entries=regional_date_entries)

        self.start_data_wait()

    def close(self, *, regional_date_entries=None):
        if not self.enabled:
            return
        if self.window_batches:
            self._report(regional_date_entries=regional_date_entries)
        if self.stack_dump_seconds > 0:
            faulthandler.cancel_dump_traceback_later()

    def _memory_stats(self):
        rss_mb = _process_rss_mb()
        stats = {
            "rss_mb": rss_mb,
            "rss_delta_mb": (
                rss_mb - self.initial_rss_mb
                if rss_mb is not None and self.initial_rss_mb is not None
                else None
            ),
        }
        if self.cuda_device is not None and torch.cuda.is_available():
            stats.update(
                {
                    "cuda_allocated_mb": torch.cuda.memory_allocated(self.cuda_device) / (1024 ** 2),
                    "cuda_reserved_mb": torch.cuda.memory_reserved(self.cuda_device) / (1024 ** 2),
                    "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(self.cuda_device) / (1024 ** 2),
                }
            )
        return stats

    def _report(self, *, regional_date_entries=None):
        timings_ms = {
            name: self.window_durations[name] / self.window_counts[name] * 1000
            for name in self.window_durations
            if self.window_counts[name]
        }
        ordered_timings = dict(
            sorted(timings_ms.items(), key=lambda item: item[1], reverse=True)
        )
        memory = self._memory_stats()
        record = {
            "batch": self.batch_index,
            "window_batches": self.window_batches,
            "timings_ms": ordered_timings,
            **memory,
            "regional_date_entries": regional_date_entries,
        }
        with open(self.output_path, "a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, ensure_ascii=True) + "\n")

        if self.console_output:
            top = list(ordered_timings.items())[: self.top_k]
            top_text = ", ".join(f"{name}={value:.1f}ms" for name, value in top)
            rss_text = (
                f"rss={memory['rss_mb']:.0f}MB "
                f"(delta={memory['rss_delta_mb']:+.0f}MB)"
                if memory["rss_mb"] is not None
                else "rss=n/a"
            )
            cuda_text = ""
            if "cuda_allocated_mb" in memory:
                cuda_text = (
                    f", cuda={memory['cuda_allocated_mb']:.0f}/"
                    f"{memory['cuda_reserved_mb']:.0f}MB"
                )
            regional_text = (
                f", regional_dates={regional_date_entries}"
                if regional_date_entries is not None
                else ""
            )
            print(
                f"[test-profile] batch={self.batch_index}: {rss_text}"
                f"{cuda_text}{regional_text}; {top_text}"
            )

        self.window_batches = 0
        self.window_durations.clear()
        self.window_counts.clear()
