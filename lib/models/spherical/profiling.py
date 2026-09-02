"""Notebook-facing performance tools for :class:`IrrepSphereUNet`.

The module deliberately separates latency measurement from tracing: CUDA events
provide low-overhead timings, while ``torch.profiler`` provides attribution and
Chrome traces.  Nsight Compute is launched in a fresh process so CUPTI replay
does not interfere with the notebook kernel.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
from dataclasses import asdict, dataclass, field, fields
from io import StringIO
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import pandas as pd
import torch
from torch import Tensor, nn
from torch.profiler import ProfilerActivity

from .irrep_layers import IrrepSphereConv, _IrrepConvGeometry
from .profile_ranges import profile_ranges, record_region


ProfileStep = Callable[[], Any]


@dataclass(frozen=True)
class IrrepProfileConfig:
    """Configuration shared by model traces and convolution microbenchmarks."""

    warmup: int = 5
    iterations: int = 20
    repeats: int = 3
    profile_iterations: int = 2
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    micro_warmup: int = 5
    micro_iterations: int = 20
    micro_repeats: int = 3
    correctness_atol: float = 5e-4
    correctness_rtol: float = 5e-4

    def __post_init__(self) -> None:
        for name in (
            "warmup", "iterations", "repeats", "profile_iterations",
            "micro_warmup", "micro_iterations", "micro_repeats",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class _ConvCase:
    name: str
    module: IrrepSphereConv
    shape: tuple[int, ...]
    device: torch.device
    dtype: torch.dtype
    geometry: _IrrepConvGeometry
    selected_path: str
    fallback_reason: str | None


@dataclass
class IrrepProfileReport:
    """Collected tables and artifact paths from a profiling session."""

    run_table: pd.DataFrame
    module_table: pd.DataFrame
    operation_table: pd.DataFrame
    kernel_table: pd.DataFrame
    backend_decisions: pd.DataFrame
    trace_paths: dict[str, Path]
    metadata: dict[str, Any]
    _cases: list[_ConvCase] = field(repr=False, default_factory=list)
    _config: IrrepProfileConfig = field(repr=False, default_factory=IrrepProfileConfig)
    _output_dir: Path = field(repr=False, default_factory=lambda: Path("profiling"))

    def display(self) -> "IrrepProfileReport":
        """Display the main tables when called from Jupyter and return ``self``."""

        try:
            from IPython.display import display
        except ImportError:
            return self
        for title, table in (
            ("Run summary", self.run_table),
            ("U-Net modules", self.module_table),
            ("Internal operations", self.operation_table),
            ("CUDA kernels", self.kernel_table),
            ("Backend decisions", self.backend_decisions),
        ):
            print(title)
            display(table)
        return self

    def export_chrome_trace(self) -> dict[str, Path]:
        """Return trace paths (traces are written during ``profile``)."""

        return dict(self.trace_paths)

    def compare_convolution_backends(self) -> "ConvolutionComparison":
        """Create a benchmark over every unique captured convolution workload."""

        return ConvolutionComparison(self._cases, self._config, self._output_dir / "microbench")


class IrrepSphereProfiler:
    """Profile an Irrep model using arbitrary no-argument notebook step callbacks."""

    def __init__(
        self,
        model: nn.Module,
        *,
        output_dir: str | Path = "profiling/irrep_sphere_unet",
        config: IrrepProfileConfig | None = None,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.config = config or IrrepProfileConfig()
        self._captured: dict[tuple[Any, ...], _ConvCase] = {}
        self._block_name_originals: list[tuple[nn.Module, Any]] = []

    def profile(self, steps: Mapping[str, ProfileStep] | ProfileStep) -> IrrepProfileReport:
        """Profile one or more inference/training steps.

        A single callable is named ``"step"``.  A mapping such as
        ``{"inference": infer_step, "training": train_step}`` keeps modes in
        separate traces and timing distributions.
        """

        if not torch.cuda.is_available():
            raise RuntimeError("IrrepSphereProfiler requires a CUDA-enabled PyTorch runtime")
        if callable(steps):
            normalized_steps: Mapping[str, ProfileStep] = {"step": steps}
        else:
            normalized_steps = steps
        if not normalized_steps:
            raise ValueError("at least one profiling step is required")

        self._captured.clear()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        originals = self._install_capture()
        run_rows: list[dict[str, Any]] = []
        module_frames: list[pd.DataFrame] = []
        operation_frames: list[pd.DataFrame] = []
        kernel_frames: list[pd.DataFrame] = []
        trace_paths: dict[str, Path] = {}
        try:
            for mode, step in normalized_steps.items():
                if not callable(step):
                    raise TypeError(f"step {mode!r} is not callable")
                run_rows.append(self._measure_step(mode, step))
                trace_path = self.output_dir / f"{_slug(mode)}.trace.json"
                module, operations, kernels = self._trace_step(mode, step, trace_path)
                module_frames.append(module)
                operation_frames.append(operations)
                kernel_frames.append(kernels)
                trace_paths[mode] = trace_path
        finally:
            self._remove_capture(originals)

        metadata = _runtime_metadata(self.model, self.config)
        if not any(not frame.empty for frame in kernel_frames):
            metadata["warnings"] = [
                "torch.profiler returned no CUDA kernel events; named range timings use "
                "CUDA Events and per-kernel hardware data remains available through Nsight Compute"
            ]
        metadata_path = self.output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        decisions = self._backend_decisions()
        report = IrrepProfileReport(
            run_table=pd.DataFrame(run_rows),
            module_table=_concat_frames(module_frames),
            operation_table=_concat_frames(operation_frames),
            kernel_table=_concat_frames(kernel_frames),
            backend_decisions=decisions,
            trace_paths=trace_paths,
            metadata=metadata,
            _cases=list(self._captured.values()),
            _config=self.config,
            _output_dir=self.output_dir,
        )
        self._write_tables(report)
        return report

    def _install_capture(self) -> list[tuple[IrrepSphereConv, Any, Any]]:
        originals = []
        for name, module in self.model.named_modules():
            if not isinstance(module, IrrepSphereConv):
                continue
            old_name = getattr(module, "_irrep_profile_name", None)
            old_capture = getattr(module, "_irrep_profile_capture", None)
            module._irrep_profile_name = name or "IrrepSphereConv"
            module._irrep_profile_capture = self._capture_case
            originals.append((module, old_name, old_capture))
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == "_IrrepSphereDoubleConv":
                self._block_name_originals.append(
                    (module, getattr(module, "_irrep_profile_name", None))
                )
                setattr(module, "_irrep_profile_name", name)
        return originals

    def _remove_capture(self, originals: Sequence[tuple[IrrepSphereConv, Any, Any]]) -> None:
        for module, old_name, old_capture in originals:
            if old_name is None:
                module.__dict__.pop("_irrep_profile_name", None)
            else:
                module._irrep_profile_name = old_name
            if old_capture is None:
                module.__dict__.pop("_irrep_profile_capture", None)
            else:
                module._irrep_profile_capture = old_capture
        for module, old_name in self._block_name_originals:
            if old_name is None:
                module.__dict__.pop("_irrep_profile_name", None)
            else:
                module._irrep_profile_name = old_name
        self._block_name_originals.clear()

    def _capture_case(self, module: IrrepSphereConv, x: Tensor, geometry: _IrrepConvGeometry) -> None:
        name = getattr(module, "_irrep_profile_name", "IrrepSphereConv")
        use_triton = module._should_use_triton(x, geometry)
        selected = "triton" if use_triton else "packed" if module._use_packed_fast_path else "blockwise"
        reason = module._triton_unsupported_reason(x, geometry)
        if module.backend == "torch":
            reason = "backend='torch'"
        elif module.backend == "auto" and not use_triton and reason is None:
            reason = "auto workload threshold selected Torch"
        signature = (
            tuple(x.shape), x.dtype, geometry.n_points, int(geometry.neighbor_idx.numel()),
            module.in_type.multiplicities, module.out_type.multiplicities,
            module.num_radial, module.quadrature, selected,
        )
        self._captured.setdefault(
            signature,
            _ConvCase(name, module, tuple(x.shape), x.device, x.dtype, geometry, selected, reason),
        )

    def _measure_step(self, mode: str, step: ProfileStep) -> dict[str, Any]:
        cfg = self.config
        for _ in range(cfg.warmup):
            step()
        torch.cuda.synchronize()
        samples: list[float] = []
        peaks: list[float] = []
        for _ in range(cfg.repeats):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(cfg.iterations):
                step()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / cfg.iterations)
            peaks.append((torch.cuda.max_memory_allocated() - baseline) / 2**20)
        median = statistics.median(samples)
        return {
            "mode": mode,
            "median_ms": median,
            "p10_ms": _quantile(samples, 0.10),
            "p90_ms": _quantile(samples, 0.90),
            "steps_per_second": 1000.0 / median,
            "peak_cuda_mib": statistics.median(peaks),
            "samples": len(samples),
        }

    def _trace_step(
        self, mode: str, step: ProfileStep, trace_path: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile_ranges() as ranges, torch.profiler.profile(
            activities=activities,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
        ) as prof:
            for _ in range(self.config.profile_iterations):
                with record_region(f"step/{mode}"):
                    step()
                prof.step()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(trace_path))
        averages = prof.key_averages(group_by_input_shape=self.config.record_shapes)
        rows = [_event_average_row(mode, event) for event in averages]
        cpu_ranges = pd.DataFrame(
            row for row in rows if row["name"].startswith("irrep::")
        )
        cuda_ranges = pd.DataFrame(ranges.cuda_rows(mode))
        if cuda_ranges.empty:
            combined_ranges = cpu_ranges
        elif cpu_ranges.empty:
            combined_ranges = cuda_ranges
        else:
            cpu_ranges = cpu_ranges.drop(
                columns=["cuda_total_ms", "self_cuda_ms"], errors="ignore",
            )
            cpu_ranges = cpu_ranges.groupby(["mode", "name"], as_index=False).agg(
                {
                    "calls": "sum", "cpu_total_ms": "sum", "self_cpu_ms": "sum",
                    "cpu_memory_mib": "sum", "cuda_memory_mib": "sum",
                }
            )
            cpu_ranges = cpu_ranges.rename(columns={"calls": "profiler_calls"})
            combined_ranges = cuda_ranges.merge(cpu_ranges, on=["mode", "name"], how="left")
        module = combined_ranges[
            combined_ranges["name"].str.startswith(("irrep::module/", "irrep::stage/"))
        ].reset_index(drop=True)
        operations = combined_ranges[
            combined_ranges["name"].str.startswith("irrep::op/")
        ].reset_index(drop=True)
        kernels = _kernel_table(mode, prof.events())
        return module, operations, kernels

    def _backend_decisions(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "module": case.name,
                "input_shape": case.shape,
                "points": case.geometry.n_points,
                "edges": int(case.geometry.neighbor_idx.numel()),
                "configured_backend": case.module.backend,
                "selected_path": case.selected_path,
                "fallback_reason": case.fallback_reason,
                "dtype": str(case.dtype).removeprefix("torch."),
            }
            for case in self._captured.values()
        )

    def _write_tables(self, report: IrrepProfileReport) -> None:
        for name, table in (
            ("run", report.run_table), ("modules", report.module_table),
            ("operations", report.operation_table), ("kernels", report.kernel_table),
            ("backends", report.backend_decisions),
        ):
            table.to_csv(self.output_dir / f"{name}.csv", index=False)


class ConvolutionComparison:
    """Correctness-guarded microbenchmarks for captured convolution shapes."""

    def __init__(
        self, cases: Sequence[_ConvCase], config: IrrepProfileConfig, output_dir: Path,
    ) -> None:
        self.cases = list(cases)
        self.config = config
        self.output_dir = output_dir
        self.table = pd.DataFrame()

    def run(self, *, include_backward: bool = True) -> pd.DataFrame:
        if not self.cases:
            raise RuntimeError("no convolution cases were captured; call profiler.profile() first")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        # Every captured shape specializes the same two Python call sites used
        # by the compiled Torch baselines below.  Forward and training also
        # need separate variants because ``requires_grad`` is guarded, so a
        # U-Net easily exceeds TorchDynamo's default per-code-object cache
        # limit (8).  Raise the limit only for this benchmark and restore it
        # on exit through the config patch context manager.
        compile_cache_limit = max(
            int(torch._dynamo.config.cache_size_limit),
            2 * len(self.cases) + 2,
        )
        with torch._dynamo.config.patch(cache_size_limit=compile_cache_limit):
            for case in self.cases:
                rows.extend(self._run_case(case, include_backward=include_backward))
        self.table = pd.DataFrame(rows)
        self.table.to_csv(self.output_dir / "backend_comparison.csv", index=False)
        return self.table

    def _run_case(self, case: _ConvCase, *, include_backward: bool) -> list[dict[str, Any]]:
        module = case.module
        module._bind_prepared_geometry(case.geometry)
        geometry = module._prepared_geometry
        assert geometry is not None
        x_base = torch.randn(case.shape, device=case.device, dtype=case.dtype)
        paths: dict[str, Callable[[Tensor], Tensor]] = {
            "blockwise": lambda x: module._forward_spatial_blockwise(x, geometry),
        }
        if module._use_packed_fast_path:
            paths["packed"] = lambda x: module._forward_spatial_packed(x, geometry)
        for path_name in ("blockwise", "packed"):
            eager_path = paths.get(path_name)
            if eager_path is not None:
                paths[f"{path_name}_compiled"] = torch.compile(
                    eager_path,
                    fullgraph=True,
                    dynamic=False,
                )
        reason = module._triton_unsupported_reason(x_base, geometry)
        if reason is None:
            def irregular_triton(x: Tensor, *, fast_path: bool) -> Tensor:
                previous = module._irregular_r1_fast_path
                module._irregular_r1_fast_path = fast_path
                try:
                    return module._forward_spatial_triton(x, geometry)
                finally:
                    module._irregular_r1_fast_path = previous

            if not module._regular_weights and module.num_radial == 1:
                paths["triton_legacy"] = lambda x: irregular_triton(
                    x, fast_path=False,
                )
            if module._regular_weights and module.num_radial == 1:
                def regular_triton(x: Tensor, *, variant: str) -> Tensor:
                    previous = module.regular_r1_variant
                    module.regular_r1_variant = variant
                    try:
                        return module._forward_spatial_triton(x, geometry)
                    finally:
                        module.regular_r1_variant = previous

                paths["triton_fused"] = lambda x: regular_triton(
                    x, variant="fused",
                )
                previous_variant = module.regular_r1_variant
                module.regular_r1_variant = "semi_packed"
                try:
                    semi_reason = module._triton_unsupported_reason(x_base, geometry)
                finally:
                    module.regular_r1_variant = previous_variant
                if semi_reason is None:
                    paths["triton_semi_packed"] = lambda x: regular_triton(
                        x, variant="semi_packed",
                    )
            paths["triton"] = lambda x: module._forward_spatial_triton(x, geometry)
        reference_name = "packed" if "packed" in paths else "blockwise"
        reference = paths[reference_name](x_base).detach()
        grad_out = torch.randn_like(reference)
        reference_grad_x: Tensor | None = None
        reference_grad_weights: tuple[Tensor, ...] | None = None
        packed_weight = getattr(module, "packed_weight", None)
        weight_parameters = (
            (packed_weight,)
            if packed_weight is not None
            else tuple(module.weights.values())
        )
        if include_backward:
            reference_x = x_base.detach().requires_grad_(True)
            reference_y = paths[reference_name](reference_x)
            reference_grads = torch.autograd.grad(
                reference_y, (reference_x, *weight_parameters), grad_out,
            )
            reference_grad_x = reference_grads[0]
            reference_grad_weights = tuple(reference_grads[1:])
            if reference_grad_x is None or any(
                grad is None for grad in reference_grad_weights
            ):
                raise RuntimeError("reference convolution returned an unused gradient")
            reference_grad_x = reference_grad_x.detach()
            reference_grad_weights = tuple(
                grad.detach() for grad in reference_grad_weights
            )
        original_grads = tuple(parameter.grad for parameter in weight_parameters)
        rows = []

        def profiled_regular_variant(
            path_name: str,
            *,
            training: bool = False,
        ) -> str | None:
            if path_name in ("triton_fused", "triton_semi_packed"):
                return path_name.removeprefix("triton_")
            if (
                path_name == "triton"
                and module._regular_weights
                and module.num_radial == 1
            ):
                if training:
                    probe = x_base.detach().requires_grad_(True)
                    return module._selected_regular_r1_variant(probe, geometry)
                with torch.inference_mode():
                    return module._selected_regular_r1_variant(x_base, geometry)
            return None

        try:
            for path_name, path in paths.items():
                try:
                    with torch.inference_mode():
                        actual = path(x_base).detach()
                    delta = (actual.float() - reference.float()).abs()
                    close = torch.allclose(
                        actual, reference,
                        atol=_dtype_tolerance(case.dtype, self.config.correctness_atol),
                        rtol=_dtype_tolerance(case.dtype, self.config.correctness_rtol),
                    )
                    def forward_step() -> Tensor:
                        with torch.inference_mode():
                            return path(x_base)

                    forward = _measure_cuda(
                        forward_step, self.config.micro_warmup,
                        self.config.micro_iterations, self.config.micro_repeats,
                    )
                    rows.append({
                        "module": case.name, "input_shape": case.shape, "path": path_name,
                        "mode": "forward", "median_ms": forward[0], "peak_cuda_mib": forward[1],
                        "regular_r1_variant": profiled_regular_variant(path_name),
                        "workspace_limit_mib": (
                            module.triton_workspace_mib if path_name.startswith("triton") else None
                        ),
                        "correct": bool(close), "max_abs_error": float(delta.max()),
                        "grad_x_correct": None, "grad_weight_correct": None,
                        "grad_x_max_abs_error": None, "grad_weight_max_abs_error": None,
                        "speedup_vs_reference": None,
                    })
                    if include_backward:
                        grad_x_close = None
                        grad_weight_close = None
                        grad_x_error = None
                        grad_weight_error = None
                        check_x = x_base.detach().requires_grad_(True)
                        check_y = path(check_x)
                        check_grads = torch.autograd.grad(
                            check_y, (check_x, *weight_parameters), grad_out,
                        )
                        check_grad_x = check_grads[0]
                        check_grad_weights = tuple(check_grads[1:])
                        grad_tolerance = _gradient_tolerance(
                            case.dtype, self.config.correctness_atol
                        )
                        assert reference_grad_x is not None
                        assert reference_grad_weights is not None
                        grad_x_close = torch.allclose(
                            check_grad_x, reference_grad_x,
                            atol=grad_tolerance, rtol=grad_tolerance,
                        )
                        grad_weight_close = all(
                            torch.allclose(
                                actual_weight_grad,
                                reference_weight_grad,
                                atol=grad_tolerance,
                                rtol=grad_tolerance,
                            )
                            for actual_weight_grad, reference_weight_grad in zip(
                                check_grad_weights, reference_grad_weights
                            )
                        )
                        grad_x_error = float(
                            (check_grad_x.float() - reference_grad_x.float()).abs().max()
                        )
                        grad_weight_error = max(
                            float(
                                (
                                    actual_weight_grad.float()
                                    - reference_weight_grad.float()
                                ).abs().max()
                            )
                            for actual_weight_grad, reference_weight_grad in zip(
                                check_grad_weights, reference_grad_weights
                            )
                        )

                        def backward_step() -> None:
                            x = x_base.detach().requires_grad_(True)
                            for parameter in weight_parameters:
                                parameter.grad = None
                            path(x).backward(grad_out)

                        backward = _measure_cuda(
                            backward_step, self.config.micro_warmup,
                            self.config.micro_iterations, self.config.micro_repeats,
                        )
                        rows.append({
                            "module": case.name, "input_shape": case.shape, "path": path_name,
                            "mode": "forward+backward", "median_ms": backward[0],
                            "peak_cuda_mib": backward[1],
                            "regular_r1_variant": profiled_regular_variant(
                                path_name, training=True,
                            ),
                            "workspace_limit_mib": (
                                module.triton_workspace_mib if path_name.startswith("triton") else None
                            ),
                            "correct": bool(close and grad_x_close is not False and grad_weight_close is not False),
                            "max_abs_error": float(delta.max()),
                            "grad_x_correct": grad_x_close,
                            "grad_weight_correct": grad_weight_close,
                            "grad_x_max_abs_error": grad_x_error,
                            "grad_weight_max_abs_error": grad_weight_error,
                            "speedup_vs_reference": None,
                        })
                except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
                    torch.cuda.empty_cache()
                    rows.append({
                        "module": case.name, "input_shape": case.shape, "path": path_name,
                        "mode": "skipped", "median_ms": None, "peak_cuda_mib": None,
                        "correct": False, "max_abs_error": None,
                        "grad_x_correct": None, "grad_weight_correct": None,
                        "grad_x_max_abs_error": None, "grad_weight_max_abs_error": None,
                        "speedup_vs_reference": None, "reason": str(error),
                    })
        finally:
            for parameter, original_grad in zip(weight_parameters, original_grads):
                parameter.grad = original_grad
        for mode in ("forward", "forward+backward"):
            reference_rows = [row for row in rows if row["path"] == reference_name and row["mode"] == mode]
            if not reference_rows:
                continue
            reference_ms = reference_rows[0]["median_ms"]
            for row in rows:
                if row["mode"] == mode and row["median_ms"] is not None:
                    row["speedup_vs_reference"] = reference_ms / row["median_ms"]
        return rows

    def export_nsight_cases(self) -> list[Path]:
        """Serialize isolated production and forced Triton workloads."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, case in enumerate(self.cases):
            probe = torch.empty(case.shape, device=case.device, dtype=case.dtype)
            if case.module._triton_unsupported_reason(probe, case.geometry) is not None:
                continue
            variants = ["configured"]
            if case.module._regular_weights and case.module.num_radial == 1:
                variants.extend(("fused", "semi_packed"))
            for variant in variants:
                payload_path = self.output_dir / (
                    f"case_{index:02d}_{_slug(case.name)}_{variant}.pt"
                )
                prepared_geometry = case.module._prepared_geometry
                case.module._prepared_geometry = None
                try:
                    module_copy = copy.deepcopy(case.module)
                finally:
                    case.module._prepared_geometry = prepared_geometry
                module_copy.clear_prepared_graph()
                if variant != "configured":
                    module_copy.regular_r1_variant = variant
                module_copy = module_copy.cpu()
                module_copy.__dict__.pop("_irrep_profile_capture", None)
                geometry_copy = _geometry_to(case.geometry, torch.device("cpu"), case.dtype)
                if variant == "semi_packed":
                    module_copy.regular_r1_variant = "semi_packed"
                    check = module_copy.triton_workspace_mib << 20
                    bytes_per_point = (
                        int(case.shape[0])
                        * case.geometry.degree_bucket
                        * (
                            int(module_copy.packed_weight.shape[1] * module_copy.packed_weight.shape[2])
                            + int(module_copy.packed_weight.shape[3] * module_copy.packed_weight.shape[4])
                        )
                        * torch.tensor([], dtype=case.dtype).element_size()
                    )
                    if check <= bytes_per_point:
                        continue
                torch.save(
                    {
                        "module": module_copy, "geometry": geometry_copy,
                        "shape": case.shape, "dtype": str(case.dtype),
                        "regular_r1_variant": variant,
                    },
                    payload_path,
                )
                paths.append(payload_path)
        return paths

    def run_nsight(
        self,
        *,
        preset: Literal["quick", "full"] = "quick",
        mode: Literal["forward", "grad_input", "grad_weight", "training"] = "forward",
        timeout_seconds: int = 1800,
    ) -> pd.DataFrame:
        """Run Nsight Compute for each captured Triton case in isolated processes."""

        payloads = self.export_nsight_cases()
        ncu = shutil.which("ncu")
        project_root = Path(__file__).resolve().parents[3]
        child_env = os.environ.copy()
        existing_pythonpath = child_env.get("PYTHONPATH")
        child_env["PYTHONPATH"] = (
            str(project_root)
            if not existing_pythonpath
            else os.pathsep.join((str(project_root), existing_pythonpath))
        )
        rows = []
        sections = ["LaunchStats", "Occupancy", "SpeedOfLight"]
        if preset == "full":
            sections.extend(["MemoryWorkloadAnalysis", "ComputeWorkloadAnalysis", "SchedulerStats"])
        kernel_filters = {
            "forward": (
                "regex:(_packed_forward_r1_kernel|_transpose_weight_r1_kernel|"
                "_gather_input_r1_kernel|"
                "_semi_matmul_r1_kernel|_reduce_output_r1_kernel)"
            ),
            "grad_input": (
                "regex:(_packed_grad_input_r1_kernel|_reduce_grad_input_r1_kernel)"
            ),
            "grad_weight": (
                "regex:(_packed_grad_weight_r1_partial_kernel|"
                "_reduce_grad_weight_partials_kernel|_grad_weight_matmul_r1_kernel|"
                "_accumulate_grad_weight_kernel)"
            ),
        }
        for payload in payloads:
            report_path = payload.with_suffix(".ncu-rep")
            csv_path = payload.with_suffix(".ncu.csv")
            command = [
                ncu or "ncu", "--profile-from-start", "off", "--target-processes", "all",
                "--force-overwrite", "--export", str(report_path.with_suffix("")),
            ]
            for section in sections:
                command.extend(["--section", section])
            if preset == "full":
                command.extend([
                    "--metrics",
                    ",".join((
                        "l1tex__t_sectors.sum",
                        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
                        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
                        "dram__bytes.sum",
                    )),
                ])
            if mode in kernel_filters:
                command.extend(["--kernel-name", kernel_filters[mode]])
            command.extend([
                sys.executable, "-m", "lib.models.spherical.profiling",
                "--nsight-case", str(payload), "--nsight-mode", mode,
            ])
            if ncu is None:
                rows.append({
                    "case": payload.name, "status": "ncu not found", "report": None,
                    "csv": None, "command": subprocess.list2cmdline(command),
                })
                continue
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=timeout_seconds,
                cwd=project_root, env=child_env,
            )
            combined_output = completed.stdout + "\n" + completed.stderr
            if "ERR_NVGPUCTRPERM" in combined_output:
                status = "counter permission denied"
                hint = (
                    "Enable NVIDIA GPU performance counters for non-admin users on the host, "
                    "or run the container with the required profiling capability (commonly SYS_ADMIN)."
                )
            else:
                status = "ok" if completed.returncode == 0 else "failed"
                hint = None
            if completed.returncode == 0:
                exported = subprocess.run(
                    [ncu, "--import", str(report_path), "--csv", "--page", "raw"],
                    check=False, capture_output=True, text=True, timeout=timeout_seconds,
                    cwd=project_root, env=child_env,
                )
                csv_path.write_text(exported.stdout, encoding="utf-8")
            row = {
                "case": payload.name,
                "status": status,
                "report": str(report_path) if report_path.exists() else None,
                "csv": str(csv_path) if csv_path.exists() else None,
                "command": subprocess.list2cmdline(command),
                "hint": hint,
                "stdout": completed.stdout[-2000:],
                "stderr": completed.stderr[-2000:],
            }
            if csv_path.exists():
                row.update(self.summarize_nsight_csv(csv_path))
            rows.append(row)
        table = pd.DataFrame(rows)
        table.to_csv(self.output_dir / "nsight_runs.csv", index=False)
        summary_columns = [
            "case", "status", "launches", "duration_ms", "sm_throughput_pct",
            "active_warps_pct", "registers_per_thread", "fma_pipe_pct",
            "tensor_pipe_pct", "l1_sectors", "dram_bytes",
        ]
        table.reindex(columns=summary_columns).to_csv(
            self.output_dir / "nsight_summary.csv", index=False,
        )
        return table

    @staticmethod
    def read_nsight_csv(path: str | Path) -> pd.DataFrame:
        """Parse the CSV table embedded in ``ncu --page raw --csv`` output."""

        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        start = next(
            (index for index, line in enumerate(lines) if "Metric Name" in line and "," in line),
            None,
        )
        if start is None:
            start = next(
                (
                    index for index, line in enumerate(lines)
                    if line.startswith('"ID",') and "gpu__time_duration" in line
                ),
                None,
            )
        if start is None:
            return pd.DataFrame()
        table = pd.read_csv(StringIO("\n".join(lines[start:])), quoting=csv.QUOTE_MINIMAL)
        if "Metric Name" not in table.columns and not table.empty:
            units = {
                column: str(table.iloc[0][column])
                for column in table.columns
                if pd.notna(table.iloc[0][column])
            }
            table = table[table["ID"].notna()].reset_index(drop=True)
            table.attrs["metric_units"] = units
        return table

    @staticmethod
    def summarize_nsight_csv(path: str | Path) -> dict[str, float | int | None]:
        """Collapse raw NCU metrics into duration-weighted kernel indicators."""

        table = ConvolutionComparison.read_nsight_csv(path)
        empty = {
            "launches": 0, "duration_ms": None, "sm_throughput_pct": None,
            "active_warps_pct": None, "registers_per_thread": None,
            "fma_pipe_pct": None, "tensor_pipe_pct": None,
            "l1_sectors": None, "dram_bytes": None,
        }
        if table.empty:
            return empty

        if "Metric Name" not in table.columns:
            def wide_number(column: str) -> pd.Series | None:
                if column not in table.columns:
                    return None
                return pd.to_numeric(
                    table[column].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                )

            duration = wide_number("gpu__time_duration.sum")
            if duration is None:
                weights = pd.Series(1.0, index=table.index)
                duration_ms = None
            else:
                unit = table.attrs.get("metric_units", {}).get(
                    "gpu__time_duration.sum", "msecond",
                ).lower()
                factor = (
                    1e-6 if "nsecond" in unit or unit == "ns"
                    else 1e-3 if "usecond" in unit or unit in ("us", "µs")
                    else 1e3 if unit == "second"
                    else 1.0
                )
                weights = duration.fillna(0.0) * factor
                duration_ms = float(weights.sum())

            def wide_weighted(*columns: str) -> float | None:
                column = next((name for name in columns if name in table.columns), None)
                if column is None:
                    return None
                series = wide_number(column)
                assert series is not None
                valid = series.notna() & weights.notna()
                denominator = float(weights[valid].sum())
                if not valid.any():
                    return None
                if denominator == 0.0:
                    return float(series[valid].mean())
                return float((series[valid] * weights[valid]).sum() / denominator)

            def wide_sum(*columns: str) -> float | None:
                column = next((name for name in columns if name in table.columns), None)
                if column is None:
                    return None
                series = wide_number(column)
                assert series is not None
                if not series.notna().any():
                    return None
                value = float(series.sum())
                if column == "dram__bytes.sum":
                    unit = table.attrs.get("metric_units", {}).get(column, "byte").lower()
                    if "kbyte" in unit:
                        value *= 1e3
                    elif "mbyte" in unit:
                        value *= 1e6
                    elif "gbyte" in unit:
                        value *= 1e9
                return value

            return {
                "launches": int(len(table)),
                "duration_ms": duration_ms,
                "sm_throughput_pct": wide_weighted(
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                ),
                "active_warps_pct": wide_weighted(
                    "sm__warps_active.avg.pct_of_peak_sustained_active",
                ),
                "registers_per_thread": wide_weighted("launch__registers_per_thread"),
                "fma_pipe_pct": wide_weighted(
                    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
                    "smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
                ),
                "tensor_pipe_pct": wide_weighted(
                    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
                    "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
                ),
                "l1_sectors": wide_sum("l1tex__t_sectors.sum"),
                "dram_bytes": wide_sum("dram__bytes.sum"),
            }

        required = {"Metric Name", "Metric Value"}
        if not required.issubset(table.columns):
            return empty

        def number(value: Any) -> float | None:
            try:
                return float(str(value).replace(",", "").replace("%", "").strip())
            except (TypeError, ValueError):
                return None

        values = table.copy()
        values["_value"] = values["Metric Value"].map(number)
        values = values[values["_value"].notna()]
        id_columns = [
            column for column in ("ID", "Kernel Name", "Kernel Time")
            if column in values.columns
        ]
        if not id_columns:
            values["_launch"] = values.index.astype(str)
        else:
            values["_launch"] = values[id_columns].astype(str).agg("|".join, axis=1)

        duration_rows = values[
            values["Metric Name"].astype(str).str.contains("gpu__time_duration", regex=False)
        ].copy()
        durations: dict[str, float] = {}
        for _, item in duration_rows.iterrows():
            duration = float(item["_value"])
            unit = str(item.get("Metric Unit", "")).lower()
            if "nsecond" in unit or unit == "ns":
                duration /= 1e6
            elif "usecond" in unit or unit in ("us", "µs"):
                duration /= 1e3
            elif "second" in unit and "msecond" not in unit:
                duration *= 1e3
            durations[str(item["_launch"])] = duration

        def matching_metric(*needles: str) -> pd.DataFrame:
            names = values["Metric Name"].astype(str)
            mask = False
            for needle in needles:
                mask = mask | names.str.contains(needle, regex=False)
            return values[mask]

        def weighted(*needles: str) -> float | None:
            rows = matching_metric(*needles)
            if rows.empty:
                return None
            numerator = 0.0
            denominator = 0.0
            for _, item in rows.iterrows():
                weight = durations.get(str(item["_launch"]), 1.0)
                numerator += float(item["_value"]) * weight
                denominator += weight
            return numerator / denominator if denominator else None

        def summed(*needles: str) -> float | None:
            rows = matching_metric(*needles)
            if rows.empty:
                return None
            return float(rows["_value"].sum())

        launch_ids = set(values["_launch"].astype(str))
        launch_ids.difference_update(duration_rows["_launch"].astype(str))
        # Metric rows share one ID per launch; fall back to duration IDs when
        # an exporter omits kernel-name columns.
        launches = len(set(durations)) or len(launch_ids)
        return {
            "launches": launches,
            "duration_ms": sum(durations.values()) if durations else None,
            "sm_throughput_pct": weighted(
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            ),
            "active_warps_pct": weighted(
                "warps_active.avg.pct_of_peak_sustained_active",
            ),
            "registers_per_thread": weighted("launch__registers_per_thread"),
            "fma_pipe_pct": weighted("pipe_fma_cycles_active"),
            "tensor_pipe_pct": weighted("pipe_tensor_cycles_active"),
            "l1_sectors": summed("l1tex__t_sectors.sum"),
            "dram_bytes": summed("dram__bytes.sum"),
        }


def make_inference_step(model: nn.Module, *args: Any, **kwargs: Any) -> ProfileStep:
    """Return an inference callback suitable for :meth:`IrrepSphereProfiler.profile`."""

    def step() -> Any:
        with torch.inference_mode():
            return model(*args, **kwargs)

    return step


def make_training_step(
    model: nn.Module,
    forward: Callable[[], Tensor],
    loss: Callable[[Tensor], Tensor] | None = None,
) -> ProfileStep:
    """Return a zero-grad/forward/backward callback without an optimizer step."""

    def step() -> Tensor:
        model.zero_grad(set_to_none=True)
        output = forward()
        objective = output.float().square().mean() if loss is None else loss(output)
        objective.backward()
        return objective

    return step


def _measure_cuda(
    fn: ProfileStep, warmup: int, iterations: int, repeats: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    samples, peaks = [], []
    for _ in range(repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
        peaks.append((torch.cuda.max_memory_allocated() - baseline) / 2**20)
    return statistics.median(samples), statistics.median(peaks)


def _event_average_row(mode: str, event: Any) -> dict[str, Any]:
    # For CPU record_function ranges, cuda_time_total includes correlated child
    # kernels while device_time_total describes device-side events themselves.
    cuda_total = getattr(event, "cuda_time_total", None)
    self_cuda = getattr(event, "self_cuda_time_total", None)
    device_total = cuda_total if cuda_total is not None else getattr(event, "device_time_total", 0.0)
    self_device = self_cuda if self_cuda is not None else getattr(event, "self_device_time_total", 0.0)
    return {
        "mode": mode, "name": event.key, "calls": event.count,
        "cpu_total_ms": event.cpu_time_total / 1000.0,
        "self_cpu_ms": event.self_cpu_time_total / 1000.0,
        "cuda_total_ms": device_total / 1000.0,
        "self_cuda_ms": self_device / 1000.0,
        "cpu_memory_mib": getattr(event, "cpu_memory_usage", 0) / 2**20,
        "cuda_memory_mib": getattr(event, "device_memory_usage", 0) / 2**20,
        "input_shapes": str(getattr(event, "input_shapes", "")),
    }


def _kernel_table(mode: str, events: Sequence[Any]) -> pd.DataFrame:
    aggregated: dict[str, dict[str, Any]] = {}
    for event in events:
        if "CUDA" not in str(getattr(event, "device_type", "")):
            continue
        name = getattr(event, "name", getattr(event, "key", "CUDA kernel"))
        duration_value = getattr(event, "device_time_total", None)
        if duration_value is None:
            duration_value = getattr(event, "cuda_time_total", None)
        if duration_value is None:
            elapsed = getattr(getattr(event, "time_range", None), "elapsed_us", None)
            duration_value = elapsed() if callable(elapsed) else 0.0
        duration = float(duration_value) if isinstance(duration_value, (int, float)) else 0.0
        row = aggregated.setdefault(name, {"mode": mode, "kernel": name, "calls": 0, "total_ms": 0.0})
        row["calls"] += 1
        row["total_ms"] += duration / 1000.0
    for row in aggregated.values():
        row["average_ms"] = row["total_ms"] / row["calls"]
    if not aggregated:
        return pd.DataFrame(columns=["mode", "kernel", "calls", "total_ms", "average_ms"])
    return pd.DataFrame(aggregated.values())


def _runtime_metadata(model: nn.Module, config: IrrepProfileConfig) -> dict[str, Any]:
    device = torch.cuda.current_device()
    return {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": torch.cuda.get_device_capability(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "config": asdict(config),
    }


def _concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    if nonempty:
        return pd.concat(nonempty, ignore_index=True)
    if frames:
        return pd.DataFrame(columns=frames[0].columns)
    return pd.DataFrame()


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _dtype_tolerance(dtype: torch.dtype, base: float) -> float:
    if dtype == torch.float16:
        return max(base, 1e-2)
    if dtype == torch.bfloat16:
        return max(base, 3e-2)
    return base


def _gradient_tolerance(dtype: torch.dtype, base: float) -> float:
    if dtype == torch.float32:
        return max(base, 5e-3)
    return _dtype_tolerance(dtype, base)


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _geometry_to(
    geometry: _IrrepConvGeometry,
    device: torch.device,
    dtype: torch.dtype,
) -> _IrrepConvGeometry:
    values: dict[str, Any] = {}
    for item in fields(geometry):
        value = getattr(geometry, item.name)
        if isinstance(value, Tensor):
            values[item.name] = value.to(
                device=device,
                dtype=dtype if value.is_floating_point() else value.dtype,
            )
        elif isinstance(value, dict):
            values[item.name] = {
                key: tensor.to(
                    device=device,
                    dtype=dtype if tensor.is_floating_point() else tensor.dtype,
                )
                for key, tensor in value.items()
            }
        else:
            values[item.name] = value
    return _IrrepConvGeometry(**values)


def _run_nsight_case(path: Path, mode: str) -> None:
    device = torch.device("cuda")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    module: IrrepSphereConv = payload["module"].to(device)
    dtype_name = payload["dtype"].removeprefix("torch.")
    dtype = getattr(torch, dtype_name)
    x = torch.randn(payload["shape"], device=device, dtype=dtype)
    geometry = _geometry_to(payload["geometry"], device, dtype)
    module._bind_prepared_geometry(geometry)
    geometry = module._prepared_geometry
    assert geometry is not None

    def forward() -> Tensor:
        return module._forward_spatial_triton(x, geometry)

    grad_out = torch.randn_like(forward())

    def training() -> None:
        module.zero_grad(set_to_none=True)
        x_train = x.detach().requires_grad_(True)
        module._forward_spatial_triton(x_train, geometry).backward(grad_out)

    step = forward if mode == "forward" else training
    for _ in range(5):
        step()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    step()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsight-case", type=Path)
    parser.add_argument(
        "--nsight-mode",
        choices=("forward", "grad_input", "grad_weight", "training"),
        default="forward",
    )
    args = parser.parse_args()
    if args.nsight_case is None:
        parser.error("--nsight-case is required")
    _run_nsight_case(args.nsight_case, args.nsight_mode)


if __name__ == "__main__":
    _main()


__all__ = [
    "ConvolutionComparison", "IrrepProfileConfig", "IrrepProfileReport",
    "IrrepSphereProfiler", "make_inference_step", "make_training_step",
]
