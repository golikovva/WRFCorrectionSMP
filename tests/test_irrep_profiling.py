from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from lib.data.spherical.sphere_geometry import build_tangent_frames
from lib.data.spherical.sphere_graph import SphereGraphGeometry
from lib.models.spherical.irrep_layers import (
    IrrepSphereConv,
    SO2IrrepFieldType,
    _gather_irrep_neighbors,
)
from lib.models.spherical.irrep_sphere_unet import IrrepSphereUNet, _IrrepSphereDoubleConv
from lib.models.spherical.sphere_grid_wrapper import FieldSchema, SphereGridModelWrapper
from lib.models.spherical.profiling import (
    ConvolutionComparison,
    IrrepProfileConfig,
    IrrepSphereProfiler,
    make_inference_step,
    make_training_step,
)


def _ring_graph(device: torch.device) -> SphereGraphGeometry:
    n_points = 12
    points = torch.randn(n_points, 3, dtype=torch.float32)
    points = points / points.norm(dim=-1, keepdim=True)
    frame_e1, frame_e2 = build_tangent_frames(points, "east_north")
    center = torch.arange(n_points).repeat_interleave(2)
    neighbor = torch.stack(
        ((torch.arange(n_points) - 1) % n_points, (torch.arange(n_points) + 1) % n_points),
        dim=1,
    ).reshape(-1)
    graph = SphereGraphGeometry.from_existing_graph(
        points, frame_e1, frame_e2, center, neighbor, radius_km=20_000.0,
    )
    return graph.to(device=device, dtype=torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_neighbor_gather_recompiles_for_a_new_grid_size() -> None:
    """Reference gathers must not read CUDA index values while recompiling."""

    compiled = torch.compile(
        _gather_irrep_neighbors,
        fullgraph=True,
        dynamic=False,
    )
    for n_points in (12, 7):
        x = torch.randn(1, n_points, 2, 3, device="cuda", requires_grad=True)
        neighbor_idx = torch.arange(n_points - 1, -1, -1, device="cuda")
        actual = compiled(x, neighbor_idx)
        expected = x.index_select(1, neighbor_idx)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_profile_config_rejects_non_positive_counts() -> None:
    with pytest.raises(ValueError, match="warmup"):
        IrrepProfileConfig(warmup=0)


def test_read_nsight_csv_extracts_metric_table(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text(
        "==PROF== Connected\n"
        '"ID","Process ID","Metric Name","Metric Unit","Metric Value"\n'
        '"1","42","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","72.5"\n',
        encoding="utf-8",
    )
    result = ConvolutionComparison.read_nsight_csv(path)
    assert list(result["Metric Name"]) == ["sm__throughput.avg.pct_of_peak_sustained_elapsed"]


def test_summarize_nsight_csv_uses_kernel_duration_weights(tmp_path: Path) -> None:
    path = tmp_path / "weighted.csv"
    path.write_text(
        '"ID","Metric Name","Metric Unit","Metric Value"\n'
        '"1","gpu__time_duration.sum","msecond","1"\n'
        '"1","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","50"\n'
        '"2","gpu__time_duration.sum","msecond","3"\n'
        '"2","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","80"\n'
        '"1","launch__registers_per_thread","register/thread","64"\n'
        '"2","launch__registers_per_thread","register/thread","96"\n',
        encoding="utf-8",
    )
    summary = ConvolutionComparison.summarize_nsight_csv(path)
    assert summary["launches"] == 2
    assert summary["duration_ms"] == pytest.approx(4.0)
    assert summary["sm_throughput_pct"] == pytest.approx(72.5)
    assert summary["registers_per_thread"] == pytest.approx(88.0)


def test_summarize_nsight_wide_csv_uses_units_row(tmp_path: Path) -> None:
    path = tmp_path / "wide.csv"
    path.write_text(
        '"ID","Kernel Name","gpu__time_duration.sum",'
        '"sm__throughput.avg.pct_of_peak_sustained_elapsed",'
        '"launch__registers_per_thread"\n'
        '"","","msecond","%","register/thread"\n'
        '"0","kernel_a","1","50","64"\n'
        '"1","kernel_b","3","80","96"\n',
        encoding="utf-8",
    )
    summary = ConvolutionComparison.summarize_nsight_csv(path)
    assert summary["launches"] == 2
    assert summary["duration_ms"] == pytest.approx(4.0)
    assert summary["sm_throughput_pct"] == pytest.approx(72.5)
    assert summary["registers_per_thread"] == pytest.approx(88.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_profile_and_compare_smoke(tmp_path: Path) -> None:
    device = torch.device("cuda")
    field_type = SO2IrrepFieldType.balanced(max_order=1, multiplicity=2)
    layer = IrrepSphereConv(
        field_type, field_type, radius_km=20_000.0, num_radial=1,
        include_self=False, backend="torch",
    ).to(device)
    layer.prepare_graph(_ring_graph(device), device=device, dtype=torch.float32)
    x = torch.randn(1, 12, field_type.total_dim, device=device)
    config = IrrepProfileConfig(
        warmup=1, iterations=1, repeats=1, profile_iterations=1,
        micro_warmup=1, micro_iterations=1, micro_repeats=1,
    )
    profiler = IrrepSphereProfiler(layer, output_dir=tmp_path, config=config)
    report = profiler.profile(
        {
            "inference": make_inference_step(layer.forward_prepared, x),
            "training": make_training_step(layer, lambda: layer.forward_prepared(x)),
        }
    )

    assert set(report.run_table["mode"]) == {"inference", "training"}
    assert not report.operation_table.empty
    assert (report.operation_table["cuda_total_ms"] > 0.0).all()
    assert report.backend_decisions.iloc[0]["selected_path"] == "packed"
    assert all(path.is_file() for path in report.trace_paths.values())

    comparison = report.compare_convolution_backends().run(include_backward=True)
    assert {
        "packed", "packed_compiled", "blockwise", "blockwise_compiled",
    }.issubset(set(comparison["path"]))
    successful = comparison[comparison["mode"] == "forward"]
    assert bool(successful["correct"].all())
    assert pd.notna(successful["median_ms"]).all()
    backward = comparison[comparison["mode"] == "forward+backward"]
    assert bool(backward["correct"].all())
    assert backward["grad_x_correct"].all()
    assert backward["grad_weight_correct"].all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_double_conv_compiles_as_fullgraph_without_profile_ranges() -> None:
    device = torch.device("cuda")
    field_type = SO2IrrepFieldType.balanced(max_order=1, multiplicity=1)
    block = _IrrepSphereDoubleConv(
        field_type,
        field_type,
        radius_km=20_000.0,
        num_radial=1,
        regular_samples=8,
        normalize_by_neighbors=True,
        include_self=False,
        self_identity_init=False,
        quadrature=False,
        quadrature_radial=None,
        quadrature_angular=8,
        quadrature_sigma_km=None,
        irrep_conv_backend="torch",
    ).to(device).eval()
    block.prepare_graph(_ring_graph(device))
    x = torch.randn(1, 12, field_type.total_dim, device=device)

    expected = block.forward_prepared(x)
    compiled = torch.compile(block.forward_prepared, fullgraph=True, dynamic=False)
    actual = compiled(x)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_unet_grid_wrapper_captures_as_fullgraph() -> None:
    input_schema = FieldSchema(
        scalars={"T2": 2, "SEAICE": 3, "HGT": 4, "HoD": 5, "DoY": 6},
        vectors={"uvmet10": (0, 1)},
    )
    output_schema = FieldSchema(
        scalars={"T2": 2},
        vectors={"uvmet10": (0, 1)},
    )
    unet = IrrepSphereUNet(
        input_schema.irrep_type,
        out_type=output_schema.irrep_type,
        multiplicities=(1, 1, 1, 1),
        max_order=1,
        radius_km=20_000.0,
        num_radial=1,
        regular_samples=4,
        include_self=False,
        irrep_conv_backend="auto",
    ).eval()
    wrapper = SphereGridModelWrapper(
        unet,
        {
            "latitude": torch.linspace(-60.0, 60.0, 8),
            "longitude": torch.linspace(-180.0, 157.5, 16),
        },
        levels=4,
        radius_km=20_000.0,
        max_neighbors=8,
        pool_ratio=0.5,
        min_points=4,
        prefer_scipy=False,
        input_schema=input_schema,
        output_schema=output_schema,
    ).eval()
    x = torch.randn(1, input_schema.num_channels, 8, 16)

    expected = wrapper(x)
    compiled = torch.compile(
        wrapper, backend="eager", fullgraph=True, dynamic=False
    )
    actual = compiled(x)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_shared_geometry_with_narrower_output_orders() -> None:
    """A geometry-bank row stride must not leak into a narrower output layer."""

    device = torch.device("cuda")
    in_type = SO2IrrepFieldType.balanced(max_order=2, multiplicity=2)
    out_type = SO2IrrepFieldType.balanced(max_order=1, multiplicity=1)
    geometry_source = IrrepSphereConv(
        in_type, in_type, radius_km=20_000.0, num_radial=1,
        include_self=False, backend="torch",
    ).to(device)
    geometry_source.prepare_graph(_ring_graph(device), device=device, dtype=torch.float32)
    assert geometry_source._prepared_geometry is not None

    layer = IrrepSphereConv(
        in_type, out_type, radius_km=20_000.0, num_radial=1,
        include_self=False, backend="triton",
    ).to(device)
    layer._bind_prepared_geometry(geometry_source._prepared_geometry)
    geometry = layer._prepared_geometry
    assert geometry is not None
    assert geometry.output_rotation_cos.stride() == (out_type.max_order, 1)

    x_reference = torch.randn(
        1, 12, in_type.total_dim, device=device, requires_grad=True,
    )
    x_triton = x_reference.detach().clone().requires_grad_(True)
    reference = layer._forward_spatial_packed(x_reference, geometry)
    actual = layer._forward_spatial_triton(x_triton, geometry)
    grad_out = torch.randn_like(reference)
    reference_grad = torch.autograd.grad(
        reference, (x_reference, layer.packed_weight), grad_out, retain_graph=True,
    )
    actual_grad = torch.autograd.grad(actual, (x_triton, layer.packed_weight), grad_out)

    assert torch.allclose(actual, reference, atol=1e-4, rtol=1e-4)
    assert torch.allclose(actual_grad[0], reference_grad[0], atol=3e-4, rtol=3e-4)
    assert torch.allclose(actual_grad[1], reference_grad[1], atol=3e-4, rtol=3e-4)
