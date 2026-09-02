from __future__ import annotations

import pytest
import torch

from lib.data.spherical.sphere_geometry import build_tangent_frames
from lib.data.spherical.sphere_graph import SphereGraphGeometry
from lib.models.spherical.irrep_layers import IrrepSphereConv, SO2IrrepFieldType
from lib.models.spherical.irrep_sphere_unet import IrrepSphereUNet
from lib.models.spherical.triton_semi_packed_irrep_conv import should_use_semi_packed


def _ring_graph(device: torch.device, *, n_points: int = 17, degree: int = 5):
    points = torch.randn(n_points, 3, dtype=torch.float32)
    points /= points.norm(dim=-1, keepdim=True)
    frame_e1, frame_e2 = build_tangent_frames(points, "east_north")
    center = torch.arange(n_points).repeat_interleave(degree)
    offsets = torch.arange(1, degree + 1)
    neighbor = ((torch.arange(n_points)[:, None] + offsets) % n_points).reshape(-1)
    graph = SphereGraphGeometry.from_existing_graph(
        points, frame_e1, frame_e2, center, neighbor, radius_km=20_000.0,
    )
    return graph.to(device=device, dtype=torch.float32)


def test_regular_r1_controls_validate_and_propagate() -> None:
    field_type = SO2IrrepFieldType.balanced(max_order=1, multiplicity=2)
    with pytest.raises(ValueError, match="regular_r1_variant"):
        IrrepSphereConv(
            field_type, field_type, 100.0, 1, regular_r1_variant="invalid",
        )
    with pytest.raises(ValueError, match="triton_workspace_mib"):
        IrrepSphereConv(
            field_type, field_type, 100.0, 1, triton_workspace_mib=-1,
        )
    model = IrrepSphereUNet(
        field_type,
        multiplicities=(2, 2, 2, 2),
        max_order=1,
        num_radial=1,
        regular_r1_variant="semi_packed",
        triton_workspace_mib=64,
    )
    convolutions = [
        layer for layer in model.modules() if isinstance(layer, IrrepSphereConv)
    ]
    assert convolutions
    assert all(layer.regular_r1_variant == "semi_packed" for layer in convolutions)
    assert all(layer.triton_workspace_mib == 64 for layer in convolutions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("variant", ["fused", "semi_packed"])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("normalize", [False, True])
def test_regular_r1_triton_variants_match_packed(
    variant: str, batch: int, normalize: bool,
) -> None:
    torch.manual_seed(101 + batch)
    device = torch.device("cuda")
    in_type = SO2IrrepFieldType.balanced(max_order=2, multiplicity=4)
    out_type = SO2IrrepFieldType.balanced(max_order=1, multiplicity=3)
    layer = IrrepSphereConv(
        in_type,
        out_type,
        radius_km=20_000.0,
        num_radial=1,
        include_self=False,
        normalize_by_neighbors=normalize,
        backend="triton",
        regular_r1_variant=variant,
        triton_workspace_mib=8,
    ).to(device)
    layer.prepare_graph(_ring_graph(device), device=device, dtype=torch.float32)
    geometry = layer._prepared_geometry
    assert geometry is not None
    x_base = torch.randn(batch, geometry.n_points, in_type.total_dim, device=device)
    grad_out = torch.randn(batch, geometry.n_points, out_type.total_dim, device=device)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        reference_x = x_base.clone().requires_grad_(True)
        reference = layer._forward_spatial_packed(reference_x, geometry)
        reference_grads = torch.autograd.grad(
            reference, (reference_x, layer.packed_weight), grad_out,
        )
        actual_x = x_base.clone().requires_grad_(True)
        actual = layer._forward_spatial_triton(actual_x, geometry)
        actual_grads = torch.autograd.grad(
            actual, (actual_x, layer.packed_weight), grad_out,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    assert torch.allclose(actual, reference, atol=1e-4, rtol=1e-4)
    assert torch.allclose(actual_grads[0], reference_grads[0], atol=3e-4, rtol=3e-4)
    assert torch.allclose(actual_grads[1], reference_grads[1], atol=3e-4, rtol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_regular_r1_auto_workspace_fallback_and_forced_error() -> None:
    device = torch.device("cuda")
    field_type = SO2IrrepFieldType.balanced(max_order=1, multiplicity=16)
    graph = _ring_graph(device)
    auto = IrrepSphereConv(
        field_type, field_type, 20_000.0, 1, include_self=False,
        backend="triton", regular_r1_variant="auto", triton_workspace_mib=0,
    ).to(device)
    auto.prepare_graph(graph, device=device, dtype=torch.float32)
    x = torch.randn(1, graph.n_points, field_type.total_dim, device=device)
    assert auto._prepared_geometry is not None
    assert auto._selected_regular_r1_variant(x, auto._prepared_geometry) == "fused"

    forced = IrrepSphereConv(
        field_type, field_type, 20_000.0, 1, include_self=False,
        backend="triton", regular_r1_variant="semi_packed", triton_workspace_mib=0,
    ).to(device)
    forced.prepare_graph(graph, device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="workspace"):
        forced.forward_prepared(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_regular_r1_auto_policy_is_mode_specific() -> None:
    x = torch.empty((1, 58_800, 80), device="cuda", dtype=torch.float32)
    weight = torch.empty((1, 16, 5, 16, 5), device="cuda", dtype=torch.float32)
    assert should_use_semi_packed(
        x, weight, 16, 512 << 20, training=False,
    )
    assert not should_use_semi_packed(
        x, weight, 16, 512 << 20, training=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_regular_r1_semi_packed_compiles_fullgraph_with_backward() -> None:
    device = torch.device("cuda")
    field_type = SO2IrrepFieldType.balanced(max_order=1, multiplicity=4)
    graph = _ring_graph(device, n_points=11, degree=3)
    layer = IrrepSphereConv(
        field_type, field_type, 20_000.0, 1, include_self=False,
        backend="triton", regular_r1_variant="semi_packed",
        triton_workspace_mib=8,
    ).to(device)
    layer.prepare_graph(graph, device=device, dtype=torch.float32)
    compiled = torch.compile(layer.forward_prepared, fullgraph=True, dynamic=False)
    x = torch.randn(1, graph.n_points, field_type.total_dim, device=device, requires_grad=True)
    grad_out = torch.randn(1, graph.n_points, field_type.total_dim, device=device)
    output = compiled(x)
    grad_x, grad_weight = torch.autograd.grad(
        output, (x, layer.packed_weight), grad_out,
    )
    assert output.shape == grad_out.shape
    assert torch.isfinite(grad_x).all()
    assert torch.isfinite(grad_weight).all()
