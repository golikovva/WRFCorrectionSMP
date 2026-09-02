from __future__ import annotations

import pytest
import torch

from lib.data.spherical.sphere_geometry import build_tangent_frames
from lib.data.spherical.sphere_graph import SphereGraphGeometry
from lib.models.spherical.irrep_layers import IrrepSphereConv, SO2IrrepFieldType


def _ring_graph(device: torch.device, *, n_points: int = 19, degree: int = 5):
    points = torch.randn(n_points, 3, dtype=torch.float32)
    points /= points.norm(dim=-1, keepdim=True)
    frame_e1, frame_e2 = build_tangent_frames(points, "east_north")
    center = torch.arange(n_points).repeat_interleave(degree)
    offsets = torch.arange(1, degree + 1)
    neighbor = (
        (torch.arange(n_points)[:, None] + offsets[None, :]) % n_points
    ).reshape(-1)
    graph = SphereGraphGeometry.from_existing_graph(
        points,
        frame_e1,
        frame_e2,
        center,
        neighbor,
        radius_km=20_000.0,
    )
    return graph.to(device=device, dtype=torch.float32)


def _make_layer(
    in_type: SO2IrrepFieldType,
    out_type: SO2IrrepFieldType,
    graph: SphereGraphGeometry,
    *,
    num_radial: int,
    backend: str,
    dtype: torch.dtype = torch.float32,
    normalize: bool = True,
) -> IrrepSphereConv:
    layer = IrrepSphereConv(
        in_type,
        out_type,
        radius_km=20_000.0,
        num_radial=num_radial,
        include_self=False,
        backend=backend,
        normalize_by_neighbors=normalize,
    ).to(device="cuda", dtype=dtype)
    layer.prepare_graph(graph, device="cuda", dtype=dtype)
    return layer


def _forward_and_grads(layer: IrrepSphereConv, x: torch.Tensor, grad_out: torch.Tensor):
    params = tuple(layer.weights.values())
    output = layer.forward_prepared(x)
    gradients = torch.autograd.grad(output, (x, *params), grad_out)
    return output, gradients


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_radial", [1, 3])
def test_irregular_triton_matches_torch_forward_and_backward(num_radial: int) -> None:
    torch.manual_seed(17 + num_radial)
    device = torch.device("cuda")
    graph = _ring_graph(device)
    in_type = SO2IrrepFieldType((3, 1))
    out_type = SO2IrrepFieldType((2, 4))
    reference_layer = _make_layer(
        in_type, out_type, graph, num_radial=num_radial, backend="torch"
    )
    triton_layer = _make_layer(
        in_type, out_type, graph, num_radial=num_radial, backend="triton"
    )
    triton_layer.load_state_dict(reference_layer.state_dict())
    x_base = torch.randn(2, graph.n_points, in_type.total_dim, device=device)
    grad_out = torch.randn(2, graph.n_points, out_type.total_dim, device=device)
    reference_x = x_base.clone().requires_grad_(True)
    triton_x = x_base.clone().requires_grad_(True)

    reference, reference_grads = _forward_and_grads(
        reference_layer, reference_x, grad_out
    )
    actual, actual_grads = _forward_and_grads(triton_layer, triton_x, grad_out)

    assert torch.allclose(actual, reference, atol=1e-3, rtol=1e-3), (
        (actual - reference).abs().max().item(),
        ((actual - reference).abs() / reference.abs().clamp_min(1e-6)).max().item(),
    )
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        assert torch.allclose(actual_grad, reference_grad, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_irregular_r1_fp32_does_not_implicitly_use_tf32() -> None:
    torch.manual_seed(19)
    device = torch.device("cuda")
    graph = _ring_graph(device, n_points=19, degree=5)
    in_type = SO2IrrepFieldType((16, 15, 14))
    out_type = SO2IrrepFieldType((16, 15, 14))
    reference_layer = _make_layer(
        in_type, out_type, graph, num_radial=1, backend="torch"
    )
    triton_layer = _make_layer(
        in_type, out_type, graph, num_radial=1, backend="triton"
    )
    triton_layer.load_state_dict(reference_layer.state_dict())
    x_base = torch.randn(2, graph.n_points, in_type.total_dim, device=device)
    grad_out = torch.randn(2, graph.n_points, out_type.total_dim, device=device)

    previous_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        reference, reference_grads = _forward_and_grads(
            reference_layer, x_base.clone().requires_grad_(True), grad_out
        )
        actual, actual_grads = _forward_and_grads(
            triton_layer, x_base.clone().requires_grad_(True), grad_out
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_allow_tf32

    assert torch.allclose(actual, reference, atol=5e-6, rtol=5e-6)
    assert torch.allclose(actual_grads[0], reference_grads[0], atol=2e-5, rtol=2e-5)
    for actual_grad, reference_grad in zip(actual_grads[1:], reference_grads[1:]):
        assert torch.allclose(
            actual_grad, reference_grad, atol=5e-4, rtol=5e-4
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_irregular_r1_triton_compiles_fullgraph_with_backward() -> None:
    torch.manual_seed(23)
    device = torch.device("cuda")
    graph = _ring_graph(device, n_points=13, degree=3)
    in_type = SO2IrrepFieldType((3, 1))
    out_type = SO2IrrepFieldType((2, 4))
    layer = _make_layer(in_type, out_type, graph, num_radial=1, backend="triton")
    compiled = torch.compile(layer.forward_prepared, fullgraph=True, dynamic=False)
    x_base = torch.randn(1, graph.n_points, in_type.total_dim, device=device)
    grad_out = torch.randn(1, graph.n_points, out_type.total_dim, device=device)
    eager_x = x_base.clone().requires_grad_(True)
    compiled_x = x_base.clone().requires_grad_(True)

    eager, eager_grads = _forward_and_grads(layer, eager_x, grad_out)
    actual = compiled(compiled_x)
    actual_grads = torch.autograd.grad(
        actual, (compiled_x, *tuple(layer.weights.values())), grad_out
    )

    assert torch.allclose(actual, eager, atol=1e-3, rtol=1e-3)
    for actual_grad, eager_grad in zip(actual_grads, eager_grads):
        assert torch.allclose(actual_grad, eager_grad, atol=5e-3, rtol=5e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("normalize", [False, True])
def test_irregular_r1_low_precision(dtype: torch.dtype, normalize: bool) -> None:
    torch.manual_seed(31)
    device = torch.device("cuda")
    graph = _ring_graph(device, n_points=11, degree=3)
    in_type = SO2IrrepFieldType((5, 1))
    out_type = SO2IrrepFieldType((8, 8, 8))
    reference_layer = _make_layer(
        in_type, out_type, graph, num_radial=1, backend="torch",
        dtype=dtype, normalize=normalize,
    )
    triton_layer = _make_layer(
        in_type, out_type, graph, num_radial=1, backend="triton",
        dtype=dtype, normalize=normalize,
    )
    triton_layer.load_state_dict(reference_layer.state_dict())
    x_base = torch.randn(
        1, graph.n_points, in_type.total_dim, device=device, dtype=dtype,
    )
    grad_out = torch.randn(
        1, graph.n_points, out_type.total_dim, device=device, dtype=dtype,
    )
    reference, reference_grads = _forward_and_grads(
        reference_layer, x_base.clone().requires_grad_(True), grad_out,
    )
    actual, actual_grads = _forward_and_grads(
        triton_layer, x_base.clone().requires_grad_(True), grad_out,
    )
    tolerance = 1e-2 if dtype == torch.float16 else 3e-2
    assert torch.allclose(actual, reference, atol=tolerance, rtol=tolerance)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        assert torch.allclose(
            actual_grad, reference_grad, atol=tolerance, rtol=tolerance,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("in_multiplicities", "out_multiplicities"),
    [((5,), (1, 8)), ((0, 5), (0, 8)), ((5, 0, 1), (0, 8))],
)
def test_irregular_r1_missing_and_single_order_cases(
    in_multiplicities: tuple[int, ...], out_multiplicities: tuple[int, ...],
) -> None:
    torch.manual_seed(37)
    device = torch.device("cuda")
    graph = _ring_graph(device, n_points=9, degree=2)
    in_type = SO2IrrepFieldType(in_multiplicities)
    out_type = SO2IrrepFieldType(out_multiplicities)
    reference_layer = _make_layer(
        in_type, out_type, graph, num_radial=1, backend="torch",
    )
    triton_layer = _make_layer(
        in_type, out_type, graph, num_radial=1, backend="triton",
    )
    triton_layer.load_state_dict(reference_layer.state_dict())
    x_base = torch.randn(1, graph.n_points, in_type.total_dim, device=device)
    grad_out = torch.randn(1, graph.n_points, out_type.total_dim, device=device)
    reference, reference_grads = _forward_and_grads(
        reference_layer, x_base.clone().requires_grad_(True), grad_out,
    )
    actual, actual_grads = _forward_and_grads(
        triton_layer, x_base.clone().requires_grad_(True), grad_out,
    )
    assert torch.allclose(actual, reference, atol=1e-3, rtol=1e-3)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        assert torch.allclose(
            actual_grad, reference_grad, atol=5e-3, rtol=5e-3,
        )
