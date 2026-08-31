"""Microbenchmark the irregular R=1 convolution, including the legacy path."""

from __future__ import annotations

import argparse
import statistics

import torch

from lib.data.spherical.sphere_geometry import build_tangent_frames
from lib.data.spherical.sphere_graph import SphereGraphGeometry
from lib.models.spherical.irrep_layers import IrrepSphereConv, SO2IrrepFieldType


def make_graph(n_points: int, degree: int, device: torch.device) -> SphereGraphGeometry:
    points = torch.randn(n_points, 3)
    points /= points.norm(dim=-1, keepdim=True)
    e1, e2 = build_tangent_frames(points, "east_north")
    center = torch.arange(n_points).repeat_interleave(degree)
    offsets = torch.arange(1, degree + 1)
    neighbor = ((torch.arange(n_points)[:, None] + offsets) % n_points).reshape(-1)
    graph = SphereGraphGeometry.from_existing_graph(
        points, e1, e2, center, neighbor, radius_km=20_000.0,
    )
    return graph.to(device=device, dtype=torch.float32)


def measure(call, *, warmup: int, iterations: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iterations):
            call()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / iterations)
    torch.cuda.reset_peak_memory_stats()
    call()
    torch.cuda.synchronize()
    return statistics.median(samples), torch.cuda.max_memory_allocated() / 2**20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", choices=("expand", "contract"), default="expand")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(101)
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    graph = make_graph(args.points, args.degree, device)
    if args.direction == "expand":
        in_type = SO2IrrepFieldType((5, 1))
        out_type = SO2IrrepFieldType((args.width,) * 3)
    else:
        in_type = SO2IrrepFieldType((args.width,) * 3)
        out_type = SO2IrrepFieldType((1, 1))

    reference = IrrepSphereConv(
        in_type, out_type, radius_km=20_000.0, num_radial=1,
        include_self=False, backend="torch",
    ).to(device=device, dtype=dtype)
    reference.prepare_graph(graph, device=device, dtype=dtype)
    state = reference.state_dict()
    x_data = torch.randn(
        args.batch, graph.n_points, in_type.total_dim, device=device, dtype=dtype,
    )

    variants = [("torch", True), ("triton_legacy", False), ("triton", True)]
    if in_type.is_regular() and out_type.is_regular():
        variants = [("torch", True), ("triton", True)]
    for backend, fast_path in variants:
        layer = IrrepSphereConv(
            in_type, out_type, radius_km=20_000.0, num_radial=1,
            include_self=False, backend="torch" if backend == "torch" else "triton",
        ).to(device=device, dtype=dtype)
        layer.load_state_dict(state)
        layer.prepare_graph(graph, device=device, dtype=dtype)
        layer._irregular_r1_fast_path = fast_path
        params = (
            (layer.packed_weight,)
            if layer._regular_weights
            else tuple(layer.weights.values())
        )
        for mode in ("eager", "compiled"):
            forward = layer.forward_prepared
            if mode == "compiled":
                forward = torch.compile(forward, fullgraph=True, dynamic=False)
            x = x_data.detach().requires_grad_(True)
            grad_out = torch.randn_like(forward(x))

            def forward_call():
                return forward(x)

            def train_call():
                y = forward(x)
                return torch.autograd.grad(y, (x, *params), grad_out)

            forward_ms, forward_mib = measure(
                forward_call, warmup=args.warmup, iterations=args.iterations,
                repeats=args.repeats,
            )
            train_ms, train_mib = measure(
                train_call, warmup=args.warmup, iterations=args.iterations,
                repeats=args.repeats,
            )
            print(
                f"{backend:14s} {mode:8s} forward={forward_ms:8.3f} ms "
                f"fwd+bwd={train_ms:8.3f} ms peak={max(forward_mib, train_mib):8.2f} MiB"
            )


if __name__ == "__main__":
    main()
