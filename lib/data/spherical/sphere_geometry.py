"""Sphere geometry helpers for intrinsic tangent-plane convolutions.

The local gauge convention used here is a right-handed frame

    W_p = [e1(p), e2(p)]

with e2 = p x e1.  A gauge rotation by gamma is represented as
W'_p = W_p R(gamma).  Therefore tangent-vector coordinates transform as
v' = R(-gamma) v for the same underlying physical tangent vector.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, Sequence, Tuple, TypeAlias

import torch
from torch import Tensor


TangentFrameStrategy: TypeAlias = (
    Literal["robust", "east_north"]
    | Callable[[Tensor], tuple[Tensor, Tensor]]
)


def _as_tensor_1d(values: Tensor | Sequence[float], *, dtype: torch.dtype) -> Tensor:
    tensor = values if isinstance(values, Tensor) else torch.as_tensor(values, dtype=dtype)
    return tensor.to(dtype=dtype).flatten()


def latlon_to_xyz(
    lat: Tensor | Sequence[float],
    lon: Tensor | Sequence[float],
    *,
    degrees: bool = True,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tuple[int, int]]:
    """Create unit-sphere xyz points from latitude and longitude arrays.

    ``lat`` and ``lon`` may be one-dimensional coordinate arrays or already
    broadcast two-dimensional grids.  Points are flattened in row-major
    ``[lat, lon]`` order, matching tensors shaped ``[H, W]``.
    """

    lat_t = torch.as_tensor(lat, dtype=dtype)
    lon_t = torch.as_tensor(lon, dtype=dtype)

    if lat_t.ndim == 1 and lon_t.ndim == 1:
        try:
            lat_grid, lon_grid = torch.meshgrid(lat_t, lon_t, indexing="ij")
        except TypeError:
            lat_grid, lon_grid = torch.meshgrid(lat_t, lon_t)
    elif lat_t.shape == lon_t.shape and lat_t.ndim == 2:
        lat_grid, lon_grid = lat_t, lon_t
    else:
        raise ValueError("lat/lon must be both 1D arrays or matching 2D grids")

    if degrees:
        lat_grid = torch.deg2rad(lat_grid)
        lon_grid = torch.deg2rad(lon_grid)

    cos_lat = torch.cos(lat_grid)
    xyz = torch.stack(
        (
            cos_lat * torch.cos(lon_grid),
            cos_lat * torch.sin(lon_grid),
            torch.sin(lat_grid),
        ),
        dim=-1,
    )
    xyz = torch.nn.functional.normalize(xyz.reshape(-1, 3), dim=-1)
    return xyz, (int(lat_grid.shape[0]), int(lat_grid.shape[1]))


def robust_tangent_frames(points_xyz: Tensor, eps: float = 1e-12) -> tuple[Tensor, Tensor]:
    """Build numerically stable local tangent frames for unit-sphere points.

    For each point we choose the reference axis among z, x, y that is least
    aligned with the surface normal.  Projecting that axis into the tangent
    plane avoids the usual east/north singularity at the poles.
    """

    if points_xyz.ndim != 2 or points_xyz.shape[-1] != 3:
        raise ValueError("points_xyz must have shape [N, 3]")

    p = torch.nn.functional.normalize(points_xyz, dim=-1)
    candidates = torch.tensor(
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        device=p.device,
        dtype=p.dtype,
    )
    alignments = torch.abs(p @ candidates.T)
    ref = candidates[torch.argmin(alignments, dim=-1)]
    e1 = ref - (ref * p).sum(dim=-1, keepdim=True) * p
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(eps)
    e2 = torch.cross(p, e1, dim=-1)
    e2 = e2 / e2.norm(dim=-1, keepdim=True).clamp_min(eps)
    return e1, e2


def _east_north_tangent_frames(points_xyz: Tensor) -> tuple[Tensor, Tensor]:
    """Build the conventional east/north frame away from the poles."""

    p = torch.nn.functional.normalize(points_xyz, dim=-1)
    rho = torch.linalg.vector_norm(p[..., :2], dim=-1)
    pole_tolerance = 32.0 * torch.finfo(p.dtype).eps
    pole_mask = rho <= pole_tolerance
    if bool(pole_mask.any()):
        indices = pole_mask.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            "frame_strategy='east_north' is undefined at the poles; "
            f"found pole point(s) at flattened indices {indices}"
        )

    east = torch.stack(
        (-p[..., 1] / rho, p[..., 0] / rho, torch.zeros_like(rho)),
        dim=-1,
    )
    north = torch.cross(p, east, dim=-1)
    north = torch.nn.functional.normalize(north, dim=-1)
    return east, north


def _validate_tangent_frames(
    points_xyz: Tensor,
    e1: Tensor,
    e2: Tensor,
    *,
    source: str,
) -> tuple[Tensor, Tensor]:
    expected_shape = tuple(points_xyz.shape)
    if tuple(e1.shape) != expected_shape or tuple(e2.shape) != expected_shape:
        raise ValueError(
            f"{source} must return e1 and e2 with shape {expected_shape}; "
            f"got {tuple(e1.shape)} and {tuple(e2.shape)}"
        )

    e1 = e1.to(device=points_xyz.device, dtype=points_xyz.dtype)
    e2 = e2.to(device=points_xyz.device, dtype=points_xyz.dtype)
    if not bool(torch.isfinite(e1).all()) or not bool(torch.isfinite(e2).all()):
        raise ValueError(f"{source} returned non-finite tangent-frame values")

    tolerance = max(1e-8, 64.0 * torch.finfo(points_xyz.dtype).eps)
    zeros = torch.zeros(points_xyz.shape[0], device=points_xyz.device, dtype=points_xyz.dtype)
    ones = torch.ones_like(zeros)
    checks = (
        ("unit-length e1", torch.linalg.vector_norm(e1, dim=-1), ones),
        ("unit-length e2", torch.linalg.vector_norm(e2, dim=-1), ones),
        ("e1 tangent to the sphere", (e1 * points_xyz).sum(dim=-1), zeros),
        ("e2 tangent to the sphere", (e2 * points_xyz).sum(dim=-1), zeros),
        ("orthogonal e1/e2", (e1 * e2).sum(dim=-1), zeros),
    )
    for description, actual, expected in checks:
        if not torch.allclose(actual, expected, atol=tolerance, rtol=tolerance):
            error = torch.max(torch.abs(actual - expected)).item()
            raise ValueError(
                f"{source} must produce {description}; maximum error was {error:.3g}"
            )

    orientation = torch.cross(e1, e2, dim=-1)
    if not torch.allclose(
        orientation,
        points_xyz,
        atol=tolerance,
        rtol=tolerance,
    ):
        error = torch.max(torch.abs(orientation - points_xyz)).item()
        raise ValueError(
            f"{source} must produce right-handed frames with cross(e1, e2) = p; "
            f"maximum error was {error:.3g}"
        )
    return e1, e2


def build_tangent_frames(
    points_xyz: Tensor,
    frame_strategy: TangentFrameStrategy = "robust",
) -> tuple[Tensor, Tensor]:
    """Build and validate local tangent frames for unit-sphere points.

    A custom strategy is called with normalized points of shape ``[N, 3]``.
    It must return a right-handed orthonormal pair ``(e1, e2)`` in the tangent
    plane.  Vector-field components are interpreted in the returned basis.
    """

    if points_xyz.ndim != 2 or points_xyz.shape[-1] != 3:
        raise ValueError("points_xyz must have shape [N, 3]")
    if not points_xyz.is_floating_point():
        raise TypeError("points_xyz must have a floating-point dtype")

    points = torch.nn.functional.normalize(points_xyz, dim=-1)
    if frame_strategy == "robust":
        e1, e2 = robust_tangent_frames(points)
        source = "frame_strategy='robust'"
    elif frame_strategy == "east_north":
        e1, e2 = _east_north_tangent_frames(points)
        source = "frame_strategy='east_north'"
    elif callable(frame_strategy):
        result = frame_strategy(points)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(
                "custom frame_strategy must return a tuple (e1, e2)"
            )
        e1, e2 = result
        if not isinstance(e1, Tensor) or not isinstance(e2, Tensor):
            raise TypeError("custom frame_strategy must return torch.Tensor values")
        source = "custom frame_strategy"
    else:
        raise ValueError(
            "frame_strategy must be 'robust', 'east_north', or a callable"
        )
    return _validate_tangent_frames(points, e1, e2, source=source)


def rotate_tangent_frames(e1: Tensor, e2: Tensor, gamma: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate frames as W' = W R(gamma).

    With this convention coordinates of tangent vectors transform by
    R(-gamma).  ``gamma`` has shape ``[N]`` or any shape broadcastable to
    ``e1[..., 0]``.
    """

    gamma = gamma.to(device=e1.device, dtype=e1.dtype)
    c = torch.cos(gamma).unsqueeze(-1)
    s = torch.sin(gamma).unsqueeze(-1)
    e1_new = c * e1 + s * e2
    e2_new = -s * e1 + c * e2
    return e1_new, e2_new


def rotation_matrices_from_angles(phi: Tensor) -> Tensor:
    """Return 2D rotation matrices R(phi) with shape ``phi.shape + [2, 2]``."""

    c = torch.cos(phi)
    s = torch.sin(phi)
    return rotation_matrices_from_cos_sin(c, s)


def rotation_matrices_from_cos_sin(cos_phi: Tensor, sin_phi: Tensor) -> Tensor:
    """Return 2D rotation matrices from cached cos/sin values."""

    row0 = torch.stack((cos_phi, -sin_phi), dim=-1)
    row1 = torch.stack((sin_phi, cos_phi), dim=-1)
    return torch.stack((row0, row1), dim=-2)


def rotate_vector_components(v: Tensor, gamma: Tensor) -> Tensor:
    """Rotate 2D vector components by R(gamma).

    This is useful in tests.  If frames are rotated by ``gamma``, components
    of the same physical tangent vector should be compared with
    ``rotate_vector_components(v, -gamma)``.
    """

    gamma = gamma.to(device=v.device, dtype=v.dtype)
    target_ndim = v.ndim - 1
    if gamma.ndim == 1 and target_ndim >= 2:
        gamma = gamma.view(1, gamma.numel(), *([1] * (target_ndim - 2)))
    else:
        while gamma.ndim < target_ndim:
            gamma = gamma.unsqueeze(-1)
    c = torch.cos(gamma)
    s = torch.sin(gamma)
    x = v[..., 0]
    y = v[..., 1]
    return torch.stack((c * x - s * y, s * x + c * y), dim=-1)


def log_map_sphere(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    """Logarithmic map log_p(q) on the unit sphere, in radians.

    ``p`` and ``q`` are unit 3D vectors with matching leading dimensions.
    The implementation returns exactly zero for coincident points, avoiding
    a tiny non-tangent artifact at self-neighbors.
    """

    cos_theta = (p * q).sum(dim=-1, keepdim=True).clamp(min=-1.0 + eps, max=1.0)
    theta = torch.acos(cos_theta)
    tangent = q - cos_theta * p
    tangent_norm = tangent.norm(dim=-1, keepdim=True)
    scale = torch.where(tangent_norm > eps, theta / tangent_norm.clamp_min(eps), torch.zeros_like(theta))
    return scale * tangent


def log_map_sphere_km(
    p: Tensor,
    q: Tensor,
    *,
    earth_radius_km: float = 6371.0,
    eps: float = 1e-12,
) -> Tensor:
    """Logarithmic map on the sphere, scaled to kilometers."""

    return earth_radius_km * log_map_sphere(p, q, eps=eps)


def parallel_transport_sphere(q: Tensor, p: Tensor, v: Tensor, eps: float = 1e-12) -> Tensor:
    """Parallel transport tangent vectors from ``T_q S^2`` to ``T_p S^2``.

    The formula is valid away from antipodal points.  Radius neighborhoods in
    this prototype are local, so antipodal transport is intentionally out of
    scope.
    """

    dot_pq = (p * q).sum(dim=-1, keepdim=True)
    dot_vp = (v * p).sum(dim=-1, keepdim=True)
    return v - (dot_vp / (1.0 + dot_pq + eps)) * (p + q)


def project_to_frame(v: Tensor, e1: Tensor, e2: Tensor) -> Tensor:
    """Project 3D tangent vectors onto local frame coordinates."""

    return torch.stack(((v * e1).sum(dim=-1), (v * e2).sum(dim=-1)), dim=-1)


def transport_matrix_q_to_p(
    p: Tensor,
    q: Tensor,
    e1_p: Tensor,
    e2_p: Tensor,
    e1_q: Tensor,
    e2_q: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """Compute the 2x2 coordinate transport matrix from q-frame to p-frame."""

    pt_e1_q = parallel_transport_sphere(q, p, e1_q, eps=eps)
    pt_e2_q = parallel_transport_sphere(q, p, e2_q, eps=eps)
    g00 = (e1_p * pt_e1_q).sum(dim=-1)
    g01 = (e1_p * pt_e2_q).sum(dim=-1)
    g10 = (e2_p * pt_e1_q).sum(dim=-1)
    g11 = (e2_p * pt_e2_q).sum(dim=-1)
    return torch.stack(
        (torch.stack((g00, g01), dim=-1), torch.stack((g10, g11), dim=-1)),
        dim=-2,
    )


def angular_distance(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    """Great-circle angular distance in radians for unit-sphere points."""

    dot = (p * q).sum(dim=-1).clamp(min=-1.0 + eps, max=1.0)
    return torch.acos(dot)


def chord_radius_from_km(radius_km: float, earth_radius_km: float = 6371.0) -> float:
    """Convert a geodesic radius to unit-sphere chord radius."""

    radius_angle = radius_km / earth_radius_km
    return 2.0 * math.sin(radius_angle / 2.0)
