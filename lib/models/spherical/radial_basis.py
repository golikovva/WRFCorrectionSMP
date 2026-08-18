"""KPConv-inspired radial influence functions."""

from __future__ import annotations

import torch
from torch import Tensor


def radial_centers(radius_km: float, num_radial: int, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Linearly spaced radial centers from 0 to the convolution radius."""

    if num_radial < 1:
        raise ValueError("num_radial must be >= 1")
    if num_radial == 1:
        return torch.zeros(1, dtype=dtype)
    return torch.linspace(0.0, float(radius_km), int(num_radial), dtype=dtype)


def default_radial_sigma(radius_km: float, num_radial: int) -> float:
    """Default triangular support width for adjacent radial centers."""

    if num_radial <= 1:
        return float(radius_km)
    return float(radius_km) / float(num_radial - 1)


def triangular_radial_basis(
    r: Tensor,
    centers: Tensor,
    sigma: float | Tensor,
    *,
    normalize: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """Evaluate triangular radial basis functions.

    This mirrors KPConv's linear influence idea, but the anchors are radial
    shells rather than unconstrained 2D or 3D kernel points.
    """

    sigma_t = torch.as_tensor(sigma, device=r.device, dtype=r.dtype).clamp_min(eps)
    centers = centers.to(device=r.device, dtype=r.dtype)
    h = torch.relu(1.0 - torch.abs(r.unsqueeze(-1) - centers) / sigma_t)
    if normalize:
        h = h / h.sum(dim=-1, keepdim=True).clamp_min(eps)
    return h
