from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.helpers.metrics import LatitudeWeightedMSE
from lib.models.loss import HeterogenousMSLoss


def _manual_lat_weighted_mse(pred, target, latitudes, spatial_ndim):
    weights = torch.cos(torch.deg2rad(torch.as_tensor(latitudes, dtype=pred.dtype)))
    weights = weights.clamp_min(0.0)
    spatial_shape = pred.shape[-spatial_ndim:]
    weights = weights.reshape(spatial_shape)
    weights = weights.reshape((1,) * (pred.ndim - spatial_ndim) + spatial_shape)
    diff = pred - target
    valid = torch.isfinite(diff)
    numerator = torch.where(valid, diff.square() * weights, torch.zeros_like(diff)).sum(
        dim=tuple(range(pred.ndim - spatial_ndim, pred.ndim))
    )
    denominator = torch.where(valid, weights, torch.zeros_like(diff)).sum(
        dim=tuple(range(pred.ndim - spatial_ndim, pred.ndim))
    )
    return numerator / denominator


def test_latitude_weighted_mse_matches_mse_when_latitudes_are_equal():
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    target = torch.zeros_like(pred)
    latitudes = torch.full((2, 2), 60.0)

    weighted = LatitudeWeightedMSE(latitudes, reduction='none')(pred, target)
    plain = F.mse_loss(pred, target, reduction='none').mean(dim=(-2, -1))

    torch.testing.assert_close(weighted, plain)


def test_latitude_weighted_mse_matches_manual_weighting_on_2d_grid():
    pred = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[2.0, 0.0], [1.0, 3.0]]]])
    target = torch.zeros_like(pred)
    latitudes = torch.tensor([[0.0, 60.0], [0.0, 60.0]])

    weighted = LatitudeWeightedMSE(latitudes, reduction='none')(pred, target)
    expected = _manual_lat_weighted_mse(pred, target, latitudes, spatial_ndim=2)

    torch.testing.assert_close(weighted, expected)


def test_latitude_weighted_mse_ignores_nan_and_inf_values():
    pred = torch.tensor([[[[1.0, float('nan')], [float('inf'), 4.0]]]])
    target = torch.zeros_like(pred)
    latitudes = torch.tensor([[0.0, 60.0], [0.0, 60.0]])

    weighted = LatitudeWeightedMSE(latitudes, reduction='none')(pred, target)
    expected = _manual_lat_weighted_mse(pred, target, latitudes, spatial_ndim=2)

    torch.testing.assert_close(weighted, expected)


def test_latitude_weighted_mse_supports_flattened_spatial_dimension():
    pred = torch.tensor([[[1.0, 2.0, 3.0]], [[2.0, 4.0, 6.0]]])
    target = torch.zeros_like(pred)
    latitudes = torch.tensor([0.0, 60.0, 60.0])

    weighted = LatitudeWeightedMSE(latitudes, reduction='none', spatial_ndim=1)(pred, target)
    expected = _manual_lat_weighted_mse(pred, target, latitudes, spatial_ndim=1)

    torch.testing.assert_close(weighted, expected)


class _FakeMeaner(torch.nn.Module):
    def __init__(self, latitudes):
        super().__init__()
        latitudes = np.asarray(latitudes, dtype=np.float32)
        self.target_coords = np.stack([np.zeros_like(latitudes), latitudes], axis=1)
        self._target_slice = torch.arange(len(latitudes))

    @property
    def target_slice(self):
        return self._target_slice

    def forward(self, x):
        return x.flatten(-2, -1)


class _FakeScatterInterpolator:
    def __init__(self, latitudes):
        latitudes = np.asarray(latitudes, dtype=np.float32).reshape(-1)
        self.q = np.stack([np.zeros_like(latitudes), latitudes], axis=1)

    def __call__(self, z):
        return z

    def calc_input_tensor_mask(self, mask_shape, distance_criterion=0.15, fill_value=0):
        return torch.ones(mask_shape)


def test_heterogenous_loss_keeps_plain_mse_as_default_era_loss():
    meaner = _FakeMeaner([0.0, 60.0])
    criterion = HeterogenousMSLoss(meaner, [1, 0, 0, 0], channels=1, k=1)
    orig = torch.zeros(1, 1, 1, 1, 2)
    corr = torch.tensor([[[[[1.0, 3.0]]]]])
    target = torch.tensor([[[[0.0, 1.0]]]])

    loss = criterion(orig, corr, target=target)
    expected = F.mse_loss(corr.flatten(-2, -1), target)

    torch.testing.assert_close(loss, expected)


def test_heterogenous_loss_can_use_latitude_weighted_era_loss():
    meaner = _FakeMeaner([0.0, 60.0])
    criterion = HeterogenousMSLoss(meaner, [1, 0, 0, 0], channels=1, k=1, era_loss='lat_weighted_mse')
    orig = torch.zeros(1, 1, 1, 1, 2)
    corr = torch.tensor([[[[[1.0, 3.0]]]]])
    target = torch.tensor([[[[0.0, 1.0]]]])

    loss = criterion(orig, corr, target=target)
    expected = LatitudeWeightedMSE([0.0, 60.0], reduction='mean', spatial_ndim=1)(
        corr.flatten(-2, -1),
        target,
    )

    torch.testing.assert_close(loss, expected)


def test_heterogenous_loss_can_use_latitude_weighted_scatter_loss_with_nans():
    meaner = _FakeMeaner([0.0, 60.0])
    scatter_interpolator = _FakeScatterInterpolator([[0.0, 60.0]])
    criterion = HeterogenousMSLoss(
        meaner,
        [0, 0, 0, 1],
        scatter_interpolator=scatter_interpolator,
        channels=2,
        k=1,
        scatter_loss='lat_weighted_mse',
    )
    orig = torch.zeros(1, 1, 2, 1, 2)
    corr = torch.tensor([[[[[1.0, 3.0]], [[5.0, 7.0]]]]])
    scatter = torch.tensor([[[[[0.0, float('nan')]], [[1.0, 5.0]]]]])
    scatter_times = torch.zeros(1, 1, 1, 2, dtype=torch.double)
    orig_dates = torch.zeros(1, dtype=torch.double)

    loss = criterion(orig, corr, scatter=scatter, scatter_times=scatter_times, orig_dates=orig_dates)
    expected = LatitudeWeightedMSE([[0.0, 60.0]], reduction='mean', spatial_ndim=2)(corr.permute(1, 0, 2, 3, 4), scatter)

    torch.testing.assert_close(loss, expected)
