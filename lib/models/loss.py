import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.model_parts import DeltaConvLayer


class HeterogenousMSLoss(nn.Module):
    def __init__(self, meaner, betas, station_interpolator=None, scatter_interpolator=None, logger=None,
                 kernel_type='gauss', channels=3, k=9, device='cpu'):
        super().__init__()
        self.mean_mse = nn.MSELoss()
        self.delta_mse = nn.MSELoss()
        self.station_mse = nn.MSELoss(reduction='none')
        self.scat_mse = nn.MSELoss()
        self.delta_conv = DeltaConvLayer(channels=channels, k=k, kernel_type=kernel_type).to(device)
        self.station_interpolator = station_interpolator
        self.scatter_interpolator = scatter_interpolator
        if scatter_interpolator:
            self.wrf_mask = scatter_interpolator.calc_input_tensor_mask([132, 430], fill_value=0)
        self.meaner = meaner
        self.betas = betas
        self.betas[2] = torch.tensor(self.betas[2], device=device)

        if logger:
            logger.set_beta(betas)

    def forward(self, orig, corr, target=None, stations=None, scatter=None, scatter_times=None, orig_dates=None, logger=None, expanded_out=False):
        device = orig.device
        mse1 = torch.tensor(0., device=device, requires_grad=True)
        if target is not None and self.betas[0] > 0:
            mean_corr = self.mean_input_to_target(corr)
            mse1 = self.mean_mse(mean_corr, target)

        delta_corr = corr - self.delta_conv(corr.view(-1, *corr.shape[-3:])).view(corr.shape)
        delta_orig = orig - self.delta_conv(orig.view(-1, *corr.shape[-3:])).view(orig.shape)
        mse2 = self.delta_mse(delta_corr, delta_orig)

        mse3 = torch.tensor(0., device=device, requires_grad=True)
        if stations is not None and self.betas[2].sum() > 0:
            pred_stations = self.station_interpolator(corr.flatten(-2, -1))
                        
            mask = ~torch.isnan(stations)
            valid_pred   = pred_stations[mask]     # 1D tensor of only the finite entries
            valid_target = stations[mask]

            if valid_pred.numel() > 0:
                mse3 = F.mse_loss(valid_pred, valid_target, reduction='mean')

        mse4 = torch.tensor(0., device=device, requires_grad=True)
        if scatter is not None and self.betas[3] > 0:
            # interpolate nwp in space
            corr_on_scat_grid = self.scatter_interpolator(corr.flatten(-2, -1)).unflatten(dim=-1, sizes=scatter.shape[-2:])[:, :, :2]
            # interpolate nwp in time
            corr_on_scat_grid = interp_nwp_in_time(corr_on_scat_grid, scatter_times, orig_dates)
            # filter NaNs
            mask = (torch.isfinite(corr_on_scat_grid)) & (torch.isfinite(scatter) & torch.isfinite(self.wrf_mask))

            corr_on_scat_grid = corr_on_scat_grid[mask]     # 1D tensor of only the finite entries
            scatter = scatter[mask]
            if corr_on_scat_grid.numel() > 0 and scatter.numel() > 0:
                mse4 = F.mse_loss(corr_on_scat_grid, scatter, reduction='mean')

        total_mse = self.betas[0] * mse1 + self.betas[1] * mse2 + self.betas[2] * mse3 \
                    + self.betas[3] * mse4

        if logger:
            logger.accumulate_stat(total_mse.item(), mse1.item(), mse2.item(), mse3.item(), mse4.item())
        if expanded_out:
            return total_mse, mse1, mse2, mse3, mse4
        return total_mse
    
    def mean_input_to_target(self, corr):
        mean_corr = self.meaner(corr)
        # t = target.flatten(-2, -1)
        # t = t[..., self.meaner.mapping.unique().long()]
        return mean_corr


def interp_nwp_in_time(
        nwp: torch.Tensor,        # (sl, bs, 2, H, W)  – on scatter grid
        meas_time: torch.Tensor,  # (bs, n, H, W)      – epoch-seconds
        nwp_t0: torch.Tensor,     # (bs,)              – epoch-seconds of step-0
        dt: int = 3600,            # NWP time-step, seconds
        batch_first: bool = False,
        return_counts: bool = False,  
) -> torch.Tensor:
    """
    Hourly NWP → per-pixel scatterometer instants (linear interp, GPU-friendly).

    Returns (bs, n, 2, H, W) with NaN where `meas_time` falls outside the NWP window.
    """
    device = nwp.device
    meas_time = meas_time.to(device)
    nwp_t0    = nwp_t0.to(device).to(meas_time.dtype)
    # -------- geometry ----------
    sl, bs, C, H, W = nwp.shape
    assert C == 2, "expect (u,v) in dim-2"
    n = meas_time.shape[1]

    # Re-order NWP: batch first so we can gather along the 'sl' axis
    nwp_b = nwp.permute(1, 0, 2, 3, 4) if not batch_first else nwp  # (bs, sl, 2, H, W)

    # -------- fractional index in the NWP time axis ----------
    rel = ((meas_time - nwp_t0[:, None, None, None]) / dt).to(nwp_b.dtype)
    idx0 = torch.floor(rel).long()                              # lower frame
    idx1 = idx0 + 1                                             # upper frame

    # Clamp to valid range so gather never steps out of bounds
    idx0 = idx0.clamp(0, sl - 1)
    idx1 = idx1.clamp(0, sl - 1)

    # Interpolation weight (0…1).  Values <0 or >1 will be masked later.
    w = (rel - idx0.to(rel.dtype)).clamp(0, 1)                  # same shape as meas_time

    # Expand indices so `take_along_dim` can broadcast across the (2, H, W) tail
    idx0_exp = idx0.unsqueeze(2)                                # (bs, n, 1, H, W)
    idx1_exp = idx1.unsqueeze(2)

    # Gather the two bracketing frames along the 'sl' axis (dim=1)
    val0 = torch.take_along_dim(nwp_b, idx0_exp, dim=1)         # (bs, n, 2, H, W)
    val1 = torch.take_along_dim(nwp_b, idx1_exp, dim=1)

    # Linear interpolation
    interp = val0 * (1.0 - w.unsqueeze(2)) + val1 * w.unsqueeze(2)

    # -------- mask times outside the NWP window ----------
    in_window = (rel >= 0) & (rel <= (sl - 1))
    interp = torch.where(in_window.unsqueeze(2), interp, torch.full_like(interp, float('nan')))
    return interp                                              # (bs, n, 2, H, W)


class SmallScaleLoss(nn.Module):
    def __init__(self, reduction='none', kernel_type='gauss', channels=3, k=9, device='cpu'):
        super().__init__()
        self.reduction = reduction
        self.delta_conv = DeltaConvLayer(channels=channels, k=k, kernel_type=kernel_type).to(device)

    def forward(self, orig, corr):
        delta_corr = corr - self.delta_conv(corr.view(-1, *corr.shape[-3:])).view(corr.shape)
        delta_orig = orig - self.delta_conv(orig.view(-1, *corr.shape[-3:])).view(orig.shape)
        return F.mse_loss(delta_corr, delta_orig, reduction=self.reduction)


def uvt_to_wt(data, c_dim=-2):
    assert data.shape[c_dim] == 3, f'assumed 3 channels (u, v, t) to be processed but got {data.shape[c_dim]}'
    u, v, t = torch.split(data, 1, dim=c_dim)
    w = torch.sqrt(torch.square(u) + torch.square(v))
    return torch.cat([w, t], dim=c_dim)


class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)  # todo исправить
        return loss


class DiffLoss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, yhat, y):
        loss = yhat - y
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


class AbsDiffLoss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, yhat, y):
        loss = abs(yhat) - abs(y)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


class WindLoss(nn.Module):
    def __init__(self, interpolator, device='cpu'):
        super().__init__()
        self.mae_uv = nn.L1Loss(reduction='mean')
        self.tmae_speed = nn.L1Loss(reduction='mean')
        self.interpolator = interpolator
        self.device = device

    def forward(self, orig, wrf, era, *args, channel_dim=-3, logger=None):
        era = torch.index_select(era, 0, torch.tensor(list(range(4, 8)), device=self.device))  # get last 4 time samples

        era = self.interpolator(era.flatten(-2, -1)).view(wrf.shape)
        uv = self.mae_uv(wrf, era)
        assert era.shape[channel_dim] == wrf.shape[channel_dim] == 2, f'{era.shape}, {wrf.shape}, but should be 2'
        speed = self.tmae_speed(self.uv_to_speed(wrf, channel_dim=channel_dim),
                                self.uv_to_speed(era, channel_dim=channel_dim))
        if logger:
            logger.accumulate_stat((uv + speed).item())
        return uv + speed

    @staticmethod
    def uv_to_speed(data, channel_dim=-3):
        return torch.sqrt(torch.square(data).sum(dim=channel_dim))
