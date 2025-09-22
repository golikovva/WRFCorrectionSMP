import warnings

import torch
import torch.nn.functional as F
from pytorch_msssim import SSIM


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )
    return out


def _custom_ssim(corr, wrf, era, data_range, win, exp_coef=(1, 1, 1), K=(0.01, 0.03)):
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0
    alpha, beta, gamma = exp_coef
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    C3 = C2 / 2

    win = win.to(corr.device, dtype=corr.dtype)

    mu1 = gaussian_filter(corr, win)
    mu2 = gaussian_filter(wrf, win)
    mu3 = gaussian_filter(era, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu3_sq = mu3.pow(2)
    mu1_mu2 = mu1 * mu2
    mu1_mu3 = mu1 * mu3

    sigma1_sq = compensation * (gaussian_filter(corr * corr, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(wrf * wrf, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(corr * wrf, win) - mu1_mu2)
    # print(luminance(mu1_sq, mu3_sq, mu1_mu3, C1).pow(alpha).mean())
    # print(contrast(sigma1_sq, sigma2_sq, C2).pow(beta).mean())
    # print(structure(sigma1_sq, sigma2_sq, sigma12, C3).pow(gamma).mean())
    ssim_map = luminance(mu1_sq, mu3_sq, mu1_mu3, C1).pow(alpha) * \
               contrast(sigma1_sq, sigma2_sq, C2).pow(beta) * \
               structure(sigma1_sq, sigma2_sq, sigma12, C3).pow(gamma)

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    return ssim_per_channel


def luminance(mu1_sq, mu2_sq, mu1_mu2, C1):
    return (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)


def contrast(sigma1_sq, sigma2_sq, C2):
    # print(sigma1_sq.min(), sigma2_sq.min(), 'sigma')
    # print((2 * sigma1_sq.pow(1 / 2) * sigma2_sq.pow(1 / 2) + C2).min(), 'up')
    # print((sigma1_sq + sigma2_sq + C2).min(), 'down')
    return (2 * torch.relu(sigma1_sq).pow(1 / 2) * torch.relu(sigma2_sq).pow(1 / 2) + C2) / (sigma1_sq + sigma2_sq + C2)


def structure(sigma1_sq, sigma2_sq, sigma12, C3):
    # print(sigma1_sq.min(), sigma2_sq.min(), 'sigma')
    # print((sigma12 + C3).min(), 'up')
    # print((sigma1_sq.pow(1 / 2) * sigma2_sq.pow(1 / 2) + C3).min(), 'down')
    return (sigma12 + C3) / (torch.relu(sigma1_sq).pow(1 / 2) * torch.relu(sigma2_sq).pow(1 / 2) + C3)


def custom_ssim(
        corr,
        wrf,
        era,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        win=None,
        exp_coef=(1, 1, 1),
        K=(0.01, 0.03),
        nonnegative_ssim=False,
):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not corr.shape == wrf.shape == era.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(corr.shape) - 1, 1, -1):
        corr = corr.squeeze(dim=d)
        wrf = wrf.squeeze(dim=d)
        era = era.squeeze(dim=d)

    if len(corr.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {corr.shape}")
    bs_sl = corr.shape[:1]
    if len(corr.shape) == 5:
        bs_sl = corr.shape[:2]
        corr = corr.flatten(0, 1)
        wrf = wrf.flatten(0, 1)
        era = era.flatten(0, 1)
    if not corr.type() == wrf.type() == era.type():
        raise ValueError("Input images should have the same dtype.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([corr.shape[1]] + [1] * (len(corr.shape) - 1))

    ssim_per_channel = _custom_ssim(corr, wrf, era, data_range=data_range, win=win, exp_coef=exp_coef, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:    
        return ssim_per_channel.mean(0)
    else:
        return ssim_per_channel.unflatten(0, bs_sl)


class CustomSSIM(torch.nn.Module):
    def __init__(
            self,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            channel=3,
            spatial_dims=2,
            exp_coefs=(1, 1, 1),
            K=(0.01, 0.03),
            nonnegative_ssim=False,
    ):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(CustomSSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.exp_coefs = exp_coefs
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, corr, wrf, era):
        return custom_ssim(
            corr,
            wrf,
            era,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            exp_coef=self.exp_coefs,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )
