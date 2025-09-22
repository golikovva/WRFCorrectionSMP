import functools

import torch
import torch.nn as nn

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


class NamedDictMetric:
    def __init__(self, metric, names):
        self.metric = metric
        self.names = names
        self.sum = 0
        self.counts = 0

    def update(self, data_dict, *args, **kwargs):
        if hasattr(self.metric, 'update'):
            self.metric.update(*[data_dict[name] for name in self.names], *args, **kwargs)
        else:
            # print(self.metric(*[data_dict[name] for name in self.names], *args, **kwargs), 'metric call')
            self.sum += self.metric(*[data_dict[name] for name in self.names], *args, **kwargs)
            self.counts += 1

    def compute(self, *args, **kwargs):
        if hasattr(self.metric, 'compute'):
            self.metric.compute(*args, **kwargs)
        else:
            return self.sum / self.counts
    
    def calculate(self, data_dict, *args, **kwargs):
        return self.metric(*[data_dict[name] for name in self.names], *args, **kwargs)
    

class MeanerMetric:
    def __init__(self, meaner, criterion):
        self.meaner = meaner
        self.criterion = criterion

    def __call__(self, *args, **kwargs):
        return self.calculate_era_loss(*args, **kwargs)

    def calculate_era_loss(self, wrf, era, meaner=None, criterion=None):
        if meaner is None:
            meaner = self.meaner
        if criterion is None:
            criterion = self.criterion
        wrf_orig = meaner(wrf)
        era = era.flatten(-2, -1)
        era = era[..., meaner.mapping.unique().long()]
        loss = criterion(wrf_orig, era)
        return loss  # loss.shape = 4, 1, 3, 8744 i.e. sl, bs, c, N


def scale(x, data_range, feature_range):
    std = (x - data_range[0]) / (data_range[1] - data_range[0])
    X_scaled = std * (feature_range[1] - feature_range[0]) + feature_range[0]
    return X_scaled


class NormSSIM(SSIM):
    def forward(self, X, Y):
        dim = list(range(X.ndim + 1))
        dim.pop(-3)
        a, b = torch.stack([X, Y]).amin(dim=dim, keepdim=True), torch.stack([X, Y]).amax(dim=dim, keepdim=True)
        X, Y = [scale(x, [a, b], [0, 1]) for x in [X, Y]]
        return super().forward(X, Y)


def normalized(forward):
    def wrapper(*data, **kwargs):
        data = torch.stack(data)
        dim = list(range(data.ndim))
        dim.pop(-3)
        a, b = data.amin(dim=dim, keepdim=True), data.amax(dim=dim, keepdim=True)
        data = [scale(d, [a, b], [0, 1])[0] for d in data]
        return forward(*data, **kwargs)

    return wrapper


def channel_meaned(metric, channel_dim=-3):
    def wrapper(*data, **kwargs):
        out = metric(*data, **kwargs)
        s = list(range(out.ndim))
        s.pop(channel_dim)
        return out.mean(s)
    return wrapper


class MulticlassAccuracy(nn.Module):
    """
    Per‐element accuracy:
      • If preds has one more dim than target (i.e. class‐scores), does argmax along `dim`.
      • Otherwise assumes preds already holds integer class guesses.
    Outputs a float tensor same shape as `target` with:
      1.0  where pred == target,
      0.0  where pred != target,
      nan  where target is nan.
    """
    def __init__(self, dim: int = 1):
        """
        Args:
          dim: the channel‐dim in preds to argmax over (only if preds is scores).
        """
        super().__init__()
        self.dim = dim

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred_labels: same shape as target, dtype long
        if preds.dim() == target.dim() + 1:
            # class‐scores → pick highest
            pred_labels = preds.argmax(dim=self.dim)
        else:
            # already discrete classes
            pred_labels = preds.to(torch.long)

        # build output
        out = torch.empty_like(target, dtype=torch.float, device=target.device)
        missing = torch.isnan(target)
        valid = ~missing

        # compare only valid entries
        if valid.any():
            tgt_int = target[valid].to(torch.long)
            out[valid] = (pred_labels[valid] == tgt_int).to(torch.float)

        # missing → nan
        out[missing] = float('nan')
        return out
