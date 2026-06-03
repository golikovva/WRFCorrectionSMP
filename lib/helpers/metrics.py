import functools

import torch
import torch.nn as nn
from typing import Optional
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


class MulticlassAccuracy_old(nn.Module):
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


class MulticlassAccuracy(nn.Module):
    """
    Per-element accuracy for possibly circular classes (e.g., 16 wind sectors).

    Shapes:
      target: ..., 1, N        (float with NaNs for missing, or int without NaNs)
      preds:  ..., K, N  (logits/scores -> argmax along `dim`)  OR
              ..., 1, N  (already integer class labels)

    Returns:
      Float tensor same shape as `target` with 1.0/0.0 and NaN for missing.
    """
    def __init__(self, dim: int = 1, tol: int = 0, circular: bool = False, K: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.tol = int(tol)
        self.circular = bool(circular)
        self.K = None if K is None else int(K)

    @staticmethod
    def _circ_dist(a: torch.Tensor, b: torch.Tensor, K: int) -> torch.Tensor:
        d = (a - b).abs()
        return torch.minimum(d, K - d)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1) Determine if preds are logits (scores) or labels
        is_scores = preds.size(self.dim) > 1  # K>1 ⇒ treat as scores
        if is_scores:
            pred_labels = preds.argmax(dim=self.dim)             # drops `dim`
            pred_labels = pred_labels.unsqueeze(self.dim)        # reinsert singleton to match target
            K_infer = preds.size(self.dim)
        else:
            pred_labels = preds.to(torch.long)                   # keep shape
            K_infer = None

        # 2) Ensure circular K if needed
        K = self.K if self.K is not None else K_infer
        if self.circular and (K is None):
            raise ValueError("For circular=True, provide K or pass logits (so K can be inferred).")

        # 3) Build masks; accept int targets (no NaNs) or float targets (with NaNs)
        if target.is_floating_point():
            missing = torch.isnan(target)
        else:
            missing = torch.zeros_like(target, dtype=torch.bool)
        valid = ~missing

        # 4) Prepare output (same shape as target)
        out = torch.empty_like(target, dtype=torch.float, device=target.device)
        out[missing] = float('nan')

        if valid.any():
            # Align dtypes and compare on valid entries
            tgt_int = target[valid].to(torch.long)
            pred_v  = pred_labels[valid]
            if self.circular and self.tol > 0:
                dist = self._circ_dist(pred_v, tgt_int, K=K)
                correct = (dist <= self.tol)
            else:
                correct = (pred_v == tgt_int)
            out[valid] = correct.to(torch.float)

        return out
    
    
class HeidkeSkillScore(nn.Module):
    """
    Multi-class Heidke Skill Score (HSS) aggregated over all valid elements.

    Shapes:
      target: ..., 1, N   (float with NaNs allowed, or int)
      preds:  ..., K, N   (logits)  OR  ..., 1, N  (labels)

    Returns:
      Scalar tensor with HSS = (P_o - P_e) / (1 - P_e).
      (HSS uses strict class equality; do not apply sector tolerance here.)
    """
    def __init__(self, dim: int = 1, K: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.K = None if K is None else int(K)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1) Pred labels
        if preds.size(self.dim) > 1:           # logits/scores
            K = preds.size(self.dim) if self.K is None else self.K
            pred_labels = preds.argmax(dim=self.dim)             # drops `dim`
        else:                                   # integer labels with singleton channel
            pred_labels = preds.squeeze(self.dim).to(torch.long) # remove singleton channel
            if self.K is None:
                # Infer K from max label across valid entries (safe upper bound)
                with torch.no_grad():
                    max_pred = pred_labels.max().item() if pred_labels.numel() > 0 else -1
                    if target.is_floating_point():
                        tmask = ~torch.isnan(target.squeeze(self.dim))
                        max_tgt = target.squeeze(self.dim)[tmask].max().item() if tmask.any() else -1
                    else:
                        max_tgt = target.squeeze(self.dim).max().item() if target.numel() > 0 else -1
                K = int(max(max_pred, max_tgt)) + 1
            else:
                K = self.K

        # 2) Targets: squeeze singleton channel
        tgt = target.squeeze(self.dim)
        # Missing mask (allow float with NaNs or int without NaNs)
        if tgt.is_floating_point():
            valid = ~torch.isnan(tgt)
            y_o = tgt[valid].to(torch.long)
            y_f = pred_labels[valid]
        else:
            valid = torch.ones_like(tgt, dtype=torch.bool)
            y_o = tgt.to(torch.long).reshape(-1)
            y_f = pred_labels.reshape(-1)

        if not valid.any():
            return torch.tensor(float('nan'), device=target.device)

        # 3) Flatten valid pairs
        if tgt.is_floating_point():
            y_f = y_f.reshape(-1)
            y_o = y_o.reshape(-1)

        # 4) Confusion matrix via bincount
        idx = (y_o * K + y_f).to(torch.long)
        C = torch.bincount(idx, minlength=K*K).reshape(K, K).to(torch.float)

        N = C.sum()
        if N <= 0:
            return torch.tensor(float('nan'), device=target.device)

        Po = torch.trace(C) / N
        no = C.sum(dim=1)
        nf = C.sum(dim=0)
        Pe = (no * nf).sum() / (N * N)

        denom = (1.0 - Pe)
        HSS = torch.where(denom > 0, (Po - Pe) / denom, torch.tensor(float('nan'), device=target.device))
        return HSS