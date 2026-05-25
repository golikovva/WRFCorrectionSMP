import numpy as np
import torch
from torch import nn


class ANOCorrector(nn.Module):
    """
    Anomaly Numerical Correction with Observations.

    The module accumulates mean target-reference errors by a calendar period and
    applies the corresponding mean error to incoming reference fields.
    """

    requires_dates = True

    def __init__(
        self,
        start_date=None,
        time_mean_period="day",
        *,
        period=None,
        n_channels=3,
        time_dim=0,
        batch_dim=1,
        time_step_hours=1,
        dtype=torch.float32,
    ):
        super().__init__()
        self.start_date = None if start_date is None else np.datetime64(start_date)
        self.period = self._normalize_period(period or time_mean_period)
        self.n_channels = int(n_channels)
        self.time_dim = time_dim
        self.batch_dim = batch_dim
        self.time_step_hours = int(time_step_hours)
        self.accumulator_dtype = dtype
        self.period_range = self._get_range()

        self.register_buffer("correction_sums", torch.empty(0, dtype=dtype))
        self.register_buffer("period_counts", torch.zeros(self.period_range, dtype=torch.float32))

    @staticmethod
    def _normalize_period(period):
        period = str(period).lower().strip()
        aliases = {
            "daily": "day",
            "dayofyear": "day",
            "doy": "day",
            "monthly": "month",
            "mon": "month",
            "constant": "all",
            "global": "all",
        }
        period = aliases.get(period, period)
        if period not in {"day", "month", "all"}:
            raise ValueError(f"Unsupported ANO period: {period!r}. Use 'day', 'month' or 'all'.")
        return period

    def _get_range(self):
        if self.period == "month":
            return 12
        if self.period == "day":
            return 366
        return 1

    def _period_indices(self, dates):
        dates = np.asarray(dates)
        if dates.size == 0:
            return dates.astype(np.int64)

        if np.issubdtype(dates.dtype, np.datetime64):
            if self.period == "month":
                periods = dates.astype("datetime64[M]").astype(int) % 12
            elif self.period == "day":
                days = dates.astype("datetime64[D]")
                periods = (days - days.astype("datetime64[Y]")).astype(int)
            else:
                periods = np.zeros(dates.shape, dtype=np.int64)
        else:
            periods = dates.astype(np.int64)

        periods = np.asarray(periods, dtype=np.int64)
        if periods.size and ((periods < 0).any() or (periods >= self.period_range).any()):
            raise ValueError(
                f"Period ids must be in [0, {self.period_range - 1}] for period={self.period!r}."
            )
        return periods

    def _expand_dates_for_sequence(self, dates, data):
        dates = np.asarray(dates)
        if not np.issubdtype(dates.dtype, np.datetime64):
            return dates

        leading_ndim = data.ndim - 3
        if dates.ndim != 1 or leading_ndim < 2:
            return dates

        time_dim = self.time_dim if self.time_dim >= 0 else leading_ndim + self.time_dim
        batch_dim = self.batch_dim if self.batch_dim >= 0 else leading_ndim + self.batch_dim
        if not (0 <= time_dim < leading_ndim and 0 <= batch_dim < leading_ndim):
            return dates
        if dates.shape[0] != data.shape[batch_dim]:
            return dates

        offsets = np.arange(data.shape[time_dim]) * np.timedelta64(self.time_step_hours, "h")
        shape = [1] * leading_ndim
        shape[time_dim] = data.shape[time_dim]
        offsets = offsets.reshape(shape)

        date_shape = [1] * leading_ndim
        date_shape[batch_dim] = dates.shape[0]
        return dates.reshape(date_shape) + offsets

    def _ensure_initialized(self, field_shape, device=None):
        field_shape = tuple(int(x) for x in field_shape)
        expected_shape = (self.period_range, *field_shape)

        if self.correction_sums.numel() == 0:
            self.correction_sums = torch.zeros(
                expected_shape,
                dtype=self.accumulator_dtype,
                device=device or self.period_counts.device,
            )
            self.period_counts = torch.zeros(
                self.period_range,
                dtype=torch.float32,
                device=device or self.period_counts.device,
            )
            return

        if tuple(self.correction_sums.shape) != expected_shape:
            raise ValueError(
                f"ANO correction field shape mismatch: expected {self.correction_sums.shape[1:]}, "
                f"got {field_shape}."
            )

    @property
    def is_fitted(self):
        return self.correction_sums.numel() > 0 and torch.any(self.period_counts > 0).item()

    def reset(self):
        if self.correction_sums.numel() > 0:
            self.correction_sums.zero_()
        self.period_counts.zero_()
        return self

    def accumulate(self, correction, dates):
        """
        Accumulate target-reference corrections.

        correction shape must be ``dates.shape + (C, H, W)``. Torch tensors and
        numpy arrays are accepted.
        """
        if torch.is_tensor(correction):
            correction_tensor = correction.detach().to(device=self.period_counts.device, dtype=self.accumulator_dtype)
        else:
            correction_tensor = torch.as_tensor(correction, dtype=self.accumulator_dtype, device=self.period_counts.device)

        if correction_tensor.ndim < 3:
            raise ValueError("correction must have at least channel, height and width dimensions.")

        dates = np.asarray(dates)
        if dates.shape != tuple(correction_tensor.shape[:-3]):
            expanded = self._expand_dates_for_sequence(dates, correction_tensor)
            if expanded.shape != tuple(correction_tensor.shape[:-3]):
                raise ValueError(
                    f"dates shape {dates.shape} is not compatible with correction shape "
                    f"{tuple(correction_tensor.shape)}."
                )
            dates = expanded

        self._ensure_initialized(correction_tensor.shape[-3:], correction_tensor.device)

        periods = self._period_indices(dates).reshape(-1)
        correction_flat = correction_tensor.reshape(len(periods), *correction_tensor.shape[-3:])
        period_ids = torch.as_tensor(periods, dtype=torch.long, device=correction_tensor.device)

        self.correction_sums.index_add_(0, period_ids, correction_flat)
        ones = torch.ones(len(periods), dtype=self.period_counts.dtype, device=self.period_counts.device)
        self.period_counts.index_add_(0, period_ids.to(self.period_counts.device), ones)
        return self

    def accumulate_corr(self, corr, dates):
        return self.accumulate(corr, dates)

    def correction_for_dates(self, dates, *, device=None, dtype=None):
        if self.correction_sums.numel() == 0:
            raise RuntimeError("ANO correction fields are empty. Call accumulate() or load_correction_fields() first.")

        periods = self._period_indices(dates)
        period_ids = torch.as_tensor(periods, dtype=torch.long, device=self.correction_sums.device)

        sums = self.correction_sums[period_ids]
        counts = self.period_counts[period_ids].to(device=sums.device, dtype=sums.dtype)
        denom = counts.reshape(*counts.shape, *([1] * (sums.ndim - counts.ndim)))
        fields = torch.where(denom > 0, sums / torch.clamp_min(denom, 1.0), torch.zeros_like(sums))

        if device is not None or dtype is not None:
            fields = fields.to(device=device or fields.device, dtype=dtype or fields.dtype)
        return fields

    def get_mean_corrections(self):
        if self.correction_sums.numel() == 0:
            return torch.empty(0, dtype=self.accumulator_dtype, device=self.period_counts.device)
        counts = self.period_counts.to(device=self.correction_sums.device, dtype=self.correction_sums.dtype)
        denom = counts.reshape(-1, *([1] * (self.correction_sums.ndim - 1)))
        return torch.where(denom > 0, self.correction_sums / torch.clamp_min(denom, 1.0), torch.zeros_like(self.correction_sums))

    def _get_mean_corrections(self):
        return self.get_mean_corrections().detach().cpu().numpy()

    def forward(self, data, dates=None):
        if dates is None:
            raise ValueError("ANOCorrector.forward requires dates. Pass dates=... or use a test() date-aware hook.")

        dates = self._expand_dates_for_sequence(dates, data)
        if not torch.is_tensor(data):
            correction = self.correction_for_dates(dates).detach().cpu().numpy()
            data_channels = data[..., : self.n_channels, :, :]
            if correction.shape[-3:] != data_channels.shape[-3:]:
                raise ValueError(
                    f"Correction field shape {tuple(correction.shape[-3:])} does not match "
                    f"input field shape {tuple(data_channels.shape[-3:])}."
                )
            return data_channels + correction

        correction = self.correction_for_dates(dates, device=data.device, dtype=data.dtype)

        data_channels = data[..., : self.n_channels, :, :]
        if correction.shape[-3:] != data_channels.shape[-3:]:
            raise ValueError(
                f"Correction field shape {tuple(correction.shape[-3:])} does not match "
                f"input field shape {tuple(data_channels.shape[-3:])}."
            )
        return data_channels + correction

    def save_correction_fields(self, path):
        np.save(path, self._get_mean_corrections())

    def load_correction_fields(self, path, *, nan_to_num=True):
        fields = np.load(path)
        if nan_to_num:
            fields = np.nan_to_num(fields, copy=False)

        fields_tensor = torch.as_tensor(fields, dtype=self.accumulator_dtype, device=self.period_counts.device)
        if fields_tensor.ndim != 4:
            raise ValueError(f"Expected correction fields with shape (period, C, H, W), got {fields_tensor.shape}.")
        if fields_tensor.shape[0] != self.period_range:
            raise ValueError(
                f"Loaded {fields_tensor.shape[0]} periods, but period={self.period!r} expects {self.period_range}."
            )

        self.correction_sums = fields_tensor.clone()
        self.period_counts = torch.ones(self.period_range, dtype=torch.float32, device=fields_tensor.device)
        return self

    @classmethod
    def from_correction_fields(cls, path, **kwargs):
        model = cls(**kwargs)
        return model.load_correction_fields(path)


class AccumCorrector(ANOCorrector):
    def __init__(self, corr_fields_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_correction_fields(corr_fields_path)


CorrAccumulator = ANOCorrector
