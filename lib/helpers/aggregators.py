import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Sequence
import warnings

class Aggregator(ABC):
    """
    Abstract base class for aggregators that collect statistics from pointwise fields.
    """

    @abstractmethod
    def init_accumulator(self, shape: Tuple[int, ...]) -> Any:
        """
        Initialize and return an accumulator structure appropriate for this aggregator.
        :param shape: The shape of incoming fields (e.g., spatial grid shape).
        """
        pass

    @abstractmethod
    def accumulate(self, acc: Any, field: np.ndarray, date: Any) -> None:
        """
        Update the accumulator with a new field at a given date.
        :param acc: The accumulator returned by init_accumulator.
        :param field: The pointwise data array to aggregate.
        :param date: The corresponding date or time identifier.
        """
        pass

    @staticmethod
    @abstractmethod
    def finalize(acc: Any) -> Any:
        """
        Compute and return the final aggregated result from the accumulator.
        """
        pass


class GlobalTemporalAggregator(Aggregator):
    """
    Computes the global spatial mean at each date and stores a time series.
    """

    def init_accumulator(self, shape: Tuple[int, ...]) -> Dict[Any, float]:
        # Use a dict date -> mean value
        return {}

    def accumulate(self, acc: Dict[Any, float], field: np.ndarray, date: Any) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mean_over_space = np.nanmean(field, axis=(-2, -1))
        acc[date] = mean_over_space
    
    @staticmethod
    def finalize(acc: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
        return acc



class RegionalTemporalAggregator(Aggregator):
    """
    Accumulate per‐region, per‐date means of a C×H×W field.

    Parameters
    ----------
    region_mask : np.ndarray[int], shape (H, W)
        0 = skip, >0 = region ID.
    region_ids : Sequence[int], optional
        Which labels to track; default = all unique >0 in mask.
    """
    def __init__(self,
                 region_mask: np.ndarray,
                 region_ids=None):
        self.region_mask = region_mask
        # derive region IDs if not given
        if region_ids is None:
            self.region_ids = np.unique(region_mask)
            self.region_ids = [int(r) for r in self.region_ids if r != 0]
        else:
            self.region_ids = list(region_ids)
        self.max_label = int(region_mask.max())

    def init_accumulator(self, shape):
        """
        Returns a dict:
          region_id -> { date1: np.ndarray(C,), date2: np.ndarray(C,), ... }
        """
        return {rid: {} for rid in self.region_ids}

    def accumulate(self, acc: Dict[int, Dict[Any, np.ndarray]],
                   field: np.ndarray,
                   date: Any) -> None:
        """
        field must have last dims (C, H, W).
        We flatten H×W, then for each channel c do a bincount over region_mask.
        """
        if field.ndim == 2:
            field = field[None]  # 2D field, assume (H, W)
        if field.ndim < 3:
            raise ValueError("Field must have at least 3 dims (C, H, W) at the end")
        C, H, W = field.shape[-3], field.shape[-2], field.shape[-1]
        if (H, W) != self.region_mask.shape:
            raise ValueError(f"Field spatial dims {(H,W)} != mask shape {self.region_mask.shape}")

        # Flatten
        flat_mask = self.region_mask.ravel()  # (H*W,)
        flat_data = field.reshape(C, -1)      # (C, H*W)

        # Prepare sums & counts arrays
        sums = np.zeros((C, self.max_label+1), dtype=float)
        counts = np.zeros((C, self.max_label+1), dtype=int)

        for c in range(C):
            vals = flat_data[c]
            # Sum (NaNs→0)
            sums[c] = np.bincount(
                flat_mask,
                weights=np.nan_to_num(vals, nan=0.0),
                minlength=self.max_label+1
            )
            # Count valid
            valid_idx = ~np.isnan(vals)
            counts[c] = np.bincount(
                flat_mask[valid_idx],
                minlength=self.max_label+1
            )

        # Compute & store per‐region mean vectors
        for rid in self.region_ids:
            # avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_vec = sums[:, rid] / counts[:, rid]
            mean_vec[counts[:, rid] == 0] = np.nan
            acc[rid][date] = mean_vec

    @staticmethod
    def finalize(acc: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
        return acc


class SpatialAggregator(Aggregator):
    """
    Computes the mean field over time for each grid cell (i.e., time-averaged spatial map).
    """

    def init_accumulator(self, shape: Tuple[int, ...]) -> Dict[str, np.ndarray]:
        return {
            'sum': torch.zeros(shape, dtype=float),
            'count': torch.zeros(shape, dtype=int)
        }

    def accumulate(self, acc: Dict[str, torch.Tensor], field: np.ndarray | torch.Tensor, date: Any) -> None:
        # Sum and count non-nan values
        x = torch.as_tensor(field, dtype=acc['sum'].dtype, device=acc['sum'].device)
        extra = x.ndim - acc['sum'].ndim

        if extra > 0:
            dims = tuple(range(extra))
            sum_x = torch.nan_to_num(x, nan=0.0).sum(dim=dims)
            count_x = (~torch.isnan(x)).sum(dim=dims)
            
        else:
            sum_x = torch.nan_to_num(x, nan=0.0)
            count_x = (~torch.isnan(x)).to(torch.int)
        acc['sum']   += sum_x
        acc['count'] += count_x
        

    @staticmethod
    def finalize(acc: Dict[str, np.ndarray]) -> np.ndarray:
        mean_field = acc['sum'] / torch.where(acc['count'] == 0, torch.nan, acc['count'])
        mean_field[acc['count'] == 0] = torch.nan
        return mean_field
    
class AverageAggregator(SpatialAggregator):
    """
    Computes the mean of all accumulated fields, ignoring NaNs.
    """

    @staticmethod
    def finalize(acc: Dict[str, np.ndarray]) -> np.ndarray:
        dims = tuple(range(1, acc['sum'].ndim))
        if len(dims) > 0:
            return torch.nansum(acc['sum'], dim=dims) / torch.nansum(acc['count'], dim=dims)
        else:
            return acc['sum'] / acc['count']

class SeasonalSpatialAggregator(Aggregator):
    """
    Aggregate 2D fields into user‑defined “seasons” (arbitrary sets of months).

    Parameters
    ----------
    season_months : Dict[str, Sequence[int]]
        Mapping from season name (e.g. 'FMA', 'DJF', 'Jan', 'All') to the
        list/tuple of month numbers (1–12) that belong to that season.
        E.g.:
            {
              'FMA': (2,3,4),
              'MJJ': (5,6,7),
              'ASO': (8,9,10),
              'NDJ': (11,12,1),
            }
    """
    DEFAULT_SEASONS = {
        'FMA': (2, 3, 4),
        'MJJ': (5, 6, 7),
        'ASO': (8, 9, 10),
        'NDJ': (11, 12, 1),
    }

    def __init__(self, season_months: Dict[str, Sequence[int]] = None):
        # Use default if not provided
        self.season_months = (
            {**self.DEFAULT_SEASONS}
            if season_months is None
            else {name: tuple(months) for name, months in season_months.items()}
        )

        # Validate months and build reverse lookup
        self._month_to_seasons: Dict[int, Tuple[str, ...]] = {}
        for name, months in self.season_months.items():
            for m in months:
                if not (1 <= m <= 12):
                    raise ValueError(f"Season '{name}' has invalid month {m}")
                self._month_to_seasons.setdefault(m, []).append(name)

    def init_accumulator(self, shape: Tuple[int, ...]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        :param shape: spatial shape of each incoming 2D field (H, W).
        Returns a dict:
            season_name -> {'sum': ndarray(H,W), 'count': ndarray(H,W)}
        """
        acc: Dict[str, Dict[str, np.ndarray]] = {}
        for season in self.season_months:
            acc[season] = {
                'sum':   np.zeros((shape), dtype=float),
                'count': np.zeros((shape), dtype=int),
            }
        return acc

    def accumulate(self,
                   acc: Dict[str, Dict[str, np.ndarray]],
                   field: np.ndarray,
                   date: Any) -> None:
        """
        Add this 2D field into all seasons that contain date.month.

        :param field: 2D array (H, W) of data or error.
        :param date: datetime.date (or anything with `.month`).
        """

        month = date.month
        seasons = self._month_to_seasons.get(month)
        if not seasons:
            raise ValueError(f"No season defined for month={month}")

        # Prepare sum/count update
        field_sum   = np.nan_to_num(field, nan=0.0)
        field_count = (~np.isnan(field)).astype(int)

        for season in seasons:
            acc_season = acc[season]
            acc_season['sum']   += field_sum
            acc_season['count'] += field_count
            
    @staticmethod
    def finalize(acc: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """
        Return a dict:
            season_name -> mean_map (2D ndarray)
        """
        result: Dict[str, np.ndarray] = {}
        for season, data in acc.items():
            s = data['sum']
            c = data['count']
            with np.errstate(divide='ignore', invalid='ignore'):
                m = s / c
            m[c == 0] = np.nan
            result[season] = m
        return result
    

