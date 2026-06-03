import math
import re
import numpy as np


def _to_datetime64_array(dates):
    """
    Convert input dates to a numpy datetime64 array.
    Keeps datetime64 dtype if already present.
    """
    dates = np.asarray(dates)

    if dates.size == 0:
        return dates.astype('datetime64[ns]')

    if np.issubdtype(dates.dtype, np.datetime64):
        return dates

    return dates.astype('datetime64[ns]')


def _normalize_years(years):
    """
    Convert years input to a 1D int numpy array.
    Accepts:
      - None
      - []
      - [2020, 2021]
      - datetime64 years
    """
    if years is None:
        return np.array([], dtype=int)

    years = np.asarray(years)

    if years.size == 0:
        return np.array([], dtype=int)

    if np.issubdtype(years.dtype, np.datetime64):
        return years.astype('datetime64[Y]').astype(int) + 1970

    return years.astype(int)


def split_given_dates_by_percents(dates, train_size, validation_size, test_size=None):
    """
    Split an already prepared dates array by fractions of sample count.

    Parameters
    ----------
    dates : array-like of datetime-like
        Existing sorted date list / array.
    train_size : float
    validation_size : float
    test_size : float or None
        Optional. If given, can be checked against sum == 1.

    Returns
    -------
    train_dates, val_dates, test_dates : np.ndarray
    """
    dates = _to_datetime64_array(dates)

    total = train_size + validation_size + (0.0 if test_size is None else test_size)
    if test_size is None:
        if total > 1.0 + 1e-12:
            raise ValueError(
                f"train_size + validation_size must be <= 1, got {train_size} + {validation_size} = {total}"
            )
    else:
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"train_size + validation_size + test_size must be 1, got {total}"
            )

    train_end = math.ceil(len(dates) * train_size)
    val_end = math.ceil(len(dates) * (train_size + validation_size))

    train = dates[:train_end]
    val = dates[train_end:val_end]
    test = dates[val_end:]

    return train, val, test


def split_given_dates_by_dates(dates, train_end, validation_end):
    """
    Split an existing dates array by explicit boundary dates.

    Semantics:
      train: date < train_end
      val:   train_end <= date < validation_end
      test:  date >= validation_end

    Parameters
    ----------
    dates : array-like of datetime-like
    train_end : datetime-like
        First timestamp of validation subset.
    validation_end : datetime-like
        First timestamp of test subset.

    Returns
    -------
    train_dates, val_dates, test_dates : np.ndarray
    """
    dates = _to_datetime64_array(dates)
    train_end = np.datetime64(train_end)
    validation_end = np.datetime64(validation_end)

    if train_end > validation_end:
        raise ValueError(
            f"train_end must be <= validation_end, got {train_end} > {validation_end}"
        )

    train_mask = dates < train_end
    val_mask = (dates >= train_end) & (dates < validation_end)
    test_mask = dates >= validation_end

    train = dates[train_mask]
    val = dates[val_mask]
    test = dates[test_mask]

    return train, val, test


def split_given_dates_by_years(dates, valid_years=None, test_years=None):
    """
    Split an existing dates array by calendar years.

    Parameters
    ----------
    dates : array-like of datetime-like
        Existing sorted date list / array.
    valid_years : list[int] or []
        Years assigned to validation.
    test_years : list[int] or []
        Years assigned to test.
    Remaining years go to train.

    Returns
    -------
    train_dates, val_dates, test_dates : np.ndarray
    """
    dates = _to_datetime64_array(dates)
    valid_years = _normalize_years(valid_years)
    test_years = _normalize_years(test_years)

    overlap = np.intersect1d(valid_years, test_years)
    if overlap.size > 0:
        raise ValueError(
            f"valid_years and test_years overlap: {overlap.tolist()}"
        )

    years = dates.astype('datetime64[Y]').astype(int) + 1970
    available_years = np.unique(years)

    missing_valid = np.setdiff1d(valid_years, available_years)
    missing_test = np.setdiff1d(test_years, available_years)

    if missing_valid.size > 0:
        raise ValueError(
            f"These validation years are not present in dates: {missing_valid.tolist()}"
        )
    if missing_test.size > 0:
        raise ValueError(
            f"These test years are not present in dates: {missing_test.tolist()}"
        )

    val_mask = np.isin(years, valid_years)
    test_mask = np.isin(years, test_years)
    train_mask = ~(val_mask | test_mask)

    train = dates[train_mask]
    val = dates[val_mask]
    test = dates[test_mask]

    return train, val, test


def parse_timedelta64(value, default_unit='D'):
    """
    Convert a duration to ``np.timedelta64``.

    Accepts numpy timedeltas, integers with a default unit, or strings like
    ``'7D'``, ``'90D'``, ``'12h'``.
    """
    if isinstance(value, np.timedelta64):
        return value

    if isinstance(value, (int, np.integer)):
        return np.timedelta64(int(value), default_unit)

    if isinstance(value, str):
        match = re.fullmatch(r'\s*(\d+)\s*([A-Za-z]+)\s*', value)
        if match is None:
            raise ValueError(f"Cannot parse duration {value!r}. Expected strings like '7D' or '12h'.")
        amount, unit = match.groups()
        return np.timedelta64(int(amount), unit)

    raise TypeError(f"Unsupported duration type: {type(value)}")


def is_all_train_size(value):
    if value is None:
        return True
    return isinstance(value, str) and value.lower().strip() in {'all', 'rest', 'remaining'}


def _infer_date_step(dates):
    unique_dates = np.unique(_to_datetime64_array(dates))
    if unique_dates.size < 2:
        return np.timedelta64(1, 'D')

    diffs = np.diff(unique_dates)
    diffs = diffs[diffs > np.timedelta64(0, 'ns')]
    if diffs.size == 0:
        return np.timedelta64(1, 'D')
    return diffs.min()


def _positive_timedelta(value):
    return max(value, np.timedelta64(0, 'ns'))


def _half_timedelta(value):
    return value // 2


def blocked_cv_splits(
    dates,
    train_size,
    test_size,
    *,
    step_size=None,
    gap='0D',
    train_position='before',
    test_starts=None,
    min_train_samples=1,
    min_test_samples=1,
):
    """
    Build rolling blocked CV splits for time series.

    With ``train_position='before'`` the split is:
      train: [test_start - gap - train_size, test_start - gap)
      test:  [test_start, test_start + test_size)

    With ``train_position='around'`` train_size is the total train duration:
      train: preferably half before and half after test, with missing edge
             duration filled from the opposite side.
      test:  [test_start, test_start + test_size)

    Parameters
    ----------
    dates : array-like of datetime-like
        Available sample start dates.
    train_size : duration-like or {'all', 'rest', 'remaining'} or None
        ``np.timedelta64``, integer days, strings like ``'30D'``, or an all/rest
        marker meaning all dates outside test and gap zones are used for train.
    test_size : duration-like
        ``np.timedelta64``, integer days, or strings like ``'30D'``.
    step_size : duration-like or None
        Distance between neighboring test anchors. Defaults to ``test_size``.
    gap : duration-like
        Temporal gap between train and test blocks.
    train_position : {'before', 'around'}
        Where to place train dates relative to the test block.
    test_starts : array-like of datetime-like or None
        Optional fixed test anchors. Use this to reuse the same test blocks
        across several train sizes.

    Returns
    -------
    list[dict]
        Each dict contains train/test date arrays and block boundaries.
    """
    dates = np.sort(_to_datetime64_array(dates))
    if dates.size == 0:
        return []

    use_all_train = is_all_train_size(train_size)
    train_delta = None if use_all_train else parse_timedelta64(train_size)
    test_delta = parse_timedelta64(test_size)
    step_delta = test_delta if step_size is None else parse_timedelta64(step_size)
    gap_delta = parse_timedelta64(gap)
    train_position = str(train_position).lower().strip()

    if train_delta is not None and train_delta <= np.timedelta64(0, 'ns'):
        raise ValueError("train_size must be positive.")
    if test_delta <= np.timedelta64(0, 'ns'):
        raise ValueError("test_size must be positive.")
    if step_delta <= np.timedelta64(0, 'ns'):
        raise ValueError("step_size must be positive.")
    if gap_delta < np.timedelta64(0, 'ns'):
        raise ValueError("gap must be non-negative.")
    if train_position not in {'before', 'around'}:
        raise ValueError("train_position must be either 'before' or 'around'.")

    date_start = dates[0]
    date_stop = dates[-1] + _infer_date_step(dates)

    if test_starts is None:
        if use_all_train or train_position == 'around':
            first_test_start = date_start
        elif train_position == 'before':
            first_test_start = date_start + train_delta + gap_delta
        last_test_start = date_stop - test_delta
        if first_test_start > last_test_start:
            return []
        test_starts = np.arange(first_test_start, last_test_start + step_delta, step_delta)
        test_starts = test_starts[test_starts <= last_test_start]
    else:
        test_starts = _to_datetime64_array(test_starts)

    splits = []
    for fold_id, test_start in enumerate(test_starts):
        test_start = np.datetime64(test_start)
        test_end = test_start + test_delta

        if use_all_train:
            train_left_start = date_start
            train_left_end = test_start - gap_delta
            train_right_start = test_end + gap_delta
            train_right_end = date_stop
            train_start = date_start
            train_end = date_stop
            train_mask = (
                (dates < train_left_end) |
                (dates >= train_right_start)
            )
        elif train_position == 'before':
            train_end = test_start - gap_delta
            train_start = train_end - train_delta
            train_mask = (dates >= train_start) & (dates < train_end)
            train_left_start = train_start
            train_left_end = train_end
            train_right_start = test_end + gap_delta
            train_right_end = train_right_start
        else:
            left_end = test_start - gap_delta
            right_start = test_end + gap_delta

            left_available = _positive_timedelta(left_end - date_start)
            right_available = _positive_timedelta(date_stop - right_start)

            preferred_left = _half_timedelta(train_delta)
            left_delta = min(preferred_left, left_available)
            right_delta = min(train_delta - left_delta, right_available)
            left_delta = min(train_delta - right_delta, left_available)

            if left_delta + right_delta < train_delta:
                continue

            train_left_start = left_end - left_delta
            train_left_end = left_end
            train_right_start = right_start
            train_right_end = right_start + right_delta
            if left_delta > np.timedelta64(0, 'ns') and right_delta > np.timedelta64(0, 'ns'):
                train_start = train_left_start
                train_end = train_right_end
            elif left_delta > np.timedelta64(0, 'ns'):
                train_start = train_left_start
                train_end = train_left_end
            else:
                train_start = train_right_start
                train_end = train_right_end
            train_mask = (
                ((dates >= train_left_start) & (dates < train_left_end)) |
                ((dates >= train_right_start) & (dates < train_right_end))
            )

        test_mask = (dates >= test_start) & (dates < test_end)

        train_dates = dates[train_mask]
        test_dates = dates[test_mask]
        if len(train_dates) < min_train_samples or len(test_dates) < min_test_samples:
            continue

        splits.append({
            'fold': fold_id,
            'train_dates': train_dates,
            'test_dates': test_dates,
            'train_start': train_start,
            'train_end': train_end,
            'train_left_start': train_left_start,
            'train_left_end': train_left_end,
            'train_right_start': train_right_start,
            'train_right_end': train_right_end,
            'test_start': test_start,
            'test_end': test_end,
            'train_position': 'all' if use_all_train else train_position,
        })

    return splits


def split_dates_dispatch(
    dates=None,
    split_mode='percents',
    *,
    start_date=None,
    end_date=None,
    time_step='1h',
    train_size=None,
    validation_size=None,
    test_size=None,
    train_end=None,
    validation_end=None,
    valid_years=None,
    test_years=None,
):
    """
    Universal dispatcher for splitting an existing dates array.

    Parameters
    ----------
    dates : array-like of datetime-like
    split_mode : str
        One of:
          - 'percents'
          - 'dates'
          - 'years'

    Returns
    -------
    train_dates, val_dates, test_dates : np.ndarray
    """
    if dates is None:
        if start_date is None or end_date is None:
            raise ValueError(
                "If dates is not provided, start_date and end_date must be given to generate the dates array."
            )
        dates = arange_dates(start_date, end_date, time_step=time_step)
    mode = split_mode.lower().strip()

    if mode in {"percents", "percent", "fractions", "fraction"}:
        if train_size is None or validation_size is None:
            raise ValueError(
                "For split_mode='percents', train_size and validation_size must be provided."
            )
        return split_given_dates_by_percents(
            dates,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
        )

    if mode in {"dates", "by_dates", "boundaries"}:
        if train_end is None or validation_end is None:
            raise ValueError(
                "For split_mode='dates', train_end and validation_end must be provided."
            )
        return split_given_dates_by_dates(
            dates,
            train_end=train_end,
            validation_end=validation_end,
        )

    if mode in {"years", "by_years"}:
        return split_given_dates_by_years(
            dates,
            valid_years=valid_years,
            test_years=test_years,
        )

    raise ValueError(
        f"Unknown split_mode='{split_mode}'. Supported modes: 'percents', 'dates', 'years'."
    )

def arange_dates(start_date, end_date, time_step='1h'):
    """
    Generate an array of dates from start_date to end_date with a given time step.

    Parameters
    ----------
    start_date : datetime-like
    end_date : datetime-like
    time_step : str
        Time step string compatible with numpy.timedelta64, e.g. '1D', '6h', '30m'.

    Returns
    -------
    np.ndarray of dtype datetime64
    """
    start = np.datetime64(start_date)
    end = np.datetime64(end_date)
    step = np.timedelta64(int(time_step[:-1]), time_step[-1])
    return np.arange(start, end, step)


# todo deprecated
def split_dates(start_date, end_date, train_size, validation_size, test_size=None, time_step='h'):
    print('Function deprecated')
    days = np.arange(start_date, end_date, np.timedelta64(1, time_step), dtype=f'datetime64[{time_step}]')  # .astype(datetime)
    train_end = math.ceil(len(days) * train_size)
    val_end = math.ceil(len(days) * (train_size + validation_size))
    train = days[:train_end]
    val = days[train_end:val_end]
    test = days[val_end:]
    return train, val, test
