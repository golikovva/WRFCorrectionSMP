import numpy as np
import pytest

from lib.data.train_test_split import (
    parse_timedelta64,
    sample_seasonally_balanced_nested_dates,
    split_dates_dispatch,
    split_given_dates_by_test_years,
)


def _complete_seasonal_year(step='1h'):
    return np.arange(
        np.datetime64('2019-03-01'),
        np.datetime64('2020-03-01'),
        np.timedelta64(int(step[:-1]), step[-1]),
    )


def test_split_given_dates_by_test_years_preserves_resolution_and_order():
    dates = np.array(
        ['2020-01-01T06', '2019-01-01T00', '2020-06-01T12'],
        dtype='datetime64[h]',
    )

    development, test = split_given_dates_by_test_years(dates, [2020])

    np.testing.assert_array_equal(
        development,
        np.array(['2019-01-01T00'], dtype='datetime64[h]'),
    )
    np.testing.assert_array_equal(
        test,
        np.array(['2020-01-01T06', '2020-06-01T12'], dtype='datetime64[h]'),
    )
    assert development.dtype == dates.dtype
    assert test.dtype == dates.dtype


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        ('h', np.timedelta64(1, 'h')),
        ('D', np.timedelta64(1, 'D')),
        ('d', np.timedelta64(1, 'D')),
        ('M', np.timedelta64(1, 'M')),
        ('7D', np.timedelta64(7, 'D')),
    ],
)
def test_parse_timedelta64_accepts_bare_numpy_units(value, expected):
    assert parse_timedelta64(value) == expected


def test_seasonal_samples_are_balanced_reproducible_and_nested():
    dates = _complete_seasonal_year()

    level_1 = sample_seasonally_balanced_nested_dates(
        dates,
        blocks_per_season=1,
        block_size='7D',
        seed=17,
    )
    level_2 = sample_seasonally_balanced_nested_dates(
        dates,
        blocks_per_season=2,
        block_size='7D',
        seed=17,
    )
    repeated = sample_seasonally_balanced_nested_dates(
        dates,
        blocks_per_season=1,
        block_size='7D',
        seed=17,
    )

    np.testing.assert_array_equal(level_1, repeated)
    assert np.all(level_1[:-1] <= level_1[1:])
    assert set(level_1).issubset(set(level_2))
    assert len(level_1) == 4 * 7 * 24
    assert len(level_2) == 2 * len(level_1)


def test_calendar_month_blocks_select_one_month_from_each_season():
    dates = _complete_seasonal_year()

    subset = sample_seasonally_balanced_nested_dates(
        dates,
        blocks_per_season=1,
        block_size='M',
        seed=3,
    )

    selected_months = np.unique(subset.astype('datetime64[M]'))
    month_numbers = selected_months.astype(np.int64) % 12 + 1
    assert len(selected_months) == 4
    assert sum(np.isin(month_numbers, [12, 1, 2])) == 1
    assert sum(np.isin(month_numbers, [3, 4, 5])) == 1
    assert sum(np.isin(month_numbers, [6, 7, 8])) == 1
    assert sum(np.isin(month_numbers, [9, 10, 11])) == 1


def test_seasonal_sampling_rejects_unavailable_nesting_level():
    dates = _complete_seasonal_year(step='1D')

    with pytest.raises(ValueError, match='not enough blocks'):
        sample_seasonally_balanced_nested_dates(
            dates,
            blocks_per_season=4,
            block_size='M',
            seed=0,
        )


def test_dispatch_seasonal_nested_returns_fixed_validation_and_test_year():
    dates = np.arange(
        np.datetime64('2019-01-01'),
        np.datetime64('2021-01-01'),
        np.timedelta64(1, 'h'),
    )
    split_kwargs = {
        'dates': dates,
        'split_mode': 'seasonal_nested',
        'test_years': [2020],
        'block_size': '7D',
        'seed': 17,
        'validation_blocks_per_season': 1,
        'validation_block_size': '14D',
        'validation_seed': 23,
    }

    train_level_1, validation, test = split_dates_dispatch(
        **split_kwargs,
        blocks_per_season=1,
    )
    train_level_2, validation_again, test_again = split_dates_dispatch(
        **split_kwargs,
        blocks_per_season=2,
    )

    assert len(train_level_1) == 4 * 7 * 24
    assert len(train_level_2) == 2 * len(train_level_1)
    assert len(validation) == 4 * 14 * 24
    assert len(test) == 366 * 24
    assert set(train_level_1).issubset(set(train_level_2))
    assert np.intersect1d(train_level_2, validation).size == 0
    assert np.intersect1d(train_level_2, test).size == 0
    np.testing.assert_array_equal(validation, validation_again)
    np.testing.assert_array_equal(test, test_again)


def test_dispatch_seasonal_nested_requires_validation_budget():
    dates = _complete_seasonal_year(step='1D')

    with pytest.raises(ValueError, match='validation_blocks_per_season'):
        split_dates_dispatch(
            dates=dates,
            split_mode='seasonal_nested',
            test_years=[2020],
            blocks_per_season=1,
        )
