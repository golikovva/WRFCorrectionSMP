import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _install_dataset_import_stubs():
    sys.modules.setdefault(
        "wrf",
        types.SimpleNamespace(ALL_TIMES=object(), getvar=None),
    )
    sys.modules.setdefault("netCDF4", types.SimpleNamespace(Dataset=None))
    sys.modules.setdefault("pygrib", types.SimpleNamespace())
    sys.modules.setdefault("xarray", types.SimpleNamespace(open_dataset=None))

    class AttrDict(dict):
        __getattr__ = dict.__getitem__

    sys.modules.setdefault("addict", types.SimpleNamespace(Dict=AttrDict))


_install_dataset_import_stubs()

from lib.data.datasets import NCs2sDataset


class ToyNCs2sDataset(NCs2sDataset):
    @staticmethod
    def _parse_date(file):
        return np.datetime64(file.stem)

    @property
    def _files_template(self):
        return "*.dummy"

    @property
    def _file_len(self):
        return "D"

    @property
    def name(self):
        return "Toy"

    def _create_grid(self):
        lon = np.array([[40.0, 41.0, 42.0], [40.5, 41.5, 42.5]], dtype=np.float32)
        lat = np.array([[60.0, 60.0, 60.0], [61.0, 61.0, 61.0]], dtype=np.float32)
        return {"longitude": lon, "latitude": lat}


def make_dataset(tmp_path, **kwargs):
    (tmp_path / "2020-01-02.dummy").write_text("placeholder")
    return ToyNCs2sDataset(tmp_path, **kwargs)


def test_empty_variables_return_zero_physical_channels(tmp_path):
    dataset = make_dataset(tmp_path, data_variables=[], seq_len=3)

    sample = dataset[np.datetime64("2020-01-02T06")]

    assert sample.shape == (3, 0, 2, 3)


def test_empty_variables_can_return_time_encoding_only(tmp_path):
    date = np.datetime64("2020-01-02T06")
    dataset = make_dataset(
        tmp_path,
        data_variables=[],
        seq_len=3,
        add_time_encoding=True,
    )

    sample = dataset[date]
    day_encoded, hour_encoded = dataset.get_day_hour_encoding(date, length=3)

    assert sample.shape == (3, 2, 2, 3)
    np.testing.assert_allclose(sample[:, 0], day_encoded)
    np.testing.assert_allclose(sample[:, 1], hour_encoded)


def test_empty_variables_can_return_coords_and_time_encoding(tmp_path):
    dataset = make_dataset(
        tmp_path,
        data_variables=[],
        seq_len=2,
        add_coords=True,
        add_time_encoding=True,
    )

    sample = dataset[np.datetime64("2020-01-02T06")]

    assert sample.shape == (2, 4, 2, 3)
    np.testing.assert_allclose(
        sample[:, 0],
        np.broadcast_to(dataset.src_grid["latitude"], sample[:, 0].shape),
    )
    np.testing.assert_allclose(
        sample[:, 1],
        np.broadcast_to(dataset.src_grid["longitude"], sample[:, 1].shape),
    )


def test_getitem_overrides_time_encoding_and_respects_length(tmp_path):
    date = np.datetime64("2020-01-02T06")
    dataset = make_dataset(
        tmp_path,
        data_variables=[],
        seq_len=4,
        add_time_encoding=False,
    )

    sample = dataset.__getitem__(date, length=2, add_time_encoding=True)

    assert sample.shape == (2, 2, 2, 3)
