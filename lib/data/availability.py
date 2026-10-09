"""Metadata-only admission of complete training sequences.

A validator lives for one preparation scan: each NetCDF file (including failures)
is inspected once, and only time-coordinate values are read.
"""
from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np


class AvailabilityConfigurationError(ValueError):
    """The dataset requires an unsupported or invalid metadata check."""


class UnavailableSequence(ValueError):
    """An otherwise supported sample is missing required data."""


@dataclass(frozen=True)
class SequenceSlice:
    file_date: np.datetime64
    indices: np.ndarray
    dates: np.ndarray


def plan_sequence_slices(date, length, time_resolution_h, file_period):
    """Plan hourly file offsets from actual timestamps, including calendar edges."""
    if int(length) != length or int(length) < 1:
        raise AvailabilityConfigurationError("sequence length must be a positive integer")
    if int(time_resolution_h) != time_resolution_h or int(time_resolution_h) < 1:
        raise AvailabilityConfigurationError("time resolution must be a positive integer number of hours")
    date = np.datetime64(date)
    if np.isnat(date) or date != date.astype("datetime64[h]"):
        raise AvailabilityConfigurationError(f"sequence start must be an exact hour: {date}")
    dates = date.astype("datetime64[h]") + np.arange(int(length)) * np.timedelta64(int(time_resolution_h), "h")
    try:
        keys = dates.astype(f"datetime64[{file_period}]")
    except (TypeError, ValueError) as exc:
        raise AvailabilityConfigurationError(f"unsupported file period: {file_period}") from exc
    boundaries = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1, len(keys)]
    return tuple(
        SequenceSlice(
            keys[start],
            (dates[start:end] - keys[start].astype("datetime64[h]")).astype("timedelta64[h]").astype(int),
            dates[start:end],
        )
        for start, end in zip(boundaries[:-1], boundaries[1:])
    )


@dataclass(frozen=True)
class VariableMetadata:
    dimensions: tuple
    shape: tuple


@dataclass
class FileMetadata:
    variables: dict
    attributes: dict
    time_coordinates: dict


def _decode_time_coordinate(variable):
    values = np.ma.asarray(variable[:])
    if np.any(np.ma.getmaskarray(values)):
        raise UnavailableSequence(f"masked values in time coordinate {variable.name}")
    values = np.asarray(values)
    if values.dtype.kind in "SUO":
        if values.ndim == 2:
            values = np.array([
                b"".join(row).decode() if values.dtype.kind == "S" else "".join(row)
                for row in values
            ])
        return np.array([
            np.datetime64((v.decode() if isinstance(v, bytes) else str(v)).replace("_", "T"), "s")
            for v in values.ravel()
        ])
    units = getattr(variable, "units", "")
    if "since" not in units:
        raise UnavailableSequence(f"time coordinate {variable.name} has no absolute time units")
    calendar = getattr(variable, "calendar", "standard")
    if calendar not in ("standard", "gregorian", "proleptic_gregorian"):
        raise AvailabilityConfigurationError(f"unsupported time calendar: {calendar}")
    decoded = netCDF4.num2date(values, units, calendar=calendar)
    return np.array([np.datetime64(v.isoformat(), "s") for v in np.asarray(decoded).ravel()])


class MetadataValidator:
    """Validate supported composed datasets without loading weather fields."""

    def __init__(self):
        self.files = {}
        self.station_metadata = {}

    def metadata(self, filename):
        key = str(Path(filename).resolve())
        if key not in self.files:
            try:
                with netCDF4.Dataset(key, "r") as nc:
                    variables = {
                        name: VariableMetadata(tuple(var.dimensions), tuple(var.shape))
                        for name, var in nc.variables.items()
                    }
                    coordinates = {}
                    for name, var in nc.variables.items():
                        # Never read a physical field here: only 1-D CF time axes
                        # or WRF's 2-D character Times array.
                        is_time = (
                            (var.ndim == 1 and (
                                "since" in str(getattr(var, "units", ""))
                                or name.lower() in ("time", "valid_time", "xtime")
                                or getattr(var, "standard_name", "") == "time"
                            ))
                            or (name == "Times" and var.ndim == 2)
                        )
                        if is_time:
                            try:
                                decoded = _decode_time_coordinate(var)
                            except AvailabilityConfigurationError:
                                raise
                            except (UnavailableSequence, ValueError, IndexError, OverflowError):
                                # Some WRF files provide a valid Times axis and
                                # an ancillary relative XTIME without CF units.
                                # Admission still requires a usable coordinate
                                # on every requested field's temporal dimension.
                                continue
                            coordinates[name] = (var.dimensions[0], decoded)
                    self.files[key] = FileMetadata(
                        variables,
                        {name: nc.getncattr(name) for name in nc.ncattrs()},
                        coordinates,
                    )
            except AvailabilityConfigurationError:
                raise
            except (OSError, RuntimeError, ValueError, IndexError, OverflowError) as exc:
                self.files[key] = UnavailableSequence(f"{key}: {exc}")
        result = self.files[key]
        if isinstance(result, Exception):
            raise result
        return result

    @staticmethod
    def _variable(metadata, name, selection, expected_shape=None, stagger=None):
        if name not in metadata.variables:
            raise UnavailableSequence(f"missing variable {name}")
        var = metadata.variables[name]
        required_ndim = 4 if stagger is not None else 3
        if len(var.shape) != required_ndim or not all(var.shape):
            raise UnavailableSequence(f"{name}: invalid dimensions {var.dimensions} {var.shape}")
        if selection.indices[-1] >= var.shape[0]:
            raise UnavailableSequence(f"{name}: time axis is too short ({var.shape[0]})")
        coordinates = [
            (coord_name, times) for coord_name, (dimension, times) in metadata.time_coordinates.items()
            if dimension == var.dimensions[0]
        ]
        if not coordinates:
            raise UnavailableSequence(f"{name}: missing temporal coordinate for {var.dimensions[0]}")
        for coord_name, times in coordinates:
            if len(times) != var.shape[0] or not np.array_equal(times[selection.indices], selection.dates):
                raise UnavailableSequence(f"{name}: time coordinate {coord_name} does not cover requested timestamps")
        shape = list(var.shape[-2:])
        if stagger is not None:
            shape[stagger] -= 1
        if expected_shape is not None and tuple(shape) != tuple(expected_shape):
            raise UnavailableSequence(f"{name}: inconsistent spatial dimensions {tuple(shape)}, expected {expected_shape}")
        return tuple(shape)

    def _uvmet10(self, metadata, selection):
        """Dependencies of wrf-python g_uvmet._get_uvmet(..., ten_m=True)."""
        variables, attrs = metadata.variables, metadata.attributes
        u = next((name for name in ("U10", "UU") if name in variables), None)
        v = next((name for name in ("V10", "VV") if name in variables), None)
        if u is None or v is None:
            raise UnavailableSequence("uvmet10 requires U10 or UU and V10 or VV")
        shape = self._variable(metadata, u, selection, stagger=None if u == "U10" else 1)
        self._variable(metadata, v, selection, shape, stagger=None if v == "V10" else 0)
        if "MAP_PROJ" not in attrs:
            raise UnavailableSequence("uvmet10 requires MAP_PROJ")
        projection = attrs["MAP_PROJ"]
        if projection in (1, 2):
            for name in ("TRUELAT1", "TRUELAT2"):
                if name not in attrs or not np.isfinite(attrs[name]):
                    raise UnavailableSequence(f"uvmet10 requires finite {name}")
            longitude = next((name for name in ("STAND_LON", "CEN_LON") if name in attrs), None)
            if longitude is None or not np.isfinite(attrs[longitude]):
                raise UnavailableSequence("uvmet10 requires finite STAND_LON or CEN_LON")
            for alternatives in (("XLAT_M", "XLAT"), ("XLONG_M", "XLONG")):
                selected = next((name for name in alternatives if name in variables), None)
                if selected is None:
                    raise UnavailableSequence(f"uvmet10 requires {' or '.join(alternatives)}")
                self._variable(metadata, selected, selection, shape)
        elif projection not in (0, 3, 6):
            raise AvailabilityConfigurationError(f"uvmet10: unsupported MAP_PROJ={projection}")
        return shape

    def _netcdf(self, dataset, date, length=None):
        selections = plan_sequence_slices(
            date, dataset.seq_len if length is None else length,
            dataset.time_res_h, dataset._file_len,
        )
        expected_grid_shape = tuple(dataset.src_grid["latitude"].shape)
        for selection in selections:
            files = dataset.dates_dict.get(selection.file_date)
            if not files:
                raise UnavailableSequence(f"no file for {selection.file_date}")
            filename = files[0]
            metadata = self.metadata(filename)
            try:
                shape = None
                for name in dataset.data_variables:
                    if name == "uvmet10" and name not in metadata.variables:
                        current = self._uvmet10(metadata, selection)
                        if shape is not None and current != shape:
                            raise UnavailableSequence("uvmet10: inconsistent spatial dimensions")
                        shape = current
                    else:
                        if name not in metadata.variables:
                            # Unknown physical variable names mean missing data.
                            # Known derived diagnostics require explicit validators.
                            from wrf.routines import _FUNC_MAP
                            if name in _FUNC_MAP:
                                raise AvailabilityConfigurationError(
                                    f"metadata validation is not implemented for derived diagnostic {name}"
                                )
                        current = self._variable(metadata, name, selection, shape)
                        shape = current
                if not dataset.data_variables:
                    # Encoding-only datasets still follow the source's time axis.
                    if not any(
                        selection.indices[-1] < len(times)
                        and np.array_equal(times[selection.indices], selection.dates)
                        for _, times in metadata.time_coordinates.values()
                    ):
                        raise UnavailableSequence("missing timestamps for encoding-only source")
                if shape is not None:
                    cropped = (
                        len(range(*dataset.lat_slice.indices(shape[0]))),
                        len(range(*dataset.lon_slice.indices(shape[1]))),
                    )
                    if cropped != expected_grid_shape:
                        raise UnavailableSequence(f"spatial dimensions {cropped} differ from dataset grid {expected_grid_shape}")
                for axis, alternatives in enumerate((("latitude", "XLAT", "XLAT_M"), ("longitude", "XLONG", "XLONG_M"))):
                    coord = next((metadata.variables[name] for name in alternatives if name in metadata.variables), None)
                    if coord is None or not coord.shape or not all(coord.shape):
                        raise UnavailableSequence(f"missing spatial coordinate {' or '.join(alternatives)}")
                    if shape is not None:
                        expected = (shape[axis],) if len(coord.shape) == 1 else shape
                        actual = coord.shape if len(coord.shape) == 1 else coord.shape[-2:]
                        if tuple(actual) != tuple(expected):
                            raise UnavailableSequence(f"inconsistent spatial coordinate {' or '.join(alternatives)}")
            except UnavailableSequence as exc:
                raise UnavailableSequence(f"{filename}: {exc}") from exc

    def check(self, dataset, date, length=None):
        # Deferred import keeps the temporal planner usable by datasets.py.
        from lib.data.datasets import (
            ConcatDataset, DictDataset, StackDataset, StackVSDataset,
            NCs2sDataset, GFSgribDataset, IFSs2sDataset,
            StationsDataset, ScatterDataset, ScatterNoneDataset, StationsNoneDataset,
        )
        if isinstance(dataset, (ScatterDataset, ScatterNoneDataset, StationsNoneDataset)):
            # No scatter observations in this window is a supported empty sample.
            return
        if isinstance(dataset, (DictDataset, ConcatDataset, StackDataset)):
            children = dataset.datasets
            children = children.items() if isinstance(children, dict) else enumerate(children)
            if isinstance(dataset, StackVSDataset):
                length = dataset.max_sl
            for name, child in children:
                try:
                    self.check(child, date, length)
                except UnavailableSequence as exc:
                    raise UnavailableSequence(f"{name}: {exc}") from exc
            return
        if isinstance(dataset, StationsDataset):
            size = dataset.seq_len if length is None else length
            expected = np.datetime64(date, "h") + np.arange(size) * np.timedelta64(1, "h")
            key = id(dataset)
            if key not in self.station_metadata:
                index = dataset.dates_dict.index
                if not index.is_monotonic_increasing:
                    raise AvailabilityConfigurationError("Stations timestamps must be ordered")
                self.station_metadata[key] = (
                    index.to_numpy(dtype="datetime64[ns]"),
                    np.asarray(dataset.dates_dict, dtype=int),
                )
            actual, all_positions = self.station_metadata[key]
            left = np.searchsorted(actual, expected[0], side="left")
            right = np.searchsorted(actual, expected[-1], side="right")
            if not np.array_equal(actual[left:right], expected):
                raise UnavailableSequence("Stations: incomplete hourly timestamp sequence")
            positions = all_positions[left:right]
            if np.any(positions < 0) or np.any(positions >= len(dataset.stations)):
                raise UnavailableSequence("Stations: timestamp positions exceed available observations")
            return
        if isinstance(dataset, (GFSgribDataset, IFSs2sDataset)):
            raise AvailabilityConfigurationError(f"metadata validation is not implemented for {type(dataset).__name__}")
        if isinstance(dataset, NCs2sDataset):
            self._netcdf(dataset, date, length)
            return
        raise AvailabilityConfigurationError(f"metadata validation is not implemented for {type(dataset).__name__}")


def filter_valid_dates(dataset, dates):
    """Return complete dates in input order and JSON-ready rejection reasons."""
    dates = np.asarray(dates)
    if dates.ndim != 1 or dates.dtype.kind != "M":
        raise AvailabilityConfigurationError("dates must be a one-dimensional datetime64 array")
    validator = MetadataValidator()
    keep, rejected = [], []
    for date in dates:
        try:
            validator.check(dataset, date)
        except UnavailableSequence as exc:
            keep.append(False)
            rejected.append({"date": str(date), "reason": str(exc)})
        else:
            keep.append(True)
    return dates[np.asarray(keep, dtype=bool)], rejected
