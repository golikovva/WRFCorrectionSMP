import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm


class IdentityScaler:
    def transform(self, tensor, *args, **kwargs):
        return tensor

    def inverse_transform(self, tensor, *args, **kwargs):
        return tensor

    def to(self, *args, **kwargs):
        return self


def normalize_region(region_name):
    return str(region_name).strip().lower()


def resolve_wrf_folder(cfg):
    ref_region = normalize_region(cfg.reference_region)
    if ref_region == "borey":
        return cfg.data.borey_folder
    if ref_region == "nestp":
        return cfg.data.nestp_folder
    raise ValueError(f"Unknown reference_region={cfg.reference_region!r}. Expected 'borey' or 'nestp'.")


def build_wrf_era_datasets(cfg):
    from lib.data.datasets import ConcatDataset, ERAMonthlyDataset, WRFs2sDataset

    seq_len = cfg.s2s.sequence_len
    wrf_dataset = WRFs2sDataset(
        resolve_wrf_folder(cfg),
        cfg.data.wrf_variables,
        seq_len=seq_len,
        add_coords=cfg.run_config.use_spatial_encoding,
        add_time_encoding=cfg.run_config.use_time_encoding,
    )

    era_dataset_uv = ERAMonthlyDataset(
        os.path.join(cfg.data.era_folder, getattr(cfg.data, "era_wind_subdir", "w10")),
        ["u10", "v10"],
        seq_len=seq_len,
        region_bbox=[55, 90, -180, 180],
    )
    era_dataset_t = ERAMonthlyDataset(
        os.path.join(cfg.data.era_folder, "t2"),
        ["t2m"],
        seq_len=seq_len,
        region_bbox=[55, 90, -180, 180],
    )
    era_dataset = ConcatDataset([era_dataset_uv, era_dataset_t])
    return wrf_dataset, era_dataset


def build_test_extra_datasets(cfg):
    from lib.data.datasets import ScatterDataset, StationsDataset

    datasets = []
    if cfg.test_config.use_stations or cfg.run_config.use_stations:
        datasets.append(
            StationsDataset(
                os.path.join(cfg.data.stations_folder, cfg.reference_region, "clear"),
                wind_variables=cfg.stations_ds.wind_variables,
                wind_format=cfg.stations_ds.wind_format,
                seq_len=cfg.s2s.sequence_len,
            )
        )
    if cfg.run_config.use_scatter:
        datasets.append(ScatterDataset(cfg.data.scatter_folder, seq_len=cfg.s2s.sequence_len))
    return datasets


def build_dataset(*datasets):
    from lib.data.datasets import DictDataset, dataset_with_indices

    return dataset_with_indices(DictDataset)(*datasets)


def build_loader(dataset, dates, cfg, *, shuffle=False, batch_size=None):
    from torch.utils.data import DataLoader
    from lib.data.data_utils import Sampler, variable_len_collate

    return DataLoader(
        dataset,
        batch_size=batch_size or cfg.run_config.batch_size,
        num_workers=cfg.run_config.num_workers,
        sampler=Sampler(np.asarray(dates), shuffle=shuffle),
        collate_fn=variable_len_collate,
        pin_memory=True,
    )


def non_overlapping_sequence_starts(dates, seq_len):
    """
    Pick sequence starts so each timestamp belongs to at most one sequence.

    The selection is done independently for each contiguous date segment, which
    keeps train/test/gap boundaries intact for blocked CV.
    """
    dates = np.sort(np.asarray(dates))
    if seq_len <= 1 or len(dates) == 0:
        return dates

    unique_dates = np.unique(dates)
    if len(unique_dates) < 2:
        return dates[:0]

    diffs = np.diff(unique_dates)
    positive_diffs = diffs[diffs > np.timedelta64(0, "ns")]
    if len(positive_diffs) == 0:
        return dates[:0]

    date_step = positive_diffs.min()
    split_points = np.where(np.diff(dates) != date_step)[0] + 1
    segments = np.split(dates, split_points)

    starts = []
    for segment in segments:
        if len(segment) < seq_len:
            continue
        starts.append(segment[: len(segment) - seq_len + 1 : seq_len])

    if not starts:
        return dates[:0]
    return np.concatenate(starts)


def grid_coords(dataset):
    grid = dataset.grid
    return np.stack([grid["longitude"].flatten(), grid["latitude"].flatten()]).T


def build_era_to_wrf_interpolator(wrf_dataset, era_dataset, device):
    from lib.helpers.interpolation import InvDistTree

    return InvDistTree(
        x=grid_coords(era_dataset),
        q=grid_coords(wrf_dataset),
        n_near=4,
        device=device,
    )


def build_test_losses(cfg, wrf_dataset, era_dataset):
    from lib.models.changeToERA5 import ClusterMapper

    meaner = ClusterMapper(
        mapping_file=None,
        target_coords=grid_coords(era_dataset),
        input_coords=grid_coords(wrf_dataset),
        weighted=cfg.run_config.weighted_meaner,
        save_mapping=False,
        device=cfg.device,
        distance_metric="euclidean",
    ).to(cfg.device)
    return SimpleNamespace(meaner=meaner)


def sequence_dates(dates, seq_len, time_step_hours):
    dates = np.asarray(dates)
    offsets = np.arange(seq_len) * np.timedelta64(int(time_step_hours), "h")
    return dates[:, None] + offsets[None, :]


def fit_ano_model(model, dataloader, era_to_wrf, cfg, *, time_step_hours=1):
    model.reset()
    model.train(False)

    for data, dates in tqdm(dataloader, desc="Fitting ANO"):
        if data is None:
            continue

        reference = data[cfg.reference_dataset]
        target = data[cfg.target_dataset]
        if reference is None or target is None:
            continue

        reference = reference.type(torch.float).to(cfg.device)[..., : model.n_channels, :, :]
        target = target.type(torch.float).to(cfg.device)

        target_on_wrf = era_to_wrf(target.flatten(-2, -1)).unflatten(-1, reference.shape[-2:])
        correction = target_on_wrf - reference
        batch_dates = sequence_dates(dates, correction.shape[1], time_step_hours)
        model.accumulate(correction, batch_dates)

    model.eval()
    return model


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
