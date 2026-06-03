import os
import sys
import yaml
import ast

sys.path.insert(0, '../../')

import torch
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from lib.config.cfg import Config
from lib.models.loss import HeterogenousMSLoss
from lib.models.changeToERA5 import ClusterMapper
from lib.data.train_test_split import split_dates_dispatch
from lib.data.data_utils import variable_len_collate, Sampler
from lib.data.datasets import (
    WRFs2sDataset,
    ERAs2sDataset,
    ERAMonthlyDataset,
    ScatterDataset,
    StationsDataset,
    StackDataset,
    dataset_with_indices,
    ConcatDataset,
    ScatterNoneDataset,
    StationsNoneDataset,
    DictDataset,
)
from lib.data.scaler import StandardScaler
from lib.pipeline.test import test
from lib.pipeline.train import train
from lib.models.build_module import build_correction_model
from lib.data.logger import WRFLogger
from lib.helpers.interpolation import InvDistTree


def _normalize_region(region_name):
    return str(region_name).strip().lower()


def main(
    cfg,
    results=None,
    *,
    folder_name=None,
    stage_name=None,
    save_metadata=True,
    baselines_only=False,
    save_dir=None,
):
    if folder_name is None:
        if cfg.run_config.run_mode == 'test' and results is not None and baselines_only:
            folder_name = 'model_comparison'
        else:
            folder_name = cfg.model_type

    if results is None:
        results = {}
    
    logger = WRFLogger(cfg, cfg.data.logs_folder, folder_name, save_dir=save_dir)
    print(logger.model_save_dir, 'model save dir')

    config_used_path = os.path.join(logger.save_dir, "config_used.yaml")
    if cfg.run_config.run_mode != 'test' or save_dir is not None or not os.path.exists(config_used_path):
        print('Saving config to', config_used_path)
        cfg.save_config(config_used_path)

    cfg.device = torch.device(cfg.device)
    print(f"Running on {cfg.device} device")

    betas = [cfg.betas.beta_era, cfg.betas.beta_ms, cfg.betas.beta_st, cfg.betas.beta_sc]
    print(betas, 'betas')

    print('Splitting train val test...')
    max_sl = cfg.s2s.sequence_len

    ref_region = _normalize_region(cfg.reference_region)
    if ref_region == 'borey':
        data_folder = cfg.data.borey_folder
    elif ref_region == 'nestp':
        data_folder = cfg.data.nestp_folder
    else:
        raise ValueError(f"Unknown reference_region={cfg.reference_region!r}. Expected 'borey' or 'nestp'.")

    print(data_folder, 'wrf folder')

    wrf_dataset = WRFs2sDataset(
        data_folder,
        cfg.data.wrf_variables,
        seq_len=max_sl,
        add_coords=cfg.run_config.use_spatial_encoding,
        add_time_encoding=cfg.run_config.use_time_encoding,
    )

    era_dataset_uv = ERAMonthlyDataset(
        os.path.join(cfg.data.era_folder, 'w10'),
        ['u10', 'v10'],
        seq_len=max_sl,
        region_bbox=[55, 90, -180, 180],
    )
    era_dataset_t = ERAMonthlyDataset(
        os.path.join(cfg.data.era_folder, 't2'),
        ['t2m'],
        seq_len=max_sl,
        region_bbox=[55, 90, -180, 180],
    )
    era_dataset = ConcatDataset([era_dataset_uv, era_dataset_t])

    st_ds = StationsDataset(
        os.path.join(cfg.data.stations_folder, cfg.reference_region, 'clear'), 
        wind_variables=cfg.stations_ds.wind_variables,
        wind_format=cfg.stations_ds.wind_format,
        seq_len=max_sl,
    )
    sc_ds = ScatterDataset(cfg.data.scatter_folder, seq_len=max_sl)

    print(len(sc_ds), len(st_ds), len(era_dataset), len(wrf_dataset))
    pipeline_datasets = [wrf_dataset, era_dataset]
    if cfg.run_config.use_stations:
        pipeline_datasets.append(st_ds)
    if cfg.run_config.use_scatter:
        pipeline_datasets.append(sc_ds)
    dataset = dataset_with_indices(DictDataset)(*pipeline_datasets)

    train_days, val_days, test_days = split_dates_dispatch(**cfg.split_config)
    test_days = test_days[::max_sl]

    train_sampler = Sampler(train_days, shuffle=True)
    val_sampler = Sampler(val_days, shuffle=False)
    test_sampler = Sampler(test_days, shuffle=False)
    collate_fn = variable_len_collate if cfg.run_config.variable_sequence_length else variable_len_collate

    train_dataloader = DataLoader(
        dataset,
        batch_size=cfg.run_config.batch_size,
        num_workers=cfg.run_config.num_workers,
        sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        dataset,
        batch_size=cfg.run_config.batch_size,
        num_workers=cfg.run_config.num_workers,
        sampler=val_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    if cfg.test_config.use_stations and not cfg.run_config.use_stations:
        pipeline_datasets.append(st_ds)
        print(pipeline_datasets)
        dataset = dataset_with_indices(DictDataset)(*pipeline_datasets)
    print(pipeline_datasets)

    test_dataloader = DataLoader(
        dataset,
        batch_size=cfg.run_config.batch_size,
        num_workers=cfg.run_config.num_workers,
        sampler=test_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    means_dict = torch.load(cfg.data.wrf_mean_path, weights_only=False)
    stds_dict = torch.load(cfg.data.wrf_std_path, weights_only=False)

    time_keys = ['day', 'hour'] if cfg.run_config.use_time_encoding else []
    landmask_key = ['LANDMASK'] if cfg.run_config.use_landmask else []
    spatial_keys = ['XLAT', 'XLONG'] if cfg.run_config.use_spatial_encoding else []

    wrf_keys = ['u10', 'v10', 'T2'] + wrf_dataset.data_variables[2:] + landmask_key + spatial_keys + time_keys
    era_keys = ['u10', 'v10', 'T2']

    print(wrf_keys, '- wrf channels to transform')
    print(era_keys, '- era channels to transform')

    era_scaler = StandardScaler()
    wrf_scaler = StandardScaler()

    era_scaler.apply_scaler_channel_params(
        torch.tensor([means_dict[x] for x in era_keys]).float().to(cfg.device),
        torch.tensor([stds_dict[x] for x in era_keys]).float().to(cfg.device),
    )
    wrf_scaler.apply_scaler_channel_params(
        torch.tensor([means_dict[x] for x in wrf_keys]).float().to(cfg.device),
        torch.tensor([stds_dict[x] for x in wrf_keys]).float().to(cfg.device),
    )

    print(wrf_scaler.means, wrf_scaler.stddevs)

    wrf_grid, era_grid = wrf_dataset.src_grid, era_dataset.src_grid
    era_coords = np.stack([era_grid['longitude'].flatten(), era_grid['latitude'].flatten()]).T
    wrf_coords = np.stack([wrf_grid['longitude'].flatten(), wrf_grid['latitude'].flatten()]).T

    scat_grid = sc_ds.src_grid
    scat_coords = np.stack([scat_grid['longitude'].flatten(), scat_grid['latitude'].flatten()]).T

    meaner = ClusterMapper(
        mapping_file=None,
        target_coords=era_coords,
        input_coords=wrf_coords,
        weighted=cfg.run_config.weighted_meaner,
        save_mapping=True,
        save_name='meaner_mapping.npy',
        device=cfg.device,
        distance_metric='euclidean',
    ).to(cfg.device)

    stations_interpolator = InvDistTree(x=wrf_coords, q=st_ds.coords, device=cfg.device)
    scat_interpolator = InvDistTree(x=wrf_coords, q=scat_coords, device=cfg.device)

    criterion = HeterogenousMSLoss(
        meaner,
        betas,
        stations_interpolator,
        scat_interpolator,
        logger=logger,
        kernel_type=cfg.loss_config.loss_kernel,
        k=cfg.loss_config.k,
        device=cfg.device,
    ).to(cfg.device).float()

    model = build_correction_model(cfg, grid=wrf_grid)
    pretrained_weights = getattr(cfg, 'pretrained_weights', None)
    if pretrained_weights is not None:
        print('Loading pretrained weights from', pretrained_weights)
        state_dict = torch.load(pretrained_weights, map_location=cfg.device)
        model.load_state_dict(state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.2)

    best_epoch = cfg.test_config.best_epoch_id

    run_mode = str(cfg.run_config.run_mode).lower()
    trained_this_run = 'train' in run_mode

    if trained_this_run:
        print(f"Started training the model: run no {logger.experiment_number}")
        best_epoch, _ = train(
            train_dataloader,
            valid_dataloader,
            model,
            optimizer,
            wrf_scaler,
            era_scaler,
            criterion,
            scheduler,
            logger,
            cfg,
        )

    if trained_this_run:
        best_model_path = os.path.join(logger.model_save_dir, f'model_{best_epoch}.pth')
    else:
        best_model_path = pretrained_weights
        if best_model_path is None and not results:
            raise ValueError(
                "run_mode='test' requires cfg.pretrained_weights when no external "
                "results are provided."
            )

    if folder_name != 'model_comparison' and best_model_path is not None:
        results[cfg.model_type] = {
            'path': config_used_path,
            'model_path': best_model_path,
        }

    models = {}
    for model_name in results:
        model_cfg = Config.fromfile(results[model_name]['path'])
        model_cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"

        loaded_model = build_correction_model(model_cfg, grid=wrf_grid)
        state_dict = torch.load(results[model_name]['model_path'], map_location=model_cfg['device'])
        loaded_model.load_state_dict(state_dict)
        loaded_model.eval()
        models[model_name] = loaded_model

    if st_ds.wind_format == 'uv':
        st_ds.wind_format = 'wd'
    test_results = test(models, criterion, wrf_scaler, era_scaler, test_dataloader, logger, cfg)
    print(test_results)

    best_epoch_metadata = int(best_epoch) if best_epoch is not None else None
    run_metadata = {
        'stage_name': stage_name if stage_name is not None else folder_name,
        'model_type': cfg.model_type,
        'run_mode': cfg.run_config.run_mode,
        'reference_region': ref_region,
        'reference_dataset': cfg.reference_dataset,
        'target_dataset': cfg.target_dataset,
        'save_dir': logger.save_dir,
        'model_save_dir': logger.model_save_dir,
        'config_used_path': config_used_path,
        'best_epoch': best_epoch_metadata,
        'best_model_path': best_model_path,
        'evaluated_model_path': best_model_path,
        'pretrained_weights': pretrained_weights,
    }

    if save_metadata:
        metadata_path = os.path.join(logger.save_dir, "run_metadata.yaml")
        with open(metadata_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(run_metadata, f, sort_keys=False, allow_unicode=True)
        print('Saved run metadata to', metadata_path)

    return run_metadata

class ConfigOverride:
    """Helper class for config overrides"""
    
    @staticmethod
    def parse_value(value_str):
        """Parse string to appropriate Python type"""
        try:
            # Try to evaluate Python literal (supports int, float, bool, None, lists, dicts)
            return ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            # If it's not a Python literal, keep as string
            return value_str
    
    @staticmethod
    def set_nested_item(config, key_path, value):
        """Set nested dictionary item using dot notation"""
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # Convert non-dict to dict if needed
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = ConfigOverride.parse_value(value)
        return config

if __name__ == '__main__':
    import os.path as osp
    import argparse

    parser = argparse.ArgumentParser(description='Training and testing script')

    # Add arguments for main function kwargs
    parser.add_argument('--cfg', type=str, default=osp.join('/home/configs/', 'train_test.yaml'),
                        help='Path to configuration file. Default /home/configs/train_test.yaml')
    parser.add_argument('--results', type=str, default=None,
                        help='Path to results file. Default /home/configs/baseline_results.yaml')
    parser.add_argument('--folder_name', type=str, default=None,
                        help='Folder name for output')
    parser.add_argument('--stage_name', type=str, default=None,
                        help='Stage name for processing')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Explicit output directory. When set, no misc_N folder is created.')
    parser.add_argument('--save_metadata', action='store_true', default=True,
                        help='Whether to save metadata (use --no-save_metadata to disable)')
    parser.add_argument('--no-save_metadata', action='store_false', dest='save_metadata',
                        help='Disable saving metadata')
    parser.add_argument('--baselines_only', action='store_true', default=False,
                        help='When testing ignore models except baselines')
    # Add other useful arguments
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                        help='Device to use (cuda/cpu). If not specified, auto-detected')
    # Allow multiple overrides
    parser.add_argument('-o', '--option', type=str, nargs=2, action='append',
                        metavar=('KEY', 'VALUE'),
                        help='Override config option. Example: -o model_args.BERTunet.n_channels 16 -o loss_config.k 7')
    # Parse arguments
    args = parser.parse_args()
    if args.results:
        results = Config.fromfile(args.cfg)
        results = results.to_dict()
    else:
        results = Config.fromfile(osp.join('/home/configs/', 'baseline_results.yaml'))
        results = results.to_dict()

    if args.cfg:
        default_cfg = Config.fromfile(args.cfg)
    else:
        default_cfg = Config.fromfile(osp.join('/home/configs/', 'train_test.yaml'))

    if args.device:
        default_cfg['device'] = args.device
    else:
        default_cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    if args.option:
        for key_path, value in args.option:
            ConfigOverride.set_nested_item(default_cfg, key_path, value)

    main_kwargs = {
        'cfg': default_cfg,
        'results': results,
        'folder_name': args.folder_name,
        'stage_name': args.stage_name,
        'save_metadata': args.save_metadata,
        'baselines_only': args.baselines_only,
        'save_dir': args.save_dir,
    }
    main(**main_kwargs)
