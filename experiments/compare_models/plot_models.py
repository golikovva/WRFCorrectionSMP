import os
import sys
sys.path.insert(0, '../../')
import torch
import numpy as np
from torch.optim import lr_scheduler
from lib.models.loss import HeterogenousMSLoss
from lib.models.changeToERA5 import ClusterMapper
from lib.data.train_test_split import split_dates
from lib.data.data_utils import variable_len_collate, Sampler
from lib.data.datasets import WRFs2sDataset, ERAs2sDataset, ERAMonthlyDataset, ScatterDataset, StationsDataset, StackDataset, DictDataset, \
    dataset_with_indices, StationsDataset, ScatterDataset, ConcatDataset, GFSncDataset, ScatterNoneDataset, StationsNoneDataset
from lib.data.scaler import StandardScaler
from torch.utils.data import DataLoader
from lib.pipeline.test import test
from lib.pipeline.train import train
from lib.models.build_module import build_correction_model
from lib.data.logger import WRFLogger
from lib.helpers.interpolation import InvDistTree
# from torch_harmonics.examples.models.s2transformer import SphericalTransformer



def main(cfg, results):

    folder_name = 'model_comparison'
    logger = WRFLogger(cfg, cfg.data.logs_folder, folder_name)
    print('Im updated main.py')
    print(logger.model_save_dir, 'model save dir')
    if not cfg.run_config.run_mode == 'test':
        print('Saving config')
        cfg.save_config(os.path.join(logger.save_dir, "config_used.yaml"))

    cfg.device = torch.device(cfg.device)
    print(f"Running on {cfg.device} device")
    betas = [cfg.betas.beta_era, cfg.betas.beta_ms, cfg.betas.beta_st, cfg.betas.beta_sc]
    print(betas, 'betas')

    print('Splitting train val test...')
    max_sl = cfg.s2s.sequence_len
    print(cfg.data.wrf_folder, 'wrf folder')
    # wrf_dataset = WRFs2sDataset(cfg.data.wrf_folder, cfg.data.wrf_variables, seq_len=max_sl, 
    #                             add_coords=cfg.run_config.use_spatial_encoding,
    #                             add_time_encoding=cfg.run_config.use_time_encoding)
    gfs_dataset = GFSncDataset(cfg.data.gfs_folder, cfg.data.gfs_variables, seq_len=max_sl,
                               time_resolution_h=6,
                               add_coords=cfg.run_config.use_spatial_encoding,
                               add_time_encoding=cfg.run_config.use_time_encoding, region_bbox=[-89.75, 90, -180, 180])
    # print(gfs_dataset.dates_dict[np.datetime64('2025-05-27T12','6h')], 'file path')
    # print(gfs_dataset[np.datetime64('2025-04-27T12')].shape)
    # era_dataset = ERAs2sDataset(cfg.data.era_folder, cfg.data.era_variables, seq_len=max_sl)
    era_dataset_uv = ERAMonthlyDataset(os.path.join(cfg.data.era_folder, 'w10'), ['u10', 'v10'], seq_len=max_sl, time_resolution_h=6, region_bbox=[-89.75, 90, -180, 180])
    era_dataset_t = ERAMonthlyDataset(os.path.join(cfg.data.era_folder, 't2'), ['t2m'], seq_len=max_sl, time_resolution_h=6, region_bbox=[-89.75, 90, -180, 180])
    era_dataset = ConcatDataset([era_dataset_uv, era_dataset_t])
    # st_ds, sc_ds = StationsNoneDataset(), ScatterNoneDataset()
    st_ds = StationsDataset(cfg.data.stations_folder, seq_len=max_sl)
    sc_ds = ScatterDataset(cfg.data.scatter_folder, seq_len=max_sl)
    dataset = dataset_with_indices(DictDataset)(gfs_dataset, era_dataset)#, st_ds, sc_ds)

    start_date, end_date = np.datetime64(cfg.data.start_date), np.datetime64(cfg.data.end_date)
    train_days, val_days, test_days = split_dates(start_date, end_date, 0.7, 0.1, 0.2, time_step='6h')
    test_days = test_days[::max_sl]

    test_sampler = Sampler(test_days, shuffle=False)
    collate_fn = variable_len_collate if cfg.run_config.variable_sequence_length else variable_len_collate

    test_dataloader = DataLoader(dataset, batch_size=cfg.run_config.batch_size, num_workers=cfg.run_config.num_workers,
                                 sampler=test_sampler, collate_fn=collate_fn, pin_memory=True)

    means_dict = torch.load(cfg.data.wrf_mean_path, weights_only=False)
    stds_dict = torch.load(cfg.data.wrf_std_path, weights_only=False)
    time_keys = ['day', 'hour'] if cfg.run_config.use_time_encoding else []
    landmask_key = ['LANDMASK'] if cfg.run_config.use_landmask else []
    spatial_keys = ['XLAT', 'XLONG'] if cfg.run_config.use_spatial_encoding else []
    wrf_keys = ['u10', 'v10', 'T2'] + landmask_key + spatial_keys + time_keys
    era_keys = ['u10', 'v10', 'T2']
    print(wrf_keys, '- wrf channels to transform')
    print(era_keys, '- era channels to transform')

    era_scaler = StandardScaler()
    wrf_scaler = StandardScaler()
    era_scaler.apply_scaler_channel_params(torch.tensor([means_dict[x] for x in era_keys]).float().to(cfg.device),
                                        torch.tensor([stds_dict[x] for x in era_keys]).float().to(cfg.device))
    wrf_scaler.apply_scaler_channel_params(torch.tensor([means_dict[x] for x in wrf_keys]).float().to(cfg.device),
                                        torch.tensor([stds_dict[x] for x in wrf_keys]).float().to(cfg.device))
    print(wrf_scaler.means, wrf_scaler.stddevs)

    # metadata = test_dataset.metadata
    wrf_grid, era_grid = gfs_dataset.src_grid, era_dataset.src_grid
    era_coords = np.stack([era_grid['longitude'].flatten(), era_grid['latitude'].flatten()]).T
    wrf_coords = np.stack([wrf_grid['longitude'].flatten(), wrf_grid['latitude'].flatten()]).T
    scat_coords = np.stack([sc_ds.src_grid['longitude'].flatten(), sc_ds.src_grid['latitude'].flatten()]).T #if False else None
    meaner = ClusterMapper(mapping_file=None,
                           target_coords=era_coords, input_coords=wrf_coords, 
                           weighted=cfg.run_config.weighted_meaner, 
                           save_mapping=True, save_name='meaner_mapping.npy', 
                           device=cfg.device, distance_metric='euclidean').to(cfg.device)

    stations_interpolator = InvDistTree(x=wrf_coords, q=st_ds.coords, device=cfg.device) #if False else None
    scat_interpolator = InvDistTree(x=wrf_coords, q=scat_coords, device=cfg.device) #if False else None
    criterion = HeterogenousMSLoss(meaner, betas, stations_interpolator, scat_interpolator,
                                    logger=logger,kernel_type=cfg.loss_config.loss_kernel,
                                    k=cfg.loss_config.k, device=cfg.device).to(cfg.device).float()
    # model = build_correction_model(cfg)
    models = {}
    for model_name in results:
        model_cfg = Config.fromfile(results[model_name]['path'])
        model_cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu" 
        print(model_cfg)

        model = build_correction_model(model_cfg, grid=gfs_dataset.src_grid)
        state_dict = torch.load(results[model_name]['model_path'])
        model.load_state_dict(state_dict)
        model.eval()
        models[model_name] = model
    
    st_ds.wind_format = 'wd'
    results = test(models, criterion, wrf_scaler, era_scaler, test_dataloader, logger, cfg)
    print(results)


if __name__ == '__main__':
    import os.path as osp
    from lib.config.cfg import Config
        
    default_cfg = Config.fromfile(osp.join('/home/configs/', 'gfs_pretrain.yaml'))
    default_cfg['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    results = {
        'TimeSformer':{
            'path': '/home/logs/TimeSformer/misc_5/config_used.yaml',
            'model_path': '/home/logs/TimeSformer/misc_5/models/model_39.pth'
        },
        'ViT 5m': {
            'path': '/home/logs/ViT/misc_13/config_used.yaml',
            'model_path': '/home/logs/ViT/misc_13/models/model_38.pth'
        },
        'ViT 11m': {
            'path': '/home/logs/ViT/misc_18/config_used.yaml',
            'model_path': '/home/logs/ViT/misc_18/models/model_38.pth'
        },
        'UNet':{
            'path': '/home/logs/UNet/misc_4/config_used.yaml',
            'model_path': '/home/logs/UNet/misc_4/models/model_15.pth'
        },
        'UNet+Transformer': {
            'path': '/home/logs/BERTunet/misc_23/config_used.yaml',
            'model_path': '/home/logs/BERTunet/misc_23/models/model_13.pth'
        },
    }
    main(default_cfg, results)

