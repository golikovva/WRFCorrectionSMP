import sys
import os
import itertools
import random
import pickle
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

sys.path.insert(0, '../../')
from lib.helpers import plot_utils
from lib.data.data_utils import get_novaya_zemlya_mask
from lib.models.loss import RMSELoss, DiffLoss, uvt_to_wt, interp_nwp_in_time, SmallScaleLoss
from lib.helpers.interpolation import InvDistTree
from lib.helpers.metrics import NormSSIM, NamedDictMetric, normalized, channel_meaned, MeanerMetric, MulticlassAccuracy, HeidkeSkillScore
from lib.helpers.aggregators import SpatialAggregator, AverageAggregator, SeasonalSpatialAggregator
from lib.helpers.ssim import CustomSSIM
from lib.helpers.paper_utils import plot_bias_correction_grid_cpy, plot_vector_bias_correction_grid_cpy, add_column_letters_on_toprow
from lib.helpers.res_table_utils import export_metrics_table
import lib.helpers.visualization as visualization
from lib.validation import metrics

def _call_model_for_test(model, test_data, dates):
    if getattr(model, "requires_dates", False):
        return model(test_data, dates=dates)
    return model(test_data)

def test(models_dict, losses, wrf_scaler, era_scaler, dataloader, logger, cfg):
    debug_mode = cfg.test_config.debug_mode
    img_format = 'pdf'
    for channel in ['u10', 'v10', 't2', 'era', 'stations', 'scatter']:
        os.makedirs(os.path.join(logger.save_dir, 'plots', channel), exist_ok=True)
    with torch.no_grad():
        for model in models_dict:
            models_dict[model].eval()
        datasets = dataloader.dataset.datasets

        # define metrics
        diff = DiffLoss(reduction='none')
        mae = torch.nn.L1Loss(reduction='none')
        mse = torch.nn.MSELoss(reduction='none')
        ssim_111 = normalized(CustomSSIM(data_range=1, size_average=False, channel=3, exp_coefs=(1, 1, 1)).forward)
        ssim_211 = normalized(CustomSSIM(data_range=1, size_average=False, channel=3, exp_coefs=(2, 1, 1)).forward)
        ssim_011 = normalized(CustomSSIM(data_range=1, size_average=False, channel=3, exp_coefs=(0, 1, 1)).forward)
        wind_err_norm = metrics.SequentialMetric(
            metrics.StatTransformed(lambda x: x[..., :2, :, :].cpu(), arity=2),
            metrics.Difference(),
            metrics.VectorNorm(arity=1, keepdims=True))
        wind_angle_norm = metrics.SequentialMetric(
            metrics.StatTransformed(lambda x: x[..., :2, :, :].cpu(), arity=2),
            metrics.AngleError(sensitivity=3, keepdims=True))
        wind_norm_diff = metrics.SequentialMetric(
            metrics.StatTransformed(lambda x: x[..., :2, :, :].cpu(), arity=2), 
            metrics.VectorNorm(arity=1, keepdims=True), 
            metrics.Difference())
        mc_acc = MulticlassAccuracy(dim=-2)
        st_dir_mae = metrics.SequentialMetric(
            metrics.CircularDifference(max_value=16),
            metrics.StatTransformed(lambda x: x*360/16, arity=1),)

        metrics_dict = {
            **{f'{model_name}_mesoscale_loss': NamedDictMetric(SmallScaleLoss(reduction='none', device=cfg.device), ['wrf', model_name]) for model_name in models_dict},
            **{f'{model_name}_era_mse': NamedDictMetric(mse, [model_name, 'era_up']) for model_name in models_dict},
            **{f'{model_name}_era_mae': NamedDictMetric(mae, [model_name, 'era_up']) for model_name in models_dict},
            **{f'{model_name}_era_err_norm': NamedDictMetric(wind_err_norm, [model_name, 'era_up']) for model_name in models_dict},
            **{f'{model_name}_era_angle_norm': NamedDictMetric(wind_angle_norm, [model_name, 'era_up']) for model_name in models_dict},
            **{f'{model_name}_era_norm_dif': NamedDictMetric(wind_norm_diff, [model_name, 'era_up']) for model_name in models_dict},
            **{f'{model_name}_mean_era_mse': NamedDictMetric(mse, [model_name + '_meaned', 'era']) for model_name in models_dict},
            **{f'{model_name}_mean_era_mae': NamedDictMetric(mae, [model_name + '_meaned', 'era']) for model_name in models_dict},
            **{f'{model_name}_ssim_custom_211':  NamedDictMetric(ssim_211,[model_name, 'wrf', 'era_up']) for model_name in models_dict},
            **{f'{model_name}_ssim_custom': NamedDictMetric(ssim_111, [model_name, 'wrf', 'era_up']) for model_name in models_dict},
            **{f'{model_name}_ssim_wrf': NamedDictMetric(ssim_111, [model_name, 'wrf', 'wrf']) for model_name in models_dict},
            **{f'{model_name}_ssim_wrf_011': NamedDictMetric(ssim_011, [model_name, 'wrf', 'wrf']) for model_name in models_dict},
            **{f'{model_name}_ssim_era': NamedDictMetric(ssim_111, [model_name, 'era_up', 'era_up']) for model_name in models_dict},
            **{f'{model_name}_spectrum': NamedDictMetric(torch.nn.Identity(), [model_name + '_spectrum']) for model_name in models_dict},

            'orig_era_mse': NamedDictMetric(mse, ['wrf', 'era_up']),
            'orig_era_mae': NamedDictMetric(mae, ['wrf', 'era_up']),
            'orig_era_err_norm': NamedDictMetric(wind_err_norm, ['wrf', 'era_up']), 
            'orig_era_angle_norm': NamedDictMetric(wind_angle_norm, ['wrf', 'era_up']), 
            'orig_era_norm_dif': NamedDictMetric(wind_norm_diff, ['wrf', 'era_up']), 

            'mean_orig_era_mse': NamedDictMetric(mse,['wrf_meaned', 'era']),
            'mean_orig_era_mae': NamedDictMetric(mae, ['wrf_meaned', 'era']),

            'orig_ssim_era': NamedDictMetric(ssim_111, ['wrf', 'era_up', 'era_up']),
            'orig_ssim_custom': NamedDictMetric(ssim_111, ['wrf', 'wrf', 'era_up']),
            'orig_ssim_custom_211': NamedDictMetric(ssim_211, ['wrf', 'wrf', 'era_up']),


            'wrf_spectrum': NamedDictMetric(torch.nn.Identity(), ['wrf_spectrum']),
            'era_spectrum': NamedDictMetric(torch.nn.Identity(), ['era_spectrum']),
        }

        if cfg.test_config.use_stations and ('Stations' in datasets):
            dir_metrics = {}
            if 'd' in datasets['Stations'].wind_format:
                dir_metrics = {
                    **{f'{model_name}_stations_accuracy': NamedDictMetric(mc_acc, [model_name + '_stations_dir', 'stations_dir']) for model_name in models_dict},
                    **{f'{model_name}_stations_dir_mae': NamedDictMetric(st_dir_mae, [model_name + '_stations_dir', 'stations_dir']) for model_name in models_dict},
                    'orig_stations_accuracy': NamedDictMetric(mc_acc, ['wrf_stations_dir', 'stations_dir']),
                    'era_stations_accuracy': NamedDictMetric(mc_acc, ['era_stations_dir', 'stations_dir']),
                    'orig_stations_dir_mae': NamedDictMetric(st_dir_mae, ['wrf_stations_dir', 'stations_dir']),
                    'era_stations_dir_mae': NamedDictMetric(st_dir_mae, ['era_stations_dir', 'stations_dir']),
                }
            metrics_dict = {**metrics_dict, **{
                **{f'{model_name}_stations_mse': NamedDictMetric(mse, [model_name + '_stations_wt', 'stations_wt']) for model_name in models_dict},
                **{f'{model_name}_stations_mae': NamedDictMetric(mae, [model_name + '_stations_wt', 'stations_wt']) for model_name in models_dict},
                'orig_stations_mse': NamedDictMetric(mse, ['wrf_stations_wt', 'stations_wt']),
                'orig_stations_mae': NamedDictMetric(mae, ['wrf_stations_wt', 'stations_wt']),
                'era_stations_mse': NamedDictMetric(mse, ['era_stations_wt', 'stations_wt']),
                'era_stations_mae': NamedDictMetric(mae, ['era_stations_wt', 'stations_wt']),
            }, **dir_metrics}

        if cfg.run_config.use_scatter and ('Scatter' in datasets):
            metrics_dict = {**metrics_dict, **{
                **{f'{model_name}_scatter_mse': NamedDictMetric(mse, [model_name + '_scatter', 'scatter']) for model_name in models_dict},
                **{f'{model_name}_scatter_mae': NamedDictMetric(mae, [model_name + '_scatter', 'scatter']) for model_name in models_dict},
                **{f'{model_name}_scatter_err_norm': NamedDictMetric(wind_err_norm,[model_name + '_scatter', 'scatter']) for model_name in models_dict},
                **{f'{model_name}_scatter_angle_norm': NamedDictMetric(wind_angle_norm,[model_name + '_scatter', 'scatter']) for model_name in models_dict},
                **{f'{model_name}_scatter_norm_diff': NamedDictMetric(wind_norm_diff,[model_name + '_scatter', 'scatter']) for model_name in models_dict},
                
                'orig_scatter_mse': NamedDictMetric(mse, ['wrf_scatter', 'scatter']),
                'orig_scatter_mae': NamedDictMetric(mae, ['wrf_scatter', 'scatter']),
                'orig_scatter_err_norm': NamedDictMetric(wind_err_norm,['wrf_scatter', 'scatter']),
                'orig_scatter_angle_norm': NamedDictMetric(wind_angle_norm,['wrf_scatter', 'scatter']),
                'orig_scatter_norm_diff': NamedDictMetric(wind_norm_diff,['wrf_scatter', 'scatter']),
    
                'era_scatter_mse': NamedDictMetric(mse, ['era_scatter', 'scatter']),
                'era_scatter_mae': NamedDictMetric(mae, ['era_scatter', 'scatter']),
                'era_scatter_err_norm': NamedDictMetric(wind_err_norm,['era_scatter', 'scatter']),
                'era_scatter_angle_norm': NamedDictMetric(wind_angle_norm,['era_scatter', 'scatter']),
                'era_scatter_norm_diff': NamedDictMetric(wind_norm_diff,['era_scatter', 'scatter']),
            }}

        wrf_grid, era_grid = datasets['WRF'].grid, datasets['ERA5'].grid

        era_coords = np.stack([era_grid['longitude'].flatten(), era_grid['latitude'].flatten()]).T
        wrf_coords = np.stack([wrf_grid['longitude'].flatten(), wrf_grid['latitude'].flatten()]).T

        era_upsampler = InvDistTree(x=era_coords, q=wrf_coords, device=cfg.device)

        if cfg.run_config.use_scatter and 'Scatter' in datasets:
            scat_grid = datasets['Scatter'].grid
            scat_coords = np.stack([scat_grid['longitude'].flatten(), scat_grid['latitude'].flatten()]).T
            scatter_interpolator = InvDistTree(x=wrf_coords, q=scat_coords, device=cfg.device)
            era_scatter_interpolator = InvDistTree(x=era_coords, q=scat_coords, device=cfg.device)
        if cfg.test_config.use_stations and 'Stations' in datasets:
            station_grid = datasets['Stations'].grid
            station_coords = np.stack([station_grid['longitude'].flatten(), station_grid['latitude'].flatten()]).T
            interpolator = InvDistTree(x=wrf_coords, q=station_coords, device=cfg.device)
            era_interpolator = InvDistTree(x=era_coords, q=station_coords, device=cfg.device)
        t = 0
        months = list(range(1, 13))

        aggregators = [SpatialAggregator(), ]
        results = {metric_name: {agg.__class__.__name__: None for agg in aggregators} for metric_name in metrics_dict}


        # =============================== costyl ========================================
        if 'ViT 11m' in models_dict:
            from lib.data.scaler import StandardScaler
            means_dict = torch.load(cfg.data.wrf_mean_path+'_global', weights_only=False)
            stds_dict = torch.load(cfg.data.wrf_std_path+'_global', weights_only=False)
            time_keys = ['day', 'hour'] if cfg.run_config.use_time_encoding else []
            landmask_key = ['LANDMASK'] if cfg.run_config.use_landmask else []
            spatial_keys = ['XLAT', 'XLONG'] if cfg.run_config.use_spatial_encoding else []
            wrf_keys = ['u10', 'v10', 'T2'] + landmask_key + spatial_keys + time_keys
            era_keys = ['u10', 'v10', 'T2']
            
            global_era_scaler = StandardScaler()
            global_wrf_scaler = StandardScaler()
            global_era_scaler.apply_scaler_channel_params(torch.tensor([means_dict[x] for x in era_keys]).float().to(cfg.device),
                                                torch.tensor([stds_dict[x] for x in era_keys]).float().to(cfg.device))
            global_wrf_scaler.apply_scaler_channel_params(torch.tensor([means_dict[x] for x in wrf_keys]).float().to(cfg.device),
                                                torch.tensor([stds_dict[x] for x in wrf_keys]).float().to(cfg.device))
        # ========================== costyl ========================================================================

        for data, dates in tqdm(dataloader): 
            if data is None:
                continue
            test_data, test_label = data.pop(cfg.reference_dataset), data.pop(cfg.target_dataset)
            test_data = torch.swapaxes(test_data.type(torch.float).to(cfg.device), 0, 1).contiguous()
            test_label = torch.swapaxes(test_label.type(torch.float).to(cfg.device), 0, 1)
            era_h, era_w = test_label.shape[-2:]

            date = dates.astype(str)

            test_data = wrf_scaler.transform(test_data, dims=2)
            
            outputs = {}
            for model_name in models_dict:
                if model_name == 'ViT 11m':
                    test_data = wrf_scaler.inverse_transform(test_data, dims=2)
                    test_data = global_wrf_scaler.transform(test_data, dims=2)

                output = _call_model_for_test(models_dict[model_name], test_data, dates)
                if model_name == 'ViT 11m':
                    output = global_era_scaler.inverse_transform(output, dims=2)
                    test_data = global_wrf_scaler.inverse_transform(test_data, dims=2)
                    test_data = wrf_scaler.transform(test_data, dims=2)
                else:
                    output = era_scaler.inverse_transform(output, dims=2)
                outputs[model_name] = output
                corr_meaned = input_to_era_map(output, losses.meaner, era_map_shape=(era_h, era_w))
                outputs[model_name + '_meaned'] = corr_meaned
                corr_spectrum = get_power_spectrum(uvt_to_wt(output, -3).cpu())[1]
                outputs[model_name + '_spectrum'] = torch.from_numpy(corr_spectrum)
            # output = model(test_data)

            # output = era_scaler.inverse_transform(output, dims=2)
            test_data = wrf_scaler.inverse_transform(test_data, dims=2)[:, :, :3]

            # ========== Interpolate WRF to others ================
            wrf_meaned = input_to_era_map(test_data, losses.meaner, era_map_shape=(era_h, era_w))
            # corr_meaned = input_to_era_map(output, losses.meaner, era_map_shape=(era_h, era_w))


            era_upsampled = era_upsampler(test_label.flatten(-2, -1)).view(test_data.shape)
            spectrum_bins, era_spectrum = get_power_spectrum(uvt_to_wt(era_upsampled, -3).cpu())

            wrf_spectrum = get_power_spectrum(uvt_to_wt(test_data, -3).cpu())[1]
            era_spectrum, wrf_spectrum = map(torch.from_numpy, [era_spectrum, wrf_spectrum])

            samples_dict = {
                **outputs,
                'wrf': test_data,
                'era_up': era_upsampled,
                'era': test_label,
                # 'corr': output,
                'wrf_meaned': wrf_meaned,
                # 'corr_meaned': corr_meaned,

                'wrf_spectrum': wrf_spectrum,
                # 'corr_spectrum': corr_spectrum,
                'era_spectrum': era_spectrum,
                }
            if cfg.test_config.use_stations and 'Stations' in datasets:
                has_dir = 'd' in datasets['Stations'].wind_format

                station = data.pop('Stations')
                station = torch.permute(station.type(torch.float).to(cfg.device), (1, 0, 3, 2))
                
                if has_dir:
                    stations_wt = station[..., [0, 2], :]
                    stations_dir = station[..., [1,], :]
                else:
                    stations_wt = station[..., [0, 1], :]

                wrf_stations = input_to_stations(test_data, interpolator)
                wrf_stations_wt, wrf_stations_dir = split_uvt_to_speed_temp_and_dir(wrf_stations)
                era_stations = input_to_stations(test_label, era_interpolator) 
                era_stations_wt, era_stations_dir = split_uvt_to_speed_temp_and_dir(era_stations)     

                stations_outputs = {}
                for model_name in models_dict:
                    corr_stations = input_to_stations(outputs[model_name], interpolator)
                    corr_stations_wt, corr_stations_dir = split_uvt_to_speed_temp_and_dir(corr_stations)
                    stations_outputs[model_name + '_stations_wt'] = corr_stations_wt
                    if has_dir:
                        stations_outputs[model_name + '_stations_dir'] = corr_stations_dir

                dir_samples = {}
                if has_dir:
                    dir_samples = {
                        'wrf_stations_dir': wrf_stations_dir,
                        'era_stations_dir': era_stations_dir,
                        'stations_dir': stations_dir,
                    }
                samples_dict = {**samples_dict, **{
                    **stations_outputs,
                    'wrf_stations_wt': wrf_stations_wt,
                    'era_stations_wt': era_stations_wt,
                    'stations_wt': stations_wt,
                }, **dir_samples}

            if cfg.run_config.use_scatter and 'Scatter' in datasets:
                scatter = data.pop('Scatter')
                batch_dates = torch.as_tensor(dates.astype('datetime64[s]').astype('float64')).to(cfg.device)
                scatter_times = scatter[0].to(cfg.device).type(torch.double)
                scatter_data = torch.stack((scatter[1], scatter[2]), dim=-3).type(torch.float).to(cfg.device)
                scatter_mask = scatter_interpolator.calc_input_tensor_mask(scatter_times.shape[-2:], 
                                                                        distance_criterion=0.15,
                                                                        fill_value=torch.nan)
                wrf_scatter = input_to_scatter(test_data, scatter_interpolator, scatter_times, batch_dates, mask=scatter_mask)
                era_scatter = input_to_scatter(test_label, era_scatter_interpolator, scatter_times, batch_dates, mask=scatter_mask)

                scatter_outputs = {}
                for model_name in models_dict:
                    corr_scatter = input_to_scatter(outputs[model_name], scatter_interpolator, scatter_times, batch_dates, mask=scatter_mask)
                    scatter_outputs[model_name + '_scatter'] = corr_scatter
                # corr_scatter = input_to_scatter(output, scatter_interpolator, scatter_times, batch_dates, mask=scatter_mask)

                samples_dict = {**samples_dict, **{
                    **scatter_outputs,
                    'wrf_scatter': wrf_scatter,
                    # 'corr_scatter': corr_scatter,
                    'era_scatter': era_scatter,
                    'scatter': scatter_data,
                    }}

            if cfg.test_config.save_outputs and (month := date[0].astype('datetime64[M]').astype(int) % 12 + 1) in months:
                months.remove(month)
                subsample = {k: samples_dict[k][0,0].cpu() - torch.tensor([0, 0, 273.15])[:, None, None] for k in ['wrf', 'era_up', *models_dict.keys()]}
                proj = visualization.get_domain_projection(cfg.reference_region)
                figsize = (10*len(subsample), 8) if cfg.reference_region == 'nestp' else None
                fig, axes, _ = plot_bias_correction_grid_cpy(
                    samples=subsample,
                    base_key="wrf",
                    target_key="era_up",
                    grid=wrf_grid,
                    channel=2,
                    proj=proj,
                    diff_sign="other_minus_base",
                    cmap_top= "RdBu_r",
                    cmap_bottom = "RdBu_r",
                    diff_centered_norm=True,
                    centered_norm=True,
                    cbar_labels=("Temperature at 2 m (C)", "Temperature diff (C)"),
                    figsize=figsize,
                )
                add_column_letters_on_toprow(axes, y=0.98)
                fig.savefig(os.path.join(logger.save_dir, 'plots', f't2_{date[0]}.png'), dpi=400, bbox_inches='tight')
                fig, axes, _ = plot_vector_bias_correction_grid_cpy(
                    samples=subsample,
                    base_key="wrf",
                    target_key="era_up",
                    grid=wrf_grid,
                    channel=None,
                    proj=proj,
                    diff_sign="other_minus_base",
                    cmap_top= "jet",
                    cmap_bottom = "RdBu_r",
                    diff_centered_norm=True,
                    centered_norm=False,
                    cbar_labels=("Wind speed (m/s)", "Wind speed diff (m/s)"),
                    figsize=figsize,
                )
                add_column_letters_on_toprow(axes, y=0.98)
                fig.savefig(os.path.join(logger.save_dir, 'plots', f'uv10_{date[0]}.png'), dpi=400, bbox_inches='tight')
                plt.close('all')
            for metric_name in metrics_dict:
                err_field = metrics_dict[metric_name].calculate(samples_dict)

                for agg in aggregators:
                    agg_name = agg.__class__.__name__
                    acc = results[metric_name][agg_name]
                    if acc is None:
                        acc = agg.init_accumulator(err_field.shape[2:])
                        results[metric_name][agg_name] = acc
                    agg.accumulate(acc, err_field, dates)

            if debug_mode and t > 5:
                break
            t += 1

        res_dict = {metric_name: AverageAggregator.finalize(results[metric_name]['SpatialAggregator']).cpu().numpy() for metric_name in metrics_dict}
        with open(os.path.join(logger.save_dir, 'experiment_metrics.pickle'), 'wb') as handle:
            pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        df = export_metrics_table(
            res_dict=res_dict,
            models=[model_name for model_name in models_dict] + ['orig'],
            metrics=['mse', 'mae', 'err_norm', 'norm_dif', 'angle_norm'],
            target='era',
            csv_path=os.path.join(logger.save_dir, 'era_metrics_table.csv'),
            image_path=os.path.join(logger.save_dir, 'plots', 'era_metrics_table.png'),
            lead_labels=['u10', 'v10', 't2'],   # optional
            precision=2,
            image_title='ERA metrics comparison',
        )

        df = export_metrics_table(
            res_dict=res_dict,
            models=[model_name for model_name in models_dict] + ['era', 'orig'],
            metrics=['mse', 'mae'],
            target='stations',
            csv_path=os.path.join(logger.save_dir, 'stations_metrics_table.csv'),
            image_path=os.path.join(logger.save_dir, 'plots', 'stations_metrics_table.png'),
            lead_labels=['w10', 't2'],   # optional
            precision=2,
            image_title='Stations metrics comparison',
        )

        if cfg.test_config.draw_spectrum:
            region_size = 210

            if cfg.reference_region == 'nestp':
                region_size = 412 
            elif cfg.reference_region == 'borey':
                region_size = 210 
            print(region_size)
            print(results['era_spectrum']['SpatialAggregator']['sum'].shape)
            era_spectrum = SpatialAggregator.finalize(results['era_spectrum']['SpatialAggregator'])
            wrf_spectrum = SpatialAggregator.finalize(results['wrf_spectrum']['SpatialAggregator'])
            models_spectrums = []
            for model_name in models_dict:
                models_spectrums.append(SpatialAggregator.finalize(results[model_name + '_spectrum']['SpatialAggregator']))
            print(spectrum_bins.shape, era_spectrum.shape, wrf_spectrum.shape, [ms.shape for ms in models_spectrums])
            # corr_spectrum = SpatialAggregator.finalize(results['corr_spectrum']['SpatialAggregator'])
            for i, c in enumerate(['w10', 't2']):
                spectrum_plot = plot_utils.power_loglog_spectrum(
                    [era_spectrum[i], wrf_spectrum[i], *[ms[i] for ms in models_spectrums]],
                    [cfg.target_dataset, cfg.reference_dataset, *[f'{mn}' for mn in models_dict]],
                    spectrum_bins/region_size/6, name=c
                )
                plt.savefig(os.path.join(logger.save_dir, 'plots', f'{c}_spectrum_plot.{img_format}'), dpi=300, bbox_inches="tight", format=img_format,)
            plt.close('all')


        if cfg.run_config.use_scatter and 'Scatter' in datasets:
            #------------- Scatter plots wrf-corr vs scatter --------------
            fig, ax = visualization.create_cartopy_axes(
                3, 3, figsize=(15, 15),)
            ax[0, 0].set_title(f'{cfg.reference_dataset} corrected')
            ax[0, 1].set_title(f'{cfg.reference_dataset} original')
            ax[0, 2].set_title(f'{cfg.target_dataset}')
            row_titles = ['diff norm', 'angle error', 'norm diff']
            for i, title in enumerate(row_titles):
                a = ax[i, 0]
                a.text(
                    -0.08, 0.5, title, transform=a.transAxes,
                    rotation=90, va="center", ha="right", fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none")
                )

                # grid=wrf_grid,)
            print(SpatialAggregator.finalize(results['scatter_err_norm']['SpatialAggregator'])[0].shape, 'shape castter err')
            print(SpatialAggregator.finalize(results['scatter_angle_norm']['SpatialAggregator'])[0].shape, 'shape castter err')
            print(SpatialAggregator.finalize(results['scatter_norm_diff']['SpatialAggregator'])[0].shape, 'shape castter err')

            res1 = SpatialAggregator.finalize(results['scatter_err_norm']['SpatialAggregator'])
            res2 = SpatialAggregator.finalize(results['orig_scatter_err_norm']['SpatialAggregator'])
            res3 = SpatialAggregator.finalize(results['era_scatter_err_norm']['SpatialAggregator'])
            vmin = min(np.nanpercentile(res1, 1), np.nanpercentile(res2, 1), np.nanpercentile(res3, 1))
            vmax = max(np.nanpercentile(res1, 99), np.nanpercentile(res2, 99), np.nanpercentile(res3, 99))
            im = visualization.visualize_scalar_field(
                scat_grid, 
                res1,
                ax=ax[0,0], vmin=vmin, vmax=vmax,)
            visualization.visualize_scalar_field(
                scat_grid, 
                res2,
                ax=ax[0,1], vmin=vmin, vmax=vmax,)
            visualization.visualize_scalar_field(
                scat_grid, 
                res3,
                ax=ax[0,2], vmin=vmin, vmax=vmax,)
            fig.colorbar(im, ax=ax[0,:], orientation='vertical', label='Wind error norm (m/s)')
        
            res1 = SpatialAggregator.finalize(results['scatter_angle_norm']['SpatialAggregator'])
            res2 = SpatialAggregator.finalize(results['orig_scatter_angle_norm']['SpatialAggregator'])
            res3 = SpatialAggregator.finalize(results['era_scatter_angle_norm']['SpatialAggregator'])
            vmin = min(np.nanpercentile(res1, 1), np.nanpercentile(res2, 1), np.nanpercentile(res3, 1))
            vmax = max(np.nanpercentile(res1, 99), np.nanpercentile(res2, 99), np.nanpercentile(res3, 99))
            im = visualization.visualize_scalar_field(
                scat_grid, 
                res1,
                ax=ax[1,0], vmin=vmin, vmax=vmax,)
            visualization.visualize_scalar_field(
                scat_grid, 
                res2,
                ax=ax[1,1], vmin=vmin, vmax=vmax,)
            visualization.visualize_scalar_field(
                scat_grid, 
                res3,
                ax=ax[1,2], vmin=vmin, vmax=vmax,)
            fig.colorbar(im, ax=ax[1,:], orientation='vertical', label='Wind angle (degrees)')

            res1 = SpatialAggregator.finalize(results['scatter_norm_diff']['SpatialAggregator'])
            res2 = SpatialAggregator.finalize(results['orig_scatter_norm_diff']['SpatialAggregator'])
            res3 = SpatialAggregator.finalize(results['era_scatter_norm_diff']['SpatialAggregator'])
            vmin = min(np.nanpercentile(res1, 1), np.nanpercentile(res2, 1), np.nanpercentile(res3, 1))
            vmax = max(np.nanpercentile(res1, 99), np.nanpercentile(res2, 99), np.nanpercentile(res3, 99))
            im = visualization.visualize_scalar_field(
                scat_grid, 
                res1,
                ax=ax[2,0], vmin=vmin, vmax=vmax,)
            visualization.visualize_scalar_field(
                scat_grid, 
                res2,
                ax=ax[2,1], vmin=vmin, vmax=vmax,)
            visualization.visualize_scalar_field(
                scat_grid, 
                res3,
                ax=ax[2,2], vmin=vmin, vmax=vmax,)
            fig.colorbar(im, ax=ax[2,:], orientation='vertical', label='Wind norm diff (m/s)')

            fig.savefig(os.path.join(logger.save_dir, 'plots', 'scatter', f'scatter_mae.png'), dpi=300, bbox_inches="tight", format="png",)
        
        # #------------- Error map plots wrf-corr vs era --------------
        base_size = 3
        # font sizes
        col_title_fs = 22
        row_title_fs = 24
        cbar_label_fs = 18
        cbar_tick_fs = 18
        proj = visualization.get_domain_projection(cfg.reference_region)
        fig, ax = visualization.create_cartopy_axes(
            4, 1+len(models_dict),
            grid=wrf_grid,
            figsize=(2*base_size*(len(models_dict)+1),4*base_size),
            add_land=False, 
            face_ocean=False,
            proj=proj,
            )

        ax[0, 0].set_title(f'{cfg.reference_dataset}', fontsize=col_title_fs)
        for i, model_name in enumerate(models_dict):
            ax[0, i+1].set_title(f'{model_name}', fontsize=col_title_fs)
        # ax[0, 1].set_title('WRF corrected')
        row_titles = ['T2 MAE', 'Diff norm', 'Angle error', 'Norm diff']
        for i, title in enumerate(row_titles):
            a = ax[i, 0]
            a.text(
                -0.08, 0.5, title, transform=a.transAxes,
                rotation=90, va="center", ha="right", fontsize=row_title_fs,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none")
            )
        
        res1 = SpatialAggregator.finalize(results['orig_era_mae']['SpatialAggregator'])[2]
        model_results = [SpatialAggregator.finalize(results[f'{mn}_era_mae']['SpatialAggregator'])[2] for mn in models_dict]

        vmin = min(np.nanpercentile(res1, 1), *[np.nanpercentile(model_res, 1) for model_res in model_results])
        vmax = max(np.nanpercentile(res1, 97), *[np.nanpercentile(model_res, 97) for model_res in model_results])
        im = visualization.visualize_scalar_field(
            wrf_grid, 
            res1,
            ax=ax[0,0], vmin=vmin, vmax=vmax)
        for i, model_res in enumerate(model_results):
            visualization.visualize_scalar_field(
                wrf_grid, 
                model_res,
                ax=ax[0,i+1], vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax[0,:], orientation='vertical')
        cbar.set_label('T2 MAE (K)', fontsize=cbar_label_fs)
        cbar.ax.tick_params(labelsize=cbar_tick_fs)

        res1 = SpatialAggregator.finalize(results['orig_era_err_norm']['SpatialAggregator'])
        model_results = [SpatialAggregator.finalize(results[f'{mn}_era_err_norm']['SpatialAggregator']) for mn in models_dict]
        vmin = min(np.nanpercentile(res1, 1), *[np.nanpercentile(model_res, 1) for model_res in model_results])
        vmax = max(np.nanpercentile(res1, 99), *[np.nanpercentile(model_res, 99) for model_res in model_results])

        im = visualization.visualize_scalar_field(
            wrf_grid, 
            res1,
            ax=ax[1,0], vmin=vmin, vmax=vmax,)
        for i, model_res in enumerate(model_results):
            visualization.visualize_scalar_field(
                wrf_grid, 
                model_res,
                ax=ax[1,i+1], vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax[1,:], orientation='vertical')
        cbar.set_label('Error norm (m/s)', fontsize=cbar_label_fs)
        cbar.ax.tick_params(labelsize=cbar_tick_fs)

        res1 = SpatialAggregator.finalize(results['orig_era_angle_norm']['SpatialAggregator'])
        model_results = [SpatialAggregator.finalize(results[f'{mn}_era_angle_norm']['SpatialAggregator']) for mn in models_dict]
        vmin = min(np.nanpercentile(res1, 1), *[np.nanpercentile(model_res, 1) for model_res in model_results])
        vmax = max(np.nanpercentile(res1, 99), *[np.nanpercentile(model_res, 99) for model_res in model_results])
        im = visualization.visualize_scalar_field(
            wrf_grid, 
            res1,
            ax=ax[2,0], vmin=vmin, vmax=vmax,)
        for i, model_res in enumerate(model_results):
            visualization.visualize_scalar_field(
                wrf_grid, 
                model_res,
                ax=ax[2,i+1], vmin=vmin, vmax=vmax)

        cbar = fig.colorbar(im, ax=ax[2,:], orientation='vertical')
        cbar.set_label('Angle (degrees)', fontsize=cbar_label_fs)
        cbar.ax.tick_params(labelsize=cbar_tick_fs)

        res1 = SpatialAggregator.finalize(results['orig_era_norm_dif']['SpatialAggregator'])
        model_results = [SpatialAggregator.finalize(results[f'{mn}_era_norm_dif']['SpatialAggregator']) for mn in models_dict]
        # res2 = SpatialAggregator.finalize(results['era_norm_dif']['SpatialAggregator'])
        vmin = min(np.nanpercentile(res1, 1), *[np.nanpercentile(model_res, 1) for model_res in model_results])
        vmax = max(np.nanpercentile(res1, 99), *[np.nanpercentile(model_res, 99) for model_res in model_results])
        import matplotlib.colors as colors
        norm = colors.CenteredNorm(vcenter=0, halfrange=max(abs(vmin), abs(vmax)))
        im = visualization.visualize_scalar_field(
            wrf_grid, 
            res1,
            ax=ax[3,0], norm=norm, cmap='seismic')  
        for i, model_res in enumerate(model_results):
            visualization.visualize_scalar_field(
                wrf_grid, 
                model_res,
                ax=ax[3,i+1], norm=norm, cmap='seismic')
            
        # visualization.visualize_scalar_field(
        #     wrf_grid, 
        #     res2,
        #     ax=ax[3,1], norm=norm, cmap='seismic')
        cbar = fig.colorbar(im, ax=ax[3,:], orientation='vertical')
        cbar.set_label('Norm diff (m/s)', fontsize=cbar_label_fs)
        cbar.ax.tick_params(labelsize=cbar_tick_fs)
        fig.savefig(os.path.join(logger.save_dir, 'plots', 'era', f'era_mae.png'), dpi=300, bbox_inches="tight", format="png",)

        if cfg.test_config.use_stations and 'Stations' in datasets:
            # for model_name in models_dict:
            #------------- Stations plots wrf-corr vs stations --------------
            fig, ax = visualization.create_cartopy_axes(
                1, len(models_dict)+1,
                grid=wrf_grid,
                add_land=True,
                add_gridlines=True, 
                face_ocean=False,
                proj=proj)
            ax[0, 0].set_title('WRF corrected')
            ax[0, 1].set_title('ERA5')
            names = [mn for mn in models_dict]
            model_results = [SpatialAggregator.finalize(results[f'{mn}_stations_mae']['SpatialAggregator']) for mn in models_dict]
            # res1 = SpatialAggregator.finalize(results[f'{model_name}_stations_mae']['SpatialAggregator'])
            res2 = SpatialAggregator.finalize(results['orig_stations_mae']['SpatialAggregator'])
            res3 = SpatialAggregator.finalize(results['era_stations_mae']['SpatialAggregator'])
            model_dir_results = []
            if has_dir:
                model_dir_results = [SpatialAggregator.finalize(results[f'{mn}_stations_dir_mae']['SpatialAggregator']) for mn in models_dict]
                dir_mae_orig = SpatialAggregator.finalize(results['orig_stations_dir_mae']['SpatialAggregator'])
                dir_mae_era = SpatialAggregator.finalize(results['era_stations_dir_mae']['SpatialAggregator'])
            print(res2.shape, res3.shape, 'stations mae')
            print(res2[..., 0], res3[..., 0], 'stations mae')
            for i, (res1, dir_res1, name) in enumerate(itertools.zip_longest(model_results, model_dir_results, names)):
                ss_corr = 1 - res1/res2
                ss_era = 1 - res3/res2
                corr_station_metrics={'w10': ss_corr[0], 't2': ss_corr[1]}
                era_station_metrics={'w10': ss_era[0], 't2': ss_era[1]}
                metric_limits={'w10': (-1, 1), 't2': (-1, 1)}
                if has_dir:
                    ss_dir_corr = 1 - dir_res1/dir_mae_orig
                    ss_dir_era = 1 - dir_mae_era/dir_mae_orig
                    corr_station_metrics['dir'] = ss_dir_corr
                    era_station_metrics['dir'] = ss_dir_era
                    metric_limits['dir'] = (-1, 1)
                visualization.draw_station_metrics(grid=wrf_grid, ax=ax[0, i], stations_grid=station_grid, station_metrics=corr_station_metrics, 
                                                metric_limits=metric_limits, title=name)
            visualization.draw_station_metrics(grid=wrf_grid, ax=ax[0, -1], stations_grid=station_grid, station_metrics=era_station_metrics, 
                                            metric_limits=metric_limits, title='ERA5')
            fig.savefig(os.path.join(logger.save_dir, 'plots', 'stations', f'{model_name}_stations_metrics.png'), dpi=300, bbox_inches="tight", format="png",)
        # exit()
    #     acc.save_data(logger.save_dir, ['power_spectrum-era',
    #                                     'power_spectrum-wrf',
    #                                     'power_spectrum-corr'])

    #     print("Drawing wrf era losses hist...")
    #     # orig_era = acc.data['wrf_orig-era']
    #     # corr_era = acc.data['wrf_corr-era']
    #     orig_era = SpatialAggregator.finalize(results['orig_era_mse']['SpatialAggregator'])
    #     corr_era = SpatialAggregator.finalize(results['era_mse']['SpatialAggregator'])
    #     # losses_plot = plot_utils.draw_losses_gist(orig_era.transpose(0, 1).mean(-1),
    #     #                                           corr_era.transpose(0, 1).mean(-1), 'ERA5')
    #     # plt.savefig(os.path.join(logger.save_dir, 'plots', 'era', f'era_losses_hist'), bbox_inches="tight", format="pdf",)
    #     # plt.close('all')

    #     print("Drawing seasonal wrf era bar plot (насколько улучшились данные wrf относительно era5)...")
    #     wrf_era_mean_loss, wrf_era_t_mean_map = get_season_mean_losses(orig_era, corr_era, acc.data['month'])
    #     season_metric_bar = plot_utils.draw_seasonal_bar_plot(_metric(*wrf_era_mean_loss), dtype="WRF on ERA5")
    #     plt.savefig(os.path.join(logger.save_dir, 'plots', 'era', f'season_era_metric_bar_plot'), bbox_inches="tight", format="pdf",)
    #     plt.close('all')

    #     # карта ошибок wrf на era5
    #     print('Drawing error map between wrf and era5...')
    #     season_maps = plot_utils.draw_seasonal_orig_corr_map(era_vector_to_map(wrf_era_t_mean_map[0], losses.meaner, era_map_shape=era_h * era_w).reshape(4, 3, era_h, era_w)[:, 2],
    #                                                          era_vector_to_map(wrf_era_t_mean_map[1], losses.meaner, era_map_shape=era_h * era_w).reshape(4, 3, era_h, era_w)[:, 2],
    #                                                          lats=era_grid['latitude'], lons=era_grid['longitude'],)
    #     season_maps.savefig(os.path.join(logger.save_dir, 'plots', 'era', f'seasonal_orig-corr_maps'), 
    #                         bbox_inches="tight", format="pdf",)
    #     orig_era_figs = plot_utils.draw_seasonal_era_error_map(era_vector_to_map(wrf_era_t_mean_map[0], losses.meaner, era_map_shape=era_h * era_w).unflatten(-1, [era_h, era_w]),
    #                                                            lats=era_grid['latitude'], lons=era_grid['longitude'],
    #                                                            dtype='WRF orig', colormap='rainbow')
    #     [f.savefig(os.path.join(logger.save_dir, 'plots', 'era', f'{name}'),
    #                 bbox_inches="tight", format="pdf",) for name, f in orig_era_figs.items()]
    #     corr_era_figs = plot_utils.draw_seasonal_era_error_map(era_vector_to_map(wrf_era_t_mean_map[1], losses.meaner, era_map_shape=era_h * era_w).unflatten(-1, [era_h, era_w]),
    #                                                            lats=era_grid['latitude'], lons=era_grid['longitude'],
    #                                                            dtype='WRF corr', colormap='rainbow')
    #     [f.savefig(os.path.join(logger.save_dir, 'plots', 'era', f'{name}'),
    #                 bbox_inches="tight", format="pdf",) for name, f in corr_era_figs.items()]
    #     wrf_era_map_metric = era_vector_to_map(_metric(wrf_era_t_mean_map[0], wrf_era_t_mean_map[1]), losses.meaner, era_map_shape=era_h * era_w).unflatten(-1, [era_h, era_w])
    #     corr_era_figs = plot_utils.draw_seasonal_era_error_map(torch.clip(wrf_era_map_metric, min=-1),
    #                                                            lats=era_grid['latitude'], lons=era_grid['longitude'],
    #                                                            dtype='WRF metric', colormap='bwr_r', vmin=-1, vmax=1)
    #     [f.savefig(os.path.join(logger.save_dir, 'plots', 'era', f'{name}'),
    #                 bbox_inches="tight", format="pdf",) for name, f in corr_era_figs.items()]
    #     plt.close('all')
    #     if use_station:
    #     # гистограмма ошибок wrf на станциях
    #         print('Drawing wrf error hist on stations...')
    #         # print(orig_stations.shape)
    #         orig_stations = acc.data['wrf_orig-stations']
    #         corr_stations = acc.data['wrf_corr-stations']
    #         losses_plot = plot_utils.draw_losses_gist(torch.nanmean(orig_stations.transpose(0, 1), dim=-1),
    #                                                 torch.nanmean(corr_stations.transpose(0, 1), dim=-1),
    #                                                 channels=['t2', 'w10'], dtype='Stations')
    #         plt.savefig(os.path.join(logger.save_dir, 'plots', 'stations', f'station_losses_hist'), bbox_inches="tight", format="pdf",)
    #         plt.close('all')

    #         # усредненная метрика wrf на станциях по сезонам
    #         print('Drawing mean seasonal wrf station metric...')
    #         wrf_st_mean_loss, wrf_st_t_mean_map = get_season_mean_losses(orig_stations, corr_stations, acc.data['month'])
    #         season_metric_bar = plot_utils.draw_seasonal_bar_plot(_metric(*wrf_st_mean_loss), channels=['t2', 'w10'],
    #                                                             dtype="WRF on Stations")
    #         plt.savefig(os.path.join(logger.save_dir, 'plots', 'stations', f'season_wrf_stations_metric_bar_plot'),
    #                     bbox_inches="tight", format="pdf",)
    #         plt.close('all')

    #         # карта метрик wrf на станциях по сезонам
    #         print("Drawing wrf seasonal metric map on stations...")
    #         st_figs = plot_utils.draw_seasonal_stations_error_map(_metric(*wrf_st_t_mean_map), metadata, output, test_label)
    #         [f.savefig(os.path.join(logger.save_dir, 'plots', 'stations', f'{name}'),
    #                     bbox_inches="tight", format="pdf",) for name, f in st_figs.items()]
    #         plt.close('all')

    #         # усредненная метрика era5 на станциях по сезонам
    #         print('Drawing mean seasonal era5 station metric')
    #         era_stations = acc.data['era-stations']
    #         era_st_mean_loss, era_st_t_mean_map = get_season_mean_losses(orig_stations, era_stations, acc.data['month'])
    #         season_metric_bar = plot_utils.draw_seasonal_bar_plot(_metric(*era_st_mean_loss),
    #                                                             channels=['t2', 'w10'], dtype="ERA5 on Stations")
    #         plt.savefig(os.path.join(logger.save_dir, 'plots', 'stations', f'season_stations_metric_bar_plot'),
    #                     bbox_inches="tight", format="pdf",)
    #         plt.close('all')

    #         # карта метрик era5 на станциях по сезонам
    #         print('Drawing era5 seasonal metric map on stations...')
    #         era_st = plot_utils.draw_seasonal_stations_error_map(_metric(*era_st_t_mean_map), metadata, output, test_label,
    #                                                             dtype='ERA5')
    #         [f.savefig(os.path.join(logger.save_dir, 'plots', 'stations', f'{name}'),
    #                     bbox_inches="tight", format="pdf",) for name, f in era_st.items()]
    #         plt.close('all')
    #     if use_scatter:
    #     # карта ошибок wrf на скаттерометре
    #         print('Drawing wrf seasonal error map on scatter...')
    #         orig_scatter_seasonal = get_season_mean_scatter(acc.data['wrf_orig-scatter'], acc.data['wrf-scatter-counts'],
    #                                                         acc.data['month'])
    #         corr_scatter_seasonal = get_season_mean_scatter(acc.data['wrf_corr-scatter'], acc.data['wrf-scatter-counts'],
    #                                                         acc.data['month'])
    #         mean_orig_scatter = torch.nanmean(orig_scatter_seasonal[:4], dim=[-1])
    #         mean_corr_scatter = torch.nanmean(corr_scatter_seasonal[:4], dim=[-1])
    #         print(mean_corr_scatter.shape, 'mean_corr_scatter.shape')
    #         scat_metric_fig = plot_utils.draw_seasonal_bar_plot(_metric(mean_orig_scatter, mean_corr_scatter),
    #                                                             channels=['u10', 'v10'],
    #                                                             dtype="WRF on Scatter")
    #         plt.savefig(os.path.join(logger.save_dir, 'plots', 'scatter', f'season_scatter_wrf_metric_bar_plot'),
    #                     bbox_inches="tight", format="pdf",)
    #         plt.close('all')
    #         scat_figs = plot_utils.draw_seasonal_scat_err_map(orig_scatter_seasonal, lons=scat_grid['longitude'],
    #                                                         lats=scat_grid['latitude'], dtype='WRF orig', colormap='common')
    #         [f.savefig(os.path.join(logger.save_dir, 'plots', 'scatter', f'{name}'),
    #                     bbox_inches="tight", format="pdf",) for name, f in scat_figs.items()]
    #         scat_figs = plot_utils.draw_seasonal_scat_err_map(corr_scatter_seasonal, lons=scat_grid['longitude'],
    #                                                         lats=scat_grid['latitude'], dtype='WRF corr', colormap='common')
    #         [f.savefig(os.path.join(logger.save_dir, 'plots', 'scatter', f'{name}'),
    #                     bbox_inches="tight", format="pdf",) for name, f in scat_figs.items()]

    #         scat_figs = plot_utils.draw_seasonal_scat_err_map(torch.clip(_metric(orig_scatter_seasonal,
    #                                                                             corr_scatter_seasonal), min=-1),
    #                                                         lons=scat_grid['longitude'],
    #                                                         lats=scat_grid['latitude'], dtype='WRF metric',
    #                                                         colormap='bwr_r', vmin=-1, vmax=1)
    #         [f.savefig(os.path.join(logger.save_dir, 'plots', 'scatter', f'{name}'),
    #                     bbox_inches="tight", format="pdf",) for name, f in scat_figs.items()]
    #         plt.close('all')

    #         # карта ошибок era5 на скаттерометре
    #         print('Drawing era5 seasonal metric map on scatter...')
    #         era_scatter_seasonal = get_season_mean_scatter(acc.data['era-scatter'], acc.data['era-scatter-counts'],
    #                                                     acc.data['month'])
    #         scat_figs = plot_utils.draw_seasonal_scat_err_map(era_scatter_seasonal, lons=scat_grid['longitude'],
    #                                                         lats=scat_grid['latitude'], dtype='ERA5', colormap='common')
    #         [f.savefig(os.path.join(logger.save_dir, 'plots', 'scatter', f'{name}'),
    #                     bbox_inches="tight", format="pdf",) for name, f in scat_figs.items()]
    #         plt.close('all')

    #         mean_era_scatter = torch.nanmean(era_scatter_seasonal[:4], dim=[-1])
    #         print(mean_corr_scatter.shape, 'mean_corr_scatter.shape')
    #         scat_metric_fig = plot_utils.draw_seasonal_bar_plot(_metric(mean_orig_scatter, mean_era_scatter),
    #                                                             channels=['u10', 'v10'],
    #                                                             dtype="ERA on Scatter")
    #         plt.savefig(os.path.join(logger.save_dir, 'plots', 'scatter', f'season_scatter_era_metric_bar_plot'),
    #                     bbox_inches="tight", format="pdf",)
    #         plt.close('all')

    #     if cfg.test_config.save_losses:
    #         acc.save_data(logger.save_dir)
    #     if use_station:
    #         st_means = torch.nanmean(acc.data['wrf_corr-stations'], dim=[0, -1]).tolist() 
    #         st_orig_means = torch.nanmean(acc.data['wrf_orig-stations'], dim=[0, -1]).tolist()
    #         st_era_means = torch.nanmean(acc.data['era-stations'], dim=[0, -1]).tolist()
    #     else:
    #         st_means = st_orig_means = st_era_means = [None] * 2
    #     if use_scatter:
    #         orig_scatter_seasonal = orig_scatter_seasonal[-1].nanmean(-1).tolist()
    #         corr_scatter_seasonal = corr_scatter_seasonal[-1].nanmean(-1).tolist()
    #         era_scatter_seasonal = era_scatter_seasonal[-1].nanmean(-1).tolist()
    #     else:
    #         orig_scatter_seasonal = corr_scatter_seasonal = era_scatter_seasonal = [None] * 2
    #     era_means = acc.data['wrf_corr-era'].mean([0, -1]).tolist()
    #     era_orig_means = acc.data['wrf_orig-era'].mean([0, -1]).tolist()
    #     test_loss = era_means + st_means + corr_scatter_seasonal + \
    #                 [acc.data['mesoscale_loss'].mean().item()]
    #     test_orig_loss = era_orig_means + st_orig_means + \
    #                      orig_scatter_seasonal + [0]
    #     era_loss = st_era_means + era_scatter_seasonal
    #     # test_loss = wrf_era_mean_loss[1].mean(0).tolist() + wrf_st_mean_loss[1].mean(0).tolist() + \
    #     #             corr_scatter_seasonal[-1].nanmean(-1).tolist() + [acc.data['mesoscale_loss'].mean().item()]
    #     # test_orig_loss = wrf_era_mean_loss[0].mean(0).tolist() + wrf_st_mean_loss[0].mean(0).tolist() + \
    #     #                  orig_scatter_seasonal[-1].nanmean(-1).tolist() + [0]

    #     a = True
    #     if a:
    #         df = pd.DataFrame([test_loss], columns=['era_u10', 'era_v10', 'era_t2', 'st_t2', 'st_w10', 'sc_u10',
    #                                                 'sc_v10', 'mesoscale_loss'])
    #         df.to_csv(os.path.join(logger.save_dir, 'mean_losses'))
    #         df = pd.DataFrame([test_orig_loss], columns=['era_u10', 'era_v10', 'era_t2', 'st_t2', 'st_w10', 'sc_u10',
    #                                                      'sc_v10', 'mesoscale_loss'])
    #         df.to_csv(os.path.join(logger.save_dir, 'mean_orig_losses'))
    #         df = pd.DataFrame([era_loss], columns=['st_t2', 'st_w10', 'sc_u10', 'sc_v10'])
    #         df.to_csv(os.path.join(logger.save_dir, 'mean_era_losses'))
    #     df_stat.to_csv(os.path.join(logger.save_dir, 'nz_stats'))
    return res_dict


def angle_to_sector_class(angle: torch.Tensor, num_sectors: int = 16) -> torch.Tensor:
    """
    Convert meteorological wind direction in degrees [0,360)
    into one of `num_sectors` integer classes [0..num_sectors-1],
    each centered at multiples of 360/num_sectors.
    """
    sector_size = 360.0 / num_sectors
    # shift by half a sector so boundaries fall midway between centers
    idx = torch.floor((angle + sector_size/2) / sector_size)
    return (idx % num_sectors).long()

def split_uvt_to_speed_temp_and_dir(data: torch.Tensor, num_sectors: int = 16):
    """
    Given data of shape [..., 3, N] with channels (u, v, t):
      - compute speed = sqrt(u^2+v^2)
      - keep temperature = t
      - compute meteorological wind direction (from which it blows)
        in degrees: angle = atan2(-u, -v) → convert to [0,360)
      - map angle → discrete sector class [0..num_sectors-1]
    
    Returns:
        data_station_wt: Tensor [..., 2, N] (speed, temperature)
        dir_class:       LongTensor [..., N]  (0..num_sectors-1)
    """
    u = data[..., 0, :]
    v = data[..., 1, :]
    t = data[..., 2, :]
    
    speed = torch.sqrt(u**2 + v**2)
    temperature = t
    
    # atan2 returns radians; meteorological direction is "from" north-clockwise
    angle = (torch.atan2(-u, -v) * 180.0 / torch.pi + 360.0) % 360.0
    dir_class = angle_to_sector_class(angle, num_sectors).unsqueeze(-2)
    
    data_station_wt = torch.stack((speed, temperature), dim=-2)
    return data_station_wt, dir_class

def append_nz_wind_statistics(res_dict, models_dict, channel_dim=-3):
    nz_polygon_array = get_novaya_zemlya_mask()
    for name in models_dict:
        model_data = uvt_to_wt(models_dict[name], channel_dim)
        model_nz = nz_polygon_array * model_data.cpu()
        model_wind = model_nz.select(channel_dim, 0).numpy()
        res_dict[name + '_nz_mean'] = np.nanmean(model_wind, axis=(-2, -1)).tolist()
        res_dict[name + '_nz_mean_sq'] = np.nanmean(model_wind ** 2, axis=(-2, -1)).tolist()
        res_dict[name + '_nz_median'] = np.nanmedian(model_wind, axis=(-2, -1)).tolist()
        res_dict[name + '_nz_std'] = np.nanstd(model_wind, axis=(-2, -1)).tolist()
        res_dict[name + '_nz_percentile_2'] = np.nanpercentile(model_wind, 2, axis=(-2, -1)).tolist()
        res_dict[name + '_nz_percentile_98'] = np.nanpercentile(model_wind, 98, axis=(-2, -1)).tolist()
    # df = pd.concat([df, pd.DataFrame(res)], ignore_index=True)
    return res_dict


def calc_station_loss(wrf, stations, interpolator, loss):
    s = wrf.shape
    wrf_interpolated = interpolator(wrf.flatten(-2, -1))
    # wrf_interpolated.shape == 4, bs, 3, 46 ; stations.shape == 4, bs, 2, 46

    t2_loss = loss(wrf_interpolated[..., 2, :], stations[..., 1, :])
    wspd = torch.sqrt(torch.square(wrf_interpolated[..., 0, :]) + torch.square(wrf_interpolated[..., 1, :]))
    w10_loss = loss(wspd, stations[..., 0, :])

    return torch.stack((t2_loss, w10_loss), dim=-2)  # sl, bs, c, N_stations


def calculate_station_metric(input_orig, input_corr, stations, interpolator_orig, interpolator_corr, loss):
    orig_loss = calc_station_loss(input_orig, stations, interpolator_orig, loss)
    corr_loss = calc_station_loss(input_corr, stations, interpolator_corr, loss)

    metric = _metric(orig_loss, corr_loss)
    mean_by_time = orig_loss.mean((0, 1)), corr_loss.mean((0, 1)), metric.mean((0, 1))  # N_stations (46), 2
    mean_by_space = orig_loss.mean(-2).flatten(-2, -1), corr_loss.mean(-2).flatten(-2, -1), \
                    metric.mean(-2).flatten(-2, -1)  # bs*4, 2
    return mean_by_space, mean_by_time


def calculate_era_loss(wrf, era, meaner, criterion):
    wrf_orig = meaner(wrf)
    era = era.flatten(-2, -1)
    era = era[..., meaner.mapping.unique().long()]
    loss = criterion(wrf_orig, era)
    return loss  # loss.shape = 4, 1, 3, 8744 i.e. sl, bs, c, N


def calculate_era_metric(wrf_orig, wrf_corr, era, meaner, criterion):
    loss_orig = calculate_era_loss(wrf_orig, era, meaner, criterion)
    loss_corr = calculate_era_loss(wrf_corr, era, meaner, criterion)
    metric = _metric(loss_orig, loss_corr)
    return loss_orig, loss_corr, metric


def get_meaned_metrics(wrf_orig, wrf_corr, era, meaner, criterion):
    # loss_orig.shape = bs, 4, 3, N
    loss_orig, loss_corr, metric = calculate_era_metric(wrf_orig, wrf_corr, era, meaner, criterion)
    mean_by_time = loss_orig.mean((0, 1)), loss_corr.mean((0, 1)), metric.mean((0, 1))  # 3, N (8744)
    mean_by_space = loss_orig.mean(-1).flatten(0, 1), loss_corr.mean(-1).flatten(0, 1), \
                    metric.mean(-1).flatten(0, 1)  # bs * 4, 3
    return mean_by_space, mean_by_time


def _metric(orig, corr):
    return (orig - corr) / orig


def get_season(month):
    return month // 3 % 4


def get_season_mean_losses(orig, corr, month, sl=4):
    seasons = get_season(month)
    orig_means_by_t, corr_means_by_t = [], []
    orig_means, corr_means = [], []
    for cur_season in range(4):
        i = torch.where(seasons == cur_season)[0] * sl
        season_ids = torch.cat([i + j for j in range(sl)])

        orig_means_by_t.append(torch.nanmean(orig[season_ids], dim=0))
        corr_means_by_t.append(torch.nanmean(corr[season_ids], dim=0))
        orig_means.append(torch.nanmean(orig[season_ids], dim=[0, -1]))
        corr_means.append(torch.nanmean(corr[season_ids], dim=[0, -1]))
    losses_meaned_by_t = list(map(torch.stack, [orig_means_by_t, corr_means_by_t]))
    losses_mean = list(map(torch.stack, [orig_means, corr_means]))
    return losses_mean, losses_meaned_by_t


def get_season_mean_scatter(losses, counts, month):
    seasons = get_season(month)
    losses_means = []
    for cur_season in range(4):
        season_ids = torch.where(seasons == cur_season)[0]
        means = losses[season_ids].sum(0) / counts[season_ids].sum(0)
        means[means == torch.inf] = torch.nan
        losses_means.append(means)
    means = losses.sum(0) / counts.sum(0)
    means[means == torch.inf] = torch.nan
    losses_means.append(means)
    return torch.stack(losses_means)


def era_vector_to_map(era_vector, meaner, era_map_shape=None):
    era_map_shape = torch.Size([era_map_shape]) if era_map_shape is not None else torch.Size([67 * 215])
    base = torch.zeros([*era_vector.shape[:-1] + era_map_shape])
    base[..., meaner.mapping.unique().long().cpu()] = era_vector.float()
    return base

def input_to_era_map(data, meaner, era_map_shape=None):
    era_map_shape = torch.Size(era_map_shape) if era_map_shape is not None else torch.Size([67, 215])
    out = (meaner(data, masked=False)*torch.where(meaner.mask, 1, torch.nan)).unflatten(-1, era_map_shape)
    return out

def input_to_stations(data, interpolator):
    return interpolator(data.flatten(-2, -1))

def input_to_scatter(data, interpolator, scatter_times, data_dates, mask=None, distance_criterion=0.15):
    data_on_scat_grid = interpolator(data.flatten(-2, -1)[..., :2, :]).unflatten(dim=-1, sizes=scatter_times.shape[-2:])
    data_on_scat_grid = interp_nwp_in_time(data_on_scat_grid, scatter_times, data_dates)
    mask = interpolator.calc_input_tensor_mask(scatter_times.shape[-2:], 
                                               distance_criterion=distance_criterion,
                                               fill_value=torch.nan) if mask is None else mask
    data_on_scat_grid = data_on_scat_grid * mask
    return data_on_scat_grid

def calc_era5_error_map(wrf, era, meaner):
    t = era.flatten(-2, -1)
    t = t[..., meaner.mapping.unique().long()]
    err = torch.nn.L1Loss(reduction='none')(t, meaner(wrf))
    base = torch.zeros_like(era.flatten(-2, -1))
    base[..., meaner.mapping.unique().long()] = err.float()
    return base


def calc_scatter_error_map(data, scatter, criterion, scatter_times, data_dates, interpolator, input_mask):
    corr_on_scat_grid = interpolator(data.flatten(-2, -1)).unflatten(dim=-1, sizes=scatter.shape[-2:])[:, :, :2]
    # interpolate nwp in time
    corr_on_scat_grid = interp_nwp_in_time(corr_on_scat_grid, scatter_times, data_dates, return_counts=True)
    corr_on_scat_grid = corr_on_scat_grid #* self.wrf_mask
    # filter NaNs
    mask = (torch.isfinite(corr_on_scat_grid)) & (torch.isfinite(scatter))

    # num_valid = mask.sum().item()
    # corr_on_scat_grid = corr_on_scat_grid[mask]     # 1D tensor of only the finite entries
    # scatter = scatter[mask]
    err = criterion(corr_on_scat_grid, scatter)

    # # todo по идее правильно спрашивать criterion по которому считаем лосс, а функцию интерполяции брать самому изнутри
    # # wrf_scattered.shape == 1, 2, 2, 56760 == bs, t, c, h*w ; counts.shape == 1, 2, 56760
    # data_scattered, scatter, counts = interpolate_input_to_scat(data[..., :2, :, :], scatter, interpolator,
    #                                                             i, start_date, input_mask, return_counts=True)
    # # scatter.shape == 1, 2, 4, 132, 430 ; err.shape == 1, 2, 2, 56760
    # # err = torch.nn.L1Loss(reduction='none')(data_scattered, scatter)
    # err = criterion(data_scattered, scatter)
    # assert not torch.isnan(err).any()
    return err, mask.sum(dim=1)


def get_power_spectrum(image):
    s = image.shape
    h, w = image.shape[-2:]
    fourier_image = np.fft.fftn(image, axes=(-2, -1))
    fourier_amplitudes = np.abs(fourier_image) ** 2
    kfreqh = np.fft.fftfreq(h) * h
    kfreqw = np.fft.fftfreq(w) * w
    kfreq2D = np.meshgrid(kfreqw, kfreqh)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.reshape(np.prod(fourier_amplitudes.shape[:-2]), h * w)
    kbins = np.arange(0.5, min(h, w) // 2 + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic="mean",
                                         bins=kbins)
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    Abins = Abins.reshape(*s[:-2], -1)
    return kvals, Abins


class LossesAccumulator:
    def __init__(self, names):
        self.data = {names[i]: [] for i in range(len(names))}

    def cat_accumulate_losses(self, names, losses):
        for i, name in enumerate(names):
            if type(losses[i]) is not torch.Tensor:
                losses[i] = torch.tensor([losses[i]])
            self.data[names[i]].append(losses[i].cpu())

    def sum_accumulate_losses(self, names, losses):
        for i in range(len(names)):
            if len(self.data[names[i]]) == 0:
                self.data[names[i]] = losses[i].cpu()
            else:
                self.data[names[i]] += losses[i].cpu()

    def cat_losses(self, names):
        for name in names:
            self.data[name] = torch.cat(self.data[name])

    def save_data(self, dir_path, keys=None):
        keys = keys if keys is not None else self.data.keys()
        for name in keys:
            torch.save(self.data[name], os.path.join(dir_path, f'{name}'))
