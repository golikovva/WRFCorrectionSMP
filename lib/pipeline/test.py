import sys
import os
import random

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
from lib.helpers.metrics import NormSSIM, NamedDictMetric, normalized, channel_meaned, MeanerMetric, MulticlassAccuracy
from lib.helpers.aggregators import SpatialAggregator, AverageAggregator, SeasonalSpatialAggregator
from lib.helpers.ssim import CustomSSIM


def test(model, losses, wrf_scaler, era_scaler, dataloader, logger, cfg):
    debug_mode = False
    img_format = 'pdf'
    for channel in ['u10', 'v10', 't2', 'era', 'stations', 'scatter']:
        os.makedirs(os.path.join(logger.save_dir, 'plots', channel), exist_ok=True)
    with torch.no_grad():
        model.eval()
        losses_to_cat = ['year', 'month', 'day', 'hour', 'mesoscale_loss',  # 'wrf_orig', 'wrf_corr',
                         'wrf_orig-era', 'wrf_corr-era', 'wrf_orig-stations', 'wrf_corr-stations',
                         'wrf_orig-scatter', 'wrf_corr-scatter', 'wrf-scatter-counts',
                         'era-stations', 'era-scatter', 'era-scatter-counts',
                         'power_spectrum-era', 'power_spectrum-wrf', 'power_spectrum-corr'
                         ]
        acc = LossesAccumulator(names=losses_to_cat)
        dataset = dataloader.dataset
        stat = {}
        df_stat = pd.DataFrame()
        diff = DiffLoss(reduction='none')
        mae = torch.nn.L1Loss(reduction='none')
        mse = torch.nn.MSELoss(reduction='none')

        metrics_dict = {
            'mesoscale_loss': NamedDictMetric(SmallScaleLoss(reduction='none', device=cfg.device), ['wrf', 'corr']),
            'era_mse': NamedDictMetric(mse, ['corr', 'era_up']),
            'era_mae': NamedDictMetric(mae, ['corr', 'era_up']),
            'orig_era_mse': NamedDictMetric(mse, ['wrf', 'era_up']),
            'orig_era_mae': NamedDictMetric(mae, ['wrf', 'era_up']),

            'mean_era_mse': NamedDictMetric(mse, ['corr_meaned', 'era']),
            'mean_era_mae': NamedDictMetric(mae, ['corr_meaned', 'era']),
            'mean_orig_era_mse': NamedDictMetric(mse,['wrf_meaned', 'era']),
            'mean_orig_era_mae': NamedDictMetric(mae, ['wrf_meaned', 'era']),

            'orig_ssim_era': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=False, channel=3).forward),
                                             ['wrf', 'era_up', 'era_up']),
            'orig_ssim_custom': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=False, channel=3, exp_coefs=(1, 1, 1)).forward),
                                               ['wrf', 'wrf', 'era_up']),
            'orig_ssim_custom_211': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=False, channel=3, exp_coefs=(2, 1, 1)).forward), 
                                                    ['wrf', 'wrf', 'era_up']),
            'ssim_custom_211': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=False, channel=3, exp_coefs=(2, 1, 1)).forward),
                                               ['corr', 'wrf', 'era_up']),
            'ssim_wrf011': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=False, channel=3,
                                                                 exp_coefs=(0, 1, 1)).forward),
                                           ['corr', 'wrf', 'wrf']),
            'ssim_wrf': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=False, channel=3, exp_coefs=(1, 1, 1)).forward),
                                        ['corr', 'wrf', 'wrf']),
            'ssim_era': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=False, channel=3, exp_coefs=(1, 1, 1)).forward),
                                        ['corr', 'era_up', 'era_up']),
            'ssim_custom': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=False, channel=3, exp_coefs=(1, 1, 1)).forward),
                                           ['corr', 'wrf', 'era_up']),

            'stations_mse': NamedDictMetric(mse, ['corr_stations_wt', 'stations_wt']),
            'stations_mae': NamedDictMetric(mae, ['corr_stations_wt', 'stations_wt']),
            'orig_stations_mse': NamedDictMetric(mse, ['wrf_stations_wt', 'stations_wt']),
            'orig_stations_mae': NamedDictMetric(mae, ['wrf_stations_wt', 'stations_wt']),
            'era_stations_mse': NamedDictMetric(mse, ['era_stations_wt', 'stations_wt']),
            'era_stations_mae': NamedDictMetric(mae, ['era_stations_wt', 'stations_wt']),
            'stations_accuracy': NamedDictMetric(MulticlassAccuracy(), ['corr_stations_dir', 'stations_dir']),
            'orig_stations_accuracy': NamedDictMetric(MulticlassAccuracy(), ['wrf_stations_dir', 'stations_dir']),
            'era_stations_accuracy': NamedDictMetric(MulticlassAccuracy(), ['era_stations_dir', 'stations_dir']),
            'scatter_mse': NamedDictMetric(mse, ['corr_scatter', 'scatter']),
            'scatter_mae': NamedDictMetric(mae, ['corr_scatter', 'scatter']),
            'orig_scatter_mse': NamedDictMetric(mse, ['wrf_scatter', 'scatter']),
            'orig_scatter_mae': NamedDictMetric(mae, ['wrf_scatter', 'scatter']),
            'era_scatter_mse': NamedDictMetric(mse, ['era_scatter', 'scatter']),
            'era_scatter_mae': NamedDictMetric(mae, ['era_scatter', 'scatter']),
            'wrf_spectrum': NamedDictMetric(torch.nn.Identity(), ['wrf_spectrum']),
            'era_spectrum': NamedDictMetric(torch.nn.Identity(), ['era_spectrum']),
            'corr_spectrum': NamedDictMetric(torch.nn.Identity(), ['corr_spectrum']),
        }
        use_scatter = hasattr(dataloader.dataset.datasets[3], 'src_grid')
        use_station = hasattr(dataloader.dataset.datasets[2], 'src_grid')
        wrf_grid, era_grid = dataloader.dataset.datasets[0].src_grid, dataloader.dataset.datasets[1].src_grid

        era_coords = np.stack([era_grid['longitude'].flatten(), era_grid['latitude'].flatten()]).T
        wrf_coords = np.stack([wrf_grid['longitude'].flatten(), wrf_grid['latitude'].flatten()]).T

        era_upsampler = InvDistTree(x=era_coords, q=wrf_coords, device=cfg.device)

        if use_scatter:
            scat_grid = dataloader.dataset.datasets[3].src_grid
            scat_coords = np.stack([scat_grid['longitude'].flatten(), scat_grid['latitude'].flatten()]).T
            scatter_interpolator = InvDistTree(x=wrf_coords, q=scat_coords, device=cfg.device)
            era_scatter_interpolator = InvDistTree(x=era_coords, q=scat_coords, device=cfg.device)
        if use_station:
            station_grid = dataloader.dataset.datasets[2].src_grid
            station_coords = np.stack([station_grid['longitude'].flatten(), station_grid['latitude'].flatten()]).T
            interpolator = InvDistTree(x=wrf_coords, q=station_coords, device=cfg.device)
            era_interpolator = InvDistTree(x=era_coords, q=station_coords, device=cfg.device)
        t = 0
        
        aggregators = [SpatialAggregator(), ]
        results = {metric_name: {agg.__class__.__name__: None for agg in aggregators} for metric_name in metrics_dict}

        for test_data, test_label, station, scatter, dates in tqdm(dataloader):
            test_data = torch.swapaxes(test_data.type(torch.float).to(cfg.device), 0, 1).contiguous()
            test_label = torch.swapaxes(test_label.type(torch.float).to(cfg.device), 0, 1)
            era_h, era_w = test_label.shape[-2:]

            if use_station:
                station = torch.permute(station.type(torch.float).to(cfg.device), (1, 0, 3, 2))

            if use_scatter:
                batch_dates = torch.as_tensor(dates.astype('datetime64[s]').astype('float64')).to(cfg.device)
                scatter_times = scatter[0].to(cfg.device).type(torch.double)
                scatter_data = torch.stack((scatter[1], scatter[2]), dim=-3).type(torch.float).to(cfg.device)

            date = dates.astype(str)
            year = dates.astype('datetime64[Y]').astype(int) + 1970
            month = dates.astype('datetime64[M]').astype(int) % 12 + 1
            day = (dates.astype('datetime64[D]') - dates.astype('datetime64[M]')).astype(int) + 1
            hour = (dates - dates.astype('datetime64[D]')).astype("m8[h]").astype(int)
            if debug_mode:
                month = random.randint(1, 12)
            test_data = wrf_scaler.transform(test_data, dims=2)

            if 'lfreq' in cfg.model_type:
                _, l_freq_corr, h_freq = model(test_data)
                output = l_freq_corr + h_freq
            else:
                output = model(test_data)

            output = era_scaler.inverse_transform(output, dims=2)
            test_data = wrf_scaler.inverse_transform(test_data, dims=2)[:, :, :3]

            mesoscale_loss = losses(output, test_data, expanded_out=True)[2].item()

            # wrf era difference
            # orig_era = calculate_era_loss(test_data, test_label, losses.meaner, rmse).flatten(0, 1)
            # corr_era = calculate_era_loss(output, test_label, losses.meaner, rmse).flatten(0, 1)


            # # wrf scatter difference
            # if use_scatter:
            #     orig_scatter, orig_counts = calc_scatter_error_map(test_data, scatter_data, rmse, scatter_times, batch_dates,
            #                                                        scatter_interpolator, losses.wrf_mask)
            #     corr_scatter, corr_counts = calc_scatter_error_map(output, scatter_data, rmse, scatter_times, batch_dates,
            #                                                        scatter_interpolator, losses.wrf_mask) 
            #     era_scatter, era_counts = calc_scatter_error_map(test_label, scatter_data, rmse, scatter_times, batch_dates,
            #                                                        scatter_interpolator, losses.wrf_mask)
            #     era_scatter = era_scatter.sum(1)
            #     orig_scatter = orig_scatter.sum(1)
            #     corr_scatter = corr_scatter.sum(1)
            # else:
            #     orig_scatter, orig_counts = torch.nan, torch.nan
            #     corr_scatter, corr_counts = torch.nan, torch.nan
            #     era_scatter, era_counts = torch.nan, torch.nan

            # if use_station:
            #     orig_stations = calc_station_loss(test_data, station, interpolator, rmse).flatten(0, 1) if use_station else None
            #     corr_stations = calc_station_loss(output, station, interpolator, rmse).flatten(0, 1) if use_station else None
            #     era_stations = calc_station_loss(test_label, station, era_interpolator, rmse).flatten(0, 1) if use_station else None
            # else:
            #     orig_stations, corr_stations, era_stations = torch.nan, torch.nan, torch.nan
            
            # print(era_stations.shape, era_scatter.shape, era_counts.shape, 'era st sc shape')
            # ========== Interpolate WRF to others ================
            wrf_meaned = input_to_era_map(test_data, losses.meaner, era_map_shape=(era_h, era_w))
            corr_meaned = input_to_era_map(output, losses.meaner, era_map_shape=(era_h, era_w))

            stations_wt = station[..., [0, 2], :]
            stations_dir = station[..., 1, :].unsqueeze(-2)
            # print(torch.unique(stations_dir, return_inverse=False, return_counts=False), 'station dir unique values')
            wrf_stations = input_to_stations(test_data, interpolator)
            wrf_stations_wt, wrf_stations_dir = split_uvt_to_speed_temp_and_dir(wrf_stations)
            corr_stations = input_to_stations(output, interpolator)
            corr_stations_wt, corr_stations_dir = split_uvt_to_speed_temp_and_dir(corr_stations)
            # print(stations_dir[0,0,:2], 'station dir first value')
            # print(corr_stations_dir[0,0], 'corr stations dir first value')
            # print(corr_stations[0,0,:2], 'corr stations first value')

            era_stations = input_to_stations(test_label, era_interpolator) 
            era_stations_wt, era_stations_dir = split_uvt_to_speed_temp_and_dir(era_stations)

            scatter_mask = scatter_interpolator.calc_input_tensor_mask(scatter_times.shape[-2:], 
                                                                       distance_criterion=0.15,
                                                                       fill_value=torch.nan)
            wrf_scatter = input_to_scatter(test_data, scatter_interpolator, scatter_times, batch_dates, mask=scatter_mask)
            corr_scatter = input_to_scatter(output, scatter_interpolator, scatter_times, batch_dates, mask=scatter_mask)

            era_scatter = input_to_scatter(test_label, era_scatter_interpolator, scatter_times, batch_dates, mask=scatter_mask)
            era_upsampled = era_upsampler(test_label.flatten(-2, -1)).view(test_data.shape)
            spectrum_bins, era_spectrum = get_power_spectrum(uvt_to_wt(era_upsampled, -3).squeeze().cpu())

            wrf_spectrum = get_power_spectrum(uvt_to_wt(test_data, -3).squeeze().cpu())[1]
            corr_spectrum = get_power_spectrum(uvt_to_wt(output, -3).squeeze().cpu())[1]
            era_spectrum, wrf_spectrum, corr_spectrum = map(torch.from_numpy,
                                                            [era_spectrum, wrf_spectrum, corr_spectrum])

            samples_dict = {'wrf': test_data,
                            'era_up': era_upsampled,
                            'era': test_label,
                            'corr': output,
                            'wrf_meaned': wrf_meaned,
                            'corr_meaned': corr_meaned,
                            # 'wrf_stations': wrf_stations.squeeze(),
                            # 'corr_stations': corr_stations.squeeze(),
                            # 'era_stations': era_stations.squeeze(),
                            'wrf_stations_wt': wrf_stations_wt,
                            'wrf_stations_dir': wrf_stations_dir,
                            'corr_stations_wt': corr_stations_wt,
                            'corr_stations_dir': corr_stations_dir,
                            'era_stations_wt': era_stations_wt,
                            'era_stations_dir': era_stations_dir,
                            'wrf_scatter': wrf_scatter,
                            'corr_scatter': corr_scatter,
                            'era_scatter': era_scatter,
                            # 'stations': station.squeeze(),
                            'stations_wt': stations_wt,
                            'stations_dir': stations_dir,
                            'scatter': scatter_data,
                            'wrf_spectrum': wrf_spectrum,
                            'corr_spectrum': corr_spectrum,
                            'era_spectrum': era_spectrum,
                            }
            for k in samples_dict:
                print(k, samples_dict[k].shape)
            save_samples = False
            if save_samples:
                np.save(os.path.join(logger.save_dir, 'plots', f'corr_sample_{date[0]}.npy'),
                        samples_dict['corr'].cpu().numpy())
                np.save(os.path.join(logger.save_dir, 'plots', f'wrf_sample_{date[0]}.npy'),
                        samples_dict['wrf'].cpu().numpy())
                np.save(os.path.join(logger.save_dir, 'plots', f'era_sample_{date[0]}.npy'),
                        samples_dict['era_up'].cpu().numpy())

            
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
        # acc.cat_losses(losses_to_cat)
        l = [(metric_name, channel) for metric_name in metrics_dict for channel in ['u10', 'v10', 't2']]
        for item in l:
            if ('scatter' in item[0]) and ('t2' in item[1]):
                l.remove(item)
            if ('spectrum' in item[0]) and ('v10' in item[1]):
                l.remove(item)
            if ('station' in item[0]) and ('v10' in item[1]):
                l.remove(item)
        for item in l:        
            if ('accuracy' in item[0]) and ('t2' in item[1]):
                # print(f'Removing {item} from metrics')
                l.remove(item)
        metrics_df = pd.DataFrame(
            torch.cat([AverageAggregator.finalize(results[metric_name]['SpatialAggregator']) for metric_name in metrics_dict]).cpu().numpy()[None],
            # torch.cat([metrics_dict[metric_name].compute() for metric_name in metrics_dict]).cpu().numpy()[None],
            columns=l, index=[logger.experiment_number])
        metrics_df.columns = pd.MultiIndex.from_tuples(metrics_df.columns, names=['metric', 'channel'])
        metrics_df.to_csv(os.path.join(logger.save_dir, 'experiment_metrics'))

        era_spectrum = SpatialAggregator.finalize(results['era_spectrum']['SpatialAggregator'])
        wrf_spectrum = SpatialAggregator.finalize(results['wrf_spectrum']['SpatialAggregator'])
        corr_spectrum = SpatialAggregator.finalize(results['corr_spectrum']['SpatialAggregator'])
        for i, c in enumerate(['w10', 't2']):
            spectrum_plot = plot_utils.power_loglog_spectrum([era_spectrum[i], wrf_spectrum[i], corr_spectrum[i]],
                                                             ['era5', 'wrf', 'wrf_corr'], spectrum_bins, name=c)
            plt.savefig(os.path.join(logger.save_dir, 'plots', f'{c}_spectrum_plot'), dpi=300, bbox_inches="tight", format="pdf",)
        plt.close('all')
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
    # return test_loss


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
    data_on_scat_grid = interpolator(data.flatten(-2, -1)).unflatten(dim=-1, sizes=scatter_times.shape[-2:])[:, :, :2]
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
