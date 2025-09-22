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
from lib.models.loss import RMSELoss, DiffLoss, uvt_to_wt
from lib.helpers.interpolation import InvDistTree, interpolate_input_to_scat
from lib.helpers.metrics import  NamedDictMetric, normalized, channel_meaned, MeanerMetric
from lib.helpers.ssim import CustomSSIM


def test(model, losses, wrf_scaler, era_scaler, dataloader, logger, cfg):
    debug_mode = False
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
        rmse = RMSELoss(reduction='none')

        metrics_dict = {
            'mean_era_mse': NamedDictMetric(channel_meaned(MeanerMetric(losses.meaner, mse), -2), ['corr', 'era']),
            'mean_era_mae': NamedDictMetric(channel_meaned(MeanerMetric(losses.meaner, mae), -2), ['corr', 'era']),
            'mean_orig_era_mse': NamedDictMetric(channel_meaned(MeanerMetric(losses.meaner, mse), -2), ['wrf', 'era']),
            'mean_orig_era_mae': NamedDictMetric(channel_meaned(MeanerMetric(losses.meaner, mae), -2), ['wrf', 'era']),
            'orig_era_mse': NamedDictMetric(channel_meaned(mse), ['wrf', 'era_up']),
            'orig_era_mae': NamedDictMetric(channel_meaned(mae), ['wrf', 'era_up']),
            'orig_ssim_era': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=True, channel=3).forward),
                                             ['wrf', 'era_up', 'era_up']),
            'orig_ssim_custom': NamedDictMetric(
                normalized(CustomSSIM(data_range=1, size_average=True, channel=3).forward),
                ['wrf', 'wrf', 'era_up']),
            'era_mse': NamedDictMetric(channel_meaned(mse), ['corr', 'era_up']),
            'era_mae': NamedDictMetric(channel_meaned(mae), ['corr', 'era_up']),
            'ssim_wrf001': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=True, channel=3,
                                                                 exp_coefs=(0, 1, 1)).forward),
                                           ['corr', 'wrf', 'wrf']),
            'ssim_wrf': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=True, channel=3).forward),
                                        ['corr', 'wrf', 'wrf']),
            'ssim_era': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=True, channel=3).forward),
                                        ['corr', 'era_up', 'era_up']),
            'ssim_custom': NamedDictMetric(normalized(CustomSSIM(data_range=1, size_average=True, channel=3).forward),
                                           ['corr', 'wrf', 'era_up']),
        }

        metadata = dataset.metadata
        era_coords = np.stack([metadata['era_xx'].flatten(), metadata['era_yy'].flatten()]).T
        scat_coords = np.stack([metadata['scat_xx'].flatten(), metadata['scat_yy'].flatten()]).T
        wrf_coords = np.stack([metadata['wrf_xx'].flatten(), metadata['wrf_yy'].flatten()]).T
        era_upsampler = InvDistTree(x=era_coords, q=wrf_coords, device=cfg.device)
        interpolator = InvDistTree(x=wrf_coords, q=metadata['coords'], device=cfg.device)
        era_interpolator = InvDistTree(x=era_coords, q=metadata['coords'], device=cfg.device)
        era_scatter_interpolator = InvDistTree(x=era_coords, q=scat_coords, device=cfg.device)
        t = 0
        for test_data, test_label, station, scatter, date_id in tqdm(dataloader, total=len(dataloader) // 4):
            test_data = torch.swapaxes(test_data.type(torch.float).to(cfg.device), 0, 1)
            test_label = torch.swapaxes(test_label.type(torch.float).to(cfg.device), 0, 1)
            station = torch.permute(station.type(torch.float).to(cfg.device), (1, 0, 3, 2))[..., [3, 1], :]
            scatter = scatter.to(cfg.device)

            datefile_id, hour = dataset.get_path_id(date_id)
            datefile_id, hour = datefile_id.item(), hour.item()
            date = dataset.wrf_files[datefile_id].split('_')[-2]
            year, month, day = map(int, date.split('-'))
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

            # конкатенируем и сохраняем все ошибки, потом их интерпретируем. требует много места на диске тк
            # сохраняются массивы соразмерные размеру датасета
            # orig ошибку нужно сохранить лишь раз а затем подгружать и пользоваться

            # wrf era difference
            orig_era = calculate_era_loss(test_data, test_label, losses.meaner, rmse).flatten(0, 1)
            corr_era = calculate_era_loss(output, test_label, losses.meaner, rmse).flatten(0, 1)

            # wrf stations difference
            orig_stations = calc_station_loss(test_data, station, interpolator, rmse).flatten(0, 1)
            corr_stations = calc_station_loss(output, station, interpolator, rmse).flatten(0, 1)
            # print(orig_stations.shape, corr_stations.shape, 'orig corr st shape')

            # wrf scatter difference
            orig_scatter, orig_counts = calc_scatter_error_map(test_data, scatter, rmse, date_id,
                                                               metadata['start_date'],
                                                               losses.scatter_interpolator, losses.wrf_mask)
            corr_scatter, corr_counts = calc_scatter_error_map(output, scatter, rmse, date_id, metadata['start_date'],
                                                               losses.scatter_interpolator, losses.wrf_mask)
            # print(orig_scatter.shape, corr_scatter.shape, corr_counts.shape, 'orig corr sc shape')
            # era stations difference
            era_stations = calc_station_loss(test_label, station, era_interpolator, rmse).flatten(0, 1)
            # era scatter difference
            era_scatter, era_counts = calc_scatter_error_map(test_label, scatter, rmse, date_id, metadata['start_date'],
                                                             era_scatter_interpolator, losses.wrf_mask)
            # print(era_stations.shape, era_scatter.shape, era_counts.shape, 'era st sc shape')
            era_upsampled = era_upsampler(test_label.flatten(-2, -1)).view(test_data.shape)
            spectrum_bins, era_spectrum = get_power_spectrum(uvt_to_wt(era_upsampled, -3).squeeze().cpu())
            wrf_spectrum = get_power_spectrum(uvt_to_wt(test_data, -3).squeeze().cpu())[1]
            corr_spectrum = get_power_spectrum(uvt_to_wt(output, -3).squeeze().cpu())[1]
            era_spectrum, wrf_spectrum, corr_spectrum = map(torch.from_numpy,
                                                            [era_spectrum, wrf_spectrum, corr_spectrum])
            samples_dict = {'wrf': test_data.squeeze(),
                            'era_up': era_upsampled.squeeze(),
                            'era': test_label.squeeze(),
                            'corr': output.squeeze()}
            for metric_name in metrics_dict:
                metrics_dict[metric_name].update(samples_dict)

            stat = append_nz_wind_statistics(stat, {'era': era_upsampled, 'wrf': test_data, 'corr': output})
            df_stat = pd.concat([df_stat, pd.DataFrame(stat)], ignore_index=True)
            acc.cat_accumulate_losses(names=losses_to_cat, losses=[year, month, day, hour, mesoscale_loss,
                                                                   # test_data, output,
                                                                   orig_era, corr_era,
                                                                   orig_stations, corr_stations,
                                                                   orig_scatter.sum(1), corr_scatter.sum(1),
                                                                   orig_counts,
                                                                   era_stations, era_scatter.sum(1), era_counts,
                                                                   era_spectrum, wrf_spectrum, corr_spectrum])

            if cfg.test_config.draw_plots:
                if date_id % 756 == 0:
                    station_metric = _metric(torch.nanmean(orig_stations, dim=[0, -1]), torch.nanmean(corr_stations, dim=[0, -1]))
                    era_metric = _metric(orig_era.mean((0, -1)), corr_era.mean((0, -1)))
                    save_output_sample=True
                    if save_output_sample:
                        torch.save(output, os.path.join(logger.save_dir, 'plots', 't2', f'sample_{date}_{hour}'))
                    for i, channel in enumerate(['u10', 'v10', 't2']):
                        st_m = station_metric[0].item() if channel == 't2' else station_metric[1].item()
                        
                        simple_plot = plot_utils.draw_simple_plots(test_data, output, test_label, i,
                                                                   orig_era.mean().item(),
                                                                   corr_era.mean().item(),
                                                                   era_metric[i].item(), st_m,
                                                                   f'{date} {hour}:00')
                        plt.savefig(os.path.join(logger.save_dir, 'plots', channel, f'plot_{date}_{hour}'))
                if date_id == len(dataset) - 1:
                    station_metric = _metric(torch.nanmean(orig_stations, dim=[0, -1]), torch.nanmean(corr_stations, dim=[0, -1]))
                    for i, channel in enumerate(['u10', 'v10', 't2']):
                        st_m = station_metric[0].item() if channel == 't2' else station_metric[1].item()
                        era_metric = _metric(orig_era.mean((0, -1)), corr_era.mean((0, -1)))
                        mega_plot = plot_utils.draw_mega_plot(test_data, test_label, output, i, date, hour,
                                                              era_metric[i].item(), st_m)
                        plt.savefig(os.path.join(logger.save_dir, 'plots', f'megaplot_{channel}'))
                plt.close('all')
            if debug_mode and t > 20:
                break
            else:
                t += 1
        acc.cat_losses(losses_to_cat)
        l = [(metric_name, channel) for metric_name in metrics_dict for channel in ['u10', 'v10', 't2']]
        metrics_df = pd.DataFrame(
            torch.cat([metrics_dict[metric_name].compute() for metric_name in metrics_dict]).cpu().numpy()[None],
            columns=l, index=[logger.experiment_number])
        metrics_df.columns = pd.MultiIndex.from_tuples(metrics_df.columns, names=['metric', 'channel'])
        metrics_df.to_csv(os.path.join(logger.save_dir, 'experiment_metrics'))

        print('Drawing power spectrums')
        era_spectrum = acc.data['power_spectrum-era'].mean(0)
        wrf_spectrum = acc.data['power_spectrum-wrf'].mean(0)
        corr_spectrum = acc.data['power_spectrum-corr'].mean(0)
        print(era_spectrum.shape, 'spectrum shapes')
        for i, c in enumerate(['w10', 't2']):
            spectrum_plot = plot_utils.power_loglog_spectrum([era_spectrum[i], wrf_spectrum[i], corr_spectrum[i]],
                                                             ['era5', 'wrf', 'wrf_corr'], spectrum_bins, name=c)
            plt.savefig(os.path.join(logger.save_dir, 'plots', f'{c}_spectrum_plot'), dpi=300, bbox_inches="tight", format="pdf",)
        plt.close('all')
        acc.save_data(logger.save_dir, ['power_spectrum-era',
                                        'power_spectrum-wrf',
                                        'power_spectrum-corr'])

        print("Drawing wrf era losses hist...")
        orig_era = acc.data['wrf_orig-era']
        corr_era = acc.data['wrf_corr-era']
        losses_plot = plot_utils.draw_losses_gist(orig_era.transpose(0, 1).mean(-1),
                                                  corr_era.transpose(0, 1).mean(-1), 'ERA5')
        plt.savefig(os.path.join(logger.save_dir, 'plots', 'era', f'era_losses_hist'), bbox_inches="tight", format="pdf",)
        plt.close('all')

        print("Drawing seasonal wrf era bar plot (насколько улучшились данные wrf относительно era5)...")
        wrf_era_mean_loss, wrf_era_t_mean_map = get_season_mean_losses(orig_era, corr_era, acc.data['month'])
        season_metric_bar = plot_utils.draw_seasonal_bar_plot(_metric(*wrf_era_mean_loss), dtype="WRF on ERA5")
        plt.savefig(os.path.join(logger.save_dir, 'plots', 'era', f'season_era_metric_bar_plot'), bbox_inches="tight", format="pdf",)
        plt.close('all')

        # карта ошибок wrf на era5
        print('Drawing error map between wrf and era5...')
        season_maps = plot_utils.draw_seasonal_orig_corr_map(era_vector_to_map(wrf_era_t_mean_map[0], losses.meaner).reshape(4, 3, 67, 215)[:, 2],
                                                             era_vector_to_map(wrf_era_t_mean_map[1], losses.meaner).reshape(4, 3, 67, 215)[:, 2],
                                                             lats=metadata['era_yy'], lons=metadata['era_xx'],)
        season_maps.savefig(os.path.join(logger.save_dir, 'plots', 'era', f'seasonal_orig-corr_maps'), 
                            bbox_inches="tight", format="pdf",)
        orig_era_figs = plot_utils.draw_seasonal_era_error_map(era_vector_to_map(wrf_era_t_mean_map[0], losses.meaner),
                                                               lats=metadata['era_yy'], lons=metadata['era_xx'],
                                                               dtype='WRF orig', colormap='rainbow')
        [f.savefig(os.path.join(logger.save_dir, 'plots', 'era', f'{name}'),
                    bbox_inches="tight", format="pdf",) for name, f in orig_era_figs.items()]
        corr_era_figs = plot_utils.draw_seasonal_era_error_map(era_vector_to_map(wrf_era_t_mean_map[1], losses.meaner),
                                                               lats=metadata['era_yy'], lons=metadata['era_xx'],
                                                               dtype='WRF corr', colormap='rainbow')
        [f.savefig(os.path.join(logger.save_dir, 'plots', 'era', f'{name}'),
                    bbox_inches="tight", format="pdf",) for name, f in corr_era_figs.items()]
        wrf_era_map_metric = era_vector_to_map(_metric(wrf_era_t_mean_map[0], wrf_era_t_mean_map[1]), losses.meaner)
        corr_era_figs = plot_utils.draw_seasonal_era_error_map(torch.clip(wrf_era_map_metric, min=-1),
                                                               lats=metadata['era_yy'], lons=metadata['era_xx'],
                                                               dtype='WRF metric', colormap='bwr_r', vmin=-1, vmax=1)
        [f.savefig(os.path.join(logger.save_dir, 'plots', 'era', f'{name}'),
                    bbox_inches="tight", format="pdf",) for name, f in corr_era_figs.items()]
        plt.close('all')

        # гистограмма ошибок wrf на станциях
        print('Drawing wrf error hist on stations...')
        # print(orig_stations.shape)
        orig_stations = acc.data['wrf_orig-stations']
        corr_stations = acc.data['wrf_corr-stations']
        losses_plot = plot_utils.draw_losses_gist(torch.nanmean(orig_stations.transpose(0, 1), dim=-1),
                                                  torch.nanmean(corr_stations.transpose(0, 1), dim=-1),
                                                  channels=['t2', 'w10'], dtype='Stations')
        plt.savefig(os.path.join(logger.save_dir, 'plots', 'stations', f'station_losses_hist'), bbox_inches="tight", format="pdf",)
        plt.close('all')

        # усредненная метрика wrf на станциях по сезонам
        print('Drawing mean seasonal wrf station metric...')
        wrf_st_mean_loss, wrf_st_t_mean_map = get_season_mean_losses(orig_stations, corr_stations, acc.data['month'])
        season_metric_bar = plot_utils.draw_seasonal_bar_plot(_metric(*wrf_st_mean_loss), channels=['t2', 'w10'],
                                                              dtype="WRF on Stations")
        plt.savefig(os.path.join(logger.save_dir, 'plots', 'stations', f'season_wrf_stations_metric_bar_plot'),
                     bbox_inches="tight", format="pdf",)
        plt.close('all')

        # карта метрик wrf на станциях по сезонам
        print("Drawing wrf seasonal metric map on stations...")
        st_figs = plot_utils.draw_seasonal_stations_error_map(_metric(*wrf_st_t_mean_map), metadata, output, test_label)
        [f.savefig(os.path.join(logger.save_dir, 'plots', 'stations', f'{name}'),
                    bbox_inches="tight", format="pdf",) for name, f in st_figs.items()]
        plt.close('all')

        # усредненная метрика era5 на станциях по сезонам
        print('Drawing mean seasonal era5 station metric')
        era_stations = acc.data['era-stations']
        era_st_mean_loss, era_st_t_mean_map = get_season_mean_losses(orig_stations, era_stations, acc.data['month'])
        season_metric_bar = plot_utils.draw_seasonal_bar_plot(_metric(*era_st_mean_loss),
                                                              channels=['t2', 'w10'], dtype="ERA5 on Stations")
        plt.savefig(os.path.join(logger.save_dir, 'plots', 'stations', f'season_stations_metric_bar_plot'),
                     bbox_inches="tight", format="pdf",)
        plt.close('all')

        # карта метрик era5 на станциях по сезонам
        print('Drawing era5 seasonal metric map on stations...')
        era_st = plot_utils.draw_seasonal_stations_error_map(_metric(*era_st_t_mean_map), metadata, output, test_label,
                                                             dtype='ERA5')
        [f.savefig(os.path.join(logger.save_dir, 'plots', 'stations', f'{name}'),
                    bbox_inches="tight", format="pdf",) for name, f in era_st.items()]
        plt.close('all')

        # карта ошибок wrf на скаттерометре
        print('Drawing wrf seasonal error map on scatter...')
        orig_scatter_seasonal = get_season_mean_scatter(acc.data['wrf_orig-scatter'], acc.data['wrf-scatter-counts'],
                                                        acc.data['month'])
        corr_scatter_seasonal = get_season_mean_scatter(acc.data['wrf_corr-scatter'], acc.data['wrf-scatter-counts'],
                                                        acc.data['month'])
        mean_orig_scatter = torch.nanmean(orig_scatter_seasonal[:4], dim=[-1])
        mean_corr_scatter = torch.nanmean(corr_scatter_seasonal[:4], dim=[-1])
        print(mean_corr_scatter.shape, 'mean_corr_scatter.shape')
        scat_metric_fig = plot_utils.draw_seasonal_bar_plot(_metric(mean_orig_scatter, mean_corr_scatter),
                                                            channels=['u10', 'v10'],
                                                            dtype="WRF on Scatter")
        plt.savefig(os.path.join(logger.save_dir, 'plots', 'scatter', f'season_scatter_wrf_metric_bar_plot'),
                     bbox_inches="tight", format="pdf",)
        plt.close('all')
        scat_figs = plot_utils.draw_seasonal_scat_err_map(orig_scatter_seasonal, lons=metadata['scat_xx'],
                                                          lats=metadata['scat_yy'], dtype='WRF orig', colormap='common')
        [f.savefig(os.path.join(logger.save_dir, 'plots', 'scatter', f'{name}'),
                    bbox_inches="tight", format="pdf",) for name, f in scat_figs.items()]
        scat_figs = plot_utils.draw_seasonal_scat_err_map(corr_scatter_seasonal, lons=metadata['scat_xx'],
                                                          lats=metadata['scat_yy'], dtype='WRF corr', colormap='common')
        [f.savefig(os.path.join(logger.save_dir, 'plots', 'scatter', f'{name}'),
                    bbox_inches="tight", format="pdf",) for name, f in scat_figs.items()]

        scat_figs = plot_utils.draw_seasonal_scat_err_map(torch.clip(_metric(orig_scatter_seasonal,
                                                                             corr_scatter_seasonal), min=-1),
                                                          lons=metadata['scat_xx'],
                                                          lats=metadata['scat_yy'], dtype='WRF metric',
                                                          colormap='bwr_r', vmin=-1, vmax=1)
        [f.savefig(os.path.join(logger.save_dir, 'plots', 'scatter', f'{name}'),
                    bbox_inches="tight", format="pdf",) for name, f in scat_figs.items()]
        plt.close('all')

        # карта ошибок era5 на скаттерометре
        print('Drawing era5 seasonal metric map on scatter...')
        era_scatter_seasonal = get_season_mean_scatter(acc.data['era-scatter'], acc.data['era-scatter-counts'],
                                                       acc.data['month'])
        scat_figs = plot_utils.draw_seasonal_scat_err_map(era_scatter_seasonal, lons=metadata['scat_xx'],
                                                          lats=metadata['scat_yy'], dtype='ERA5', colormap='common')
        [f.savefig(os.path.join(logger.save_dir, 'plots', 'scatter', f'{name}'),
                    bbox_inches="tight", format="pdf",) for name, f in scat_figs.items()]
        plt.close('all')

        mean_era_scatter = torch.nanmean(era_scatter_seasonal[:4], dim=[-1])
        print(mean_corr_scatter.shape, 'mean_corr_scatter.shape')
        scat_metric_fig = plot_utils.draw_seasonal_bar_plot(_metric(mean_orig_scatter, mean_era_scatter),
                                                            channels=['u10', 'v10'],
                                                            dtype="ERA on Scatter")
        plt.savefig(os.path.join(logger.save_dir, 'plots', 'scatter', f'season_scatter_era_metric_bar_plot'),
                     bbox_inches="tight", format="pdf",)
        plt.close('all')

        if cfg.test_config.save_losses:
            acc.save_data(logger.save_dir)

        st_means = torch.nanmean(acc.data['wrf_corr-stations'], dim=[0, -1]).tolist()
        st_orig_means = torch.nanmean(acc.data['wrf_orig-stations'], dim=[0, -1]).tolist()
        st_era_means = torch.nanmean(acc.data['era-stations'], dim=[0, -1]).tolist()

        era_means = acc.data['wrf_corr-era'].mean([0, -1]).tolist()
        era_orig_means = acc.data['wrf_orig-era'].mean([0, -1]).tolist()
        test_loss = era_means + st_means + corr_scatter_seasonal[-1].nanmean(-1).tolist() + \
                    [acc.data['mesoscale_loss'].mean().item()]
        test_orig_loss = era_orig_means + st_orig_means + \
                         orig_scatter_seasonal[-1].nanmean(-1).tolist() + [0]
        era_loss = st_era_means + era_scatter_seasonal[-1].nanmean(-1).tolist()
        # test_loss = wrf_era_mean_loss[1].mean(0).tolist() + wrf_st_mean_loss[1].mean(0).tolist() + \
        #             corr_scatter_seasonal[-1].nanmean(-1).tolist() + [acc.data['mesoscale_loss'].mean().item()]
        # test_orig_loss = wrf_era_mean_loss[0].mean(0).tolist() + wrf_st_mean_loss[0].mean(0).tolist() + \
        #                  orig_scatter_seasonal[-1].nanmean(-1).tolist() + [0]

        a = True
        if a:
            df = pd.DataFrame([test_loss], columns=['era_u10', 'era_v10', 'era_t2', 'st_t2', 'st_w10', 'sc_u10',
                                                    'sc_v10', 'mesoscale_loss'])
            df.to_csv(os.path.join(logger.save_dir, 'mean_losses'))
            df = pd.DataFrame([test_orig_loss], columns=['era_u10', 'era_v10', 'era_t2', 'st_t2', 'st_w10', 'sc_u10',
                                                         'sc_v10', 'mesoscale_loss'])
            df.to_csv(os.path.join(logger.save_dir, 'mean_orig_losses'))
            df = pd.DataFrame([era_loss], columns=['st_t2', 'st_w10', 'sc_u10', 'sc_v10'])
            df.to_csv(os.path.join(logger.save_dir, 'mean_era_losses'))
        df_stat.to_csv(os.path.join(logger.save_dir, 'nz_stats'))
    return test_loss


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
    era_map_shape = era_map_shape if era_map_shape is not None else torch.Size([67 * 215])
    base = torch.zeros(era_vector.shape[:-1] + era_map_shape)
    base[..., meaner.mapping.unique().long()] = era_vector.float()
    return base


def calc_era5_error_map(wrf, era, meaner):
    t = era.flatten(-2, -1)
    t = t[..., meaner.mapping.unique().long()]
    err = torch.nn.L1Loss(reduction='none')(t, meaner(wrf))
    base = torch.zeros_like(era.flatten(-2, -1))
    base[..., meaner.mapping.unique().long()] = err.float()
    return base


def calc_scatter_error_map(data, scatter, criterion, i, start_date, interpolator, input_mask):
    # todo по идее правильно спрашивать criterion по которому считаем лосс, а функцию интерполяции брать самому изнутри
    # wrf_scattered.shape == 1, 2, 2, 56760 == bs, t, c, h*w ; counts.shape == 1, 2, 56760
    data_scattered, scatter, counts = interpolate_input_to_scat(data[..., :2, :, :], scatter, interpolator,
                                                                i, start_date, input_mask, return_counts=True)
    # scatter.shape == 1, 2, 4, 132, 430 ; err.shape == 1, 2, 2, 56760
    # err = torch.nn.L1Loss(reduction='none')(data_scattered, scatter)
    err = criterion(data_scattered, scatter)
    assert not torch.isnan(err).any()
    return err, counts


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
