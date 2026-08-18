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
from lib.helpers.metrics import (
    NormSSIM,
    normalized,
    channel_meaned,
    MeanerMetric,
    MulticlassAccuracy,
    HeidkeSkillScore,
    LatitudeWeightedMSE,
    LatitudeWeightedMAE,
)
from lib.helpers.aggregators import SpatialAggregator, AverageAggregator, SeasonalSpatialAggregator, RegionalTemporalAggregator
from lib.helpers.ssim import CustomSSIM
from lib.helpers.paper_utils import plot_bias_correction_grid_cpy, plot_vector_bias_correction_grid_cpy, add_column_letters_on_toprow
from lib.helpers.res_table_utils import export_metrics_table
from lib.pipeline.test_metrics import build_test_metrics
from lib.pipeline.test_profiler import TestLoopProfiler, count_temporal_entries
import lib.helpers.visualization as visualization
from lib.validation import metrics

def _call_model_for_test(model, test_data, dates):
    if getattr(model, "requires_dates", False):
        return model(test_data, dates=dates)
    return model(test_data)


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _regional_metrics_config(cfg):
    return _cfg_get(_cfg_get(cfg, "test_config"), "regional_metrics")


def _regional_metrics_enabled(cfg):
    regional_cfg = _regional_metrics_config(cfg)
    return bool(regional_cfg) and bool(_cfg_get(regional_cfg, "enabled", False))


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_filename(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value)).strip("_")


def _grid_lat_lon(grid):
    if isinstance(grid, dict):
        lat = grid.get("latitude", grid.get("lat"))
        lon = grid.get("longitude", grid.get("lon"))
    else:
        lat = getattr(grid, "latitude", getattr(grid, "lat", None))
        lon = getattr(grid, "longitude", getattr(grid, "lon", None))
    if lat is None or lon is None:
        raise ValueError("Could not extract latitude/longitude from grid for mask plotting.")
    return np.asarray(lat).squeeze(), np.asarray(lon).squeeze()


def _plot_regional_masks(regional_aggregator, grids, logger, cfg):
    masks_dir = os.path.join(logger.save_dir, "plots", "regional_masks")
    os.makedirs(masks_dir, exist_ok=True)

    domain_proj = visualization.get_domain_projection(cfg.reference_region)
    src_crs = visualization.ccrs.PlateCarree()

    for grid_name, masks in regional_aggregator.masks_by_grid.items():
        grid = grids.get(grid_name)
        if grid is None:
            continue

        grid_dir = os.path.join(masks_dir, _safe_filename(grid_name))
        os.makedirs(grid_dir, exist_ok=True)

        for mask_name, mask in masks.items():
            mask = np.asarray(mask, dtype=float)
            mask_sum = int(np.nansum(mask))
            out_path = os.path.join(grid_dir, f"{_safe_filename(mask_name)}.png")

            try:
                if mask.ndim == 2:
                    fig, ax = visualization.create_cartopy_axes(
                        grid=grid,
                        proj=domain_proj,
                        figsize=(8, 8),
                    )
                    im = visualization.visualize_scalar_field(
                        grid,
                        mask,
                        ax=ax,
                        cmap="viridis",
                        vmin=0,
                        vmax=1,
                    )
                    fig.colorbar(im, ax=ax, orientation="vertical", label="mask")
                elif mask.ndim == 1:
                    extent_grid = grids.get("wrf") or grid
                    fig, ax = visualization.create_cartopy_axes(
                        grid=extent_grid,
                        proj=domain_proj,
                        figsize=(8, 8),
                    )
                    lat, lon = _grid_lat_lon(grid)
                    scatter = ax.scatter(
                        lon.ravel(),
                        lat.ravel(),
                        c=mask.ravel(),
                        cmap="viridis",
                        vmin=0,
                        vmax=1,
                        s=28,
                        edgecolors="black",
                        linewidths=0.3,
                        transform=src_crs,
                        zorder=10,
                    )
                    fig.colorbar(scatter, ax=ax, orientation="vertical", label="mask")
                else:
                    print(f"Warning: cannot plot regional mask {mask_name!r} on {grid_name!r}; ndim={mask.ndim}.")
                    continue

                ax.set_title(f"{grid_name}: {mask_name} (n={mask_sum})")
                fig.savefig(out_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
            except Exception as exc:
                plt.close("all")
                print(f"Warning: failed to plot regional mask {mask_name!r} on {grid_name!r}: {exc}")


def _finalize_regional_metrics(results, metrics_dict, regional_aggregator):
    summary = {}
    rows = []
    agg_name = regional_aggregator.name
    for metric_name, metric_results in results.items():
        if agg_name not in metric_results or metric_results[agg_name] is None:
            continue

        metric = metrics_dict[metric_name]
        grid_name = metric.meta.grid
        if hasattr(regional_aggregator, "finalize_reduced_time"):
            finalized = regional_aggregator.finalize_reduced_time(metric_results[agg_name])
        else:
            finalized = regional_aggregator.finalize(metric_results[agg_name])
        for region_name, data in finalized.items():
            values = np.atleast_1d(_to_numpy(data['value']))
            counts = np.atleast_1d(_to_numpy(data['valid_count']))
            summary.setdefault(grid_name, {}).setdefault(region_name, {})[metric_name] = {
                'value': values,
                'valid_count': counts,
            }
            for channel_idx, value in enumerate(values):
                rows.append({
                    'region': region_name,
                    'grid': grid_name,
                    'metric_name': metric_name,
                    'channel': channel_idx,
                    'value': float(value),
                    'valid_count': int(counts[channel_idx]),
                })
    return summary, pd.DataFrame(rows, columns=['region', 'grid', 'metric_name', 'channel', 'value', 'valid_count'])


def _regional_summary_to_res_dicts(summary):
    region_res_dicts = {}
    for _grid_name, region_data in summary.items():
        for region_name, metrics_data in region_data.items():
            res_dict = region_res_dicts.setdefault(region_name, {})
            for metric_name, data in metrics_data.items():
                res_dict[metric_name] = data['value']
    return region_res_dicts


def _has_any_metric(res_dict, models, target, metrics):
    keys = {
        f"{model}_{target}_{metric}"
        for model in models
        for metric in metrics
    }
    return any(key in res_dict for key in keys)


def _export_regional_metrics_tables(summary, model_names, save_dir):
    model_names = list(model_names)
    region_res_dicts = _regional_summary_to_res_dicts(summary)
    if not region_res_dicts:
        return

    tables_dir = os.path.join(save_dir, 'regional_metrics_tables')
    plots_dir = os.path.join(save_dir, 'plots', 'regional_metrics_tables')

    era_models = model_names + ['orig']
    era_metrics = ['mse', 'mae', 'err_norm', 'norm_dif', 'angle_norm']
    station_models = model_names + ['era', 'orig']
    station_metrics = ['mse', 'mae']

    for region_name, res_dict in region_res_dicts.items():
        safe_region = _safe_filename(region_name)

        if _has_any_metric(res_dict, era_models, 'era', era_metrics):
            export_metrics_table(
                res_dict=res_dict,
                models=era_models,
                metrics=era_metrics,
                target='era',
                csv_path=os.path.join(tables_dir, f'{safe_region}_era_metrics_table.csv'),
                image_path=os.path.join(plots_dir, f'{safe_region}_era_metrics_table.png'),
                lead_labels=['u10', 'v10', 't2'],
                precision=2,
                image_title=f'ERA metrics comparison ({region_name})',
            )

        if _has_any_metric(res_dict, station_models, 'stations', station_metrics):
            export_metrics_table(
                res_dict=res_dict,
                models=station_models,
                metrics=station_metrics,
                target='stations',
                csv_path=os.path.join(tables_dir, f'{safe_region}_stations_metrics_table.csv'),
                image_path=os.path.join(plots_dir, f'{safe_region}_stations_metrics_table.png'),
                lead_labels=['w10', 't2'],
                precision=2,
                image_title=f'Stations metrics comparison ({region_name})',
            )


def _save_regional_metrics(results, metrics_dict, regional_aggregator, save_dir, model_names):
    summary, df_long = _finalize_regional_metrics(results, metrics_dict, regional_aggregator)

    with open(os.path.join(save_dir, 'regional_metrics.pickle'), 'wb') as handle:
        pickle.dump(summary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    long_path = os.path.join(save_dir, 'regional_metrics_long.csv')
    df_long.to_csv(long_path, index=False)

    wide_path = os.path.join(save_dir, 'regional_metrics_wide.csv')
    if df_long.empty:
        pd.DataFrame().to_csv(wide_path, index=False)
        return summary

    value_wide = df_long.pivot_table(
        index=['region', 'grid', 'metric_name'],
        columns='channel',
        values='value',
        aggfunc='first',
    ).add_prefix('value_channel_')
    count_wide = df_long.pivot_table(
        index=['region', 'grid', 'metric_name'],
        columns='channel',
        values='valid_count',
        aggfunc='first',
    ).add_prefix('valid_count_channel_')
    df_wide = pd.concat([value_wide, count_wide], axis=1).reset_index()
    df_wide.to_csv(wide_path, index=False)
    _export_regional_metrics_tables(summary, model_names, save_dir)
    return summary


def test(models_dict, losses, wrf_scaler, era_scaler, dataloader, logger, cfg):
    debug_mode = cfg.test_config.debug_mode
    img_format = 'pdf'
    for channel in ['u10', 'v10', 't2', 'era', 'stations', 'scatter']:
        os.makedirs(os.path.join(logger.save_dir, 'plots', channel), exist_ok=True)
    with torch.no_grad():
        for model in models_dict:
            models_dict[model].eval()
        datasets = dataloader.dataset.datasets

        wrf_grid, era_grid = datasets['WRF'].grid, datasets['ERA5'].grid
        scat_grid = datasets['Scatter'].grid if cfg.run_config.use_scatter and 'Scatter' in datasets else None

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

        metric_fns = {
            "mesoscale_loss": SmallScaleLoss(reduction='none', device=cfg.device),
            "mse": mse,
            "mae": mae,
            "wind_err_norm": wind_err_norm,
            "wind_angle_norm": wind_angle_norm,
            "wind_norm_diff": wind_norm_diff,
            "ssim_111": ssim_111,
            "ssim_211": ssim_211,
            "ssim_011": ssim_011,
            "identity": torch.nn.Identity(),
            "mc_acc": mc_acc,
            "st_dir_mae": st_dir_mae,
            "era_lat_weighted_mse": LatitudeWeightedMSE(
                era_grid['latitude'],
                reduction='none',
                spatial_ndim=2,
            ).to(cfg.device),
            "era_lat_weighted_mae": LatitudeWeightedMAE(
                era_grid['latitude'],
                reduction='none',
                spatial_ndim=2,
            ).to(cfg.device),
        }
        if scat_grid is not None:
            metric_fns["scatter_lat_weighted_mse"] = LatitudeWeightedMSE(
                scat_grid['latitude'],
                reduction='none',
                spatial_ndim=2,
            ).to(cfg.device)
            metric_fns["scatter_lat_weighted_mae"] = LatitudeWeightedMAE(
                scat_grid['latitude'],
                reduction='none',
                spatial_ndim=2,
            ).to(cfg.device)
        draw_spectrum = bool(_cfg_get(cfg.test_config, "draw_spectrum", False))
        metrics_dict = build_test_metrics(
            models_dict.keys(),
            metric_fns,
            include_stations=cfg.test_config.use_stations and ('Stations' in datasets),
            stations_has_dir=('Stations' in datasets) and ('d' in datasets['Stations'].wind_format),
            include_scatter=cfg.run_config.use_scatter and ('Scatter' in datasets),
            include_spectrum=draw_spectrum,
        )

        era_coords = np.stack([era_grid['longitude'].flatten(), era_grid['latitude'].flatten()]).T
        wrf_coords = np.stack([wrf_grid['longitude'].flatten(), wrf_grid['latitude'].flatten()]).T

        era_upsampler = InvDistTree(x=era_coords, q=wrf_coords, device=cfg.device)

        if cfg.run_config.use_scatter and 'Scatter' in datasets:
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
        regional_aggregator = None
        if _regional_metrics_enabled(cfg):
            regional_grids = {
                'wrf': wrf_grid,
                'era': era_grid,
                'scatter': scat_grid if cfg.run_config.use_scatter and 'Scatter' in datasets else None,
                'stations': station_grid if cfg.test_config.use_stations and 'Stations' in datasets else None,
            }
            regional_aggregator = RegionalTemporalAggregator(
                grids=regional_grids,
                regional_config=_regional_metrics_config(cfg),
                temporal_resolution=_cfg_get(
                    _regional_metrics_config(cfg),
                    "temporal_resolution",
                ),
                reduce_time=False,
            )
            _plot_regional_masks(regional_aggregator, regional_grids, logger, cfg)
            aggregators.append(regional_aggregator)

        results = {
            metric_name: {agg.name: None for agg in aggregators if agg.supports(metric)}
            for metric_name, metric in metrics_dict.items()
        }

        profiler = TestLoopProfiler(cfg, logger.save_dir)
        profiler.start_data_wait()
        for data, dates in tqdm(dataloader): 
            profiler.batch_received()
            batch_timer = profiler.start()
            if data is None:
                profiler.stop("batch_processing", batch_timer)
                profiler.finish_batch()
                continue

            input_timer = profiler.start()
            test_data, test_label = data.pop(cfg.reference_dataset), data.pop(cfg.target_dataset)
            test_data = torch.swapaxes(test_data.type(torch.float).to(cfg.device), 0, 1).contiguous()
            test_label = torch.swapaxes(test_label.type(torch.float).to(cfg.device), 0, 1)
            era_h, era_w = test_label.shape[-2:]

            date = dates.astype(str)

            test_data = wrf_scaler.transform(test_data, dims=2)
            profiler.stop("input_preparation", input_timer)
            
            outputs = {}
            for model_name in models_dict:
                model_inference_timer = profiler.start()
                output = _call_model_for_test(models_dict[model_name], test_data, dates)
                profiler.stop(f"model_inference/{model_name}", model_inference_timer)

                model_postprocess_timer = profiler.start()
                output = era_scaler.inverse_transform(output, dims=2)
                outputs[model_name] = output
                corr_meaned = input_to_era_map(output, losses.meaner, era_map_shape=(era_h, era_w))
                outputs[model_name + '_meaned'] = corr_meaned
                profiler.stop(f"model_postprocess/{model_name}", model_postprocess_timer)

                if draw_spectrum:
                    model_spectrum_timer = profiler.start()
                    corr_spectrum = get_power_spectrum(uvt_to_wt(output, -3).cpu())[1]
                    outputs[model_name + '_spectrum'] = torch.from_numpy(corr_spectrum)
                    profiler.stop(f"model_spectrum/{model_name}", model_spectrum_timer)

            base_interpolation_timer = profiler.start()
            test_data = wrf_scaler.inverse_transform(test_data, dims=2)[:, :, :3]

            # ========== Interpolate WRF to others ================
            wrf_meaned = input_to_era_map(test_data, losses.meaner, era_map_shape=(era_h, era_w))

            era_upsampled = era_upsampler(test_label.flatten(-2, -1)).view(test_data.shape)
            profiler.stop("base_interpolation", base_interpolation_timer)

            spectrum_samples = {}
            if draw_spectrum:
                base_spectrum_timer = profiler.start()
                spectrum_bins, era_spectrum = get_power_spectrum(
                    uvt_to_wt(era_upsampled, -3).cpu()
                )
                wrf_spectrum = get_power_spectrum(uvt_to_wt(test_data, -3).cpu())[1]
                era_spectrum, wrf_spectrum = map(
                    torch.from_numpy,
                    [era_spectrum, wrf_spectrum],
                )
                spectrum_samples = {
                    'wrf_spectrum': wrf_spectrum,
                    'era_spectrum': era_spectrum,
                }
                profiler.stop("base_spectrum", base_spectrum_timer)

            samples_dict = {
                **outputs,
                'wrf': test_data,
                'era_up': era_upsampled,
                'era': test_label,
                # 'corr': output,
                'wrf_meaned': wrf_meaned,
                # 'corr_meaned': corr_meaned,

                **spectrum_samples,
                }
            if cfg.test_config.use_stations and 'Stations' in datasets:
                stations_timer = profiler.start()
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
                profiler.stop("stations", stations_timer)

            if cfg.run_config.use_scatter and 'Scatter' in datasets:
                scatter_timer = profiler.start()
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
                profiler.stop("scatter", scatter_timer)

            if cfg.test_config.plot_samples and (month := date[0].astype('datetime64[M]').astype(int) % 12 + 1) in months:
                plot_timer = profiler.start()
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
                profiler.stop("plot_samples", plot_timer)
            for metric_name, metric in metrics_dict.items():
                metric_timer = profiler.start()
                err_field = metric.calculate(samples_dict)
                profiler.stop(f"metric/{metric_name}", metric_timer)

                for agg in aggregators:
                    agg_name = agg.name
                    if agg_name not in results[metric_name]:
                        continue
                    acc = results[metric_name][agg_name]
                    if acc is None:
                        acc = agg.init_accumulator(err_field.shape[2:], metric=metric)
                        results[metric_name][agg_name] = acc
                    aggregate_timer = profiler.start()
                    agg.accumulate(acc, err_field, dates, metric=metric)
                    profiler.stop(f"aggregate/{agg_name}/{metric_name}", aggregate_timer)

            if debug_mode and t > 5:
                profiler.stop("batch_processing", batch_timer)
                regional_entries = (
                    count_temporal_entries(results, regional_aggregator.name)
                    if profiler.enabled and regional_aggregator is not None
                    else None
                )
                profiler.finish_batch(regional_date_entries=regional_entries)
                break
            t += 1
            profiler.stop("batch_processing", batch_timer)
            regional_entries = (
                count_temporal_entries(results, regional_aggregator.name)
                if profiler.enabled and regional_aggregator is not None
                else None
            )
            profiler.finish_batch(regional_date_entries=regional_entries)

        regional_entries = (
            count_temporal_entries(results, regional_aggregator.name)
            if profiler.enabled and regional_aggregator is not None
            else None
        )
        profiler.close(regional_date_entries=regional_entries)

        res_dict = {metric_name: AverageAggregator.finalize(results[metric_name]['SpatialAggregator']).cpu().numpy() for metric_name in metrics_dict}
        with open(os.path.join(logger.save_dir, 'experiment_metrics.pickle'), 'wb') as handle:
            pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if regional_aggregator is not None:
            _save_regional_metrics(results, metrics_dict, regional_aggregator, logger.save_dir, models_dict.keys())

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

        mean_era_models = [f"{model_name}_mean" for model_name in models_dict] + ["mean_orig"]
        mean_era_display_names = {
            **{f"{model_name}_mean": model_name for model_name in models_dict},
            "mean_orig": "orig",
        }
        df = export_metrics_table(
            res_dict=res_dict,
            models=mean_era_models,
            metrics=['mse', 'mae', 'lat_weighted_mse', 'lat_weighted_mae'],
            target='era',
            key_template='{model}_{target}_{metric}',
            model_display_names=mean_era_display_names,
            csv_path=os.path.join(logger.save_dir, 'mean_era_metrics_table.csv'),
            image_path=os.path.join(logger.save_dir, 'plots', 'mean_era_metrics_table.png'),
            lead_labels=['u10', 'v10', 't2'],
            precision=2,
            image_title='ERA-grid metrics comparison',
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

        if cfg.run_config.use_scatter and 'Scatter' in datasets:
            df = export_metrics_table(
                res_dict=res_dict,
                models=[model_name for model_name in models_dict] + ['era', 'orig'],
                metrics=['mse', 'mae', 'lat_weighted_mse', 'lat_weighted_mae', 'err_norm', 'angle_norm', 'norm_diff'],
                target='scatter',
                csv_path=os.path.join(logger.save_dir, 'scatter_metrics_table.csv'),
                image_path=os.path.join(logger.save_dir, 'plots', 'scatter_metrics_table.png'),
                lead_labels=['u10', 'v10'],
                precision=2,
                image_title='Scatter metrics comparison',
            )

        if draw_spectrum:
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


def calc_station_loss(wrf, stations, interpolator, loss):
    s = wrf.shape
    wrf_interpolated = interpolator(wrf.flatten(-2, -1))
    # wrf_interpolated.shape == 4, bs, 3, 46 ; stations.shape == 4, bs, 2, 46

    t2_loss = loss(wrf_interpolated[..., 2, :], stations[..., 1, :])
    wspd = torch.sqrt(torch.square(wrf_interpolated[..., 0, :]) + torch.square(wrf_interpolated[..., 1, :]))
    w10_loss = loss(wspd, stations[..., 0, :])

    return torch.stack((t2_loss, w10_loss), dim=-2)  # sl, bs, c, N_stations


def calculate_era_loss(wrf, era, meaner, criterion):
    wrf_orig = meaner(wrf)
    era = era.flatten(-2, -1)
    era = era[..., meaner.mapping.unique().long()]
    loss = criterion(wrf_orig, era)
    return loss  # loss.shape = 4, 1, 3, 8744 i.e. sl, bs, c, N


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
