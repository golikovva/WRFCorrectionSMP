import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Arc
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap

import numpy as np
import torch


def draw_borey_basemap(era_season_map, lats, lons, dtype='WRF', date='2019-01-01', channel='t2', colormap='common'):
    figs = {}
    fig, ax = plt.subplots(1, dpi=300)
    fig.suptitle(f'{dtype} ERA5 {channel} {date} difference map', fontsize=16)

    ax.set_xticks([])
    ax.set_yticks([])
    m = Basemap(
        projection='laea',
        resolution='i',
        lat_0=71.0,
        lon_0=80.0,
        llcrnrlon=42.27414,
        llcrnrlat=62.22973,
        urcrnrlon=72.22717,
        urcrnrlat=78.75766,

    )
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(50, 90, 2), labels=[1, 0, 0, 0], color='gray')
    m.drawmeridians(np.arange(-180, 180, 5), labels=[0, 0, 0, 1], color='gray')
    x, y = m(lons, lats)
    vmin, vmax, mean = np.nanmin(era_season_map), np.nanmax(era_season_map), np.nanmean(era_season_map)
    if colormap == 'common':
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = 'viridis'
    else:
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cmap = 'bwr'
    img = m.pcolor(x, y, np.squeeze(era_season_map), cmap=cmap, norm=norm)
    # img = m.pcolormesh(lons, lats, np.squeeze(era_season_map), latlon=True, cmap='bwr')
    cbar = m.colorbar(img, location='bottom', pad="10%")
    fig.tight_layout()
    figs[f'{dtype.lower()}_era_{date}'] = fig
    return figs, ax, m


def borey_basemap_ax(data, lats, lons, colormap='common', ax=None, vmin=None, vmax=None, if_cbar=True,
                     return_basemap=False):
    if ax is None:
        ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])
    m = Basemap(
        projection='laea',
        resolution='i',
        lat_0=71.0,
        lon_0=80.0,
        llcrnrlon=42.27414,
        llcrnrlat=62.22973,
        urcrnrlon=72.22717,
        urcrnrlat=78.75766,
        ax=ax,
    )
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(50, 90, 2), labels=[1, 0, 0, 0], color='gray')
    m.drawmeridians(np.arange(-180, 180, 5), labels=[0, 0, 0, 1], color='gray')
    x, y = m(lons, lats)

    vmin = vmin if vmin else np.nanmin(data)
    vmax = vmax if vmax else np.nanpercentile(data, 95, method='closest_observation')

    if 'bwr' in colormap:
        norm = colors.TwoSlopeNorm(vmin=min(-0.001, vmin), vcenter=0, vmax=vmax)
        cmap = colormap
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = 'viridis' if colormap == 'common' else colormap

    img = m.pcolor(x, y, np.squeeze(data), cmap=cmap, norm=norm)
    if if_cbar:
        cbar = m.colorbar(img, location='bottom', pad="10%")
    if return_basemap:
        return ax, m
    return ax


def draw_seasonal_orig_corr_map(orig, corr, lats, lons, colormap='magma_r'):
    vmin = min(orig.min(), corr.min())
    vmax = max(orig.max(), corr.max())
    cmap = plt.cm.get_cmap(colormap)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    print(vmin, vmax)
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    fig, ax = plt.subplots(4, 2, figsize=(10, 16), layout="constrained")

    # Создадим пустую переменную для хранения последнего img
    ax[0, 0].set_title('WRF', fontsize=24)
    ax[0, 1].set_title('BERTUnet', fontsize=24)
    for i, season in enumerate(seasons):
        ax[i][0].set_ylabel(season, fontsize=20)
        ax[i][0].yaxis.set_label_coords(-0.1, 0.5)

        borey_basemap_ax(orig[i], lats, lons, ax=ax[i][0], vmin=vmin, vmax=vmax, colormap=colormap, if_cbar=False)
        borey_basemap_ax(corr[i], lats, lons, ax=ax[i][1], vmin=vmin, vmax=vmax, colormap=colormap, if_cbar=False)

    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[-1, :], orientation='horizontal',
                        aspect=50)
    cbar.ax.tick_params(labelsize=14)
    return fig


def draw_mega_plot(wrf_tensor, era_tensor, wrfcorr_tensor, channel, date, hour, era_metric=None, station_metric=None):
    fig, axs = plt.subplots(4, 4, dpi=600)
    imgs_to_draw = list(map(lambda x: torch.Tensor.numpy(torch.Tensor.cpu(x)),
                            [wrf_tensor, wrfcorr_tensor, era_tensor, wrfcorr_tensor - wrf_tensor]))

    for i, title in enumerate(['Original WRF image', 'Corrected WRF image', 'ERA5 image', 'Correction']):
        axs[i][0].set_title(title)
        if title == 'Correction':
            vmin, vmax = (imgs_to_draw[i][:, :, channel].min(), imgs_to_draw[i][:, :, channel].max())
        else:
            vmin, vmax = (imgs_to_draw[0][:, :, channel].min() - 5, imgs_to_draw[0][:, :, channel].max() + 5)
        for t in range(4):
            im = axs[i][t].imshow(imgs_to_draw[i][t, 0, channel], interpolation='none',
                                  extent=[0, 280, 0, 210], vmin=vmin, vmax=vmax)
            axs[i][t].axison = False
        if title == 'Corrected WRF image':
            fig.colorbar(im, ax=axs[:3], orientation='vertical', fraction=0.1, aspect=21)
        elif title == 'Correction':
            fig.colorbar(im, ax=axs[3], orientation='vertical', fraction=0.1, aspect=7)
    if era_metric:
        axs[i][1].text(10, 10, f'era metric={round(era_metric, 3)}', size='xx-small')
    if station_metric:
        axs[i][2].text(10, 10, f'station metric={round(station_metric, 3)}', size='xx-small')
    axs[i][0].text(10, 10, f'date: {date} {hour}:00', size='xx-small')


def draw_station_err_map(metadata, wrf_sample, station_err_map):
    t2_err = station_err_map[:, 0]
    w10_err = station_err_map[:, 1]
    fig, ax = plt.subplots(dpi=800)
    cmap = plt.cm.get_cmap('Reds')
    norm = colors.Normalize(vmin=0, vmax=1.3)

    plt.scatter(metadata['coords'][:, 0], metadata['coords'][:, 1], label='t2/w10 station metric', s=0.2)
    plt.pcolormesh(metadata['wrf_xx'], metadata['wrf_yy'], wrf_sample, shading='auto')

    theta1, theta2 = 90, 90 + 180
    radius = 0.2

    arcs(metadata['coords'][:, 0], metadata['coords'][:, 1], 3 * radius, radius, theta1=theta1, theta2=theta2,
         color=cmap(norm(t2_err.cpu().numpy())))
    arcs(metadata['coords'][:, 0], metadata['coords'][:, 1], 3 * radius, radius, theta1=theta2, theta2=theta1,
         color=cmap(norm(w10_err.cpu().numpy())))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.legend()

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', fraction=0.1)


def draw_era_error_map(era_err_map):
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    for i in range(3):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        im1 = axs[i].imshow(era_err_map[i].reshape(67, 215), interpolation='none',
                            extent=[0, 280, 0, 210], )  # vmin=0, vmax=0.7)
        # im2 = axs[1].imshow(era_err_map[2].reshape(67, 215), interpolation='none', extent=[0, 280, 0, 210], vmin=0,
        #                     vmax=2.9)
        fig.colorbar(im1, ax=axs[i], orientation='vertical', fraction=0.1)
    return fig, axs


def draw_seasonal_era_error_map(era_season_map, lats=None, lons=None, dtype='WRF', colormap='common', vmin=None,
                                vmax=None, channels=None):
    channels = ['u10', 'v10', 't2'] if channels is None else channels
    vmin_arg, vmax_arg = vmin, vmax
    figs = {}
    for c, channel in enumerate(channels):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'{dtype} ERA5 {channel} seasonal error maps', fontsize=16)
        for t, season in enumerate(['Winter', 'Spring', 'Summer', 'Autumn']):
            row, col = t // 2, t % 2
            axs[row][col].set_xticks([])
            axs[row][col].set_yticks([])
            vmin = vmin_arg if vmin_arg else 0
            vmax = vmax_arg if vmax_arg else np.percentile(era_season_map[:, c], 99)

            if lats is not None and lons is not None:
                borey_basemap_ax(era_season_map[t, c].reshape(67, 215), lats=lats, lons=lons, vmin=vmin, vmax=vmax,
                                 ax=axs[row][col], colormap=colormap)
            else:
                im1 = axs[row][col].imshow(era_season_map[t, c].reshape(67, 215), interpolation='none',
                                           extent=[0, 280, 0, 210], vmin=vmin, vmax=vmax, colormap=colormap)
                divider = make_axes_locatable(axs[row][col])
                colorbar_axes = divider.append_axes("right", size="10%", pad=0.1)
                fig.colorbar(im1, cax=colorbar_axes)
            axs[row][col].set_title(season)

        figs[f'{dtype.lower()}_era_seasonal_{channel}'] = fig

        fig, ax = plt.subplots(1)
        fig.suptitle(f'{dtype} ERA5 {channel} error map', fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        vmin = vmin if vmin else 0
        vmax = vmax if vmax else era_season_map.mean(0)[c].max()
        if lats is not None and lons is not None:
            borey_basemap_ax(era_season_map.mean(0)[c].reshape(67, 215), lats=lats, lons=lons, vmin=vmin, vmax=vmax,
                             ax=ax, colormap=colormap)
        else:
            im1 = ax.imshow(era_season_map.mean(0)[c].reshape(67, 215), interpolation='none',
                            extent=[0, 280, 0, 210], vmin=vmin, vmax=vmax, colormap=colormap)
            divider = make_axes_locatable(ax)
            colorbar_axes = divider.append_axes("right", size="10%", pad=0.1)
            fig.colorbar(im1, cax=colorbar_axes)
        figs[f'{dtype.lower()}_era_{channel}'] = fig

    return figs


def draw_seasonal_stations_error_map(station_season_map, metadata, wrf_sample, era_sample, dtype='WRF'):
    figs = {}
    for i, s in enumerate(['Winter', 'Spring', 'Summer', 'Autumn']):
        f = draw_station_metrics(metadata, era_sample, wrf_sample, station_season_map[i, 0], station_season_map[i, 1],
                                 title=f'{s} {dtype} stations t2,w10 metric')
        figs[f'{s.lower()}_{dtype.lower()}-stations-metric'] = f
    mean_map = station_season_map.mean(0)
    f = draw_station_metrics(metadata, era_sample, wrf_sample, mean_map[0], mean_map[1],
                             title=f'{dtype} stations t2,w10 metric')
    figs[f'mean_{dtype.lower()}-stations-metric'] = f
    return figs


def draw_seasonal_1d_stations_error_map(station_season_map, metadata, wrf_sample, era_sample, dtype='WRF'):
    figs = {}
    for i, s in enumerate(['Winter', 'Spring', 'Summer', 'Autumn']):
        f = draw_station_one_channel_metrics(metadata, era_sample, wrf_sample, station_season_map[i, 0],
                                             title=f'{s} {dtype} stations t2,w10 metric')
        figs[f'{s.lower()}_{dtype.lower()}-stations-metric'] = f
    mean_map = station_season_map.mean(0)
    f = draw_station_one_channel_metrics(metadata, era_sample, wrf_sample, mean_map[0],
                                         title=f'{dtype} stations t2,w10 metric')
    figs[f'mean_{dtype.lower()}-stations-metric'] = f
    return figs


def draw_scat_err_map(scat_err_map, lons, lats, title='Scatterometer error map', colormap='common', **kwargs):
    fig, axs = plt.subplots(2, dpi=300, figsize=(10, 10))
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title('U10')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_title('V10')
    borey_basemap_ax(scat_err_map[0].reshape(132, 430), lons=lons, lats=lats, ax=axs[0], colormap=colormap, **kwargs)
    borey_basemap_ax(scat_err_map[1].reshape(132, 430), lons=lons, lats=lats, ax=axs[1], colormap=colormap, **kwargs)
    # im1 = axs[0].imshow(scat_err_map[0].reshape(132, 430), interpolation='none', extent=[0, 280, 0, 210], )
    # im2 = axs[1].imshow(scat_err_map[1].reshape(132, 430), interpolation='none', extent=[0, 280, 0, 210], )
    # fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.1)
    # fig.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.1)
    fig.suptitle(title)
    return fig


def draw_seasonal_scat_err_map(seasonal_scat, lons, lats, dtype='WRF', colormap='common', **kwargs):
    figs = {}
    for i, s in enumerate(['Winter', 'Spring', 'Summer', 'Autumn']):
        f = draw_scat_err_map(seasonal_scat[i], lons=lons, lats=lats, title=f'{s} {dtype} scatterometer error map',
                              colormap=colormap, **kwargs)
        figs[f'{s.lower()}_{dtype.lower()}_scatter_error_map'] = f
    f = draw_scat_err_map(seasonal_scat[4], lons=lons, lats=lats, title=f'{dtype} scatterometer error map',
                          colormap=colormap, **kwargs)
    figs[f'mean_{dtype.lower()}_scatter_error_map'] = f
    return figs


def draw_simple_plots(wrf_tensor, wrfcorr_tensor, era_tensor, channel=2,
                      input_loss=-1, test_loss=-1, era_metric=None, station_metric=None, date=None):
    fig, axs = plt.subplots(4, 1, figsize=(5, 14))
    vmin = min(wrf_tensor[:, :, channel].min(), wrfcorr_tensor[:, :, channel].min(), era_tensor[:, :, channel].min())
    vmax = max(wrf_tensor[:, :, channel].max(), wrfcorr_tensor[:, :, channel].max(), era_tensor[:, :, channel].max())
    im = axs[0].imshow(wrf_tensor[0, 0, channel].cpu().numpy(), interpolation='none', vmin=vmin, vmax=vmax)
    axs[1].imshow(wrfcorr_tensor[0, 0, channel].cpu().numpy(), interpolation='none', vmin=vmin, vmax=vmax)
    axs[2].imshow(era_tensor[0, 0, channel].cpu().numpy(), interpolation='none',
                  extent=[0, 280, 0, 210], vmin=vmin, vmax=vmax)
    imc = axs[3].imshow(wrfcorr_tensor[0, 0, channel].cpu().numpy() - wrf_tensor[0, 0, channel].cpu().numpy(),
                        interpolation='none', extent=[0, 280, 0, 210], )
    axs[0].set_xlabel('Original WRF data')
    axs[0].text(50, 28, f'loss={round(input_loss, 3)}')
    axs[1].set_xlabel('Corrected WRF data')
    axs[1].text(50, 28, f'loss={round(test_loss, 3)}')
    axs[2].set_xlabel('ERA5 reanalysis')
    axs[3].set_xlabel('Correction')
    for i in range(4):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].xaxis.set_label_coords(.5, -.01)
    if era_metric:
        axs[3].text(0, -85, f'era metric={round(era_metric, 3)}')
    if station_metric:
        axs[3].text(0, -60, f'station metric={round(station_metric, 3)}')
    if date:
        axs[3].text(0, 5, f'{date}')
    fig.colorbar(im, ax=axs[:3], orientation='vertical', fraction=0.1, aspect=21)
    fig.colorbar(imc, ax=axs[3], orientation='vertical', fraction=0.1, aspect=6)
    return fig, axs


def draw_station_metrics(metadata, era_sample, wrf_sample, stations_t2_metric, stations_w10_metric,
                         title='Stations metrics'):
    fig, ax = plt.subplots(dpi=400)
    cmap = plt.cm.get_cmap('bwr_r')
    norm = colors.Normalize(vmin=-1, vmax=1)

    plt.scatter(metadata['coords'][:, 0], metadata['coords'][:, 1], label='t2/w10 station metric')
    # plt.pcolormesh(metadata['era_xx'], metadata['era_yy'], era_sample[0, 0, 2].cpu().numpy(), shading='auto')
    ax, m = borey_basemap_ax(wrf_sample[0, 0, 2].cpu().numpy(), lats=metadata['wrf_yy'], lons=metadata['wrf_xx'],
                             colormap='common', if_cbar=False, return_basemap=True)

    # plt.pcolormesh(metadata['wrf_xx'], metadata['wrf_yy'], wrf_sample[0, 0, 2].cpu().numpy(), shading='auto')

    theta1, theta2 = 90, 90 + 180
    radius = 25000
    basemap_station_coords = m(metadata['coords'][:, 0], metadata['coords'][:, 1])
    arcs(basemap_station_coords[0], basemap_station_coords[1], 1 * radius, radius, theta1=theta1, theta2=theta2,
         color=cmap(norm(stations_t2_metric.cpu().numpy())), zorder=10)
    arcs(basemap_station_coords[0], basemap_station_coords[1], 1 * radius, radius, theta1=theta2, theta2=theta1,
         color=cmap(norm(stations_w10_metric.cpu().numpy())), zorder=10)
    plt.title(title)
    plt.legend(loc='upper right')

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', fraction=0.1)
    return fig


def draw_station_one_channel_metrics(metadata, era_sample, wrf_sample, stations_metric,
                                     title='Stations metrics'):
    fig, ax = plt.subplots(dpi=400)
    cmap = plt.cm.get_cmap('bwr_r')
    norm = colors.Normalize(vmin=-1, vmax=1)

    plt.scatter(metadata['coords'][:, 0], metadata['coords'][:, 1], label='t2/w10 station metric')
    # plt.pcolormesh(metadata['era_xx'], metadata['era_yy'], era_sample[0, 0, 2].cpu().numpy(), shading='auto')
    ax, m = borey_basemap_ax(wrf_sample[0, 0, 0].cpu().numpy(), lats=metadata['wrf_yy'], lons=metadata['wrf_xx'],
                             colormap='common', if_cbar=False, return_basemap=True)

    # plt.pcolormesh(metadata['wrf_xx'], metadata['wrf_yy'], wrf_sample[0, 0, 2].cpu().numpy(), shading='auto')

    theta1, theta2 = 0, 360
    radius = 25000
    basemap_station_coords = m(metadata['coords'][:, 0], metadata['coords'][:, 1])
    arcs(basemap_station_coords[0], basemap_station_coords[1], 1 * radius, radius, theta1=theta1, theta2=theta2,
         color=cmap(norm(stations_metric.cpu().numpy())), zorder=10)
    plt.title(title)
    plt.legend(loc='upper right')

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', fraction=0.1)
    return fig


def draw_losses_gist(orig, corr, dtype, channels=None):
    channels = ['u10', 'v10', 't2'] if channels is None else channels
    fig, axs = plt.subplots(1, len(channels), figsize=(len(channels) * 4, 4), squeeze=False)
    for i, channel in enumerate(channels):
        axs[0][i].hist(orig[i], bins=50, label='original')
        axs[0][i].hist(corr[i], bins=50, label='corrected')
        axs[0][i].set_xlabel(f'{dtype} {channel} losses')
    plt.legend()
    axs[0][0].set_ylabel('Num samples')
    fig.suptitle('Loss distribution')
    return fig, axs


def draw_seasonal_bar_plot(metric_mean, channels=None, dtype="ERA5"):
    channels = ['u10', 'v10', 't2'] if channels is None else channels
    fig, axs = plt.subplots(1)
    width = 0.25
    ind = np.arange(4)
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    for i, channel in enumerate(channels):
        axs.bar(ind + width * i, metric_mean[:, i], width, label=channel)

    axs.set_xlabel("Dates")
    axs.set_ylabel('Metric')
    axs.set_title(f"Seasonal {dtype} metric by channel")

    axs.set_xticks(ind + width * (len(channels) / 2 - 0.5), seasons)
    plt.legend()
    return fig, axs

from fractions import Fraction

def power_loglog_spectrum(
    spectrums,
    labels,
    bins,
    name,
    figsize=None,
    # ---- slope guide options ----
    ref_slopes=None,         # e.g., [3, 5/3]
    ref_k=None,              # anchor k (defaults to geometric mean of x-lims)
    ref_source=0,            # spectrum index used for amplitude normalization
    slope_style=None,        # kwargs for guide lines
    slope_labels=True,       # draw labels on the lines
    label_pos=0.80,          # fractional log-x position in [0, 1]
    label_bbox=None,         # bbox dict for readability
    # ---- NEW: vertical adjustments ----
    slope_gains=None,        # multiplicative gain(s); float or list(len=ref_slopes)
    slope_shifts=None        # decade shift(s); float or list; gain = 10**shift
):
    """
    Plot spectra and reference slope lines ~ k^{-p}, with optional vertical shifts.
    - slope_gains: multiply each guide by this factor (up if >1, down if <1).
                   Can be a single float for all, or a list aligned with ref_slopes.
    - slope_shifts: shift by decades (Δ). gain = 10**Δ (e.g., Δ=+0.3 ≈ ×2, Δ=-0.5 ≈ ×0.316).
                    Can be float or list aligned with ref_slopes.
    If both gains and shifts are given, they are multiplied together.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot spectra
    for spectrum, label in zip(spectrums, labels):
        linestyle=None
        if label == 'ERA5':
            linestyle = 'dashed'
        if label == 'WRF':
            linestyle = 'dashed'
        ax.loglog(bins, spectrum, label=label, linestyle=linestyle)

    # Cosmetics
    ax.grid(linestyle='--')
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel(r"$k$ (cycle/km)", fontsize=14)
    ax.set_ylabel(r"$P(k)$ ($m^{2}/s^{2}$)" if 'w' in name else r"$P(k)$", fontsize=14)
    ax.set_title(f'Frequency power spectrum ({name})', fontsize=14)

    # Legend BEFORE slope guides so guides aren’t included
    ax.legend(fontsize=14)

    # ---- Reference slope guides ----
    if ref_slopes:
        k = np.asarray(bins)
        xmin, xmax = ax.get_xlim()
        k0 = np.sqrt(xmin * xmax) if ref_k is None else float(np.clip(ref_k, xmin, xmax))

        ref_idx = int(np.argmin(np.abs(k - k0)))
        ref_spec = np.asarray(spectrums[ref_source])
        P0 = float(ref_spec[ref_idx])

        if slope_style is None:
            slope_style = dict(linestyle='--', linewidth=1.5, alpha=0.85, color='0.35')
        if label_bbox is None:
            label_bbox = dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.2)

        # Resolve per-slope gains
        def as_list(x, n):
            if x is None:
                return [1.0] * n
            if isinstance(x, (int, float)):
                return [float(x)] * n
            if len(x) != n:
                raise ValueError("Length of slope_gains/slope_shifts must match ref_slopes.")
            return list(map(float, x))

        gains = as_list(slope_gains, len(ref_slopes))
        shifts = as_list(slope_shifts, len(ref_slopes))
        # Combine: total_gain_i = gains[i] * 10**shifts[i]
        total_gains = [g * (10.0**s) for g, s in zip(gains, shifts)]

        # Geometric interpolation helper for label x-position
        def geo_interp(a, b, t):
            return a * (b / a) ** t

        x_label = geo_interp(xmin, xmax, label_pos)

        def nice_exp_str(p):
            if abs(p - round(p)) < 1e-10:
                return f"{int(round(p))}"
            frac = Fraction(p).limit_denominator(10)
            return f"{frac.numerator}/{frac.denominator}" if abs(frac - p) < 1e-10 else f"{p:.2f}"

        for p, gtot in zip(ref_slopes, total_gains):
            p = float(p)
            # Base line through (k0, P0): y_base = P0 * (k/k0)^(-p)
            y_base = P0 * (k / k0) ** (-p)
            y = gtot * y_base  # vertical shift via multiplicative gain
            line, = ax.loglog(k[1:-40], y[1:-40], **slope_style)
            line.set_label("_nolegend_")

            if slope_labels:
                y_label = gtot * P0 * (x_label / k0) ** (-p)
                x2 = min(xmax * 0.999, x_label * 1.1)
                y2 = gtot * P0 * (x2 / k0) ** (-p)

                (X1, Y1) = ax.transData.transform((x_label, y_label))
                (X2, Y2) = ax.transData.transform((x2, y2))
                angle_deg = np.degrees(np.arctan2(Y2 - Y1, X2 - X1))+17

                exp_str = nice_exp_str(p)
                label_text = rf"$k^{{-{exp_str}}}$"

                ax.text(
                    x_label, y_label,
                    label_text,
                    rotation=angle_deg,
                    rotation_mode='anchor',
                    ha='left', va='center',
                    fontsize=12,
                    bbox=label_bbox
                )

    # ---- Secondary bottom axis: wavelength (km) from k ----
    def k_to_lambda(kvals):
        return 1 / np.asarray(kvals)

    ax2 = ax.twiny()
    ax2.set_xscale('log')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Wavelength (km)", fontsize=14)
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')

    xmin, xmax = ax.get_xlim()
    decades = np.arange(np.floor(np.log10(xmin)), np.ceil(np.log10(xmax)) + 1)
    k_ticks = 10.0 ** decades
    k_ticks = k_ticks[(k_ticks >= xmin * 0.999) & (k_ticks <= xmax * 1.001)]
    ax2.set_xticks(k_ticks)
    ax2.set_xticklabels([f"{lam:.1f}" for lam in k_to_lambda(k_ticks)], fontsize=14)

    plt.tight_layout()
    return fig, ax

def power_loglog_spectrum_old(spectrums, labels, bins, name):
    fig, ax = plt.subplots()
    for spectrum, label in zip(spectrums, labels):
        ax.loglog(bins, spectrum, label=label)
    ax.grid(linestyle='--')
    plt.xlabel("$k$ (cycle/km)")
    if 'w' in name:
        plt.ylabel("$P(k)$ ($m^{2}/s^{2}$)")
    else:
        plt.ylabel("$P(k)$")
    plt.title(f'Frequency power spectrum ({name})')
    plt.legend()

    # Добавление второй оси для длины волны (снизу графика)
    def k_to_lambda(k):
        return 105 * 6 / k

    ax2 = ax.twiny()  # Создаем ось сверху
    ax2.set_xscale('log')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Wavelength (km)")

    # Перенос оси ax2 снизу
    ax2.spines['bottom'].set_position(('outward', 45))  # Отодвигаем ось вниз
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')

    # Настройка значений для длины волны
    k_ticks = np.array([1, 10, 100])  # Примерные значения волнового числа
    lambda_ticks = k_to_lambda(k_ticks)
    ax2.set_xticks(k_ticks)
    ax2.set_xticklabels([f'{l:.1f}' for l in lambda_ticks])  # Форматируем метки длины волны

    plt.tight_layout()
    return fig, ax


def arcs(x, y, w, h=None, rot=0.0, theta1=0.0, theta2=360.0,
         c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter plot of Arcs.
    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Center of ellipses.
    w, h : scalar or array_like, shape (n, )
        Total length (diameter) of horizontal/vertical axis.
        `h` is set to be equal to `w` by default, ie. circle.
    rot : scalar or array_like, shape (n, )
        Rotation in degrees (anti-clockwise).
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.
    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
    Examples
    --------
    a = np.arange(11)
    arcs(a, a, w=4, h=a, rot=a*30, theta1=0.0, theta2=180.0,
         c=a, alpha=0.5, ec='none')
    plt.colorbar()
    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    if h is None:
        h = w

    zipped = np.broadcast(x, y, w, h, rot, theta1, theta2)
    patches = [Arc((x_, y_), w_, h_, angle=rot_, theta1=t1_, theta2=t2_)
               for x_, y_, w_, h_, rot_, t1_, t2_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection
