import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Dict, Optional, Sequence, Tuple
import warnings


def _letters(i: int) -> str:
    # 0 -> 'a', 25 -> 'z', 26 -> 'aa', ...
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return s

def add_column_letters_on_toprow(
    ax_grid,
    labels=None,
    x: float = 0.02,         # left padding in axes coords
    y: float = 0.98,         # near top inside the axes
    fontsize: int = 12,
    weight: str = "bold",
    box_fc: str = "white",   # box facecolor
    box_alpha: float = 0.8,  # box opacity
    box_ec: str = "none",    # box edgecolor
    box_pad: float = 0.2,    # padding in "boxstyle" units
    box_round: float = 0.1   # corner rounding in "boxstyle" units
):
    """
    Add (a), (b), (c), ... to the top-row axes. Labels are placed INSIDE the axes
    with a semi-transparent white rounded bbox for readability on dark imagery.
    """
    ax_arr = np.asarray(ax_grid)
    top = ax_arr[0] if ax_arr.ndim > 1 else ax_arr
    ncols = len(top)

    if labels is None:
        labels = [f'({_letters(j)})' for j in range(ncols)]

    texts = []
    for j, ax in enumerate(top):
        t = ax.text(
            x, y, labels[j],
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=fontsize, fontweight=weight,
            zorder=10, clip_on=False,
            bbox=dict(
                facecolor=box_fc,
                edgecolor=box_ec,
                alpha=box_alpha,
                boxstyle=f"round,pad={box_pad},rounding_size={box_round}"
            ),
        )
        texts.append(t)
    return texts



def plot_bias_correction_grid_cpy(
    samples: Dict[str, torch.Tensor],
    base_key: str,
    target_key: str,
    grid,  # Added grid parameter for lat/lon information
    proj = None,
    channel: int = 0,
    diff_sign: str = "other_minus_base",  # or "base_minus_other"
    order: Optional[Sequence[str]] = None,
    cmap_top: str = "viridis",
    cmap_bottom: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    diff_vmax: Optional[float] = None,
    centered_norm=False,
    diff_centered_norm=False,
    cbar_labels: Tuple[str, str] = ("Temperature at 2 m [K]", "Correction [K]"),
    figsize: Tuple[float, float] = None,
    nan_color: str = "lightgray",
    cbar_width_ratio: float = 0.04,
):
    # --- проверки ---
    if base_key not in samples:
        raise ValueError(f"`base_key='{base_key}'` not found in samples.")
    if target_key not in samples:
        raise ValueError(f"`target_key='{target_key}'` not found in samples.")


    # --- подготовка данных ---
    data_np = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
               for k, v in samples.items()}
    data_np = {k: arr[channel] if arr.ndim == 3 else arr for k, arr in data_np.items() if (arr.ndim == 2) or (arr.shape[0] >= channel + 1)}
    
    if figsize is None:
        figsize = (4*len(data_np), 6)
        
    # --- оставшиеся проверки ---
    if order is None:
        order = list(data_np.keys())
    if order[0] != base_key:
        raise ValueError("Первая колонка должна быть базовой моделью (to be corrected).")
    if len(order) < 2 or order[1] != target_key:
        raise ValueError("Вторая колонка должна быть целевой моделью (target).")
        
    # --- colormap с цветом для NaN ---
    top_cmap = plt.get_cmap(cmap_top).copy()
    top_cmap.set_bad(nan_color)
    bottom_cmap = plt.get_cmap(cmap_bottom).copy()
    bottom_cmap.set_bad(nan_color)
    
    
    # --- границы верхнего ряда ---
    if vmin is None or vmax is None:
        finite_vals = np.concatenate([np.ravel(a[np.isfinite(a)]) for a in data_np.values()])
        if vmin is None:
            vmin = float(np.nanpercentile(finite_vals, 1))
        if vmax is None:
            vmax = float(np.nanpercentile(finite_vals, 99))
    norm = None
    if centered_norm:
        halfrange=max(abs(vmin), abs(vmax))
        # norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        norm = colors.CenteredNorm(vcenter=0, halfrange=halfrange)
        vmin = None
        vmax = None
    
    # --- разности и их границы ---
    diffs = {}
    base_field = data_np[base_key]
    for k in order:
        if k == base_key:
            diffs[k] = None
        else:
            other = data_np[k]
            d = (other - base_field) if diff_sign == "other_minus_base" else (base_field - other)
            diffs[k] = d

    if diff_vmax is None:
        all_d = np.concatenate([np.ravel(d[np.isfinite(d)]) for d in diffs.values() if d is not None])
        diff_vmax = float(np.nanpercentile(np.abs(all_d), 99)) if all_d.size else 1.0
    diff_vmin = -diff_vmax
    diff_norm = None
    if diff_centered_norm:
        diff_norm = colors.TwoSlopeNorm(vmin=diff_vmin, vcenter=0, vmax=diff_vmax)
        diff_vmin = None
        diff_vmax = None
    
    # --- разметка: отдельный столбец под colorbar для КАЖДОЙ строки ---
    n_cols = len(order)
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2, ncols=n_cols + 1,
        width_ratios=[1]*n_cols + [cbar_width_ratio],
        height_ratios=[1, 1],
        wspace=0.02, hspace=0.02
    )
    projection = ccrs.LambertAzimuthalEqualArea(central_longitude=80.0, central_latitude=71.0) if proj is None else proj
    # Create axes with Cartopy projections
    axes = np.empty((2, n_cols), dtype=object)
    for r in range(2):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(
                gs[r, c], 
                projection=projection
            )


    # оси colorbar строго соответствуют каждой строке
    cax_top = fig.add_subplot(gs[0, -1])
    cax_bottom = fig.add_subplot(gs[1, -1])

    ims_top = []
    ims_bottom = []

    # --- верхний ряд: поля ---
    for j, key in enumerate(order):
        ax = axes[0, j]
        # Use pcolormesh instead of imshow for geographic data
        im = ax.pcolormesh(
            grid.longitude, grid.latitude, data_np[key],
            transform=ccrs.PlateCarree(),
            vmin=vmin, vmax=vmax, cmap=top_cmap, norm=norm
        )
        ims_top.append(im)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add geographic features
        # ax.add_feature(cfeature.COASTLINE, linewidth=0.05)
        
        # Configure gridlines with labels only on external edges
        gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
        gl.ylocator = mticker.FixedLocator(np.arange(-85, 90, 5))
        gl.left_labels = False  # Only left edge columns
        gl.right_labels = False
        gl.top_labels = False       # Top row
        gl.bottom_labels = False   # No bottom labels on top row
        
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(key, fontsize=16)

    # --- нижний ряд: коррекции ---
    for j, key in enumerate(order):
        ax = axes[1, j]
        if diffs[key] is None:
            ax.axis('off')
            ims_bottom.append(None)
        else:
            # Use pcolormesh instead of imshow for geographic data
            im = ax.pcolormesh(
                grid.longitude, grid.latitude, diffs[key],
                transform=ccrs.PlateCarree(),
                vmin=diff_vmin, vmax=diff_vmax, cmap=bottom_cmap, norm=diff_norm
            )
            ims_bottom.append(im)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add geographic features
            # ax.add_feature(cfeature.COASTLINE, linewidth=0.01)
            
            # Configure gridlines with labels only on external edges
            gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
            gl.ylocator = mticker.FixedLocator(np.arange(-85, 90, 5))
            gl.left_labels = (j == 1)  # Only left edge columns
            gl.right_labels = False
            gl.top_labels = False      # No top labels on bottom row
            gl.bottom_labels = True    # Bottom row

    # --- подписи строк как y-label у первой колонки ---
    axes[0, 0].set_ylabel("Model output", fontsize=16)
    axes[1, 1].set_ylabel("Correction", fontsize=16) 
    axes[1, 1].yaxis.set_label_coords(-0.2, 0.5)  # Move label further left

    # --- colorbars, выровненные по строкам ---
    mappable_top = ims_top[-1]
    fig.colorbar(mappable_top, cax=cax_top).set_label(cbar_labels[0], fontsize=14)

    # возьмём последний валидный mappable снизу
    mappable_bottom = next(im for im in reversed(ims_bottom) if im is not None)
    fig.colorbar(mappable_bottom, cax=cax_bottom).set_label(cbar_labels[1], fontsize=14)

    # небольшая подгонка полей
    # Note: Spine visibility is handled differently with Cartopy projections

    return fig, axes, (cax_top, cax_bottom)


def plot_vector_bias_correction_grid_cpy(
    samples: Dict[str, torch.Tensor],
    base_key: str,
    target_key: str,
    grid,  # Added grid parameter for lat/lon information
    proj = None,
    channel: int = 0,
    diff_sign: str = "other_minus_base",  # or "base_minus_other"
    order: Optional[Sequence[str]] = None,
    cmap_top: str = "viridis",
    cmap_bottom: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    diff_vmax: Optional[float] = None,
    centered_norm=False,
    diff_centered_norm=False,
    cbar_labels: Tuple[str, str] = ("Wind Speed [m/s]", "Speed Correction [m/s]"),
    figsize: Tuple[float, float] = None,
    nan_color: str = "lightgray",
    cbar_width_ratio: float = 0.04,
    quiver_step: int = 16,
    quiver_key_length: int = 10,
    quiver_key_units: str = "m/s",
    quiver_key_color: str = "black",
    quiver_scale: Optional[float] = None,
    diff_quiver_scale: Optional[float] = None,
):
    """
    Plot bias correction results for vector fields (wind) in a (2+N columns, 2 rows) grid.
    
    Parameters:
        samples: Dict mapping model names to tensors of shape (2, height, width) for vector fields
        base_key: Key for the base model (to be corrected)
        target_key: Key for the target model
        grid: Grid object with latitude/longitude information
        channel: Channel index (if data has more than 3 dimensions)
        diff_sign: Direction of difference calculation
        order: Order of models to display
        cmap_top: Colormap for wind speed magnitude
        cmap_bottom: Colormap for corrections
        vmin, vmax: Color scale limits for wind speed
        diff_vmax: Color scale limits for corrections (symmetric around 0)
        centered_norm, diff_centered_norm: Whether to use centered normalization
        cbar_labels: Labels for colorbars
        figsize: Figure size
        nan_color: Color for NaN values
        cbar_width_ratio: Width ratio for colorbar columns
        quiver_step: Step size for quiver subsampling
        quiver_key_length: Length of quiver key
        quiver_key_units: Units for quiver key
        quiver_key_color: Color of quiver key
        quiver_scale: Scale for quiver arrows
        diff_quiver_scale: Scale for diff quiver arrows
    """
    
    # --- проверки ---
    if base_key not in samples:
        raise ValueError(f"`base_key='{base_key}'` not found in samples.")
    if target_key not in samples:
        raise ValueError(f"`target_key='{target_key}'` not found in samples.")
    if order is None:
        order = list(samples.keys())
    if order[0] != base_key:
        raise ValueError("Первая колонка должна быть базовой моделью (to be corrected).")
    if len(order) < 2 or order[1] != target_key:
        raise ValueError("Вторая колонка должна быть целевой моделью (target).")
    if figsize is None:
        figsize = (4*len(samples), 6)
    # --- подготовка данных ---
    data_np = {k: (v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
               for k, v in samples.items()}
    
    # Extract u and v components and handle channel dimension
    data_uv = {}
    for k, arr in data_np.items():
        if arr.ndim >= 3:
            # Assume shape is (2, height, width) or (..., 2, height, width)
            if arr.shape[-3] >= 2:  # Vector components dimension
                u_data = arr[..., 0, :, :] if arr.ndim > 3 else arr[0, :, :]
                v_data = arr[..., 1, :, :] if arr.ndim > 3 else arr[1, :, :]
            else:
                raise ValueError(f"Expected at least 2 vector components (to take first two), got shape {arr.shape}")
        else:
            raise ValueError(f"Expected at least 3D array (2, H, W), got shape {arr.shape}")
        
        # Apply channel selection if needed
        if u_data.ndim > 2 and channel < u_data.ndim - 2:
            u_data = u_data[..., channel, :, :]
            v_data = v_data[..., channel, :, :]
        elif u_data.ndim == 3 and channel < u_data.shape[0]:
            u_data = u_data[channel]
            v_data = v_data[channel]
            
        data_uv[k] = (u_data, v_data)

    # Calculate wind speed magnitude for visualization
    data_speed = {k: np.sqrt(u**2 + v**2) for k, (u, v) in data_uv.items()}

    # --- colormap с цветом для NaN ---
    top_cmap = plt.get_cmap(cmap_top).copy()
    top_cmap.set_bad(nan_color)
    bottom_cmap = plt.get_cmap(cmap_bottom).copy()
    bottom_cmap.set_bad(nan_color)
    
    # --- границы верхнего ряда ---
    if vmin is None or vmax is None:
        finite_vals = np.concatenate([np.ravel(a[np.isfinite(a)]) for a in data_speed.values()])
        if vmin is None:
            vmin = float(np.nanpercentile(finite_vals, 1))
        if vmax is None:
            vmax = float(np.nanpercentile(finite_vals, 99))
    norm = None
    if centered_norm:
        halfrange=max(abs(vmin), abs(vmax))
        # norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        norm = colors.CenteredNorm(vcenter=0, halfrange=halfrange)
        vmin = None
        vmax = None
    
    # --- разности и их границы ---
    diffs_speed = {}
    diffs_uv = {}
    base_u, base_v = data_uv[base_key]
    base_speed = data_speed[base_key]
    
    for k in order:
        if k == base_key:
            diffs_speed[k] = None
            diffs_uv[k] = None
        else:
            other_u, other_v = data_uv[k]
            other_speed = data_speed[k]
            
            # Speed difference
            speed_diff = (other_speed - base_speed) if diff_sign == "other_minus_base" else (base_speed - other_speed)
            diffs_speed[k] = speed_diff
            
            # Vector difference (u and v components)
            u_diff = (other_u - base_u) if diff_sign == "other_minus_base" else (base_u - other_u)
            v_diff = (other_v - base_v) if diff_sign == "other_minus_base" else (base_v - other_v)
            diffs_uv[k] = (u_diff, v_diff)

    if diff_vmax is None:
        all_d = np.concatenate([np.ravel(d[np.isfinite(d)]) for d in diffs_speed.values() if d is not None])
        diff_vmax = float(np.nanpercentile(np.abs(all_d), 99)) if all_d.size else 1.0
    diff_vmin = -diff_vmax
    diff_norm = None
    if diff_centered_norm:
        diff_norm = colors.TwoSlopeNorm(vmin=diff_vmin, vcenter=0, vmax=diff_vmax)
        diff_vmin = None
        diff_vmax = None
    
    # --- разметка: отдельный столбец под colorbar для КАЖДОЙ строки ---
    n_cols = len(order)
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2, ncols=n_cols + 1,
        width_ratios=[1]*n_cols + [cbar_width_ratio],
        height_ratios=[1, 1],
        wspace=0.02, hspace=0.02
    )

    # Create axes with Cartopy projections
    projection = ccrs.LambertAzimuthalEqualArea(central_longitude=80.0, central_latitude=71.0) if proj is None else proj
    axes = np.empty((2, n_cols), dtype=object)
    for r in range(2):
        for c in range(n_cols):
            axes[r, c] = fig.add_subplot(
                gs[r, c], 
                projection=projection
            )

    # оси colorbar строго соответствуют каждой строке
    cax_top = fig.add_subplot(gs[0, -1])
    cax_bottom = fig.add_subplot(gs[1, -1])

    ims_top = []
    ims_bottom = []

    # --- верхний ряд: поля скорости с векторами ---
    for j, key in enumerate(order):
        ax = axes[0, j]
        
        # Plot wind speed magnitude as color field
        im = ax.pcolormesh(
            grid.longitude, grid.latitude, data_speed[key],
            transform=ccrs.PlateCarree(),
            vmin=vmin, vmax=vmax, cmap=top_cmap, norm=norm
        )
        ims_top.append(im)
        
        # Plot wind vectors
        u, v = data_uv[key]
        visualize_vector_field(ax, grid, (u, v), 
                              key_length=quiver_key_length,
                              key_units=quiver_key_units,
                              key_color=quiver_key_color,
                              step=quiver_step,
                              scale=quiver_scale)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Configure gridlines with labels only on external edges
        gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
        gl.ylocator = mticker.FixedLocator(np.arange(-85, 90, 5))
        gl.left_labels = False  # Only left edge columns
        gl.right_labels = False
        gl.top_labels = False       # Top row
        gl.bottom_labels = False   # No bottom labels on top row
        
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(key, fontsize=16)

    # --- нижний ряд: коррекции скорости с векторами коррекции ---
    for j, key in enumerate(order):
        ax = axes[1, j]
        if diffs_speed[key] is None:
            ax.axis('off')
            ims_bottom.append(None)
        else:
            # Plot speed correction as color field
            im = ax.pcolormesh(
                grid.longitude, grid.latitude, diffs_speed[key],
                transform=ccrs.PlateCarree(),
                vmin=diff_vmin, vmax=diff_vmax, cmap=bottom_cmap, norm=diff_norm
            )
            ims_bottom.append(im)
            
            # Plot vector correction
            u_diff, v_diff = diffs_uv[key]
            visualize_vector_field(ax, grid, (u_diff, v_diff),
                                  key_length=quiver_key_length,
                                  key_units=quiver_key_units,
                                  key_color=quiver_key_color,
                                  step=quiver_step,
                                  scale=diff_quiver_scale)
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Configure gridlines with labels only on external edges
            gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
            gl.ylocator = mticker.FixedLocator(np.arange(-85, 90, 5))
            gl.left_labels = (j == 1)  # Only left edge columns
            gl.right_labels = False
            gl.top_labels = False      # No top labels on bottom row
            gl.bottom_labels = True    # Bottom row

    # --- подписи строк как y-label у первой колонки ---
    axes[0, 0].set_ylabel("Model output", fontsize=16)
    # Move the 'Correction' ylabel to the left to avoid interference with gridline labels
    axes[1, 1].set_ylabel("Correction", fontsize=16)
    # Adjust the position of the Correction ylabel to the left
    axes[1, 1].yaxis.set_label_coords(-0.15, 0.5)  # Move label further left

    # --- colorbars, выровненные по строкам ---
    mappable_top = ims_top[-1]
    fig.colorbar(mappable_top, cax=cax_top).set_label(cbar_labels[0], fontsize=14)

    # возьмём последний валидный mappable снизу
    mappable_bottom = next(im for im in reversed(ims_bottom) if im is not None)
    fig.colorbar(mappable_bottom, cax=cax_bottom).set_label(cbar_labels[1], fontsize=14)

    return fig, axes, (cax_top, cax_bottom)


def fix_quiver_bug(field, lat):
    """
    Fixes a bug in quiver vector plots where the u-component (eastward) vector is distorted
    due to the curvature of latitude in polar projections.
    """
    ufield, vfield = field
    # Compute the original magnitude of the vector field
    old_magnitude = np.sqrt(ufield ** 2 + vfield ** 2)
    # Adjust the u-component by accounting for latitude distortion
    ufield_fixed = ufield / np.cos(np.radians(lat))
    # Compute the new magnitude after fixing the u-component
    new_magnitude = np.sqrt(ufield_fixed ** 2 + vfield ** 2)
    # Rescale the vector field to maintain original magnitudes
    field_fixed = np.stack([ufield_fixed, vfield]) * old_magnitude / new_magnitude.clip(min=1e-6)
    return field_fixed


def block_average(arr, step, min_valid=None):
    """
    Downsample a 2D array by taking the mean over non-overlapping blocks of size step x step.
    Crops the array to make its dimensions divisible by step. Only includes averaged values where
    the number of valid (non-NaN) pixels is at least `min_valid`, otherwise sets result to NaN.
    """
    h, w = arr.shape
    h2 = (h // step) * step
    w2 = (w // step) * step
    cropped = arr[:h2, :w2]
    reshaped = cropped.reshape(h2 // step, step, w2 // step, step)
    valid_counts = np.sum(~np.isnan(reshaped), axis=(1, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        means = np.nanmean(reshaped, axis=(1, 3))
    min_valid = min_valid if min_valid is not None else step * step / 2 
    means[valid_counts < min_valid] = np.nan
    return means


def visualize_vector_field(ax, grid, field, key_length=50, key_units='cm/s', key_color='black', 
                           from_polar=False, from_direction=True, step=64, use_pooling=True, min_valid=5,
                           scale=None, width=0.002, headwidth=3, headlength=5):
    """
    Visualizes a vector field on the map using quiver arrows, with optional block average pooling.
    Uses block-averaging for vector components but selects the geographic center of each block for
    lon/lat, ensuring no seam artifacts at the 180° meridian.
    """
    import warnings
    
    if from_polar:
        norm, angle = field
        u, v = polar_to_cartesian(norm, angle, from_direction=from_direction)
    else:
        u, v = field

    u_fixed, v_fixed = fix_quiver_bug((u, v), grid.latitude)
    h, w = grid.longitude.shape

    if use_pooling and step > 1:
        # Average vector components
        u_p = block_average(u_fixed, step, min_valid)
        v_p = block_average(v_fixed, step, min_valid)

        # Determine block centers for coordinates (no averaging lon/lat values)
        h2 = (h // step) * step
        w2 = (w // step) * step
        # indices at center of each block
        i_centers = (np.arange(step//2, h2, step)).astype(int)
        j_centers = (np.arange(step//2, w2, step)).astype(int)
        lon_p = grid.longitude[i_centers[:, None], j_centers[None, :]]
        lat_p = grid.latitude[i_centers[:, None], j_centers[None, :]]

        mask = (~np.isnan(u_p) & ~np.isnan(v_p))
        layer = ax.quiver(
            lon_p[mask], lat_p[mask], u_p[mask], v_p[mask],
            transform=ccrs.PlateCarree(), color=key_color,
            scale=scale,
            scale_units='height',
            width=width,
            headwidth=headwidth,
            headlength=headlength
        )
    else:
        layer = ax.quiver(
            grid.longitude[::step, ::step], grid.latitude[::step, ::step],
            u_fixed[::step, ::step], v_fixed[::step, ::step],
            transform=ccrs.PlateCarree(), color=key_color,
            scale_units='height',
            width=width,
            headwidth=headwidth,
            headlength=headlength
        )

    # ax.quiverkey(layer, X=0.69, Y=0.2, U=key_length, label=f'{key_length} {key_units}',
    #              labelpos='E', coordinates='axes')
    return layer


def polar_to_cartesian(norm, angle, from_direction=True):
    """Convert polar coordinates to cartesian."""
    if from_direction:
        # angle is meteorological direction (degrees clockwise from north)
        u = -norm * np.sin(np.radians(angle))
        v = -norm * np.cos(np.radians(angle))
    else:
        # angle is mathematical angle (degrees counterclockwise from east)
        u = norm * np.cos(np.radians(angle))
        v = norm * np.sin(np.radians(angle))
    return u, v


def cartesian_to_polar(u, v, to_direction=True):
    """Convert cartesian coordinates to polar."""
    norm = np.sqrt(u**2 + v**2)
    if to_direction:
        # Convert to meteorological direction (degrees clockwise from north)
        angle = np.degrees(np.arctan2(-u, -v)) % 360
    else:
        # Convert to mathematical angle (degrees counterclockwise from east)
        angle = np.degrees(np.arctan2(v, u)) % 360
    return norm, angle