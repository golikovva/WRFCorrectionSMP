# Add "import cartopy" to the top of your Jupyter notebook,
# before using these functions, or visualizations will fail.
import datetime
from datetime import date
from typing import Dict, Optional
import calendar
import warnings
from collections.abc import Mapping
import cartopy.crs as ccrs  # For cartographic projections in visualizations
import cartopy.feature as cfeature  # For adding geographic features (land, oceans, etc.)
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting
import matplotlib.dates as mdates
import matplotlib.colors as colors
import matplotlib.ticker as mticker
from matplotlib.path import Path
from matplotlib.cm import get_cmap
from matplotlib.axes import Axes as MplAxes
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Arc
from mpl_toolkits.axes_grid1 import make_axes_locatable
try:
    from cartopy.mpl.geoaxes import GeoAxes
except Exception:
    GeoAxes = None


def get_domain_projection(domain_name):
    name = domain_name.lower()
    if 'borey' in name:
        return ccrs.LambertAzimuthalEqualArea(
            central_longitude=80.0,
            central_latitude=71.0
        )
    elif 'pan' in name or 'arctic' in name:
        return ccrs.NorthPolarStereo(central_longitude=120.0)
    elif 'smp' in name or 'nestp' in name:
        return ccrs.NorthPolarStereo(central_longitude=120.0)
    elif any(k in name for k in ['glorys', 'global', 'world']):
        return ccrs.Robinson(central_longitude=120.0)
    else:
        return None

def get_domain_extent(domain_name):
    name = domain_name.lower()

    if 'borey' in name:
        return [-1850799.028266253, -169147.24810465102,
                 -390064.97663009784, 881589.4853213852]
    elif 'pan' in name or 'arctic' in name:
        return None   # todo fill when ready
    elif 'smp' in name or 'nestp' in name:
        return [-3178711.951944511, 3178711.952368256,
                -2252806.5461317636, 239082.4972454039]
    elif any(k in name for k in ['glorys', 'global', 'world']):
        return [-180, 180, -90, 90]
    else:
        return None

def set_domain_extent(ax, domain, grid=None):
    src = ccrs.PlateCarree()

    if isinstance(domain, str):
        domain_name = domain
        domain_proj = get_domain_projection(domain_name)
        if domain_proj is None:
            raise ValueError(f'Unknown domain name: {domain_name}')
    else:
        domain_name = None
        domain_proj = domain

    if grid is None:
        if domain_name is None:
            raise ValueError(
                "When grid is None, `domain` must be a string domain name "
                "so that extent can be looked up."
            )
        extent_proj = get_domain_extent(domain_name)
        if extent_proj is None:
            raise ValueError(f'No predefined extent for domain: {domain_name}')
    else:
        lat2d, lon2d = lat_lon_from_grid(grid)

        xy = domain_proj.transform_points(src, lon2d, lat2d)
        x = xy[..., 0]
        y = xy[..., 1]

        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            raise ValueError("No finite projected coordinates found for grid")

        extent_proj = [
            np.nanmin(x[mask]),
            np.nanmax(x[mask]),
            np.nanmin(y[mask]),
            np.nanmax(y[mask]),
        ]

    ax.set_extent(extent_proj, crs=domain_proj)

def lat_lon_from_grid(grid):
    lat_names = ['lat', 'latitude', 'XLAT', 'Latitude']
    lon_names = ['lon', 'long', 'longitude', 'XLON', 'XLONG', 'Longitude']

    def _looks_like_array(x):
        # Reject addict auto-created empties / dicts
        if isinstance(x, Mapping):
            return False
        # Accept numpy arrays and array-like objects
        return isinstance(x, np.ndarray) or hasattr(x, "shape")

    def _get(obj, names):
        # 1) Mapping path (safe for addict.Dict)
        if isinstance(obj, Mapping):
            for name in names:
                if name in obj:              # does NOT create in addict
                    val = obj[name]
                    if _looks_like_array(val):
                        return val

        # 2) Attribute path (for non-mapping grid objects)
        for name in names:
            try:
                val = getattr(obj, name)
            except AttributeError:
                continue
            if _looks_like_array(val):
                return val

        return None

    lat = _get(grid, lat_names)
    lon = _get(grid, lon_names)
    return lat, lon

def fix_quiver_bug(field, lat):
    """
    Fixes a bug in quiver vector plots where the u-component (eastward) vector is distorted
    due to the curvature of latitude in polar projections.

    Args:
        field (tuple of np.array): Tuple containing the u and v components of the vector field.
        lat (np.array): Array of latitudes corresponding to the field.

    Returns:
        np.array: Fixed vector field with adjusted u-component.
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

def _nice_step(span_deg):
    """Pick a meridian/parallel step that gives ~5–8 lines."""
    for s in [0.25, 0.5, 1, 2, 5, 10, 15, 20, 30, 45, 60]:
        if span_deg / s <= 8:
            return s
    return 60

def _outer_ring(lon, lat):
    """Counter-clockwise outer ring of a 2D lon/lat grid."""
    top    = np.c_[lon[0, :],          lat[0, :]]
    right  = np.c_[lon[1:, -1],        lat[1:, -1]]
    bottom = np.c_[lon[-1, -2::-1],    lat[-1, -2::-1]]
    left   = np.c_[lon[-2:0:-1, 0],    lat[-2:0:-1, 0]]
    return np.vstack([top, right, bottom, left])

def _tight_projected_limits(lon, lat, target_crs, src_crs=ccrs.PlateCarree()):
    """Compute tight (xmin,xmax,ymin,ymax) in target_crs for a lon/lat grid."""
    ring = _outer_ring(lon, lat)
    xy   = target_crs.transform_points(src_crs, ring[:, 0], ring[:, 1])
    x, y = xy[:, 0], xy[:, 1]
    return np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)

def _add_graticule(ax, lon, lat):
    """Smart meridian/parallel labels for the current data domain."""
    src = ccrs.PlateCarree()
    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    gl = ax.gridlines(crs=src, draw_labels=True, linewidth=0.6,
                      linestyle=':', color='k', alpha=0.35,
                      x_inline=False, y_inline=True)
    gl.right_labels = False
    gl.top_labels   = False
    gl.rotate_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    gl.xlocator = mticker.MultipleLocator(_nice_step(lon_max - lon_min))
    gl.ylocator = mticker.MultipleLocator(_nice_step(lat_max - lat_min))
    return gl

def add_colorbar_aligned(ax, mappable, where="right", size="4.5%", pad=0.04,
                         label=None, orientation="vertical"):
    """
    Perfectly aligned colorbar next to a GeoAxes (no manual shrink).
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(where, size=size, pad=pad, axes_class=MplAxes)
    cb = ax.figure.colorbar(mappable, cax=cax, orientation=orientation)
    if label:
        cb.ax.set_ylabel(label)
    return cb


def create_cartopy_axes(
    nrows: int = 1,
    ncols: int = 1,
    *,
    coastline_resolution: str = '110m',
    central_longitude: float = 120.0,
    figsize=None,
    ax_size: float = 6.0,
    grid=None,                 # optional: object with .latitude/.longitude to auto-aspect single map
    add_land: bool = True,
    add_gridlines: bool = True,
    face_ocean: bool = True,
    proj = None,
):
    """
    Create one or a grid of Cartopy maps (North Polar Stereographic).
    Always returns (fig, axes) where axes is a 2D array of GeoAxes.

    Parameters
    ----------
    nrows, ncols : int
        Grid dimensions.
    coastline_resolution : {'110m','50m','10m'}
        Natural Earth scale for land feature (and optional coastlines).
    central_longitude : float
        Central meridian of the NorthPolarStereo projection.
    figsize : (W,H) or None
        Figure size in inches. If None:
          * (ax_size*ncols*aspect, ax_size) if grid provided, else (ax_size*ncols, ax_size*nrows)
    ax_size : float
        Size (inches) of each panel’s height when figsize is None.
    grid : object or dict with .latitude/.longitude (or ['lat','lon'])
        Used only for single-panel auto-aspect calculation.
    add_land, face_ocean : bool
        Styling toggles.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray shape (nrows, ncols) of cartopy.mpl.geoaxes.GeoAxes if nrows + ncols > 1, else cartopy.mpl.geoaxes.GeoAxes
    """

    # Decide figsize
    # proj = ccrs.NorthPolarStereo(central_longitude=central_longitude)
    if proj is None:
        proj = ccrs.Robinson(central_longitude=central_longitude)
    if figsize is None:
        aspect = 1
        if grid is not None:
            # allow dict-like or attribute access
            lat = np.asarray(getattr(grid, 'latitude', grid.get('lat', grid.get('latitude'))))
            lon = np.asarray(getattr(grid, 'longitude', grid.get('lon', grid.get('longitude'))))
            xmin, xmax, ymin, ymax = _tight_projected_limits(lon, lat, proj)
            print(xmin, xmax, ymin, ymax, 'plot boundaries')
            aspect = (xmax - xmin) / (ymax - ymin) if (ymax > ymin) else 1.0

        figsize = (ax_size * ncols * aspect, ax_size * nrows)
    print(figsize)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                             subplot_kw={'projection': proj}, squeeze=False)
    for row in axes:
        for ax in row:
            if face_ocean:
                ax.set_facecolor(cfeature.COLORS['water'])
            if add_land:
                land = cfeature.NaturalEarthFeature(
                    'physical', 'land', coastline_resolution,
                    edgecolor=cfeature.COLORS['land'],
                    facecolor=cfeature.COLORS['land']
                )
                ax.add_feature(land, zorder=0)
            if add_gridlines:
                ax.gridlines(draw_labels=True, color='gray', zorder=9)
                
    if nrows == 1 and ncols == 1:
        axes = axes[0, 0]
    return fig, axes

def true_ndim(a) -> int:
    a = np.asarray(a)
    shp = a.shape
    return int(np.count_nonzero(np.array(shp) > 1))

def visualize_scalar_field(grid, field, *,
                           ax=None,
                           lat=None, lon=None,
                           add_graticule=True, add_colorbar=False, cbar_label='Variable units', **kwargs):
    """
    Plot a scalar field on the provided GeoAxes, make the domain fill the axes,
    and (optionally) add a perfectly aligned colorbar.
    """
    # pull lat/lon
    if lat is None and lon is None:
        if hasattr(grid, 'latitude'): lat = grid.latitude
        elif hasattr(grid, 'lat'):    lat = grid.lat
        else:
            key = [k for k in grid.keys() if 'lat' in k.lower()][0]
            lat = grid[key]
    if lon is None:
        if hasattr(grid, 'longitude'): lon = grid.longitude
        elif hasattr(grid, 'lon'):      lon = grid.lon
        else:
            key = [k for k in grid.keys() if 'lon' in k.lower()][0]
            lon = grid[key]
            
    if ax is None:
        _, ax = create_cartopy_axes(grid=grid)

    if field.ndim >= 3 and true_ndim(field) == 2:
        field = np.squeeze(field)
    assert field.ndim == 2, "Field must be 2D after squeezing"
    
    src_crs = ccrs.PlateCarree()
    m = ax.pcolormesh(lon, lat, field, transform=src_crs, shading='auto', **kwargs)

    # # compute tight projected limits and set rectangular boundary
    # xmin, xmax, ymin, ymax = _tight_projected_limits(lon, lat, ax.projection, src_crs)
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    # rect_path = Path([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)],
    #                  [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
    # ax.set_boundary(rect_path, transform=ax.transData)
    # ax.set_aspect('equal', adjustable='box')

    # # grid
    # if add_graticule:
    #     _add_graticule(ax, lon, lat)

    # # colorbar
    # cb = None
    # if add_colorbar:
    #     cb = add_colorbar_aligned(ax, m, where="right", size="4.5%", pad=0.04,
    #                               label=cbar_label, orientation='vertical')

    return m


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


def visualize_vector_field(ax, grid, field, key_length=50, draw_quiverkey=True, key_units='cm/s', key_color='black', 
                           from_polar=False, from_direction=True, step=64, use_pooling=True, min_valid=5,
                           scale=None, width=0.002, headwidth=3, headlength=5):
    """
    Visualizes a vector field on the map using quiver arrows, with optional block average pooling.
    Uses block-averaging for vector components but selects the geographic center of each block for
    lon/lat, ensuring no seam artifacts at the 180° meridian.
    """
    if from_polar:
        norm, angle = field
        u, v = polar_to_cartesian(norm, angle, from_direction=from_direction)
    else:
        u, v = field

    u_fixed, v_fixed = fix_quiver_bug((u, v), grid.lat)
    h, w = grid.lon.shape

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
        lon_p = grid.lon[i_centers[:, None], j_centers[None, :]]
        lat_p = grid.lat[i_centers[:, None], j_centers[None, :]]

        mask = (~np.isnan(u_p) & ~np.isnan(v_p))
        layer = ax.quiver(
            lon_p[mask], lat_p[mask], u_p[mask], v_p[mask],
            transform=ccrs.PlateCarree(), color=key_color,
            scale=scale,
            width=width,
            headwidth=headwidth,
            headlength=headlength
        )
    else:
        layer = ax.quiver(
            grid.lon[::step, ::step], grid.lat[::step, ::step],
            u_fixed[::step, ::step], v_fixed[::step, ::step],
            transform=ccrs.PlateCarree(), color=key_color,
            scale=scale,
            width=width,
            headwidth=headwidth,
            headlength=headlength
        )
    if draw_quiverkey:
        ax.quiverkey(layer, X=0.69, Y=0.2, U=key_length, label=f'{key_length} {key_units}',
                    labelpos='E', coordinates='axes')
    return layer


def get_color_params(metric_name, vmin, vmax):
    params = {
        'norm': None,
        'cmap': None,
        'vmin': None,
        'vmax': None,
    }
    if metric_name in plt.colormaps:
        params['cmap'] = plt.colormaps[metric_name]
        params['vmin'] = vmin
        params['vmax'] = vmax
    if 'mse' in metric_name or 'mae' in metric_name:
        params['cmap'] = plt.colormaps['magma_r']
        params['vmin'] = vmin
        params['vmax'] = vmax
    elif 'diff' in metric_name:
        params['cmap'] = plt.colormaps['RdBu_r']
        abs_max = max(abs(vmin), abs(vmax))
        params['norm'] = colors.TwoSlopeNorm(vmin=-abs_max,
                                             vcenter=0,
                                             vmax=abs_max)
    elif any(m in metric_name for m in ['ice', 'identity']):
        import cmocean
        params['cmap'] = cmocean.cm.ice
        params['vmin'] = vmin
        params['vmax'] = vmax
    else:
        params['vmin'] = vmin
        params['vmax'] = vmax
    return params


def plot_error_evolution(
    error_data: Dict[date, float],
    title: str = "Error Evolution Over Time",
    xlabel: str = "Date",
    ylabel: str = "Error",
    color: str = "tab:blue",
    figsize: tuple = (12, 6),
    grid: bool = True,
    marker: str = None,
    date_format: str = "%Y-%m-%d",
    label_rotation: int = 45,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot one or multiple error evolution charts on the same figure.
    
    Args:
        error_data: Dictionary mapping dates to error values
        ax: Existing axes to plot on (for multiple datasets)
        label: Legend label for this dataset
        ... (other parameters remain the same)
    """
    # Extract and sort dates and errors
    start_date = min(error_data)
    end_date = max(error_data)
    dates = [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    errors=[]
    for d in dates:
        value = error_data.get(d, np.array([np.nan]))
        try:
            value = value.item() # Convert to scalar if possible
        except AttributeError:
            pass
        errors.append(value)

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        new_plot = True
    else:
        fig = ax.figure
        new_plot = False

    # Plot the data
    ax.plot(dates, errors, marker=marker, linestyle="-", 
            color=color, label=label)
    if label:
        ax.legend()
    # Only configure axis properties for new plots
    if new_plot:
        # Configure date formatting
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        
        # Rotate and align labels
        plt.setp(ax.get_xticklabels(), rotation=label_rotation, ha="right")

        # Set labels and title
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        
        # Add grid
        if grid:
            ax.grid(True, alpha=0.3)

        # Adjust layout
        fig.tight_layout()
    return fig, ax


def plot_error_cycle(
    error_data: Dict[date, float],
    cycle: str = "month",  # "month" or "daily"
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Error",
    color: str = "tab:blue",
    figsize: tuple = (12, 6),
    grid: bool = True,
    marker: Optional[str] = None,
    linestyle: str = "-",
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    interquantile_range: bool = False,
    interdecile_range: bool = False,
    std_range: bool = False,
    aggregation_func: str = 'nanmean',
    label_rotation: int = 45,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot aggregated error over a year cycle, either by calendar month or by day-of-year,
    and always align month-cycle points to the correct day-of-year positions so that
    monthly and daily curves overlay properly.

    Args:
        error_data: Dict mapping datetime.date -> float error.
        cycle: 'month' for monthly means (plotted at each month's start doy),
               'daily' for day-of-year means (1..365).
        interquantile_range: show 25–75 percentile band
        interdecile_range: show 10–90 percentile band
        std_range: show ±2 standard deviation band
        aggregation_func: NumPy function name like 'nanmean', 'nanmedian', etc.
    Returns:
        (fig, ax): the matplotlib figure and axes.
    """
    # Validate cycle
    if cycle not in {"month", "daily"}:
        raise ValueError("cycle must be either 'month' or 'daily'")

    # Prepare binning and x positions
    if cycle == "month":
        bins = list(range(1, 13))  # months
        xs = [date(2001, m, 1).timetuple().tm_yday for m in bins]
        default_title = "Monthly Error Cycle"
        default_xlabel = "Month"
        xticks = xs
        xtick_labels = [calendar.month_abbr[m] for m in bins]
    else:
        bins = list(range(1, 366))  # day-of-year
        xs = bins
        default_title = "Daily Error Cycle"
        default_xlabel = "Day of Year"
        xticks = [date(2001, m, 1).timetuple().tm_yday for m in range(1, 13)]
        xtick_labels = [calendar.month_abbr[m] for m in range(1, 13)]

    # Group errors into bins
    grouped: Dict[int, list] = {b: [] for b in bins}
    for d, err in error_data.items():
        b = d.month if cycle == "month" else d.timetuple().tm_yday
        if b in grouped:
            grouped[b].append(err)

    # Get aggregation function
    try:
        agg_func = getattr(np, aggregation_func)
    except AttributeError:
        raise ValueError(f"aggregation_func '{aggregation_func}' not found in numpy")

    # Compute aggregated values per bin
    agg_vals = [agg_func(grouped[b]) if grouped[b] else np.nan for b in bins]
    # Compute optional bands
    if interquantile_range:
        p25 = [np.nanpercentile(grouped[b], 25) if grouped[b] else np.nan for b in bins]
        p75 = [np.nanpercentile(grouped[b], 75) if grouped[b] else np.nan for b in bins]
    if interdecile_range:
        p10 = [np.nanpercentile(grouped[b], 10) if grouped[b] else np.nan for b in bins]
        p90 = [np.nanpercentile(grouped[b], 90) if grouped[b] else np.nan for b in bins]
    if std_range:
        std = [np.nanstd(grouped[b]) if grouped[b] else np.nan for b in bins]
        lower = [m - 2*s if not np.isnan(m) and not np.isnan(s) else np.nan for m, s in zip(agg_vals, std)]
        upper = [m + 2*s if not np.isnan(m) and not np.isnan(s) else np.nan for m, s in zip(agg_vals, std)]

    # Prepare figure/axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot main curve at correct x positions
    ax.plot(xs, agg_vals, marker=marker, linestyle=linestyle,
            color=color, label=label or aggregation_func)

    # Shade percentile and std bands
    base_rgba = colors.to_rgba(color)
    if interdecile_range:
        ax.fill_between(xs, p10, p90, color=base_rgba, alpha=0.15, label="10–90 %")
    if interquantile_range:
        ax.fill_between(xs, p25, p75, color=base_rgba, alpha=0.25, label="25–75 %")
    if std_range:
        ax.fill_between(xs, lower, upper, color=base_rgba, alpha=0.2, label="±2 σ")

    # Formatting
    ax.set(
        title=title or default_title,
        xlabel=xlabel or default_xlabel,
        ylabel=ylabel
    )
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=label_rotation, ha="right")
    if grid:
        ax.grid(alpha=0.3)
    if label or interquantile_range or interdecile_range or std_range:
        ax.legend()

    fig.tight_layout()
    return fig, ax

def polar_to_cartesian(norm: np.ndarray, angle_deg: np.ndarray, from_direction=True) -> tuple:
    """
    Converts polar coordinates (norm, angle) to Cartesian (u, v).
    
    Parameters:
        norm (np.ndarray): Magnitude of the vector (same shape as angle).
        angle_deg (np.ndarray): Angle in degrees.
        from_direction (bool): If True, assumes meteorological convention (angle is the direction FROM which the vector comes).
        
    Returns:
        tuple: u and v components of the vector.
    """
    angle_rad = np.radians(angle_deg)
    
    if from_direction:
        # Meteorological: wind FROM direction → invert angle
        angle_rad = np.radians(270 - angle_deg)
    
    u = norm * np.cos(angle_rad)
    v = norm * np.sin(angle_rad)
    return u, v

def cartesian_to_polar(u: np.ndarray, v: np.ndarray, to_direction=True) -> tuple:
    """
    Converts Cartesian vector components (u, v) to polar coordinates (norm, angle).
    
    Parameters:
        u (np.ndarray): Zonal (eastward) component.
        v (np.ndarray): Meridional (northward) component.
        to_direction (bool): If True, converts to meteorological direction (angle the vector comes FROM).
        
    Returns:
        tuple: 
            - norm (np.ndarray): Magnitude of the vector.
            - angle_deg (np.ndarray): Angle in degrees. 
                If to_direction=True, angle is meteorological "FROM" direction (0° = from north).
                If to_direction=False, angle is standard math angle (0° = east, counter-clockwise).
    """
    norm = np.sqrt(u**2 + v**2)
    
    # Get standard angle in radians: 0 = east, pi/2 = north, etc.
    angle_rad = np.arctan2(v, u)  # Range: [-π, π]
    angle_deg = np.degrees(angle_rad)  # Convert to degrees

    # Convert to [0, 360)
    angle_deg = (angle_deg + 360) % 360

    if to_direction:
        # Convert to meteorological FROM direction (0° = from north, clockwise)
        # u = norm * cos(theta), v = norm * sin(theta)
        # So the direction the vector comes FROM is 270 - angle
        angle_deg = (270 - angle_deg) % 360

    return norm, angle_deg

def visualize_full_vector_field(ax, grid, field, key_length=50, key_units='cm/s', key_color='black', draw_quiverkey=True,
                                from_polar=False, from_direction=True, step=32, use_pooling=True,
                                min_valid=5, scale=None, width=0.002, headwidth=3, headlength=5,
                                **kwargs):


    if from_polar:
        norm, angle = field
        u, v = polar_to_cartesian(norm, angle, from_direction=from_direction)
    else:
        u, v = field
        norm, angle = cartesian_to_polar(u, v, to_direction=from_direction)

    layer = visualize_scalar_field(ax, grid, norm, if_colorbar=False, **kwargs)
    visualize_vector_field(ax, grid, (u, v), key_length, draw_quiverkey, key_units, key_color, False, from_direction, step,
                           use_pooling=use_pooling, min_valid=min_valid, scale=scale, width=width,
                           headwidth=headwidth, headlength=headlength)
    return layer

def plot_vector_field_scatter(errors, units='cm/s'):
    """
    Plots vector field errors with aligned marginal distributions and standard deviation circle.
    
    Parameters:
    errors (numpy.ndarray): Array of shape (N, 2) where each row is (u_error, v_error)
    """
    # Validate input shape
    if errors.shape[1] != 2:
        raise ValueError("Input array must have shape (N, 2)")
    
    # Calculate statistics
    mean_u, mean_v = np.mean(errors, axis=0)
    std_u, std_v = np.std(errors, axis=0)
    radius = np.sqrt(std_u**2 + std_v**2)
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 4)
    
    # Main scatter plot
    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    
    # Plot error samples
    ax_scatter.scatter(errors[:, 0], errors[:, 1], s=10, alpha=0.5, color='blue')
    
    # Draw standard deviation circle
    circle = plt.Circle((0, 0), radius=radius, fill=False, color='red', 
                        linewidth=2, linestyle='-', label=f'Std Dev Circle (r={radius:.1f} {units})')
    ax_scatter.add_patch(circle)
    
    # Mark origin and mean
    ax_scatter.plot(0, 0, 'k+', markersize=12, label='No Bias (0,0)')
    ax_scatter.plot(mean_u, mean_v, 'ro', markersize=8, label=f'Mean Error ({mean_u:.1f}, {mean_v:.1f})')
    
    # Set axis limits (symmetric)
    max_val = max(15, np.max(np.abs(errors)) * 1.1, radius * 1.1)
    ax_scatter.set_xlim(-max_val, max_val)
    ax_scatter.set_ylim(-max_val, max_val)
    
    # Labels and grid
    ax_scatter.set_xlabel(f'Bias along U direction {units}', fontsize=12)
    ax_scatter.set_ylabel(f'Bias along V direction {units}', fontsize=12)
    ax_scatter.grid(True, linestyle='--', alpha=0.7)
    ax_scatter.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_scatter.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax_scatter.legend(loc='best')
    
    # Marginal histograms
    ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)
    
    # U-error histogram (top)
    ax_histx.hist(errors[:, 0], bins=50, color='blue', alpha=0.7, density=True,
                  range=(-max_val, max_val))
    ax_histx.axvline(mean_u, color='red', linestyle='-', label=f'Mean = {mean_u:.1f}')
    ax_histx.axvline(mean_u - std_u, color='green', linestyle='--', label=f'±1σ = {std_u:.1f}')
    ax_histx.axvline(mean_u + std_u, color='green', linestyle='--')
    ax_histx.set_title('U-error Distribution', fontsize=10)
    ax_histx.legend(fontsize=8)
    ax_histx.set_yticks([])
    
    # V-error histogram (right)
    ax_histy.hist(errors[:, 1], bins=50, color='blue', alpha=0.7, 
                 orientation='horizontal', density=True,
                 range=(-max_val, max_val))
    ax_histy.axhline(mean_v, color='red', linestyle='-', label=f'Mean = {mean_v:.1f}')
    ax_histy.axhline(mean_v - std_v, color='green', linestyle='--', label=f'±1σ = {std_v:.1f}')
    ax_histy.axhline(mean_v + std_v, color='green', linestyle='--')
    ax_histy.set_title('V-error Distribution', fontsize=10)
    ax_histy.legend(fontsize=8)
    ax_histy.set_xticks([])
    
    # Remove tick labels from histograms to avoid duplication
    plt.setp(ax_histx.get_xticklabels(), visible=False)
    plt.setp(ax_histy.get_yticklabels(), visible=False)
    
    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    return fig, (ax_scatter, ax_histx, ax_histy)


# ---------- helper: draw many colored arc sectors on a given Axes ----------
def _arcs_on_ax(ax, x, y, w, h=None, theta1=0.0, theta2=360.0, colors=None, lw=1.6, zorder=10, **kwargs):
    """
    Vectorized arc drawer that *does not* rely on global plt.gca().
    x,y,w,h,theta1,theta2 can be broadcastable arrays.
    'colors' can be a single RGBA or an (N,4) array for per-arc colors.
    """
    if h is None:
        h = w
    # Broadcast all to the same shape
    x, y, w, h, t1, t2 = np.broadcast_arrays(x, y, w, h, theta1, theta2)
    patches = [Arc((x[i], y[i]), float(w[i]), float(h[i]),
                   angle=0.0, theta1=float(t1[i]), theta2=float(t2[i]))
               for i in range(x.size)]
    coll = PatchCollection(patches, facecolor='none', edgecolor='k', linewidths=lw, zorder=zorder, **kwargs)
    if colors is not None:
        # If array of RGBA provided, Matplotlib interprets it per-patch
        coll.set_edgecolor(colors)
    ax.add_collection(coll)
    ax.autoscale_view()
    return coll

# ---------- main: Cartopy-native station metrics overlay ----------
def draw_station_metrics(
    *,
    # Background (optional but recommended for tight limits)
    grid=None,          # dict-like or object with .latitude/.longitude (2D arrays)
    sample_field=None,         # 2D field to pcolormesh using your visualize_scalar_field
    central_longitude=120.0,
    ax=None,

    # Station positions
    stations_grid=None,      # 1D array (if None, try to infer from metadata)

    # Metrics to plot per station
    station_metrics=None,   # dict: { "t2": 1D array, "v10": 1D array, ... }
    metric_limits=None,     # dict: { "t2": (vmin, vmax), ... } or None → global vmin/vmax
    cmap='bwr_r',

    # Appearance
    diameter_m=50_000,      # arc diameter in *projection units* (meters for NorthPolarStereo)
    sector_gap_deg=6.0,     # small gap between adjacent sectors, degrees
    start_angle_deg=90.0,   # where the first sector starts (12 o'clock)
    station_marker=True,    # draw station centers
    station_marker_kw=None, # dict for ax.scatter (size, color, etc.)

    # Decorations
    title='Stations metrics',
    add_graticule=True,
    add_colorbars=True,
    cbar_size="3.5%",       # slightly narrower bars when multiple
    cbar_pad=0.04
):
    """
    Plot (optional) sample field background and overlay station metrics as colored arc sectors
    on a Cartopy GeoAxes using your helper functions.

    Returns
    -------
    fig, ax, artists : tuple
        artists: dict with keys 'sectors' (list of PatchCollections), 'station_scatter' (PathCollection or None), 'colorbars' (list)
    """
    assert station_metrics is not None and len(station_metrics) > 0, "station_metrics must be a non-empty dict"
    metric_names = list(station_metrics.keys())
    K = len(metric_names)

    # Make / fetch axes
    if ax is None:
        fig, ax = create_cartopy_axes(
            nrows=1, ncols=1,
            central_longitude=central_longitude,
            grid=grid if grid is not None else None,
            add_land=True, face_ocean=True
        )
    else:
        fig = ax.figure

    # Background
    if grid is not None and sample_field is not None:
        # Use your helper; it already sets tight limits + boundary
        visualize_scalar_field(
            grid, sample_field, ax=ax,
            add_graticule=add_graticule,
            add_colorbar=False
        )
    else:
        # If no background, at least set face and optional graticule
        if add_graticule:
            src = ccrs.PlateCarree()
            # Mild default grid for orientation
            gl = ax.gridlines(crs=src, draw_labels=True, linewidth=0.6,
                              linestyle=':', color='k', alpha=0.3,
                              x_inline=False, y_inline=True)
            gl.right_labels = False
            gl.top_labels = False

    station_lats = np.asarray(stations_grid.latitude).ravel()
    station_lons = np.asarray(stations_grid.longitude).ravel()
    N = station_lats.size
    assert all(len(np.asarray(v).ravel()) == N for v in station_metrics.values()), "All metric arrays must match station count"

    # Transform station coords into projection coordinates (meters in NorthPolarStereo)
    src = ccrs.PlateCarree()
    pts = ax.projection.transform_points(src, station_lons, station_lats)
    xs, ys = pts[:, 0], pts[:, 1]

    # Optionally, if no background grid was provided, set tight rectangular limits around stations
    if grid is None:
        pad = 300_000.0  # 300 km padding
        xmin, xmax = xs.min() - pad, xs.max() + pad
        ymin, ymax = ys.min() - pad, ys.max() + pad
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Set rectangular boundary so the axes are "filled"
        from matplotlib.path import Path
        rect_path = Path([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)],
                         [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
        ax.set_boundary(rect_path, transform=ax.transData)
        ax.set_aspect('equal', adjustable='box')

    # Station center marker
    scatter_art = None
    if station_marker:
        kw = dict(s=8, color='k', zorder=12, transform=None)  # transform=None → data/projection coords
        if station_marker_kw:
            kw.update(station_marker_kw)
        # We already transformed to projection coords (xs, ys); plot in data coords
        scatter_art = ax.scatter(xs, ys, **kw)

    # Prepare colormaps per metric
    cm = get_cmap(cmap)
    norms = {}
    if metric_limits is not None:
        for name in metric_names:
            vmin, vmax = metric_limits.get(name, (None, None))
            if vmin is None or vmax is None:
                vals = np.asarray(station_metrics[name]).ravel()
                vmin = np.nanmin(vals) if vmin is None else vmin
                vmax = np.nanmax(vals) if vmax is None else vmax
            norms[name] = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        # Global limits across all metrics (default to [-1, 1] if that’s your usual metric)
        all_vals = np.concatenate([np.asarray(v).ravel() for v in station_metrics.values()])
        vmin, vmax = (float(np.nanmin(all_vals)), float(np.nanmax(all_vals)))
        # If the range is too tight, fall back to symmetric ±1
        if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-6:
            vmin, vmax = -1.0, 1.0
        norms = {name: colors.Normalize(vmin=vmin, vmax=vmax) for name in metric_names}

    # Build sector angles
    full = 360.0
    base_span = full / K
    gap = float(sector_gap_deg)
    span = max(0.0, base_span - gap)  # leave a small gap between sectors

    collections = []
    colorbars = []

    for k, name in enumerate(metric_names):
        vals = np.asarray(station_metrics[name]).ravel()
        # Colors per station for this metric
        rgba = cm(norms[name](vals))

        # Sector angles for this metric (vectorized per station)
        theta1 = start_angle_deg + k * base_span + 0.5 * gap
        theta2 = start_angle_deg + (k + 1) * base_span - 0.5 * gap
        # Make them arrays broadcastable to stations
        theta1_arr = np.full_like(xs, theta1, dtype=float)
        theta2_arr = np.full_like(xs, theta2, dtype=float)

        coll = _arcs_on_ax(
            ax, xs, ys,
            w=np.full_like(xs, float(diameter_m)),
            h=np.full_like(xs, float(diameter_m)),
            theta1=theta1_arr,
            theta2=theta2_arr,
            colors=rgba,
            lw=3.8,
            zorder=13  # above the background & station dots
        )
        collections.append(coll)

    if add_colorbars:
        # A standalone ScalarMappable drives the colorbar
        import matplotlib as mpl
        sm = mpl.cm.ScalarMappable(norm=norms[name], cmap=cm)
        sm.set_array([])
        cb = add_colorbar_aligned(ax, sm, where="right", size=cbar_size, pad=cbar_pad, label='', orientation='vertical')
        colorbars.append(cb)

    if title:
        ax.set_title(title)

    
    return fig, ax, {'sectors': collections, 'station_scatter': scatter_art, 'colorbars': colorbars}
