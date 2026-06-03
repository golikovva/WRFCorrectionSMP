import math
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


def _exists(value):
    return value is not None


def _cast_tuple(value, repeat):
    return value if isinstance(value, tuple) else (value,) * repeat


def _grid_tensor(grid, key):
    if key not in grid:
        raise KeyError(f"Grid must contain {key!r}")
    return torch.as_tensor(grid[key], dtype=torch.float64)


def _real_spherical_harmonics(lon_deg, lat_deg, legendre_polys):
    lon_rad = torch.deg2rad(lon_deg)
    lat_rad = torch.deg2rad(lat_deg)
    x = torch.sin(lat_rad)
    one_minus_x2 = torch.clamp(1.0 - x.square(), min=0.0)
    sin_theta = torch.sqrt(one_minus_x2)

    features = []
    p_cache = {}
    for l in range(legendre_polys):
        for m_signed in range(-l, l + 1):
            m = abs(m_signed)
            p_lm = _associated_legendre(l, m, x, sin_theta, p_cache)
            norm = _spherical_harmonic_norm(l, m)

            if m_signed == 0:
                y_lm = norm * p_lm
            elif m_signed > 0:
                y_lm = math.sqrt(2.0) * norm * p_lm * torch.cos(m * lon_rad)
            else:
                y_lm = math.sqrt(2.0) * norm * p_lm * torch.sin(m * lon_rad)

            features.append(y_lm)

    return torch.stack(features, dim=-1)


def _associated_legendre(l, m, x, sin_theta, cache):
    key = (l, m)
    if key in cache:
        return cache[key]

    if l == 0 and m == 0:
        value = torch.ones_like(x)
    elif l == m:
        prev = _associated_legendre(m - 1, m - 1, x, sin_theta, cache)
        value = -(2 * m - 1) * sin_theta * prev
    elif l == m + 1:
        prev = _associated_legendre(m, m, x, sin_theta, cache)
        value = (2 * m + 1) * x * prev
    else:
        p_lm1 = _associated_legendre(l - 1, m, x, sin_theta, cache)
        p_lm2 = _associated_legendre(l - 2, m, x, sin_theta, cache)
        value = ((2 * l - 1) * x * p_lm1 - (l + m - 1) * p_lm2) / (l - m)

    cache[key] = value
    return value


def _spherical_harmonic_norm(l, m):
    log_norm = (
        math.log((2 * l + 1) / (4.0 * math.pi))
        + math.lgamma(l - m + 1)
        - math.lgamma(l + m + 1)
    )
    return math.exp(0.5 * log_norm)


class SphericalHarmonicsGrid(nn.Module):
    def __init__(self, grid, legendre_polys):
        super().__init__()
        self.legendre_polys = int(legendre_polys)
        if self.legendre_polys <= 0:
            raise ValueError("legendre_polys must be positive")

        lon = _grid_tensor(grid, "longitude")
        lat = _grid_tensor(grid, "latitude")
        if lon.shape != lat.shape:
            raise ValueError(f"longitude and latitude shapes differ: {lon.shape} vs {lat.shape}")
        if lon.ndim != 2:
            raise ValueError(f"Expected 2D grid, got shape {tuple(lon.shape)}")

        self.grid_shape = tuple(lon.shape)
        lonlat = torch.stack((lon.reshape(-1), lat.reshape(-1)), dim=-1)
        self.register_buffer("lonlat", lonlat, persistent=False)

    @property
    def embedding_dim(self):
        return self.legendre_polys * self.legendre_polys

    def forward(self):
        lon = self.lonlat[:, 0]
        lat = self.lonlat[:, 1]
        basis = _real_spherical_harmonics(lon, lat, self.legendre_polys)
        return basis.to(dtype=torch.float32)


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=1.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
        dropout=False,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first
        self.dropout = dropout

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self._init_weights(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def _init_weights(self, weight, bias, c, w0):
        bound = (1.0 / self.dim_in) if self.is_first else (math.sqrt(c / self.dim_in) / w0)
        weight.uniform_(-bound, bound)
        if _exists(bias):
            bias.uniform_(-bound, bound)

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        if self.dropout:
            p = 0.5 if isinstance(self.dropout, bool) else float(self.dropout)
            x = F.dropout(x, p=p, training=self.training)
        return self.activation(x)


class SirenNet(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=1.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
        dropout=False,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.dim_hidden = int(dim_hidden)

        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            is_first = index == 0
            self.layers.append(
                Siren(
                    dim_in=dim_in if is_first else self.dim_hidden,
                    dim_out=self.dim_hidden,
                    w0=w0_initial if is_first else w0,
                    use_bias=use_bias,
                    is_first=is_first,
                    dropout=dropout,
                )
            )

        final_activation = nn.Identity() if final_activation is None else final_activation
        self.last_layer = Siren(
            dim_in=self.dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation,
            dropout=False,
        )

    def forward(self, x, mods=None):
        mods = _cast_tuple(mods, self.num_layers)
        for layer, mod in zip(self.layers, mods):
            x = layer(x)
            if _exists(mod):
                x = x * mod.reshape(1, -1)
        return self.last_layer(x)


class SirenSHGridEncoder(nn.Module):
    def __init__(
        self,
        grid,
        legendre_polys=20,
        out_channels=8,
        dim_hidden=64,
        num_layers=2,
        dropout=False,
        w0=1.0,
        w0_initial=30.0,
    ):
        super().__init__()
        harmonics = SphericalHarmonicsGrid(grid, legendre_polys=legendre_polys)
        self.grid_shape = harmonics.grid_shape
        self.out_channels = int(out_channels)
        self.register_buffer("sh_grid", harmonics(), persistent=False)
        self.siren = SirenNet(
            dim_in=harmonics.embedding_dim,
            dim_hidden=dim_hidden,
            dim_out=self.out_channels,
            num_layers=num_layers,
            dropout=dropout,
            w0=w0,
            w0_initial=w0_initial,
        )

    def forward(self, *, device=None, dtype=None):
        param = next(self.siren.parameters())
        device = param.device if device is None else device
        siren_dtype = param.dtype
        sh_grid = self.sh_grid.to(device=device, dtype=siren_dtype)
        encoded = self.siren(sh_grid)
        h, w = self.grid_shape
        encoded = encoded.reshape(h, w, self.out_channels).permute(2, 0, 1).contiguous()
        if dtype is not None:
            encoded = encoded.to(dtype=dtype)
        return encoded


class InputChannelAppender(nn.Module):
    def __init__(self, model, encoder):
        super().__init__()
        self.model = model
        self.encoder = encoder
        if hasattr(model, "requires_dates"):
            self.requires_dates = model.requires_dates

    def forward(self, x, *args, **kwargs):
        x = self.append_channels(x)
        return self.model(x, *args, **kwargs)

    def append_channels(self, x):
        if isinstance(x, PackedSequence):
            data = self._append_dense(x.data)
            return PackedSequence(data, x.batch_sizes, x.sorted_indices, x.unsorted_indices)
        return self._append_dense(x)

    def _append_dense(self, x):
        if x.ndim < 4:
            raise ValueError(f"Expected at least 4D tensor with channel dim -3, got {tuple(x.shape)}")

        h, w = x.shape[-2:]
        if (h, w) != self.encoder.grid_shape:
            raise ValueError(
                f"Input grid {(h, w)} does not match location encoder grid {self.encoder.grid_shape}"
            )

        encoding = self.encoder(device=x.device, dtype=x.dtype)
        leading_shape = x.shape[:-3]
        view_shape = (1,) * len(leading_shape) + tuple(encoding.shape)
        expand_shape = tuple(leading_shape) + tuple(encoding.shape)
        encoding = encoding.reshape(view_shape).expand(expand_shape)
        return torch.cat((x, encoding), dim=-3)


def build_siren_sh_encoder(grid, cfg):
    params = deepcopy(dict(cfg))
    params.pop("enabled", None)
    encoding_type = params.pop("type", "siren_sh")
    if encoding_type != "siren_sh":
        raise ValueError(f"Unsupported location_encoding.type={encoding_type!r}")
    if grid is None:
        raise ValueError("cfg.location_encoding.enabled=True requires build_correction_model(..., grid=...)")
    return SirenSHGridEncoder(grid=grid, **params)
