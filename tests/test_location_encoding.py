import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from lib.models.build_module import build_correction_model
from lib.models.location_encoding import InputChannelAppender, SirenSHGridEncoder


def make_grid(h=8, w=8):
    lon = np.linspace(60.0, 80.0, w, dtype=np.float32)
    lat = np.linspace(65.0, 75.0, h, dtype=np.float32)
    lon, lat = np.meshgrid(lon, lat)
    return {"longitude": lon, "latitude": lat}


def make_unet_cfg(base_channels=3, enabled=True, out_channels=2):
    return SimpleNamespace(
        device="cpu",
        model_type="UNet",
        model_args=SimpleNamespace(
            UNet={
                "n_channels": base_channels,
                "n_classes": 3,
                "bilinear": True,
            }
        ),
        location_encoding={
            "enabled": enabled,
            "type": "siren_sh",
            "legendre_polys": 3,
            "out_channels": out_channels,
            "dim_hidden": 8,
            "num_layers": 1,
            "dropout": False,
            "w0": 1.0,
            "w0_initial": 30.0,
        },
    )


def test_siren_sh_grid_encoder_shape_dtype_and_state_dict():
    encoder = SirenSHGridEncoder(
        make_grid(h=2, w=3),
        legendre_polys=3,
        out_channels=4,
        dim_hidden=8,
        num_layers=1,
    )

    encoded = encoder(dtype=torch.float64)

    assert encoded.shape == (4, 2, 3)
    assert encoded.dtype == torch.float64
    assert "sh_grid" not in encoder.state_dict()


def test_input_channel_appender_dense_and_packed_sequence():
    encoder = SirenSHGridEncoder(
        make_grid(h=2, w=3),
        legendre_polys=2,
        out_channels=2,
        dim_hidden=8,
        num_layers=1,
    )
    appender = InputChannelAppender(nn.Identity(), encoder)

    dense = torch.randn(2, 4, 5, 2, 3)
    dense_out = appender(dense)
    assert dense_out.shape == (2, 4, 7, 2, 3)

    packed = PackedSequence(torch.randn(4, 5, 2, 3), torch.tensor([2, 2]))
    packed_out = appender(packed)
    assert isinstance(packed_out, PackedSequence)
    assert packed_out.data.shape == (4, 7, 2, 3)


def test_build_unet_modes_adjust_input_channels():
    grid = make_grid(h=32, w=32)

    raw_cfg = make_unet_cfg(base_channels=5, enabled=False, out_channels=2)
    raw_model = build_correction_model(raw_cfg)
    assert raw_model.unet.inc.double_conv[0].in_channels == 5

    replacement_cfg = make_unet_cfg(base_channels=3, enabled=True, out_channels=2)
    replacement_model = build_correction_model(replacement_cfg, grid=grid)
    assert isinstance(replacement_model.unet, InputChannelAppender)
    assert replacement_model.unet.model.inc.double_conv[0].in_channels == 5

    combined_cfg = make_unet_cfg(base_channels=5, enabled=True, out_channels=2)
    combined_model = build_correction_model(combined_cfg, grid=grid)
    assert combined_model.unet.model.inc.double_conv[0].in_channels == 7


def test_unet_forward_backward_and_checkpoint_roundtrip():
    grid = make_grid(h=32, w=32)
    cfg = make_unet_cfg(base_channels=3, enabled=True, out_channels=2)
    model = build_correction_model(cfg, grid=grid)

    x = torch.randn(2, 2, 3, 32, 32)
    out = model(x)
    loss = out.square().mean()
    loss.backward()

    siren_grads = [
        param.grad
        for name, param in model.named_parameters()
        if "encoder.siren" in name and param.requires_grad
    ]
    assert siren_grads
    assert all(grad is not None for grad in siren_grads)

    state_dict = copy.deepcopy(model.state_dict())
    assert not any("sh_grid" in key for key in state_dict)

    reloaded = build_correction_model(cfg, grid=grid)
    reloaded.load_state_dict(state_dict)


def test_ropeunet_forward_backward_smoke():
    pytest.importorskip("timm")
    grid = make_grid(h=32, w=32)
    cfg = SimpleNamespace(
        device="cpu",
        model_type="RoPEUNet",
        model_args=SimpleNamespace(
            RoPEUNet={
                "n_channels": 3,
                "n_classes": 3,
                "bilinear": True,
                "chan_factor": 8,
                "batch_first": False,
                "max_T": 4,
                "vit_depth": 1,
                "vit_heads": 4,
                "vit_mlp_ratio": 2.0,
                "vit_drop": 0.0,
                "vit_attn_drop": 0.0,
                "vit_drop_path": 0.0,
                "vit_rope_theta": 10000.0,
                "vit_rope_mode": "axial",
            }
        ),
        location_encoding={
            "enabled": True,
            "type": "siren_sh",
            "legendre_polys": 2,
            "out_channels": 2,
            "dim_hidden": 8,
            "num_layers": 1,
            "dropout": False,
            "w0": 1.0,
            "w0_initial": 30.0,
        },
    )
    model = build_correction_model(cfg, grid=grid)

    out = model(torch.randn(2, 1, 3, 32, 32))
    out.square().mean().backward()

    assert out.shape == (2, 1, 3, 32, 32)
    assert any(
        param.grad is not None
        for name, param in model.named_parameters()
        if "encoder.siren" in name
    )
