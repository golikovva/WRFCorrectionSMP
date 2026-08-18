import copy
import torch
from lib.models.model import Corrector, InputChannelSelector, LowFreqCorrector, i2itos2s
from lib.models.location_encoding import InputChannelAppender, build_siren_sh_encoder


def _cfg_get(section, key, default=None):
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    try:
        return getattr(section, key)
    except (AttributeError, KeyError):
        return default


def _enabled_location_encoding(cfg):
    location_cfg = _cfg_get(cfg, "location_encoding")
    if location_cfg is None or not bool(_cfg_get(location_cfg, "enabled", False)):
        return None
    return location_cfg


def _location_out_channels(location_cfg):
    return int(_cfg_get(location_cfg, "out_channels", 8)) if location_cfg is not None else 0


def _model_input_drop_first_channels(cfg):
    input_cfg = _cfg_get(cfg, "model_input")
    if input_cfg is None:
        return 0
    drop_first_channels = int(_cfg_get(input_cfg, "drop_first_channels", 0) or 0)
    if drop_first_channels < 0:
        raise ValueError("cfg.model_input.drop_first_channels must be non-negative")
    return drop_first_channels


def _model_args(args, channel_delta=0, channel_keys=("n_channels",)):
    args = copy.deepcopy(dict(args))
    if channel_delta == 0:
        return args

    for key in channel_keys:
        if key in args:
            args[key] = int(args[key]) + channel_delta
            if args[key] <= 0:
                raise ValueError(f"{key} must stay positive after model input channel selection")
            return args
    raise KeyError(f"Cannot add location encoding channels; none of {channel_keys} found in model args")


def _wrap_with_location_encoder(model, location_cfg, grid):
    if location_cfg is None:
        return model
    encoder = build_siren_sh_encoder(grid, location_cfg)
    return InputChannelAppender(model, encoder)


def _wrap_with_input_channel_selector(model, drop_first_channels):
    if drop_first_channels <= 0:
        return model
    return InputChannelSelector(model, drop_first_channels=drop_first_channels)


def _convnext_args(args, channel_delta):
    args = _model_args(args, channel_delta, ("in_chans", "n_channels"))
    if "n_channels" in args and "in_chans" not in args:
        args["in_chans"] = args.pop("n_channels")
    if "n_classes" in args and "out_channel" not in args:
        args["out_channel"] = args.pop("n_classes")
    return args


def build_correction_model(cfg, grid=None):
    model_type = cfg.model_type.lower()
    location_cfg = _enabled_location_encoding(cfg)
    extra_channels = _location_out_channels(location_cfg)
    drop_first_channels = _model_input_drop_first_channels(cfg)
    channel_delta = extra_channels - drop_first_channels
    print(extra_channels, 'extra_channels')
    print(drop_first_channels, 'drop_first_channels')
    print(channel_delta, 'channel_delta')
    if model_type == "bertunet":
        from lib.models.bertunet import BERTUNet
        unet = BERTUNet(**_model_args(cfg.model_args.BERTunet, channel_delta, ("n_channels",)))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        unet = _wrap_with_input_channel_selector(unet, drop_first_channels)
        model = Corrector(unet).to(cfg.device)
    elif model_type == "bertunet_raw":
        from lib.models.bertunet import BERTUNet
        model = BERTUNet(**_model_args(cfg.model_args.BERTunet, channel_delta, ("n_channels",)))
        model = _wrap_with_location_encoder(model, location_cfg, grid).to(cfg.device)
        model = _wrap_with_input_channel_selector(model, drop_first_channels).to(cfg.device)
    elif model_type == 'bertunet_lfreq':
        from lib.models.bertunet import BERTUNet
        unet = BERTUNet(**_model_args(cfg.model_args.BERTunet, channel_delta, ("n_channels",)))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        unet = _wrap_with_input_channel_selector(unet, drop_first_channels)
        model = LowFreqCorrector(unet).to(cfg.device)
    elif model_type == 'vsbertunet':
        from lib.models.bertunet import S2SBERTUnet
        unet = S2SBERTUnet(**_model_args(cfg.model_args.VSBERTunet, channel_delta, ("n_channels",)))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        unet = _wrap_with_input_channel_selector(unet, drop_first_channels)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'unet':
        from lib.models.unet import UNet
        print('Building UNet model...')
        unet = i2itos2s(UNet)(**_model_args(cfg.model_args.UNet, channel_delta, ("n_channels",)))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        unet = _wrap_with_input_channel_selector(unet, drop_first_channels)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'vit':
        from lib.models.vit import ViT
        print('Building ViT model...')
        unet = i2itos2s(ViT)(**_model_args(cfg.model_args.ViT, channel_delta, ("in_channels",)))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        unet = _wrap_with_input_channel_selector(unet, drop_first_channels)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'swinlstm_b':
        print('Building ViT model...')
        from .swinLSTM_B import SwinLSTM
        unet = SwinLSTM(**_model_args(cfg.model_args.SwinLSTM_B, channel_delta, ("in_chans",)))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        unet = _wrap_with_input_channel_selector(unet, drop_first_channels)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'swinlstm_d':
        print('Building swinlstm_d model...')
        from .swinLSTM_D import SwinLSTM
        unet = SwinLSTM(**_model_args(cfg.model_args.SwinLSTM_D, channel_delta, ("in_chans",)))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        unet = _wrap_with_input_channel_selector(unet, drop_first_channels)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'timesformer':
        from lib.models.timesformer import TimeSformer
        print('Building TimeSformer model...')
        unet = TimeSformer(**_model_args(cfg.model_args.TimeSformer, channel_delta, ("in_chans",)))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        unet = _wrap_with_input_channel_selector(unet, drop_first_channels)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'cnn2d':
        from torchcnnbuilder.models import ForecasterBase
        backbone = i2itos2s(ForecasterBase)(**_model_args(
            cfg.model_args.CNN2D,
            channel_delta,
            ("in_channels", "input_channels", "n_channels", "in_chans"),
        ))
        backbone = _wrap_with_location_encoder(backbone, location_cfg, grid)
        backbone = _wrap_with_input_channel_selector(backbone, drop_first_channels)
        model = Corrector(backbone).to(cfg.device)
    elif model_type == 'cnn3d':
        from torchcnnbuilder.models import ForecasterBase
        backbone = i2itos2s(ForecasterBase)(**_model_args(
            cfg.model_args.CNN3D,
            channel_delta,
            ("in_channels", "input_channels", "n_channels", "in_chans"),
        ))
        backbone = _wrap_with_location_encoder(backbone, location_cfg, grid)
        backbone = _wrap_with_input_channel_selector(backbone, drop_first_channels)
        model = Corrector(backbone).to(cfg.device)
    elif model_type == 'ropeunet':
        from lib.models.unet_rope import RoPEUNet
        model = RoPEUNet(**_model_args(cfg.model_args.RoPEUNet, channel_delta, ("n_channels",)))
        model = _wrap_with_location_encoder(model, location_cfg, grid).to(cfg.device)
        model = _wrap_with_input_channel_selector(model, drop_first_channels).to(cfg.device)
        # model = Corrector(backbone).to(cfg.device)
    elif model_type == 'convnext':
        from lib.models.convnext import ConvNeXtV2
        unet = i2itos2s(ConvNeXtV2)(**_convnext_args(cfg.model_args.ConvNext, channel_delta))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        unet = _wrap_with_input_channel_selector(unet, drop_first_channels)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'ropeconvnext':
        from lib.models.convnext import ConvNeXtV2LatentVit
        model = ConvNeXtV2LatentVit(**_model_args(cfg.model_args.RoPEConvNeXtV2, channel_delta, ("in_chans",)))
        model = _wrap_with_location_encoder(model, location_cfg, grid).to(cfg.device)
        model = _wrap_with_input_channel_selector(model, drop_first_channels).to(cfg.device)
    elif model_type == 'irrepsphereunet':
        from lib.models.spherical.steerable_layers import SO2IrrepFieldType
        from lib.models.spherical.irrep_sphere_unet import IrrepSphereUNet
        from lib.models.spherical.sphere_grid_wrapper import SphereGridModelWrapper
        unet_cfg = cfg.model_args.IrrepSphereUNet

        input_schema = _build_field_schema(
            unet_cfg.input_schema
        )

        output_schema_cfg = unet_cfg.get("output_schema")
        output_schema = (
            _build_field_schema(output_schema_cfg)
            if output_schema_cfg is not None
            else None
        )
        unet = IrrepSphereUNet(
            input_schema.irrep_type,
            out_type=(
                output_schema.irrep_type
                if output_schema is not None
                else None
            ),
            **unet_cfg.model,
        )

        model = i2itos2s(SphereGridModelWrapper)(
            unet,
            grid,
            input_schema=input_schema,
            output_schema=output_schema,
            **unet_cfg.graph,
        ).to(
            device=cfg.device,
            dtype=torch.float32,
        )

    elif model_type == 'aurora':
        pass
    else:
        raise TypeError(f"Unknown model_type={cfg.model_type!r}")
    return model

def _build_field_schema(schema_cfg):
    from lib.models.spherical.sphere_grid_wrapper import FieldSchema

    scalars = {
        str(name): int(channel)
        for name, channel in schema_cfg.get("scalars", {}).items()
    }
    vectors = {
        str(name): tuple(int(channel) for channel in channels)
        for name, channels in schema_cfg.get("vectors", {}).items()
    }

    return FieldSchema(
        scalars=scalars,
        vectors=vectors,
    )

def build_inference_correction_model(cfg):
    if cfg['model_type'] == "BERTunet":
        from lib.models.bertunet import BERTUNet
        unet = BERTUNet(n_channels=9, n_classes=3, bilinear=True)
        model = Corrector(unet).to(cfg['device'])
        state_dict = torch.load(cfg['model_weights'])
        model.load_state_dict(state_dict)
    else:
        raise TypeError
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def count_flops(model, input_tensor):
    from thop import profile
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    return flops, params

def count_flops_summary(model, input_shape):
    from torchsummary import summary
    out = summary(model, input_shape)
    return out

# def count_flops_stat(model, input_shape):
#     from torchstat import stat
#     out = stat(model, input_shape)
#     return out
