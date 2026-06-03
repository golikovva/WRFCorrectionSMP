import copy
import torch
from lib.models.model import Corrector, LowFreqCorrector, i2itos2s
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


def _model_args(args, extra_channels=0, channel_keys=("n_channels",)):
    args = copy.deepcopy(dict(args))
    if extra_channels <= 0:
        return args

    for key in channel_keys:
        if key in args:
            args[key] = int(args[key]) + extra_channels
            return args
    raise KeyError(f"Cannot add location encoding channels; none of {channel_keys} found in model args")


def _wrap_with_location_encoder(model, location_cfg, grid):
    if location_cfg is None:
        return model
    encoder = build_siren_sh_encoder(grid, location_cfg)
    return InputChannelAppender(model, encoder)


def _convnext_args(args, extra_channels):
    args = _model_args(args, extra_channels, ("in_chans", "n_channels"))
    if "n_channels" in args and "in_chans" not in args:
        args["in_chans"] = args.pop("n_channels")
    if "n_classes" in args and "out_channel" not in args:
        args["out_channel"] = args.pop("n_classes")
    return args


def build_correction_model(cfg, grid=None):
    model_type = cfg.model_type.lower()
    location_cfg = _enabled_location_encoding(cfg)
    extra_channels = _location_out_channels(location_cfg)
    if model_type == "bertunet":
        from lib.models.bertunet import BERTUNet
        unet = BERTUNet(**_model_args(cfg.model_args.BERTunet, extra_channels, ("n_channels",)))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        model = Corrector(unet).to(cfg.device)
    elif model_type == "bertunet_raw":
        from lib.models.bertunet import BERTUNet
        model = BERTUNet(**_model_args(cfg.model_args.BERTunet, extra_channels, ("n_channels",)))
        model = _wrap_with_location_encoder(model, location_cfg, grid).to(cfg.device)
    elif model_type == 'bertunet_lfreq':
        from lib.models.bertunet import BERTUNet
        unet = BERTUNet(**_model_args(cfg.model_args.BERTunet, extra_channels, ("n_channels",)))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        model = LowFreqCorrector(unet).to(cfg.device)
    elif model_type == 'vsbertunet':
        from lib.models.bertunet import S2SBERTUnet
        unet = S2SBERTUnet(**_model_args(cfg.model_args.VSBERTunet, extra_channels, ("n_channels",)))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'unet':
        from lib.models.unet import UNet
        print('Building UNet model...')
        unet = i2itos2s(UNet)(**_model_args(cfg.model_args.UNet, extra_channels, ("n_channels",)))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'vit':
        from lib.models.vit import ViT
        print('Building ViT model...')
        unet = i2itos2s(ViT)(**_model_args(cfg.model_args.ViT, extra_channels, ("in_channels",)))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'swinlstm_b':
        print('Building ViT model...')
        from .swinLSTM_B import SwinLSTM
        unet = SwinLSTM(**_model_args(cfg.model_args.SwinLSTM_B, extra_channels, ("in_chans",)))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'swinlstm_d':
        print('Building swinlstm_d model...')
        from .swinLSTM_D import SwinLSTM
        unet = SwinLSTM(**_model_args(cfg.model_args.SwinLSTM_D, extra_channels, ("in_chans",)))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'timesformer':
        from lib.models.timesformer import TimeSformer
        print('Building TimeSformer model...')
        unet = TimeSformer(**_model_args(cfg.model_args.TimeSformer, extra_channels, ("in_chans",)))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'cnn2d':
        from torchcnnbuilder.models import ForecasterBase
        backbone = i2itos2s(ForecasterBase)(**_model_args(
            cfg.model_args.CNN2D,
            extra_channels,
            ("in_channels", "input_channels", "n_channels", "in_chans"),
        ))
        backbone = _wrap_with_location_encoder(backbone, location_cfg, grid)
        model = Corrector(backbone).to(cfg.device)
    elif model_type == 'cnn3d':
        from torchcnnbuilder.models import ForecasterBase
        backbone = i2itos2s(ForecasterBase)(**_model_args(
            cfg.model_args.CNN3D,
            extra_channels,
            ("in_channels", "input_channels", "n_channels", "in_chans"),
        ))
        backbone = _wrap_with_location_encoder(backbone, location_cfg, grid)
        model = Corrector(backbone).to(cfg.device)
    elif model_type == 'ropeunet':
        from lib.models.unet_rope import RoPEUNet
        model = RoPEUNet(**_model_args(cfg.model_args.RoPEUNet, extra_channels, ("n_channels",)))
        model = _wrap_with_location_encoder(model, location_cfg, grid).to(cfg.device)
        # model = Corrector(backbone).to(cfg.device)
    elif model_type == 'convnext':
        from lib.models.convnext import ConvNeXtV2
        unet = i2itos2s(ConvNeXtV2)(**_convnext_args(cfg.model_args.ConvNext, extra_channels))
        unet = _wrap_with_location_encoder(unet, location_cfg, grid)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'ropeconvnext':
        from lib.models.convnext import ConvNeXtV2LatentVit
        model = ConvNeXtV2LatentVit(**_model_args(cfg.model_args.RoPEConvNeXtV2, extra_channels, ("in_chans",)))
        model = _wrap_with_location_encoder(model, location_cfg, grid).to(cfg.device)
    elif model_type == 'aurora':
        pass
    else:
        raise TypeError(f"Unknown model_type={cfg.model_type!r}")
    return model


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
