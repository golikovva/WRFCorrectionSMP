import torch
from lib.models.unet import UNet
from lib.models.bertunet import BERTUNet, S2SBERTUnet
from lib.models.vit import ViT
from lib.models.timesformer import TimeSformer
from lib.models.model import Corrector, LowFreqCorrector, i2itos2s
from lib.models.convnext import ConvNeXtV2, ConvNeXtV2LatentVit

def build_correction_model(cfg):
    model_type = cfg.model_type.lower()
    if model_type == "bertunet":
        unet = BERTUNet(**cfg.model_args.BERTunet)
        model = Corrector(unet).to(cfg.device)
    elif model_type == "bertunet_raw":
        model = BERTUNet(**cfg.model_args.BERTunet).to(cfg.device)
    elif model_type == 'bertunet_lfreq':
        unet = BERTUNet(**cfg.model_args.BERTunet)
        model = LowFreqCorrector(unet).to(cfg.device)
    elif model_type == 'vsbertunet':
        unet = S2SBERTUnet(**cfg.model_args.VSBERTunet)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'unet':
        print('Building UNet model...')
        unet = i2itos2s(UNet)(**cfg.model_args.UNet)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'vit':
        print('Building ViT model...')
        unet = i2itos2s(ViT)(**cfg.model_args.ViT)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'swinlstm_b':
        print('Building ViT model...')
        from .swinLSTM_B import SwinLSTM
        unet = SwinLSTM(**cfg.model_args.SwinLSTM_B)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'swinlstm_d':
        print('Building swinlstm_d model...')
        from .swinLSTM_D import SwinLSTM
        unet = SwinLSTM(**cfg.model_args.SwinLSTM_D)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'timesformer':
        print('Building TimeSformer model...')
        unet = TimeSformer(**cfg.model_args.TimeSformer)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'cnn2d':
        from torchcnnbuilder.models import ForecasterBase
        backbone = i2itos2s(ForecasterBase)(**cfg.model_args.CNN2D)
        model = Corrector(backbone).to(cfg.device)
    elif model_type == 'cnn3d':
        from torchcnnbuilder.models import ForecasterBase
        backbone = i2itos2s(ForecasterBase)(**cfg.model_args.CNN3D)
        model = Corrector(backbone).to(cfg.device)
    elif model_type == 'ropeunet':
        from lib.models.unet_rope import RoPEUNet
        model = RoPEUNet(**cfg.model_args.RoPEUNet).to(cfg.device)
        # model = Corrector(backbone).to(cfg.device)
    elif model_type == 'convnext':
        unet = i2itos2s(ConvNeXtV2)(**cfg.model_args.ConvNext)
        model = Corrector(unet).to(cfg.device)
    elif model_type == 'ropeconvnext':
        model = ConvNeXtV2LatentVit(**cfg.model_args.RoPEConvNeXtV2).to(cfg.device)
    elif model_type == 'aurora':
        pass
    else:
        raise TypeError
    return model


def build_inference_correction_model(cfg):
    if cfg['model_type'] == "BERTunet":
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