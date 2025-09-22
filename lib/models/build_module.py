import torch
from lib.models.unet import UNet, S2SBERTUnet
from lib.models.model import Corrector, LowFreqCorrector, i2itos2s
# from lib.models.convnext import ConvNeXtV2

def build_correction_model(cfg):
    if cfg.model_type == "BERTunet":
        unet = UNet(*cfg.model_args.BERTunet.values())
        model = Corrector(unet).to(cfg.device)
    elif cfg.model_type == "BERTunet_raw":
        model = UNet(*cfg.model_args.BERTunet.values()).to(cfg.device)
    elif cfg.model_type == 'BERTunet_lfreq':
        unet = UNet(*cfg.model_args.BERTunet.values())
        model = LowFreqCorrector(unet).to(cfg.device)
    elif cfg.model_type == 'VSBERTunet':
        unet = S2SBERTUnet(*cfg.model_args.VSBERTunet.values())
        model = Corrector(unet).to(cfg.device)
    # elif cfg.model_type == 'ConvNext':
    #     unet = i2itos2s(ConvNeXtV2)(*cfg.model_args.ConvNext.values())
    #     model = Corrector(unet).to(cfg.device)
    elif cfg.model_type == 'Aurora':
        pass
    else:
        raise TypeError
    return model


def build_inference_correction_model(cfg):
    if cfg['model_type'] == "BERTunet":
        unet = UNet(n_channels=9, n_classes=3, bilinear=True)
        model = Corrector(unet).to(cfg['device'])
        state_dict = torch.load(cfg['model_weights'])
        model.load_state_dict(state_dict)
    else:
        raise TypeError
    return model
