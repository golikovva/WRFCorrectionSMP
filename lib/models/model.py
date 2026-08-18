import functools
from torch import nn
import torch.nn.functional as F
import torch
from lib.models.model_parts import DeltaConvLayer


class Corrector(nn.Module):
    def __init__(self, model, channels=6):
        super().__init__()
        self.channels = channels
        self.unet = model

    def forward(self, x_orig):
        x = x_orig
        unet_out = self.unet(x)
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            o_input = torch.split(x_orig.data, 3, dim=-3)
            res = o_input[0] + unet_out.data
            return torch.nn.utils.rnn.PackedSequence(res, x_orig.batch_sizes, x_orig.sorted_indices, x_orig.unsorted_indices)
        else:
            o_input = torch.split(x_orig, 3, dim=-3)
            # print(unet_out.shape, 'out shape')
            # print(o_input[0].shape)
            return o_input[0] + unet_out.view(x.shape[0], x.shape[1], 3, x.shape[3], x.shape[4])


class InputChannelSelector(nn.Module):
    def __init__(self, model, drop_first_channels=0):
        super().__init__()
        self.model = model
        self.drop_first_channels = int(drop_first_channels)
        if self.drop_first_channels < 0:
            raise ValueError("drop_first_channels must be non-negative")
        if hasattr(model, "requires_dates"):
            self.requires_dates = model.requires_dates

    def forward(self, x, *args, **kwargs):
        return self.model(self.select_channels(x), *args, **kwargs)

    def select_channels(self, x):
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            data = self._select_dense(x.data)
            return torch.nn.utils.rnn.PackedSequence(data, x.batch_sizes, x.sorted_indices, x.unsorted_indices)
        return self._select_dense(x)

    def _select_dense(self, x):
        if self.drop_first_channels == 0:
            return x
        if x.ndim < 3:
            raise ValueError(f"Expected tensor with channel dim -3, got {tuple(x.shape)}")

        channels = x.shape[-3]
        if self.drop_first_channels >= channels:
            raise ValueError(
                f"Cannot drop {self.drop_first_channels} channels from input with {channels} channels"
            )
        return x.narrow(dim=-3, start=self.drop_first_channels, length=channels - self.drop_first_channels)


class LowFreqCorrector(nn.Module):
    def __init__(self, model, in_channels=8, out_channels=3, k=7, inference_mode=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = model
        self.freq_conv = DeltaConvLayer(k, 'gauss')
        self.inference_mode = inference_mode

    def forward(self, x_orig):
        s = x_orig.shape
        with torch.no_grad():
            o_input, metadata = torch.split(x_orig, [self.out_channels, s[-3] - self.out_channels], dim=-3)

            so = o_input.shape
            l_freq = self.freq_conv(o_input.view(-1, *so[-3:])).view(so)

            h_freq = o_input - l_freq
        l_freq_corr = self.model(torch.cat((l_freq, metadata), dim=-3))
        l_freq_corr = l_freq_corr.view(*s[:2], *l_freq_corr.shape[1:])

        return l_freq, l_freq_corr, h_freq


def i2itos2s(cls):
    class Seq2SeqModel(cls):
        def forward(self, x):
            s = x.shape
            out = super().forward(x.flatten(0, 1))
            return out.unflatten(0, s[:2])
    return Seq2SeqModel
