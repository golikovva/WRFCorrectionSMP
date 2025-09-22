import torch
import torch.nn as nn
import numpy as np
import scipy


class DeltaConvLayer(nn.Module):
    def __init__(self, channels=3, k=21, kernel_type='gauss'):
        super(DeltaConvLayer, self).__init__()
        self.conv = nn.Conv2d(channels, channels, k, stride=1, padding=k // 2,
                              bias=False, groups=channels, padding_mode='replicate')
        self.k = k
        self.kernel_type = kernel_type
        self.weights_init(k)

    def forward(self, x):
        return self.conv(x)

    def weights_init(self, k):
        if self.kernel_type == 'mean':
            w = torch.ones_like(self.delta_conv.weight) / self.k / self.k
            self.conv.weight = nn.Parameter(w)
        elif self.kernel_type == 'gauss':
            n = np.zeros((self.k, self.k))
            n[self.k // 2, self.k // 2] = 1
            sigma = k / 3
            kk = scipy.ndimage.gaussian_filter(n, sigma=sigma)
            for name, f in self.named_parameters():
                f.data.copy_(torch.from_numpy(kk))
