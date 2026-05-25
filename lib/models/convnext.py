
import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
import numpy.random as random
from timm.layers import trunc_normal_, DropPath
from lib.models.vit_latent import LatentViT

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):

        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3,  out_channel=3,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.num_stage = len(depths)
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(self.num_stage - 1):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(self.num_stage):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.out_conv = nn.Conv2d(int(dims[0]/2), out_channel, kernel_size=1) # final classifier conv
        self.upsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers

        for i in reversed(range(self.num_stage)):
            upsample_layer = nn.Sequential(
                    LayerNorm(dims[i]*2, eps=1e-6, data_format="channels_first"),
                    nn.ConvTranspose2d(dims[i]*2, int(dims[i]/2), kernel_size=2, stride=2),
            )
            self.upsample_layers.append(upsample_layer)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def Encoder(self, x):
        down_features = []
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            down_features.append(x)

        return x, down_features

    def Decoder(self, x, down_features):
        # [print(f.shape, 'fi') for f in down_features]
        for i in range(self.num_stage):
            df = down_features.pop()
            x = torch.cat([pad(x, df), df], dim=1)
            x = self.upsample_layers[i](x)
        return x

    def forward(self, x):
        x, down_features = self.Encoder(x)
        x = self.Decoder(x, down_features)
        x = self.out_conv(x)

        return x


def pad(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2], mode='reflect')
    return x1


class ConvNeXtV2LatentVit(ConvNeXtV2):
    """
    ConvNeXtV2 U-Net-like model with a RoPE ViT operating in latent space.

    Input:
        batch_first=True  -> (B, T, C, H, W)
        batch_first=False -> (T, B, C, H, W)

    Output:
        same temporal layout as input, with channel dim = out_channel
    """
    def __init__(
        self,
        in_chans: int = 3,
        out_channel: int = 3,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        head_init_scale: float = 1.0,
        *,
        batch_first: bool = True,
        max_T: int = 10,
        vit_depth: int = 4,
        vit_heads: int = 4,
        vit_mlp_ratio: float = 4.0,
        vit_drop: float = 0.0,
        vit_attn_drop: float = 0.0,
        vit_drop_path: float = 0.0,
        vit_rope_theta: float = 100.0,
        vit_rope_mode: str = "mixed",
    ):
        super().__init__(
            in_chans=in_chans,
            out_channel=out_channel,
            depths=list(depths),
            dims=list(dims),
            drop_path_rate=drop_path_rate,
            head_init_scale=head_init_scale,
        )

        self.in_chans = in_chans
        self.out_channel = out_channel
        self.batch_first = batch_first

        self.unet_mode = "train"
        self.transformer_mode = "train"

        # bottleneck channel count after ConvNeXtV2 encoder
        self.latent_dim = dims[-1]

        if self.latent_dim % vit_heads != 0:
            raise ValueError(
                f"latent_dim={self.latent_dim} must be divisible by vit_heads={vit_heads}"
            )

        head_dim = self.latent_dim // vit_heads
        if head_dim % 4 != 0:
            raise ValueError(
                f"LatentViT SpatialRoPE requires head_dim % 4 == 0, "
                f"but got head_dim={head_dim} (latent_dim={self.latent_dim}, vit_heads={vit_heads})"
            )

        self.latent_vit = LatentViT(
            dim=self.latent_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            max_T=max_T,
            mlp_ratio=vit_mlp_ratio,
            drop=vit_drop,
            attn_drop=vit_attn_drop,
            drop_path=vit_drop_path,
            rope_theta=vit_rope_theta,
            rope_mode=vit_rope_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            if batch_first=True:  (B, T, Cin, H, W)
            if batch_first=False: (T, B, Cin, H, W)

        returns:
            same temporal layout, shape (..., Cout, H_out, W_out)
        """
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4).contiguous()

        if x.ndim != 5:
            raise ValueError(f"Expected 5D input, got shape {tuple(x.shape)}")

        B, T, Cin, H, W = x.shape
        if Cin != self.in_chans:
            raise ValueError(f"Expected Cin={self.in_chans}, got {Cin}")

        # apply ConvNeXt encoder independently to each time slice
        x_2d = x.reshape(B * T, Cin, H, W)

        # x_latent: (B*T, C_lat, h_lat, w_lat)
        # down_features: list of skip tensors, each (B*T, C_i, h_i, w_i)
        x_latent, down_features = self.Encoder(x_2d)

        C_lat, h_lat, w_lat = x_latent.shape[1:]
        if C_lat != self.latent_dim:
            raise RuntimeError(
                f"Latent channel mismatch: got {C_lat}, expected {self.latent_dim}"
            )

        # temporal latent transformer
        x_latent = x_latent.reshape(B, T, C_lat, h_lat, w_lat)
        x_latent = self.latent_vit(x_latent)
        x_latent = x_latent.reshape(B * T, C_lat, h_lat, w_lat)

        # decode framewise using stored spatial skips
        # copy the list because Decoder pops from it
        x_dec = self.Decoder(x_latent, down_features.copy())
        logits = self.out_conv(x_dec)

        H_out, W_out = logits.shape[-2:]
        logits = logits.reshape(B, T, self.out_channel, H_out, W_out)

        if not self.batch_first:
            logits = logits.permute(1, 0, 2, 3, 4).contiguous()

        return logits

    def freeze_backbone(self):
        """
        Freeze ConvNeXt encoder/decoder and keep latent transformer trainable.
        """
        for name, param in self.named_parameters():
            if not name.startswith("latent_vit"):
                param.requires_grad = False
        print("ConvNeXtV2 backbone frozen. latent_vit remains trainable.")

    def set_mode(self, unet_mode: str = "eval", transformer_mode: str = "train"):
        """
        Set separate train/eval modes for ConvNeXtV2 part and latent transformer.
        """
        self.unet_mode = unet_mode
        self.transformer_mode = transformer_mode

        for name, module in self.named_modules():
            if name == "":
                continue
            if name.startswith("latent_vit"):
                module.train() if transformer_mode == "train" else module.eval()
            else:
                module.train() if unet_mode == "train" else module.eval()

        print(
            f"ConvNeXtV2 backbone set to {unet_mode}. "
            f"Latent transformer set to {transformer_mode}."
        )

    def train(self, mode: bool = True):
        """
        Preserve global train(mode) behavior, while allowing separate backbone/transformer modes.
        """
        for name, module in self.named_modules():
            if name == "":
                module.training = mode
                continue
            if name.startswith("latent_vit"):
                module.train(self.transformer_mode == "train" and mode)
            else:
                module.train(self.unet_mode == "train" and mode)
        return self
    

if __name__ == "__main__":
    # model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
    model = ConvNeXtV2()

    # print(model)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)


    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))