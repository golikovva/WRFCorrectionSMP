import torch
from torch import nn

from lib.models.unet_parts import DoubleConv, Down, Up, OutConv
from lib.models.vit_latent import LatentViT


class RoPEUNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        *,
        batch_first: bool = True,
        bilinear: bool = True,
        chan_factor: int = 2,
        # latent transformer params
        max_T: int = 10,
        vit_depth: int = 4,
        vit_heads: int = 4,
        vit_mlp_ratio: float = 4.0,
        vit_drop: float = 0.0,
        vit_attn_drop: float = 0.0,
        vit_drop_path: float = 0.0,
        vit_rope_theta: float = 10000.0,
        vit_rope_mode: str = "axial",  # "axial" | "mixed" (если ты добавил switch)
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        print("Model:", n_channels, "->", n_classes)

        # Режимы
        self.unet_mode = "train"
        self.transformer_mode = "train"

        self.batch_first = batch_first
        self.chan_factor = chan_factor
        factor = 2 if bilinear else 1

        # --- UNet encoder ---
        self.inc = DoubleConv(n_channels, 64 // chan_factor)
        self.down1 = Down(64 // chan_factor, 128 // chan_factor)
        self.down2 = Down(128 // chan_factor, 256 // chan_factor)
        self.down3 = Down(256 // chan_factor, 512 // chan_factor)
        self.down4 = Down(512 // chan_factor, 1024 // chan_factor // factor)

        # latent dim на выходе down4 у тебя сейчас = 1024//chan_factor//factor.
        # Для bilinear=True, chan_factor=2, factor=2 => 1024/2/2 = 256 (совпадает с 512//2 раньше).
        self.latent_dim = 1024 // chan_factor // factor

        # --- Latent transformer (B, T, C, h, w) -> (B, T, C, h, w) ---
        # Если в LatentViT есть rope_mode, прокинь его; если нет — убери аргумент.

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

        # --- UNet decoder ---
        self.up1 = Up(1024 // chan_factor, 512 // chan_factor // factor, bilinear)
        self.up2 = Up(512 // chan_factor, 256 // chan_factor // factor, bilinear)
        self.up3 = Up(256 // chan_factor, 128 // chan_factor // factor, bilinear)
        self.up4 = Up(128 // chan_factor, 64 // chan_factor)
        self.outc = OutConv(64 // chan_factor, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, Cin, H, W)
        returns: (B, T, Cout, H, W)
        """
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)

        orig_shape = x.shape
        assert x.ndim == 5, f"Expected (B,T,C,H,W), got {tuple(x.shape)}"
        
        B, T, Cin, H, W = orig_shape

        # encoder на каждом времени независимо: (B*T, Cin, H, W)
        x_ = x.reshape(B * T, Cin, H, W)

        x1 = self.inc(x_)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # (B*T, C_lat, h_lat, w_lat)

        # latent transformer expects (B, T, C_lat, h_lat, w_lat)
        C_lat, h_lat, w_lat = x5.shape[1], x5.shape[2], x5.shape[3]
        # sanity: C_lat должен совпасть с self.latent_dim
        # (если в unet_parts поменяется, лучше ловить тут)
        if C_lat != self.latent_dim:
            raise RuntimeError(f"latent channels mismatch: got {C_lat}, expected {self.latent_dim}")

        x5 = x5.view(B, T, C_lat, h_lat, w_lat)
        x5 = self.latent_vit(x5)
        x5 = x5.view(B * T, C_lat, h_lat, w_lat)

        # decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x).view(B, T, self.n_classes, H, W)
        if not self.batch_first:
            logits = logits.permute(1, 0, 2, 3, 4).contiguous()
        return logits

    def freeze_backbone(self):
        """
        Замораживает параметры U-net части модели, оставляя latent transformer для обучения.
        """
        for name, param in self.named_parameters():
            # оставляем обучаемым latent_vit
            if not name.startswith("latent_vit"):
                param.requires_grad = False
        print("U-net backbone заморожен. Параметры latent transformer остаются обучаемыми.")

    def set_mode(self, unet_mode: str = "eval", transformer_mode: str = "train"):
        """
        Устанавливает режимы для U-net и трансформера.
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

        print(f"U-net переведен в режим {unet_mode}. Transformer переведен в режим {transformer_mode}.")

    def train(self, mode: bool = True):
        """
        Сохраняем твою идею: общий mode + раздельные режимы для подмодулей.
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