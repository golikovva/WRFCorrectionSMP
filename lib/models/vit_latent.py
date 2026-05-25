import math
import torch
import torch.nn as nn
from timm.layers import DropPath
from timm.layers import Mlp as TimmMlp


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: (..., d), d even
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def build_inv_freq(dim: int, theta: float, device, dtype) -> torch.Tensor:
    # стандартная RoPE-частота: inv_freq[i] = theta^(-2i/dim)
    # здесь мы работаем с dim (реальная размерность), пары -> шаг 2
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    return inv.to(dtype=torch.float32)


@torch.no_grad()
def build_spatial_cos_sin_axial(
    H: int,
    W: int,
    dim_x: int,
    dim_y: int,
    theta: float,
    device,
    out_dtype,
):
    """
    Возвращает cos/sin для пространственной решётки (H*W) отдельно для x-части и y-части.

    cos_x, sin_x: [1,1,1,H*W, dim_x]  (broadcast на B, heads, T)
    cos_y, sin_y: [1,1,1,H*W, dim_y]
    """
    assert dim_x % 2 == 0 and dim_y % 2 == 0, "dim_x and dim_y must be even for RoPE pairing."

    # координаты для row-major flatten: index = y*W + x
    pos_x = torch.arange(W, device=device, dtype=torch.float32).repeat(H)                 # [H*W]
    pos_y = torch.arange(H, device=device, dtype=torch.float32).repeat_interleave(W)     # [H*W]

    inv_x = build_inv_freq(dim_x, theta, device, out_dtype)  # [dim_x/2] (float32)
    inv_y = build_inv_freq(dim_y, theta, device, out_dtype)  # [dim_y/2] (float32)

    # angles: [H*W, dim/2]
    ang_x = pos_x[:, None] * inv_x[None, :]
    ang_y = pos_y[:, None] * inv_y[None, :]

    cos_x = ang_x.cos()
    sin_x = ang_x.sin()
    cos_y = ang_y.cos()
    sin_y = ang_y.sin()

    # expand to match last dim (repeat each freq twice, for pair dims)
    cos_x = torch.repeat_interleave(cos_x, repeats=2, dim=-1)  # [H*W, dim_x]
    sin_x = torch.repeat_interleave(sin_x, repeats=2, dim=-1)
    cos_y = torch.repeat_interleave(cos_y, repeats=2, dim=-1)  # [H*W, dim_y]
    sin_y = torch.repeat_interleave(sin_y, repeats=2, dim=-1)

    # reshape for broadcast over (B, heads, T, HW, dim_part)
    cos_x = cos_x[None, None, None, :, :].to(dtype=out_dtype)
    sin_x = sin_x[None, None, None, :, :].to(dtype=out_dtype)
    cos_y = cos_y[None, None, None, :, :].to(dtype=out_dtype)
    sin_y = sin_y[None, None, None, :, :].to(dtype=out_dtype)

    return cos_x, sin_x, cos_y, sin_y


class SpatialRoPE2D(nn.Module):
    """
    2D axial RoPE по пространству (H,W), игнорирует время:
    применяется одинаковая spatial-сетка для каждого t.
    """
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0, "For clean axial split, require head_dim % 4 == 0."
        self.head_dim = head_dim
        self.theta = float(theta)
        self._cache = {}  # (H,W,device_index,dtype) -> (cos_x,sin_x,cos_y,sin_y)

    def _get_cached(self, H: int, W: int, device: torch.device, dtype: torch.dtype):
        key = (H, W, device.type, device.index, str(dtype))
        if key not in self._cache:
            dim_x = self.head_dim // 2
            dim_y = self.head_dim - dim_x
            cos_x, sin_x, cos_y, sin_y = build_spatial_cos_sin_axial(
                H=H, W=W, dim_x=dim_x, dim_y=dim_y, theta=self.theta,
                device=device, out_dtype=dtype
            )
            self._cache[key] = (cos_x, sin_x, cos_y, sin_y)
        return self._cache[key]

    def forward(self, q: torch.Tensor, k: torch.Tensor, T: int, H: int, W: int):
        """
        q,k: [B, heads, N, head_dim], N=T*H*W
        """
        B, heads, N, d = q.shape
        assert d == self.head_dim
        assert N == T * H * W

        cos_x, sin_x, cos_y, sin_y = self._get_cached(H, W, q.device, q.dtype)

        # reshape to expose (T, HW) and broadcast cos/sin over T
        q = q.view(B, heads, T, H * W, d)
        k = k.view(B, heads, T, H * W, d)

        dim_x = d // 2
        dim_y = d - dim_x

        qx, qy = q[..., :dim_x], q[..., dim_x:]
        kx, ky = k[..., :dim_x], k[..., dim_x:]

        # RoPE on x-part uses pos_x; on y-part uses pos_y
        qx = (qx * cos_x) + (rotate_half(qx) * sin_x)
        kx = (kx * cos_x) + (rotate_half(kx) * sin_x)

        qy = (qy * cos_y) + (rotate_half(qy) * sin_y)
        ky = (ky * cos_y) + (rotate_half(ky) * sin_y)

        q = torch.cat([qx, qy], dim=-1).view(B, heads, N, d)
        k = torch.cat([kx, ky], dim=-1).view(B, heads, N, d)
        return q, k


class SpatialRoPE2DMixed(nn.Module):
    """
    Mixed 2D RoPE (как в RoPE-ViT):
      - обучаемые частоты per-head: freqs_x[h, d_c], freqs_y[h, d_c]
      - angle(x,y) = x*freqs_x + y*freqs_y
      - применяем RoPE к q,k (реальные пары) для каждого (x,y) токена
    Игнорирует время: одна spatial-сетка повторяется для каждого t.

    Требование: head_dim % 4 == 0 (чтобы корректно построить "mag" длины head_dim/4 и
    получить dim_c = head_dim/2 комплексных компонент).
    """
    def __init__(self, head_dim: int, num_heads: int, theta: float = 10.0, rotate_init: bool = True):
        super().__init__()
        assert head_dim % 4 == 0, "Mixed RoPE init here assumes head_dim % 4 == 0."
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.theta = float(theta)

        # dim_c = число комплексных компонент на голову = head_dim/2
        dim_c = head_dim // 2

        # mag: базовые частоты, длина head_dim/4 (как в референсе)
        # mag[i] = theta^(- (4i)/head_dim ) эквивалентно шагу torch.arange(0, head_dim, 4)/head_dim
        mag = 1.0 / (self.theta ** (torch.arange(0, head_dim, 4, dtype=torch.float32)[: head_dim // 4] / head_dim))
        # соберём freqs_x/freqs_y: [heads, dim_c]
        freqs_x = []
        freqs_y = []

        for _ in range(num_heads):
            a = torch.rand(1) * (2 * math.pi) if rotate_init else torch.zeros(1)
            # длина: 2*(head_dim/4) = head_dim/2 = dim_c
            fx = torch.cat([mag * torch.cos(a), mag * torch.cos((math.pi / 2) + a)], dim=-1)
            fy = torch.cat([mag * torch.sin(a), mag * torch.sin((math.pi / 2) + a)], dim=-1)
            freqs_x.append(fx)
            freqs_y.append(fy)

        freqs_x = torch.stack(freqs_x, dim=0)  # [heads, dim_c]
        freqs_y = torch.stack(freqs_y, dim=0)  # [heads, dim_c]

        # обучаемые параметры (обычно без weight_decay)
        self.freqs_x = nn.Parameter(freqs_x, requires_grad=True)
        self.freqs_y = nn.Parameter(freqs_y, requires_grad=True)

        # кешируем только координаты (они не зависят от параметров)
        self._pos_cache = {}  # (H,W,device)->(pos_x,pos_y)

    @torch.no_grad()
    def _get_pos_xy(self, H: int, W: int, device: torch.device):
        key = (H, W, device.type, device.index)
        if key not in self._pos_cache:
            # row-major flatten: idx = y*W + x
            pos_x = torch.arange(W, device=device, dtype=torch.float32).repeat(H)                 # [HW]
            pos_y = torch.arange(H, device=device, dtype=torch.float32).repeat_interleave(W)     # [HW]
            self._pos_cache[key] = (pos_x, pos_y)
        return self._pos_cache[key]

    def forward(self, q: torch.Tensor, k: torch.Tensor, T: int, H: int, W: int):
        """
        q,k: [B, heads, N, head_dim], N=T*H*W
        returns rotated q,k with same shape
        """
        B, heads, N, d = q.shape
        assert heads == self.num_heads
        assert d == self.head_dim
        assert N == T * H * W

        pos_x, pos_y = self._get_pos_xy(H, W, q.device)  # [HW], [HW]
        HW = H * W
        dim_c = d // 2

        # reshape, чтобы применить одну и ту же spatial фразу для каждого t
        q = q.view(B, heads, T, HW, d)
        k = k.view(B, heads, T, HW, d)

        # angles: [heads, HW, dim_c] (float32 для стабильности)
        # angle[h, n, j] = pos_x[n]*freqs_x[h,j] + pos_y[n]*freqs_y[h,j]
        fx = self.freqs_x.to(dtype=torch.float32, device=q.device)  # [heads, dim_c]
        fy = self.freqs_y.to(dtype=torch.float32, device=q.device)  # [heads, dim_c]

        angles = (
            torch.einsum("n,hd->hnd", pos_x, fx) +
            torch.einsum("n,hd->hnd", pos_y, fy)
        )  # [heads, HW, dim_c]

        cos = angles.cos()
        sin = angles.sin()

        # расширяем комплексные компоненты до реальных пар: [heads, HW, head_dim]
        cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1)

        # broadcast на (B, heads, T, HW, d)
        cos = cos[None, :, None, :, :].to(dtype=q.dtype)
        sin = sin[None, :, None, :, :].to(dtype=q.dtype)

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        q = q.view(B, heads, N, d)
        k = k.view(B, heads, N, d)
        return q, k

class SpatialRoPEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope_theta: float = 10000.0,
        rope_mode: str = "axial",   # "axial" | "mixed"
        mixed_rotate_init: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        rope_mode = rope_mode.lower()
        if rope_mode == "mixed":
            self.rope = SpatialRoPE2DMixed(
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                theta=rope_theta,
                rotate_init=mixed_rotate_init,
            )
        elif rope_mode == "axial":
            self.rope = SpatialRoPE2D(head_dim=self.head_dim, theta=rope_theta)
        else:
            raise ValueError(f"Unknown rope_mode={rope_mode}")

    def forward(self, x: torch.Tensor, *, T: int, H: int, W: int):
        B, N, C = x.shape
        assert C == self.dim
        assert N == T * H * W

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]

        q, k = self.rope(q, k, T=T, H=H, W=W)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class LatentBlock(nn.Module):
    """
    Аналог timm ViT Block, но attention принимает (T,H,W), чтобы применить spatial RoPE.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        rope_theta: float = 10000.0,
        rope_mode: str = "axial",  # "axial" | "mixed"
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpatialRoPEAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, rope_theta=rope_theta, rope_mode=rope_mode
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = TimmMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, *, T: int, H: int, W: int):
        x = x + self.drop_path(self.attn(self.norm1(x), T=T, H=H, W=W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LatentViT(nn.Module):
    """
    "LatentViT" энкодер:
      - вход:  (B, T, C, H, W)
      - APE по времени (learned)
      - flatten -> (B, T*H*W, C)
      - Transformer blocks с spatial RoPE (2D axial), без учёта t в RoPE
      - выход: (B, T, C, H, W)
    """
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        max_T: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        rope_theta: float = 10000.0,
        rope_mode: str = "axial",  # "axial" | "mixed"
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.max_T = max_T

        self.time_embed = nn.Embedding(max_T, dim)  # APE по времени
        self.pos_drop = nn.Dropout(drop)

        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([
            LatentBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop, attn_drop=attn_drop, drop_path=dpr[i],
                rope_theta=rope_theta, rope_mode=rope_mode, norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape
        assert C == self.dim, f"Expected C={self.dim}, got {C}"
        if T > self.max_T:
            raise ValueError(f"T={T} exceeds max_T={self.max_T} for time embedding.")

        # add time APE
        t_ids = torch.arange(T, device=x.device)
        t_emb = self.time_embed(t_ids)  # [T, C]

        x = x.permute(0, 1, 3, 4, 2)  # [B, T, H, W, C]
        x = x + t_emb[None, :, None, None, :]  # broadcast over H,W
        x = x.reshape(B, T * H * W, C)          # [B, N, C]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, T=T, H=H, W=W)

        x = self.norm(x)
        x = x.view(B, T, H, W, C).permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
        return x