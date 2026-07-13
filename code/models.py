"""CNN, ViT and CNN-ViT hybrid models used by the course project."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """MobileNetV2-style local feature block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expand: int = 2) -> None:
        super().__init__()
        hidden = in_channels * expand
        self.use_residual = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            ConvBNAct(in_channels, hidden, kernel_size=1),
            ConvBNAct(hidden, hidden, kernel_size=3, stride=stride, groups=hidden),
            ConvBNAct(hidden, out_channels, kernel_size=1, activation=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return x + y if self.use_residual else y


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=depth, norm=nn.LayerNorm(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MobileViTBlock(nn.Module):
    """A compact local-global block inspired by MobileViT.

    Convolutions encode local structure. The feature map is then rearranged into
    P sequences, each containing the same relative pixel from every patch. A
    transformer models long-range relations along each sequence. Finally the
    representation is folded back and fused with the input feature map.
    """

    def __init__(
        self,
        channels: int,
        transformer_dim: int,
        depth: int,
        heads: int,
        patch_size: int = 2,
        fusion: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_h = patch_size
        self.patch_w = patch_size
        self.fusion = fusion
        self.local_rep = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3),
            ConvBNAct(channels, transformer_dim, kernel_size=1),
        )
        self.transformer = TransformerEncoder(transformer_dim, depth, heads, dropout=dropout)
        self.project = ConvBNAct(transformer_dim, channels, kernel_size=1)
        self.fuse = ConvBNAct(channels * 2, channels, kernel_size=3) if fusion else nn.Identity()

    def _unfold(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        b, d, h, w = x.shape
        new_h = math.ceil(h / self.patch_h) * self.patch_h
        new_w = math.ceil(w / self.patch_w) * self.patch_w
        pad_h, pad_w = new_h - h, new_w - w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        nh, nw = new_h // self.patch_h, new_w // self.patch_w
        # [B, D, Nh, Ph, Nw, Pw] -> [B*P, N, D]
        x = x.reshape(b, d, nh, self.patch_h, nw, self.patch_w)
        x = x.permute(0, 3, 5, 2, 4, 1).contiguous()
        x = x.reshape(b * self.patch_h * self.patch_w, nh * nw, d)
        return x, (h, w, new_h, new_w)

    def _fold(self, x: torch.Tensor, shape: tuple[int, int, int, int], batch_size: int) -> torch.Tensor:
        h, w, new_h, new_w = shape
        nh, nw = new_h // self.patch_h, new_w // self.patch_w
        d = x.shape[-1]
        x = x.reshape(batch_size, self.patch_h, self.patch_w, nh, nw, d)
        x = x.permute(0, 5, 3, 1, 4, 2).contiguous()
        x = x.reshape(batch_size, d, new_h, new_w)
        return x[:, :, :h, :w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        local = self.local_rep(x)
        tokens, shape = self._unfold(local)
        tokens = self.transformer(tokens)
        global_features = self._fold(tokens, shape, x.shape[0])
        global_features = self.project(global_features)
        if self.fusion:
            return self.fuse(torch.cat((residual, global_features), dim=1))
        return global_features


class HybridCNNViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        patch_size: int = 2,
        fusion: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 32, 3, 1),
            InvertedResidual(32, 48, stride=2, expand=2),
            MobileViTBlock(48, 64, depth=2, heads=4, patch_size=patch_size, fusion=fusion, dropout=dropout),
            InvertedResidual(48, 64, stride=2, expand=2),
            MobileViTBlock(64, 96, depth=2, heads=4, patch_size=patch_size, fusion=fusion, dropout=dropout),
            InvertedResidual(64, 96, stride=2, expand=2),
            MobileViTBlock(96, 128, depth=2, heads=4, patch_size=patch_size, fusion=fusion, dropout=dropout),
            ConvBNAct(96, 160, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(160, num_classes))
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


class TinyCNN(nn.Module):
    """Lightweight CNN baseline with the same downsampling schedule."""

    def __init__(self, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(3, 32, 3, 1),
            InvertedResidual(32, 48, stride=2, expand=2),
            InvertedResidual(48, 48, expand=2),
            InvertedResidual(48, 64, stride=2, expand=2),
            InvertedResidual(64, 64, expand=2),
            InvertedResidual(64, 96, stride=2, expand=2),
            InvertedResidual(96, 96, expand=2),
            ConvBNAct(96, 160, kernel_size=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(160, num_classes))
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)).flatten(1))


class TinyViT(nn.Module):
    """Small pure ViT baseline for 32-224 pixel inputs."""

    def __init__(
        self,
        num_classes: int,
        image_size: int = 32,
        patch_size: int = 4,
        dim: int = 128,
        depth: int = 6,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size for vit_tiny")
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(dim, depth, heads, mlp_ratio=4.0, dropout=dropout)
        self.head = nn.Linear(dim, num_classes)
        self.apply(_init_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.dropout(x + self.pos_embed[:, : x.shape[1]])
        x = self.encoder(x)
        return self.head(x[:, 0])


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


@dataclass(frozen=True)
class ModelInfo:
    name: str
    description: str


MODEL_INFO = {
    "cnn_tiny": ModelInfo("cnn_tiny", "Lightweight convolution-only baseline"),
    "vit_tiny": ModelInfo("vit_tiny", "Pure vision transformer baseline"),
    "hybrid_tiny": ModelInfo("hybrid_tiny", "CNN-ViT local-global hybrid"),
    "hybrid_no_fusion": ModelInfo("hybrid_no_fusion", "Hybrid without residual feature fusion"),
}


def build_model(
    name: str,
    num_classes: int,
    image_size: int = 32,
    patch_size: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    if name == "cnn_tiny":
        return TinyCNN(num_classes, dropout)
    if name == "vit_tiny":
        vit_patch = 4 if image_size <= 64 else 16
        return TinyViT(num_classes, image_size, vit_patch, dropout=dropout)
    if name == "hybrid_tiny":
        return HybridCNNViT(num_classes, patch_size, True, dropout)
    if name == "hybrid_no_fusion":
        return HybridCNNViT(num_classes, patch_size, False, dropout)
    raise ValueError(f"Unknown model: {name}. Choices: {', '.join(MODEL_INFO)}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

