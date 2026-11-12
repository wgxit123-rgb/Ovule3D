"""Development-stage V2 raw/SAM topology pyramids, added after initial screening.

The initial 31k-parameter head and all its results remain separate V1 evidence.
These new models are not preregistered first-round models and have no assumed
accuracy advantage. They output three topology logits, not SAM-decoded instances.

Shared raw path: native XY detail (16 channels) -> XY/2 (32) -> Z/2,XY/4 (64)
-> gated low-grid context fusion -> skip decoder -> native-grid logits. Context
uses small 3D convolutions on a low XY grid; no depthwise 3D stack runs on the
full-resolution grid. SAM features are detached. The raw control replaces SAM
context input with pooled raw encoder features and retains the same fusion and
decoder. Its input projection has fewer parameters, so capacity is similar but
not exactly matched; ``architecture_spec`` reports the actual count.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    return next(group for group in (8, 4, 2, 1) if channels % group == 0)


class ChannelNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, xy_only: bool = False):
        super().__init__()
        kernel, padding = ((1, 3, 3), (0, 1, 1)) if xy_only else ((3, 3, 3), (1, 1, 1))
        self.body = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel, padding=padding, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels), nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel, padding=padding, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.body(x) + self.shortcut(x))


def _resize_xy(x: Tensor, target: tuple[int, int]) -> Tensor:
    if tuple(x.shape[-2:]) == tuple(target):
        return x
    batch, channels, depth, height, width = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(batch * depth, channels, height, width)
    x = F.interpolate(x, size=target, mode="bilinear", align_corners=False)
    return x.reshape(batch, depth, channels, *target).permute(0, 2, 1, 3, 4).contiguous()


class _TopologyPyramid(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3, base_channels: int = 16,
                 feature_channels: int = 256, initial_gate: float = -2.0, *, use_sam: bool):
        super().__init__()
        if min(in_channels, out_channels, base_channels, feature_channels) < 1:
            raise ValueError("Channel counts must be positive")
        self.requires_features = use_sam
        self.feature_channels = feature_channels
        self.base_channels = base_channels
        self.initial_gate = float(initial_gate)
        width0, width1, width2 = base_channels, base_channels * 2, base_channels * 4
        self.stem = nn.Sequential(nn.Conv3d(in_channels, width0, (1, 3, 3), padding=(0, 1, 1)),
                                  ResidualBlock(width0, width0, xy_only=True))
        self.down_xy = nn.Sequential(
            nn.Conv3d(width0, width1, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            ResidualBlock(width1, width1))
        self.down_zyx = nn.Sequential(
            nn.Conv3d(width1, width2, 3, stride=(2, 2, 2), padding=1),
            ResidualBlock(width2, width2))
        context_input_channels = feature_channels if use_sam else width1
        self.context = nn.Sequential(
            nn.Conv3d(context_input_channels, width1, 1), ChannelNorm(width1), nn.GELU(),
            ResidualBlock(width1, width1),
            # Z downsampling matches the raw down_zyx convolution's centres;
            # subsequent resizing is XY-only, avoiding a half-slice Z shift.
            nn.Conv3d(width1, width1, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
        )
        self.context_gate_logit = nn.Parameter(torch.tensor(float(initial_gate)))
        self.fusion = nn.Sequential(nn.Conv3d(width2 + width1, width2, 1),
                                    ResidualBlock(width2, width2))
        self.decode_xy = ResidualBlock(width2 + width1, width1)
        self.decode_native = ResidualBlock(width1 + width0, width0, xy_only=True)
        self.classifier = nn.Conv3d(width0, out_channels, 1)

    def forward(self, image: Tensor, features: Tensor | None = None) -> Tensor:
        if image.ndim != 5:
            raise ValueError("image must be [B,C,Z,Y,X]")
        if self.requires_features:
            if features is None or features.ndim != 5:
                raise ValueError("dtpa3d_pyramid requires frozen SAM features [B,C,Z,h,w]")
            if features.shape[0] != image.shape[0] or features.shape[2] != image.shape[2]:
                raise ValueError("Image/features batch and Z lengths must match")
            if features.shape[1] != self.feature_channels:
                raise ValueError(f"Expected {self.feature_channels} SAM feature channels")
        native = self.stem(image)
        middle = self.down_xy(native)
        bottleneck = self.down_zyx(middle)
        if self.requires_features:
            context_input = features.detach()
        else:
            # Match the cached embedding's nominal stride-16 XY context grid.
            context_input = F.adaptive_avg_pool3d(
                middle, (image.shape[2], max(1, math.ceil(image.shape[3] / 16)),
                         max(1, math.ceil(image.shape[4] / 16))))
        context = self.context(context_input)
        if context.shape[2] != bottleneck.shape[2]:
            raise AssertionError("Context and raw Z downsampling must align exactly")
        context = _resize_xy(context, bottleneck.shape[-2:])
        context = self.context_gate_logit.sigmoid() * context
        fused = bottleneck + self.fusion(torch.cat((bottleneck, context), dim=1))
        decoded = F.interpolate(fused, size=middle.shape[-3:], mode="trilinear", align_corners=False)
        decoded = self.decode_xy(torch.cat((decoded, middle), dim=1))
        decoded = F.interpolate(decoded, size=native.shape[-3:], mode="trilinear", align_corners=False)
        return self.classifier(self.decode_native(torch.cat((decoded, native), dim=1)))

    def architecture_spec(self) -> dict:
        count = sum(parameter.numel() for parameter in self.parameters())
        return {
            "name": "dtpa3d_pyramid" if self.requires_features else "pyramid3d_raw",
            "experiment_stage": "V2 development extension after first-round screening; not original preregistration",
            "trainable_parameters": count,
            "raw_encoder_widths": [self.base_channels * scale for scale in (1, 2, 4)],
            "raw_downsampling": [[1, 2, 2], [2, 2, 2]],
            "context_channels": self.base_channels * 2,
            "context_source": "detached frozen SAM embeddings" if self.requires_features else "pooled trainable raw encoder features",
            "context_initial_gate_probability": float(torch.tensor(self.initial_gate).sigmoid()),
            "context_current_gate_probability": float(self.context_gate_logit.detach().sigmoid().cpu()),
            "context_z_alignment": "stride-(2,1,1) convolution; XY-only interpolation into raw bottleneck",
            "capacity_control": "same raw/fusion/decoder; context input projections differ, not exactly capacity matched",
            "sam_encoder_parameters_included": False,
            "sam_mask_decoder_implemented": False,
            "output": "native-grid background/interior/wall logits",
            "first_round_results_retained": True,
        }


class DTPA3DPyramid(_TopologyPyramid):
    requires_features = True

    def __init__(self, in_channels=1, out_channels=3, base_channels=16,
                 feature_channels=256, initial_gate=-2.0):
        super().__init__(in_channels, out_channels, base_channels, feature_channels, initial_gate, use_sam=True)


class Pyramid3DRaw(_TopologyPyramid):
    requires_features = False

    def __init__(self, in_channels=1, out_channels=3, base_channels=16,
                 feature_channels=256, initial_gate=-2.0):
        super().__init__(in_channels, out_channels, base_channels, feature_channels, initial_gate, use_sam=False)


def build_pyramid_model(name: str, in_channels: int = 1, out_channels: int = 3,
                        base_channels: int = 16, feature_channels: int = 256,
                        initial_gate: float = -2.0) -> _TopologyPyramid:
    factories = {"dtpa3d_pyramid": DTPA3DPyramid, "pyramid3d_raw": Pyramid3DRaw}
    if name not in factories:
        raise ValueError(f"Unknown pyramid model {name!r}; choose {tuple(factories)}")
    return factories[name](in_channels, out_channels, base_channels, feature_channels, initial_gate)
