"""Reproducible voxel baselines and an experimental DTPA-derived 3D head.

All models return *logits* in ``[B, C, Z, Y, X]`` and expose
``forward(image, features=None)``.  ``features`` are optional only for voxel
baselines.  The feature models consume frozen, slice-wise SAM embeddings; they
do not implement a SAM encoder, mask decoder, prompt router, or correction
controller and must not be described as a full reproduction of DTPA.

``convnext3d_custom`` is a local 3D adaptation of ConvNeXt-style blocks, not an
official ConvNeXt implementation/checkpoint.  MONAI models are optional; their
installed version and configuration must be recorded with each experiment.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    return next(g for g in (8, 4, 2, 1) if channels % g == 0)


def _validate_image(image: Tensor) -> None:
    if image.ndim != 5:
        raise ValueError(f"image must be [B,C,Z,Y,X], received {tuple(image.shape)}")


def _resize_xy(tensor: Tensor, target: tuple[int, int]) -> Tensor:
    """Resize each slice independently; never interpolate between Z planes."""
    if tensor.shape[-2:] == target:
        return tensor
    b, c, z, y, x = tensor.shape
    slices = tensor.permute(0, 2, 1, 3, 4).reshape(b * z, c, y, x)
    slices = F.interpolate(slices, size=target, mode="bilinear", align_corners=False)
    return slices.reshape(b, z, c, *target).permute(0, 2, 1, 3, 4).contiguous()


class ChannelLayerNorm3D(nn.Module):
    """Normalize channels per voxel, without sharing statistics across slices."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()


class ResidualConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.body(x) + self.skip(x))


class AnisotropicUNet3D(nn.Module):
    """Local residual U-Net reference; the first downsampling preserves Z."""

    def __init__(self, in_channels: int = 1, out_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        widths = [base_channels * scale for scale in (1, 2, 4, 8)]
        self.encoders = nn.ModuleList([ResidualConv3D(in_channels, widths[0])])
        self.downsamples = nn.ModuleList()
        for index, stride in enumerate(((1, 2, 2), (2, 2, 2), (2, 2, 2))):
            self.downsamples.append(nn.Conv3d(widths[index], widths[index + 1], 3, stride=stride, padding=1))
            self.encoders.append(ResidualConv3D(widths[index + 1], widths[index + 1]))
        self.decoders = nn.ModuleList([
            ResidualConv3D(widths[index + 1] + widths[index], widths[index]) for index in (2, 1, 0)
        ])
        self.classifier = nn.Conv3d(widths[0], out_channels, 1)

    def forward(self, image: Tensor, features: Tensor | None = None) -> Tensor:
        _validate_image(image)
        skip_features = [self.encoders[0](image)]
        for downsample, encoder in zip(self.downsamples, self.encoders[1:]):
            skip_features.append(encoder(downsample(skip_features[-1])))
        x = skip_features[-1]
        for decoder, skip in zip(self.decoders, reversed(skip_features[:-1])):
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.classifier(x)


class ConvNeXtBlock3D(nn.Module):
    """Custom anisotropic depthwise ConvNeXt-style residual block."""

    def __init__(self, channels: int, expansion: int = 4, layer_scale: float = 1e-6) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(channels, channels, (3, 7, 7), padding=(1, 3, 3), groups=channels)
        self.norm = ChannelLayerNorm3D(channels)
        self.pointwise = nn.Sequential(
            nn.Conv3d(channels, channels * expansion, 1), nn.GELU(),
            nn.Conv3d(channels * expansion, channels, 1),
        )
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1, 1), layer_scale))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.scale * self.pointwise(self.norm(self.depthwise(x)))


class CustomConvNeXt3D(nn.Module):
    """Original-volume custom ConvNeXt encoder/decoder; not an official model."""

    def __init__(self, in_channels: int = 1, out_channels: int = 3, base_channels: int = 16, blocks_per_stage: int = 2) -> None:
        super().__init__()
        widths = [base_channels * scale for scale in (1, 2, 4, 8)]
        self.stem = nn.Conv3d(in_channels, widths[0], (1, 3, 3), padding=(0, 1, 1))
        self.encoders = nn.ModuleList([
            nn.Sequential(*(ConvNeXtBlock3D(width) for _ in range(blocks_per_stage))) for width in widths
        ])
        self.downsamples = nn.ModuleList([
            nn.Sequential(ChannelLayerNorm3D(widths[index]),
                          nn.Conv3d(widths[index], widths[index + 1], 3, stride=stride, padding=1))
            for index, stride in enumerate(((1, 2, 2), (2, 2, 2), (2, 2, 2)))
        ])
        self.decoders = nn.ModuleList([
            nn.Sequential(nn.Conv3d(widths[index + 1] + widths[index], widths[index], 1),
                          ConvNeXtBlock3D(widths[index])) for index in (2, 1, 0)
        ])
        self.classifier = nn.Conv3d(widths[0], out_channels, 1)

    def forward(self, image: Tensor, features: Tensor | None = None) -> Tensor:
        _validate_image(image)
        skip_features = [self.encoders[0](self.stem(image))]
        for downsample, encoder in zip(self.downsamples, self.encoders[1:]):
            skip_features.append(encoder(downsample(skip_features[-1])))
        x = skip_features[-1]
        for decoder, skip in zip(self.decoders, reversed(skip_features[:-1])):
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        return self.classifier(x)


class AnisotropicFeatureBlock(nn.Module):
    """Separate XY and Z fusion; no-Z mode has strictly slice-local outputs."""

    def __init__(self, channels: int, mix_z: bool = True) -> None:
        super().__init__()
        axial_kernel = (3, 1, 1) if mix_z else (1, 1, 1)
        axial_padding = (1, 0, 0) if mix_z else (0, 0, 0)
        self.body = nn.Sequential(
            ChannelLayerNorm3D(channels),
            nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1), groups=channels),
            nn.Conv3d(channels, channels, axial_kernel, padding=axial_padding, groups=channels),
            nn.Conv3d(channels, channels * 2, 1), nn.GELU(),
            nn.Conv3d(channels * 2, channels, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.body(x)


class _FrozenFeatureModel(nn.Module):
    requires_features = True

    def __init__(self, feature_channels: int) -> None:
        super().__init__()
        self.feature_channels = feature_channels

    def _check_features(self, image: Tensor, features: Tensor | None) -> Tensor:
        _validate_image(image)
        if features is None:
            raise ValueError("This model requires frozen slice-wise SAM features [B,C,Z,h,w].")
        if features.ndim != 5 or features.shape[:1] != image.shape[:1]:
            raise ValueError("SAM features must be 5D and have the same batch size as image.")
        if features.shape[1] != self.feature_channels:
            raise ValueError(f"Expected {self.feature_channels} feature channels, received {features.shape[1]}.")
        if features.shape[2] != image.shape[2]:
            raise ValueError("Each input Z slice requires its own embedding; image/features Z lengths must match.")
        return features.detach()


class DTPA3DFeatureFusion(_FrozenFeatureModel):
    """Unvalidated DTPA-derived prototype, trained on frozen 2D SAM embeddings.

    XY and axial convolutions learn 3D topology. A slice-local image branch
    retains native XY details. The output is a topology probability precursor,
    not a 3D instance mask; downstream instance reconstruction is still needed.
    ``mix_z=False`` and ``raw_detail=False`` define matched ablations.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 3, base_channels: int = 16,
                 feature_channels: int = 256, fusion_depth: int = 3,
                 mix_z: bool = True, raw_detail: bool = True) -> None:
        super().__init__(feature_channels)
        self.mix_z = mix_z
        self.raw_detail = raw_detail
        width = base_channels * 2
        self.feature_projection = nn.Conv3d(feature_channels, width, 1)
        self.feature_fusion = nn.Sequential(*(AnisotropicFeatureBlock(width, mix_z) for _ in range(fusion_depth)))
        if raw_detail:
            self.image_branch = nn.Sequential(
                nn.Conv3d(in_channels, base_channels, (1, 3, 3), padding=(0, 1, 1)),
                ChannelLayerNorm3D(base_channels), nn.GELU(),
                nn.Conv3d(base_channels, base_channels, (1, 3, 3), padding=(0, 1, 1)), nn.GELU(),
            )
        self.output_fusion = nn.Sequential(
            nn.Conv3d(width + (base_channels if raw_detail else 0), width, 1),
            AnisotropicFeatureBlock(width, mix_z),
            nn.Conv3d(width, out_channels, 1),
        )

    def forward(self, image: Tensor, features: Tensor | None = None) -> Tensor:
        frozen_features = self._check_features(image, features)
        x = self.feature_fusion(self.feature_projection(frozen_features))
        x = _resize_xy(x, image.shape[-2:])
        if self.raw_detail:
            x = torch.cat((x, self.image_branch(image)), dim=1)
        return self.output_fusion(x)


class Frozen2DTopologyHead(_FrozenFeatureModel):
    """Original DTPA topology-head structure applied independently to Z slices.

    Defaults reproduce the 25,475-parameter LN(256)->Linear(96)->GELU->
    Dropout(0.1)->Linear(3) head structure. This is a head baseline only, not a
    reproduction of the complete DTPA/SAM prompting and correction pipeline.
    """

    def __init__(self, feature_channels: int = 256, out_channels: int = 3,
                 hidden_channels: int = 96, dropout: float = 0.1) -> None:
        super().__init__(feature_channels)
        self.head = nn.Sequential(nn.LayerNorm(feature_channels),
                                  nn.Linear(feature_channels, hidden_channels), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_channels, out_channels))

    def forward(self, image: Tensor, features: Tensor | None = None) -> Tensor:
        frozen_features = self._check_features(image, features)
        logits = self.head(frozen_features.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3).contiguous()
        return _resize_xy(logits, image.shape[-2:])


class MONAIVoxelModel(nn.Module):
    """Interface adapter; Swin inputs are padded to multiples of 32, then cropped."""

    def __init__(self, network: nn.Module, pad_multiple: int = 1,
                 constructor_options: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.network = network
        self.pad_multiple = pad_multiple
        self.constructor_options = constructor_options or {}

    def forward(self, image: Tensor, features: Tensor | None = None) -> Tensor:
        _validate_image(image)
        original_shape = image.shape[-3:]
        if self.pad_multiple > 1:
            pad_z, pad_y, pad_x = [(-length) % self.pad_multiple for length in original_shape]
            image = F.pad(image, (0, pad_x, 0, pad_y, 0, pad_z), mode="replicate")
        logits = self.network(image)
        return logits[..., :original_shape[0], :original_shape[1], :original_shape[2]]


MODEL_NAMES = (
    "unet3d", "convnext3d_custom", "segresnet", "swinunetr",
    "dtpa2d_topology", "dtpa3d_feature_fusion", "dtpa3d_no_z", "dtpa3d_no_raw_detail",
    "dtpa3d_pyramid", "pyramid3d_raw",
    "dtpa3d_pyramid_no_z", "dtpa3d_pyramid_no_raw_input", "plantseg_arch_topology22",
)


def build_model(name: str, in_channels: int = 1, out_channels: int = 3,
                base_channels: int = 16, feature_channels: int = 256, **kwargs: Any) -> nn.Module:
    """Build a model with explicit optional dependencies and strict arguments.

    MONAI ``SwinUNETR`` API differences are checked at runtime. ``feature_size``
    defaults to 24 (must be divisible by 12). Group normalization makes small
    Z-patches viable after padding. Pass ``use_checkpoint=True`` for lower
    activation memory. No pretrained weights are downloaded by this factory.
    """
    name = name.lower().replace("-", "_")
    if min(in_channels, out_channels, base_channels, feature_channels) < 1:
        raise ValueError("Channel counts must be positive.")
    if name in ("dtpa3d_pyramid", "pyramid3d_raw"):
        from .pyramid import build_pyramid_model
        return build_pyramid_model(name, in_channels, out_channels, base_channels, feature_channels, **kwargs)
    if name in ("dtpa3d_pyramid_no_z", "dtpa3d_pyramid_no_raw_input"):
        from .pyramid_ablations import build_pyramid_ablation
        return build_pyramid_ablation(name, in_channels, out_channels, base_channels, feature_channels, **kwargs)
    if name == "plantseg_arch_topology22":
        if in_channels != 1 or out_channels != 3 or kwargs:
            raise ValueError("Matched PlantSeg architecture requires one image/three topology channels and fixed official widths")
        from .plantseg import PlantSegTopologyReference
        return PlantSegTopologyReference()
    if name == "unet3d":
        if kwargs:
            raise TypeError(f"Unexpected unet3d arguments: {sorted(kwargs)}")
        return AnisotropicUNet3D(in_channels, out_channels, base_channels)
    if name == "convnext3d_custom":
        return CustomConvNeXt3D(in_channels, out_channels, base_channels, **kwargs)
    if name == "dtpa2d_topology":
        return Frozen2DTopologyHead(feature_channels=feature_channels, out_channels=out_channels, **kwargs)
    if name in ("dtpa3d_feature_fusion", "dtpa3d_no_z", "dtpa3d_no_raw_detail"):
        expected = {"mix_z": name != "dtpa3d_no_z", "raw_detail": name != "dtpa3d_no_raw_detail"}
        for key, value in expected.items():
            if key in kwargs and kwargs[key] != value:
                raise ValueError(f"Model name {name} requires {key}={value}.")
            kwargs[key] = value
        return DTPA3DFeatureFusion(in_channels, out_channels, base_channels, feature_channels, **kwargs)
    if name in ("segresnet", "swinunetr"):
        try:
            from monai.networks.nets import SegResNet, SwinUNETR
        except ImportError as error:
            raise ImportError(f"Model {name} requires MONAI. Install the version pinned in the project environment.") from error
        if name == "segresnet":
            options = dict(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                           init_filters=base_channels, norm=("GROUP", {"num_groups": _groups(base_channels)}))
            options.update(kwargs)
            return MONAIVoxelModel(SegResNet(**options), constructor_options=options)
        options = dict(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                       feature_size=24, norm_name=("group", {"num_groups": 8}),
                       use_checkpoint=False)
        if "img_size" in inspect.signature(SwinUNETR).parameters:
            options["img_size"] = (96, 96, 96)
        options.update(kwargs)
        return MONAIVoxelModel(SwinUNETR(**options), pad_multiple=32, constructor_options=options)
    raise ValueError(f"Unknown model {name!r}. Available models: {', '.join(MODEL_NAMES)}")


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {"total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}
