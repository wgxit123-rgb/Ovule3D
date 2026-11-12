"""Matched-family controls; the original V2 pyramid implementation is untouched.

No-Z removes ALL cross-slice information paths, including normalization's Z
statistics, not just convolution depth. Its capacity therefore differs. The
zero-raw-input control retains the entire nominal architecture but masks raw
intensities; it is an input ablation, not a smaller SAM-only architecture.
"""
from __future__ import annotations

from torch import Tensor, nn
import torch

from .pyramid import DTPA3DPyramid


class SliceGroupNorm(nn.GroupNorm):
    """GroupNorm over XY and grouped channels, independently per B,Z slice."""

    def forward(self, value: Tensor) -> Tensor:
        batch, channels, depth, height, width = value.shape
        slices = value.permute(0, 2, 1, 3, 4).reshape(batch * depth, channels, height, width)
        slices = super().forward(slices)
        return slices.reshape(batch, depth, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()


def remove_z_paths(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv3d) and (child.kernel_size[0] != 1 or child.stride[0] != 1):
            # Fresh initialization at the smaller fan-in. These are newly
            # trained controls, never cropped weights from a trained model.
            replacement = nn.Conv3d(child.in_channels, child.out_channels,
                                    (1, *child.kernel_size[1:]), stride=(1, *child.stride[1:]),
                                    padding=(0, *child.padding[1:]), dilation=(1, *child.dilation[1:]),
                                    groups=child.groups, bias=child.bias is not None,
                                    padding_mode=child.padding_mode)
            setattr(module, name, replacement)
        elif isinstance(child, nn.GroupNorm):
            setattr(module, name, SliceGroupNorm(child.num_groups, child.num_channels,
                                                eps=child.eps, affine=child.affine))
        else:
            remove_z_paths(child)


class PyramidNoZ(DTPA3DPyramid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        remove_z_paths(self)

    def architecture_spec(self):
        return {**super().architecture_spec(), "name": "dtpa3d_pyramid_no_z",
                "ablation": "all learned convolutions and normalization are slice-local; no Z downsampling",
                "raw_downsampling": [[1, 2, 2], [1, 2, 2]],
                "context_z_alignment": "Z preserved; resizing leaves Z unchanged",
                "capacity_control": "same widths, branches and XY scales; fewer axial parameters; normalization excludes Z",
                "interpretation": "3D-awareness ablation, not an isolated single-convolution effect"}


class PyramidNoRawInput(DTPA3DPyramid):
    def forward(self, image: Tensor, features: Tensor | None = None) -> Tensor:
        return super().forward(torch.zeros_like(image), features)

    def architecture_spec(self):
        return {**super().architecture_spec(), "name": "dtpa3d_pyramid_no_raw_input",
                "ablation": "zero raw input; retain SAM input and all nominal raw/fusion/decoder modules",
                "capacity_control": "identical nominal parameter count; raw-input weights cannot learn image-dependent cues",
                "interpretation": "raw-intensity information ablation, not removal of the raw-branch architecture"}


def build_pyramid_ablation(name, in_channels=1, out_channels=3, base_channels=16,
                           feature_channels=256, **kwargs):
    factories = {"dtpa3d_pyramid_no_z": PyramidNoZ,
                 "dtpa3d_pyramid_no_raw_input": PyramidNoRawInput}
    if name not in factories:
        raise ValueError(f"Unknown matched pyramid ablation: {name}")
    return factories[name](in_channels, out_channels, base_channels, feature_channels, **kwargs)
