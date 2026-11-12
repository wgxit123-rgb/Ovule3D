"""Pinned official PlantSeg boundary network, distinct from its graph pipeline."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from .io import sha256

ARCHITECTURE_SHA256 = "2da9f7149e144b63fee715fb8fe5154e96bf9aa230d176c91e923d528ef0c4e1"
WEIGHTS_SHA256 = "0f11cbd0608d12ef161892586408c6ec1a074f1f27ea6afef9d1ceece3dd3c1f"


class PlantSegTopologyReference(torch.nn.Module):
    """Official UNet architecture, fresh 3-class topology training, NOT PlantSeg.

    Same 22 public train files, loss/normalization and postprocessor as the new
    topology models. No official pretrained weights and no native GASP pipeline.
    """

    requires_features = False

    def __init__(self, directory=None):
        super().__init__()
        directory = Path(directory) if directory else Path(__file__).resolve().parents[1] / "models/plantseg_ds2x"
        source = directory / "unet.py"
        if sha256(source) != ARCHITECTURE_SHA256:
            raise ValueError("Pinned official PlantSeg source checksum mismatch")
        spec = importlib.util.spec_from_file_location("ovule3d_matched_plantseg_unet", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.network = module.UNet3D(in_channels=1, out_channels=3, f_maps=32, layer_order="gcr",
                                     num_groups=8, final_sigmoid=False, is_segmentation=True, testing=False)

    def forward(self, image, features=None):
        return self.network(image)

    def architecture_spec(self):
        return {"name": "plantseg_arch_topology22", "source_doi": "10.5281/zenodo.7805434",
                "architecture_sha256": ARCHITECTURE_SHA256, "feature_maps": [32, 64, 128, 256],
                "trainable_parameters": sum(p.numel() for p in self.parameters()),
                "initialization": "fresh; no official pretrained weights",
                "output": "three unactivated background/interior/wall logits",
                "changed_from_official_recipe": "three output channels; shared topology loss, percentile normalization and 22-file training budget",
                "native_plantseg_pipeline": False, "native_gasp": False,
                "base_channels_argument": "ignored; official f_maps=32 fixed explicitly"}


def load_boundary_model(directory: str | Path, device="cpu"):
    directory = Path(directory)
    if sha256(directory / "unet.py") != ARCHITECTURE_SHA256 or sha256(directory / "model.pytorch") != WEIGHTS_SHA256:
        raise ValueError("Pinned official PlantSeg source/weight checksum mismatch")
    spec = importlib.util.spec_from_file_location("ovule3d_pinned_plantseg_unet", directory / "unet.py")
    module = importlib.util.module_from_spec(spec)
    # The exact source was reviewed before implementing this adapter.
    spec.loader.exec_module(module)
    model = module.UNet3D(in_channels=1, out_channels=1, f_maps=32, layer_order="gcr", num_groups=8,
                         final_sigmoid=True, is_segmentation=True, testing=False)
    state = torch.load(directory / "model.pytorch", map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def global_intensity_moments(raw, z_slice=slice(None), *, z_block=8):
    """Float64 moments on the full raw prepared field, never an annotation ROI."""
    start, stop, step = z_slice.indices(raw.shape[0])
    if step != 1 or stop <= start:
        raise ValueError("Nonempty contiguous source Z crop required")
    count, total, squared = 0, 0.0, 0.0
    for first in range(start, stop, z_block):
        chunk = np.asarray(raw[first:min(first+z_block, stop)], dtype=np.float64)
        if not np.isfinite(chunk).all():
            raise ValueError("Non-finite raw intensity")
        count += chunk.size
        total += float(chunk.sum())
        squared += float(np.square(chunk).sum())
    mean = total/count
    std = float(np.sqrt(max(0, squared/count - mean**2)))
    return {"mean": mean, "std": std, "count": count, "z_start": start, "z_stop": stop,
            "std_floor": 1e-10, "normalization": "(original raw - global prepared-field mean)/max(std,1e-10)"}
