"""Volume-separated sampling, ignore-aware topology targets, aligned features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries
from torch.nn import functional as F
from torch.utils.data import Dataset

IGNORE = 255


def topology_target(labels: np.ndarray, ignore_mask: np.ndarray | None = None) -> np.ndarray:
    """0 true background, 1 cell interior, 2 six-connected thick instance wall.

    Negative labels and explicitly ignored voxels never contribute to loss.
    Exclude one six-connected voxel around ignore to avoid invented boundaries.
    External array faces are not automatically treated as cell walls.
    """
    valid = labels >= 0
    if ignore_mask is not None:
        valid &= ~ignore_mask
    safe = np.where(valid, labels, 0)
    boundary = find_boundaries(safe, connectivity=1, mode="thick")
    target = np.zeros(labels.shape, dtype=np.uint8)
    target[safe > 0] = 1
    target[boundary] = 2
    invalid_halo = ndi.binary_dilation(~valid, structure=ndi.generate_binary_structure(3, 1))
    target[invalid_halo] = IGNORE
    return target


def feature_crop(features: np.ndarray, full_shape: tuple[int, int, int],
                 start: tuple[int, int, int], patch: tuple[int, int, int], stride: int = 16) -> torch.Tensor:
    """Crop global full-slice embeddings using consistent pixel-center coordinates.

    Cached features are [Z,C,h,w] and correspond to the complete working slice.
    This operation selects Z exactly and interpolates XY only. Out-of-image
    padding repeats feature borders, matching image padding.
    """
    z0, y0, x0 = start
    dz, dy, dx = patch
    z_indices = np.clip(np.arange(z0, z0 + dz), 0, full_shape[0] - 1)
    gy, gx = max(1, int(np.ceil(dy / stride))), max(1, int(np.ceil(dx / stride)))
    yy = y0 + (torch.arange(gy, dtype=torch.float32) + 0.5) * dy / gy
    xx = x0 + (torch.arange(gx, dtype=torch.float32) + 0.5) * dx / gx
    # Read only the feature-grid bounding box required for bilinear sampling.
    # Converting complete Z slices here needlessly copies tens/hundreds of MB
    # per tiny patch; the coordinate transformation below is equivalent.
    fy, fx = features.shape[-2:]
    coord_y = yy / full_shape[1] * fy - 0.5
    coord_x = xx / full_shape[2] * fx - 0.5
    y_low = int(np.clip(np.floor(float(coord_y.min())), 0, fy - 1))
    x_low = int(np.clip(np.floor(float(coord_x.min())), 0, fx - 1))
    y_high = int(np.clip(np.ceil(float(coord_y.max())), 0, fy - 1)) + 1
    x_high = int(np.clip(np.ceil(float(coord_x.max())), 0, fx - 1)) + 1
    local = torch.from_numpy(np.asarray(features[z_indices, :, y_low:y_high, x_low:x_high], dtype=np.float32))
    yy = (coord_y - y_low + 0.5) / (y_high - y_low) * 2 - 1
    xx = (coord_x - x_low + 0.5) / (x_high - x_low) * 2 - 1
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)[None].expand(dz, -1, -1, -1)
    return F.grid_sample(local, grid, mode="bilinear", padding_mode="border", align_corners=False).permute(1, 0, 2, 3)


def _extract(array: np.ndarray, start: tuple[int, ...], patch: tuple[int, ...], fill: int | None = None) -> np.ndarray:
    stops = [min(s + p, n) for s, p, n in zip(start, patch, array.shape)]
    cropped = np.array(array[tuple(slice(s, t) for s, t in zip(start, stops))], copy=True)
    padding = [(0, p - n) for p, n in zip(patch, cropped.shape)]
    if any(p[1] for p in padding):
        cropped = np.pad(cropped, padding, mode="edge" if fill is None else "constant",
                         **({} if fill is None else {"constant_values": fill}))
    return cropped


class VolumePatchDataset(Dataset):
    """Deterministic stateless patch sampler. Seed+index fixes volume and crop.

    Sampling is uniform by volume, never a random slice split. Validation uses
    a fixed indexed panel. Training samples may retry weakly annotated crops;
    all selected volume IDs and source paths remain in the run configuration.
    """

    def __init__(self, records: list[dict], patch: tuple[int, int, int], length: int,
                 seed: int, training: bool, features: bool = False):
        if not records:
            raise ValueError("No volume records")
        if any(record["split"] == "test" for record in records):
            raise ValueError("Training/validation patch dataset never accepts test records")
        self.records, self.patch, self.length = records, tuple(patch), length
        self.seed, self.training, self.features = seed, training, features
        self.arrays = {}
        for record in records:
            self.arrays[record["id"]] = {
                key: np.load(record[key], mmap_mode="r") for key in ("image", "target")
            }
            if features:
                feature_path = record.get("features")
                if not feature_path or not Path(feature_path).is_file():
                    raise FileNotFoundError(f"Frozen features missing for {record['id']}")
                self.arrays[record["id"]]["features"] = np.load(feature_path, mmap_mode="r")

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        record_index = int(rng.integers(len(self.records))) if self.training else int(index % len(self.records))
        record = self.records[record_index]
        arrays = self.arrays[record["id"]]
        for _ in range(32):
            start = tuple(int(rng.integers(max(1, n - p + 1))) for n, p in zip(arrays["image"].shape, self.patch))
            target = _extract(arrays["target"], start, self.patch, fill=IGNORE)
            if np.mean(target != IGNORE) >= 0.1 and np.count_nonzero(target == 1) > 32:
                break
        image = _extract(arrays["image"], start, self.patch).astype(np.float32)
        feature_tensor = feature_crop(arrays["features"], arrays["image"].shape, start, self.patch) if self.features else None
        image = torch.from_numpy(image[None])
        target = torch.from_numpy(target.astype(np.int64))
        if self.training:
            # Cache-preserving spatial augmentation. No intensity transform is
            # applied to raw alone, since embeddings would no longer match.
            for axis in range(3):
                if rng.random() < 0.5:
                    image = image.flip(axis + 1)
                    target = target.flip(axis)
                    if feature_tensor is not None:
                        feature_tensor = feature_tensor.flip(axis + 1)
            if self.patch[1] == self.patch[2]:
                turns = int(rng.integers(4))
                image = image.rot90(turns, (-2, -1))
                target = target.rot90(turns, (-2, -1))
                if feature_tensor is not None:
                    feature_tensor = feature_tensor.rot90(turns, (-2, -1))
        result = {"image": image.contiguous(), "target": target.contiguous(), "volume_id": record["id"], "start": torch.tensor(start)}
        if feature_tensor is not None:
            result["features"] = feature_tensor.contiguous()
        return result
