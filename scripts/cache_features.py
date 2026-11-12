"""Freeze slice-wise SAM embeddings without reading annotation arrays."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn import functional as F

from ovule3d.io import read_json, sha256, utc_now, write_json


def resample_sam_embedding(encoded, input_shape, grid_shape, image_size=1024):
    """Map native-image cell centers to continuous SAM token coordinates."""
    yy = (torch.arange(grid_shape[0], device=encoded.device, dtype=torch.float32) + 0.5)
    xx = (torch.arange(grid_shape[1], device=encoded.device, dtype=torch.float32) + 0.5)
    yy = yy / grid_shape[0] * input_shape[0] / image_size * 2 - 1
    xx = xx / grid_shape[1] * input_shape[1] / image_size * 2 - 1
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)[None].expand(len(encoded), -1, -1, -1)
    return F.grid_sample(encoded.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--checkpoint", type=Path, default=Path(r"D:\cell\models\micro_sam_cache\models\vit_b_lm"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--limit-volumes", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    checkpoint_hash = sha256(args.checkpoint)
    grid_rule_id = "sam_continuous_unpadded_coordinates_v1"
    manifest_path = args.root / "cache/prepared_manifest.json"
    manifest = read_json(manifest_path)
    records = [r for r in manifest["records"] if r["split"] in args.splits]
    if args.limit_volumes:
        records = records[:args.limit_volumes]
    # Verified official local state_dict. weights_only avoids arbitrary pickle execution.
    sam = sam_model_registry["vit_b"]()
    sam.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    sam.requires_grad_(False).eval().to(device)
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    for record in records:
        metadata_path = Path(record["image"]).with_name("metadata.json")
        feature_path = Path(record["image"]).with_name("sam_features.npy")
        provenance_path = feature_path.with_suffix(".json")
        if feature_path.exists() and provenance_path.exists():
            provenance = read_json(provenance_path)
            if provenance["checkpoint_sha256"] != checkpoint_hash or provenance["source_sha256"] != record["source_sha256"]:
                raise RuntimeError(f"Feature provenance differs: {feature_path}")
            if provenance.get("grid_rule_id") != grid_rule_id:
                raise RuntimeError(f"Cached feature-grid convention differs: {feature_path}")
        else:
            image = np.load(record["image"], mmap_mode="r")
            z_length, height, width = image.shape
            grid_shape = (math.ceil(height / 16), math.ceil(width / 16))
            temporary = feature_path.with_suffix(".partial.npy")
            output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                              shape=(z_length, 256, *grid_shape))
            started = time.monotonic()
            for z0 in range(0, z_length, args.batch_size):
                values = np.asarray(image[z0:z0 + args.batch_size], dtype=np.float32)
                values = np.rint(values * 255).clip(0, 255)
                batch = torch.from_numpy(values[:, None]).expand(-1, 3, -1, -1).to(device)
                with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                           enabled=device.type == "cuda"):
                    resized = transform.apply_image_torch(batch)
                    input_shape = resized.shape[-2:]
                    encoded = sam.image_encoder(sam.preprocess(resized))
                    # Use continuous unpadded coordinates. Integer-rounded token
                    # crops stretch the short axis by several native pixels.
                    encoded = resample_sam_embedding(encoded, input_shape, grid_shape, sam.image_encoder.img_size)
                    if not torch.isfinite(encoded).all():
                        raise FloatingPointError(f"Non-finite frozen SAM features at {record['id']} z={z0}")
                output[z0:z0 + len(values)] = encoded.cpu().numpy().astype(np.float16)
                if z0 % (args.batch_size * 20) == 0:
                    print(f"features {record['id']} {min(z0 + len(values), z_length)}/{z_length} "
                          f"{time.monotonic() - started:.1f}s", flush=True)
            output.flush()
            del output
            temporary.replace(feature_path)
            write_json(provenance_path, {
                "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
                "model_type": "vit_b_lm", "source_sha256": record["source_sha256"],
                "shape": [z_length, 256, *grid_shape], "normalization": record["normalization"],
                "created_utc": utc_now(), "duration_seconds": time.monotonic() - started,
                "decoder_used": False, "pretraining_overlap_status": "See provenance/pretraining_audit.md",
                "grid_rule_id": grid_rule_id,
                "feature_grid_rule": "Continuous resized image coordinates in full padded SAM token grid; bilinear XY to ceil(native H/16,W/16); no integer token crop, no Z resampling",
            })
        metadata = read_json(metadata_path)
        metadata["features"] = str(feature_path.resolve())
        metadata["feature_provenance"] = str(provenance_path.resolve())
        write_json(metadata_path, metadata)
    # Reload to preserve any other records and regenerate fields from sidecars.
    manifest = read_json(manifest_path)
    for record in manifest["records"]:
        metadata_path = Path(record["image"]).with_name("metadata.json")
        record.update(read_json(metadata_path))
    manifest["features_updated_utc"] = utc_now()
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
