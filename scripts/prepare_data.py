"""Prepare train/validation caches; never reads test label arrays."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np

from ovule3d.data import topology_target
from ovule3d.io import read_json, sha256, utc_now, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--allow-partial", action="store_true", help="Smoke only, never a full-data result")
    args = parser.parse_args()
    manifest = read_json(args.root / "provenance/manifest.json")
    records = [r for r in manifest["records"] if r["split"] in ("train", "val")]
    if not args.allow_partial and (manifest["status"] != "completed" and manifest["status"] != "complete"):
        raise RuntimeError(f"Download manifest incomplete: {manifest['status']}")
    output_records = []
    for record in records:
        source = Path(record["path"])
        if not source.is_file():
            if args.allow_partial:
                continue
            raise FileNotFoundError(f"Verified source file is missing: {source}")
        folder = args.root / "cache/volumes" / record["split"] / record["id"]
        meta_path = folder / "metadata.json"
        if meta_path.is_file():
            metadata = read_json(meta_path)
            if metadata["source_sha256"] != record["sha256"]:
                raise RuntimeError(f"Changed source for cache {source}")
            if not all(Path(metadata[key]).is_file() for key in ("image", "target", "labels")):
                raise RuntimeError(f"Incomplete existing volume cache: {folder}")
            output_records.append(metadata)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        crop = slice(*record.get("crop_z", [None, None]))
        print(f"Preparing {record['split']} {record['id']}", flush=True)
        with h5py.File(source, "r") as stream:
            raw = stream["raw"][crop]
            key = "label_with_ignore" if "label_with_ignore" in stream else "label"
            labels = stream[key][crop].astype(np.int32)
            if raw.shape != labels.shape:
                raise ValueError(f"Shape mismatch after crop: {source}")
        # Per-volume, label-free normalization. Subsample deterministic strided
        # raw values only to estimate percentiles and save exact values.
        sample = raw.ravel()[::max(1, raw.size // 2_000_000)]
        low, high = np.percentile(sample, (0.5, 99.8)).astype(float)
        image = np.clip((raw.astype(np.float32) - low) / max(high - low, 1e-6), 0, 1).astype(np.float16)
        target = topology_target(labels)
        for name, values in (("image", image), ("target", target), ("labels", labels)):
            np.save(folder / f"{name}.npy", values)
        metadata = {
            "id": record["id"], "split": record["split"], "group_id": record.get("group_id", record["id"]),
            "source": str(source.resolve()), "source_sha256": record["sha256"], "crop_z": record.get("crop_z"),
            "shape": list(raw.shape), "label_key": key, "normalization": {"p0.5": low, "p99.8": high},
            "spacing_zyx_um": record.get("spacing_zyx_um"), "spacing_status": record.get("spacing_status"),
            "image": str((folder / "image.npy").resolve()), "target": str((folder / "target.npy").resolve()),
            "labels": str((folder / "labels.npy").resolve()),
            "target_class_counts": np.bincount(target.ravel(), minlength=256).tolist(),
            "target_rule": "6-connected thick instance boundary; one voxel ignore halo; 255 ignore",
            "prepared_utc": utc_now(),
        }
        write_json(meta_path, metadata)
        output_records.append(metadata)
        del raw, labels, target, image
    train_groups = {r["group_id"] for r in output_records if r["split"] == "train"}
    val_groups = {r["group_id"] for r in output_records if r["split"] == "val"}
    if train_groups & val_groups:
        raise RuntimeError(f"Train/val specimen overlap: {train_groups & val_groups}")
    if not args.allow_partial:
        expected = {split: sum(r["split"] == split for r in records) for split in ("train", "val")}
        observed = {split: sum(r["split"] == split for r in output_records) for split in ("train", "val")}
        if observed != expected or observed != {"train": 22, "val": 2}:
            raise RuntimeError(f"Incomplete full-data cache: expected22/2, observed {observed}")
    write_json(args.root / "cache/prepared_manifest.json", {
        "status": "partial_smoke_only" if args.allow_partial else "prepared",
        "created_utc": utc_now(), "source_manifest_sha256": sha256(args.root / "provenance/manifest.json"),
        "records": output_records,
    })
    print({split: sum(r["split"] == split for r in output_records) for split in ("train", "val")}, flush=True)


if __name__ == "__main__":
    main()
