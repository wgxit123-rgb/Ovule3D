"""Fetch immutable PlantSeg Ovules H5 files, verifying upstream SHA-256.

Use the project Python or the existing cell environment; only stdlib is needed
for download. H5 inspection additionally uses h5py/numpy. Test arrays are never
read by this script: only file bytes and dataset metadata are audited.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "plantseg_ovules"
PROVENANCE = ROOT / "provenance"
SPLITS = {"train": "x9yns", "val": "xp5uf", "test": "8jz7e"}
TEST_SOURCE = Path(r"D:\cell\external_data\plantseg_ovules_test")
CROP_Z = {
    "N_487_ds2x.h5": (17, None),
    "N_535_ds2x.h5": (None, -1),
    "N_534_ds2x.h5": (None, -1),
    "N_451_ds2x.h5": (None, -1),
    "N_425_ds2x.h5": (None, -1),
    "N_420_ds2x.h5": (None, -1),
}


def request_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ovule3d-research/0.1"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def fetch_record(item):
    split, entry = item
    attrs = entry["attributes"]
    name = attrs["name"]
    expected = attrs["extra"]["hashes"]["sha256"]
    destination = TEST_SOURCE / name if split == "test" else DATA / split / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if split == "test" and not destination.is_file():
        raise FileNotFoundError(f"Locked existing test file not found: {destination}")
    if not destination.is_file():
        temporary = destination.with_suffix(f".h5.{os.getpid()}.part")
        for attempt in range(5):
            try:
                # Fresh attempts avoid silently concatenating OSF responses that
                # do not honour Range requests. Final file moves only on SHA pass.
                req = urllib.request.Request(entry["links"]["download"], headers={"User-Agent": "ovule3d-research/0.1"})
                with urllib.request.urlopen(req, timeout=180) as response, temporary.open("wb") as stream:
                    while chunk := response.read(2 * 1024 * 1024):
                        stream.write(chunk)
                actual = sha256(temporary)
                if temporary.stat().st_size != attrs["size"] or actual != expected:
                    raise ValueError(f"Size/SHA256 mismatch for {name}: {actual}")
                temporary.replace(destination)
                break
            except Exception as error:
                print(json.dumps({"event": "retry", "file": name, "attempt": attempt + 1, "error": str(error)}), flush=True)
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 10))
    actual = sha256(destination)
    if actual != expected or destination.stat().st_size != attrs["size"]:
        raise ValueError(f"Existing file does not match OSF: {destination}")
    record = {
        "id": name.removesuffix(".h5"), "split": split,
        "path": str(destination), "size_bytes": attrs["size"],
        "sha256": actual, "upstream_sha256": expected,
        "source_url": entry["links"]["download"],
        "osf_node": SPLITS[split], "osf_version": attrs["extra"]["version"],
        "crop_z": list(CROP_Z.get(name, (None, None))),
        "crop_policy": "torch-em upstream correction recorded, not applied to raw H5",
        "spacing_zyx_um": None,
        "spacing_status": "unconfirmed; do not infer native spacing from ds2x filename",
        "locked_test": split == "test", "label_array_read": False,
        "group_id": name.split("_")[1],
    }
    print(json.dumps({"event": "verified", "split": split, "file": name, "size": attrs["size"]}), flush=True)
    atomic_json(PROVENANCE / "verified_records" / f"{split}_{name}.json", record)
    return record


def inspect_metadata(record):
    import h5py
    import numpy as np
    def convert(value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value
    with h5py.File(record["path"], "r") as volume:
        record["h5_attrs"] = {k: convert(v) for k, v in volume.attrs.items()}
        record["datasets"] = {}
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                record["datasets"][name] = {
                    "shape": list(obj.shape), "dtype": str(obj.dtype),
                    "chunks": list(obj.chunks) if obj.chunks else None,
                    "attrs": {k: convert(v) for k, v in obj.attrs.items()},
                }
        volume.visititems(visitor)
        if record["split"] != "test":
            # A single central slice per training/validation key is a schema
            # investigation, not a full-volume performance evaluation.
            record["label_slice_probe"] = {}
            for key in ("label", "label_with_ignore"):
                if key not in volume:
                    continue
                dataset = volume[key]
                sample = dataset[dataset.shape[0] // 2]
                values, counts = np.unique(sample, return_counts=True)
                record["label_slice_probe"][key] = {
                    "slice_z": dataset.shape[0] // 2,
                    "minimum": int(sample.min()), "maximum": int(sample.max()),
                    "unique_count": int(len(values)),
                    "zero_voxels": int(np.count_nonzero(sample == 0)),
                    "minus_one_voxels": int(np.count_nonzero(sample == -1)),
                    "most_common_ids": [[int(values[i]), int(counts[i])] for i in np.argsort(counts)[-8:][::-1]],
                }
            record["label_array_read"] = True
            record["label_array_read_scope"] = "central training/validation slice only"
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    args = parser.parse_args()
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    entries = []
    metadata = {}
    for split, node in SPLITS.items():
        if args.split is not None and split != args.split:
            continue
        url = f"https://files.de-1.osf.io/v1/resources/{node}/providers/osfstorage/"
        payload = request_json(url)
        metadata[split] = {"node": node, "url": url, "response": payload}
        atomic_json(PROVENANCE / f"osf_{split}_listing.json", payload)
        entries.extend((split, entry) for entry in payload["data"] if entry["attributes"]["kind"] == "file")
    atomic_json(PROVENANCE / ("download_inventory.json" if args.split is None else f"download_inventory_{args.split}.json"), metadata)
    records, errors = [], []
    # Validation comes first so a real smoke test can run while the rest of
    # training downloads. The held-out test is only verified from existing files.
    priority = {"val": 0, "train": 1, "test": 2}
    entries.sort(key=lambda item: (priority[item[0]], item[1]["attributes"]["size"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(fetch_record, item): item for item in entries}
        for future in concurrent.futures.as_completed(future_map):
            split, entry = future_map[future]
            try:
                record = inspect_metadata(future.result())
                records.append(record)
                atomic_json(PROVENANCE / ("manifest.json" if args.split is None else f"manifest_{args.split}.json"), {
                    "schema_version": "1.0", "status": "in_progress", "records": sorted(records, key=lambda r: (r["split"], r["id"])),
                    "errors": errors,
                })
            except Exception as error:
                issue = {"split": split, "file": entry["attributes"]["name"], "error": repr(error)}
                errors.append(issue)
                print(json.dumps({"event": "error", **issue}), flush=True)
    manifest = {"schema_version": "1.0", "status": "complete" if not errors and len(records) == len(entries) else "incomplete", "expected_files": len(entries), "records": sorted(records, key=lambda r: (r["split"], r["id"])), "errors": errors}
    atomic_json(PROVENANCE / ("manifest.json" if args.split is None else f"manifest_{args.split}.json"), manifest)
    print(json.dumps({"event": "finished", "status": manifest["status"], "verified_files": len(records), "errors": errors}), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
