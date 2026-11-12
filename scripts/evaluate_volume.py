"""Sliding-window 3D validation; held-out test access requires a frozen record.

Only complete fused topology probability volumes are converted to instances.
Annotation masks never affect predictions, foreground, seeds, or watershed.
Development centre crops are explicit and cannot be reported as whole volumes.
"""
from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Callable

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ovule3d.data import feature_crop
from ovule3d.io import read_json, sha256, utc_now, write_json
from ovule3d.metrics import summarize_instances
from ovule3d.models import build_model
from ovule3d.postprocess import probabilities_to_instances


def axis_starts(length: int, patch: int) -> list[int]:
    """Half-patch strides plus the last aligned window cover every voxel."""
    if length < 1 or patch < 1:
        raise ValueError("Spatial dimensions must be positive")
    last = max(0, length - patch)
    return sorted(set([*range(0, last + 1, max(1, patch // 2)), last]))


def gaussian_patch_weights(patch: tuple[int, int, int]) -> np.ndarray:
    axes = [np.exp(-0.5 * (np.linspace(-1, 1, size, dtype=np.float32) / 0.5) ** 2)
            if size > 1 else np.ones(1, dtype=np.float32) for size in patch]
    weights = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return np.maximum(weights / weights.max(), 1e-3).astype(np.float32)


def centre_crop_bounds(shape: tuple[int, int, int], max_shape: tuple[int, int, int] | None):
    selected = shape if max_shape is None else tuple(min(n, p) for n, p in zip(shape, max_shape))
    origin = tuple((n - p) // 2 for n, p in zip(shape, selected))
    return origin, tuple(start + length for start, length in zip(origin, selected))


def sliding_window_predict(model: torch.nn.Module, image: np.ndarray, patch: tuple[int, int, int],
                           device: torch.device, *, features: np.ndarray | None = None,
                           feature_full_shape: tuple[int, int, int] | None = None,
                           feature_origin: tuple[int, int, int] = (0, 0, 0),
                           amp: bool = False, work_dir: Path | None = None,
                           progress: Callable[[int, int], None] | None = None):
    """Fuse probabilities with 50% overlap, cropping padding only at image faces.

    ``image`` is already the selected full-volume/crop array. Cached features
    must describe precisely that same spatial frame in [Z,C,h,w] order. XY
    sampling is delegated to the training dataset's ``feature_crop`` helper.
    No ground truth is accepted by this function. Scratch accumulation may be
    memory-mapped; the caller owns its lifetime until the returned array closes.
    """
    if image.ndim != 3 or min(image.shape) < 1 or len(patch) != 3 or min(patch) < 1:
        raise ValueError("Nonempty 3D image and three positive patch dimensions required")
    shape = tuple(int(size) for size in image.shape)
    feature_frame = tuple(feature_full_shape or shape)
    if features is not None and (features.ndim != 4 or features.shape[0] != feature_frame[0]):
        raise ValueError("features must be [Z,C,h,w] in the declared feature coordinate frame")
    if any(a < 0 or a + n > f for a, n, f in zip(feature_origin, shape, feature_frame)):
        raise ValueError("Evaluated crop lies outside declared feature coordinate frame")
    needs_features = getattr(model, "requires_features", False)
    if needs_features and features is None:
        raise ValueError("Frozen SAM features are required by this model")
    starts = [axis_starts(length, size) for length, size in zip(shape, patch)]
    total = int(np.prod([len(axis) for axis in starts]))
    if work_dir is None:
        probabilities = np.zeros((3, *shape), dtype=np.float32)
        weight_sum = np.zeros(shape, dtype=np.float32)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
        probabilities = np.lib.format.open_memmap(work_dir / "probability_sum.npy", mode="w+",
                                                 dtype=np.float32, shape=(3, *shape))
        weight_sum = np.lib.format.open_memmap(work_dir / "weight_sum.npy", mode="w+",
                                             dtype=np.float32, shape=shape)
        probabilities[:] = 0
        weight_sum[:] = 0
    importance = gaussian_patch_weights(patch)
    model.eval()
    with torch.inference_mode():
        for index, start in enumerate(itertools.product(*starts), 1):
            stops = tuple(min(a + size, length) for a, size, length in zip(start, patch, shape))
            slices = tuple(slice(a, b) for a, b in zip(start, stops))
            actual = tuple(b - a for a, b in zip(start, stops))
            raw_patch = np.asarray(image[slices], dtype=np.float32).copy()
            padding = [(0, size - length) for size, length in zip(patch, actual)]
            if any(right for _, right in padding):
                raw_patch = np.pad(raw_patch, padding, mode="edge")
            tensor = torch.from_numpy(raw_patch[None, None]).to(device)
            feature_tensor = None
            if needs_features:
                global_start = tuple(a + b for a, b in zip(start, feature_origin))
                feature_tensor = feature_crop(features, feature_frame, global_start, patch)[None].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(tensor, features=feature_tensor)
            if tuple(logits.shape) != (1, 3, *patch):
                raise RuntimeError(f"Model returned {tuple(logits.shape)}, expected {(1, 3, *patch)}")
            prediction = logits.float().softmax(1)[0].cpu().numpy()
            if not np.isfinite(prediction).all():
                raise FloatingPointError("Non-finite probability patch")
            patch_slices = tuple(slice(0, size) for size in actual)
            weights = importance[patch_slices]
            probabilities[(slice(None), *slices)] += prediction[(slice(None), *patch_slices)] * weights[None]
            weight_sum[slices] += weights
            if progress and (index == 1 or index % 100 == 0 or index == total):
                progress(index, total)
    minimum_weight = float(weight_sum.min())
    if minimum_weight <= 0:
        raise AssertionError("Sliding windows left uncovered voxels")
    for z0 in range(0, shape[0], 16):
        probabilities[:, z0:z0 + 16] /= weight_sum[None, z0:z0 + 16]
    if isinstance(probabilities, np.memmap):
        probabilities.flush()
        weight_sum.flush()
    diagnostics = {"patch_zyx": list(patch), "overlap": 0.5,
                   "stride_zyx": [max(1, size // 2) for size in patch],
                   "weighting": "positive Gaussian, sigma=0.5 in [-1,1], minimum weight=0.001",
                   "window_count": total, "minimum_coverage_weight": minimum_weight,
                   "all_voxels_covered": True, "patch_boundary_policy": "blend probabilities; no per-patch walls or instances",
                   "padding": "replicate only where image/crop is smaller than a patch"}
    del weight_sum
    return probabilities, diagnostics


def code_hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256(path)
            for folder in ("scripts", "ovule3d") for path in sorted((root / folder).glob("*.py"))}


def inference_settings(args, patch):
    return {"patch": list(patch), "overlap": 0.5, "weighting": "gaussian",
            "amp": not args.no_amp and str(args.device).startswith("cuda"),
            "postprocess": {"units": args.postprocess_unit,
                            "foreground_threshold": args.foreground_threshold,
                            "interior_seed_threshold": args.interior_seed_threshold,
                            "wall_seed_threshold": args.wall_seed_threshold,
                            "min_seed_distance": args.min_seed_distance,
                            "seed_smoothing": args.seed_smoothing,
                            "min_cell_volume": args.min_cell_volume,
                            "wall_weight": args.wall_weight, "connectivity": 1},
            "metric_iou_threshold": args.iou_threshold, "overlap_threshold": args.overlap_threshold,
            "measure_morphology": args.measure_morphology, "morphology_max_instances": args.morphology_max_instances,
            "morphology_seed": args.morphology_seed,
            "spacing_override": list(args.spacing_zyx) if args.spacing_zyx else None,
            "spacing_override_unit": args.spacing_unit}


def validate_test_freeze(freeze: dict | None, *, checkpoint_hash: str, current_code_hashes: dict[str, str],
                         manifest_hash: str, records: list[dict], settings: dict,
                         max_shape=None, limit_volumes=None) -> None:
    """Reject unfrozen settings before any held-out annotation array is opened."""
    if freeze is None or freeze.get("status") != "frozen_for_test":
        raise ValueError("Test evaluation requires --freeze-record with status=frozen_for_test")
    if max_shape is not None or limit_volumes is not None:
        raise ValueError("Frozen test evaluation does not permit development crops or volume limits")
    if freeze.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Test freeze checkpoint SHA-256 mismatch")
    recorded_code = {key.replace("\\", "/"): value for key, value in freeze.get("code_sha256", {}).items()}
    if recorded_code != current_code_hashes:
        raise ValueError("Test freeze current code SHA-256 inventory mismatch")
    if freeze.get("prepared_manifest_sha256") != manifest_hash:
        raise ValueError("Test freeze prepared manifest SHA-256 mismatch")
    selector = freeze.get("test_selector", {})
    expected_ids = sorted(record["id"] for record in records)
    if selector.get("split") != "test" or selector.get("volume_ids") != expected_ids or not expected_ids:
        raise ValueError("Test selector must fix exactly the sorted nonempty test volume ID list")
    if freeze.get("inference") != settings:
        raise ValueError("Test freeze inference/postprocessing/metric settings mismatch")


def voxel_units(value):
    """Remove micrometre labels when source calibration is unavailable."""
    if isinstance(value, dict):
        return {key.replace("_um3", "_vox3").replace("_um2", "_vox2").replace("_um", "_vox"):
                voxel_units(item) for key, item in value.items()}
    if isinstance(value, list):
        return [voxel_units(item) for item in value]
    if isinstance(value, str):
        return value.replace("physical", "voxel-grid")
    return value


def resolve_spacing(record: dict, args):
    if args.spacing_zyx is not None:
        spacing, unit = np.asarray(args.spacing_zyx, dtype=float), args.spacing_unit
        source = "explicit command-line override; must be supported by recorded calibration evidence"
    elif record.get("spacing_zyx_um") is not None:
        spacing, unit = np.asarray(record["spacing_zyx_um"], dtype=float), "um"
        source = record.get("spacing_status", "prepared manifest calibration")
    else:
        spacing, unit = np.ones(3), "voxel"
        source = "physical voxel spacing unconfirmed; no micrometre morphology claims permitted"
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Spacing must contain three positive finite Z,Y,X values")
    return spacing, unit, source


def write_prediction_h5(output_path, probabilities, instances, labels, metadata, *, retain_probabilities=True):
    """Storage choice only; instance prediction and metric computation precede it."""
    shape = instances.shape
    chunks = (1, min(16, shape[0]), min(128, shape[1]), min(128, shape[2]))
    with h5py.File(output_path, "w") as stream:
        if retain_probabilities:
            stream.create_dataset("probabilities", data=probabilities, dtype="float32", chunks=chunks,
                                  compression="gzip", compression_opts=1, shuffle=True)
        stream.create_dataset("instances", data=instances, dtype="int32", chunks=chunks[1:],
                              compression="gzip", compression_opts=1, shuffle=True)
        stream.create_dataset("metric_valid_mask", data=(labels >= 0), chunks=chunks[1:],
                              compression="gzip", compression_opts=1)
        stream.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        stream.attrs["probability_channel_order"] = "background,interior,wall"
        stream.attrs["probabilities_retained"] = retain_probabilities
        stream.attrs["prediction_conditioned_on_gt"] = False
        stream.attrs["spacing_unit"] = metadata["spacing_unit"]
        stream.attrs["spacing_zyx"] = metadata["spacing_zyx"]


def evaluate_record(model, record, args, patch, output: Path, settings: dict):
    started = time.monotonic()
    image = np.load(record["image"], mmap_mode="r")
    full_shape = tuple(image.shape)
    origin, stop = centre_crop_bounds(full_shape, tuple(args.max_shape) if args.max_shape else None)
    slices = tuple(slice(a, b) for a, b in zip(origin, stop))
    evaluated_image = image[slices]
    crop_shape = tuple(evaluated_image.shape)
    features = None
    feature_frame, feature_origin = full_shape, origin
    if getattr(model, "requires_features", False):
        if not record.get("features"):
            raise FileNotFoundError(f"No frozen feature cache for {record['id']}")
        features = np.load(record["features"], mmap_mode="r")
        if tuple(features.shape[:1]) != tuple(full_shape[:1]):
            raise ValueError("Cached feature Z differs from raw image Z")
        if (origin != (0, 0, 0) or crop_shape != full_shape) and any(a < b for a, b in zip(crop_shape, patch)):
            # Remap to one explicit crop grid first; subsequent feature_crop
            # calls then use the same crop coordinate frame and edge padding.
            features = feature_crop(features, full_shape, origin, crop_shape).permute(1, 0, 2, 3).numpy()
            feature_frame, feature_origin = crop_shape, (0, 0, 0)
    spacing, spacing_unit, spacing_source = resolve_spacing(record, args)
    if args.postprocess_unit == "um" and spacing_unit != "um":
        raise ValueError("Physical postprocessing requires confirmed micrometre spacing")
    post_spacing = spacing if args.postprocess_unit == "um" else np.ones(3)
    safe_id = record["id"].replace("/", "_").replace("\\", "_")
    with tempfile.TemporaryDirectory(prefix=f"{safe_id}_", dir=output) as temporary:
        probabilities, window_info = sliding_window_predict(
            model, evaluated_image, patch, torch.device(args.device), features=features,
            feature_full_shape=feature_frame, feature_origin=feature_origin,
            amp=settings["amp"], work_dir=Path(temporary),
            progress=lambda current, total: print(json.dumps({"event": "inference", "id": record["id"],
                                                               "window": current, "windows": total}), flush=True))
        print(json.dumps({"event": "stage", "id": record["id"], "stage": "instance_reconstruction",
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
        # GT is deliberately not loaded and no GT-derived ROI is passed here.
        instances, reconstruction = probabilities_to_instances(
            probabilities, post_spacing, foreground_threshold=args.foreground_threshold,
            interior_seed_threshold=args.interior_seed_threshold, wall_seed_threshold=args.wall_seed_threshold,
            min_seed_distance_um=args.min_seed_distance, seed_smoothing_um=args.seed_smoothing,
            min_cell_volume_um3=args.min_cell_volume, wall_weight=args.wall_weight,
            connectivity=1, return_diagnostics=True)
        if args.postprocess_unit == "voxel":
            reconstruction = voxel_units(reconstruction)
        reconstruction["units"] = args.postprocess_unit
        reconstruction["ground_truth_used"] = False
        # Only after prediction/reconstruction are annotations opened for metrics.
        labels_array = np.load(record["labels"], mmap_mode="r")
        if tuple(labels_array.shape) != full_shape:
            raise ValueError("Image and GT label volume shapes differ")
        labels = np.asarray(labels_array[slices])
        print(json.dumps({"event": "stage", "id": record["id"], "stage": "instance_metrics",
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
        metrics = summarize_instances(instances, labels, spacing,
                                      iou_threshold=args.iou_threshold, overlap_threshold=args.overlap_threshold,
                                      measure_morphology=args.measure_morphology,
                                      morphology_max_instances=args.morphology_max_instances,
                                      morphology_seed=args.morphology_seed, morphology_exclude_boundary=True)
        if spacing_unit == "voxel":
            metrics = voxel_units(metrics)
        metrics["spatial_unit"] = spacing_unit
        scope = "development_centre_crop" if args.max_shape else "complete_prepared_volume"
        metadata = {"id": record["id"], "group_id": record.get("group_id", record["id"]),
                    "split": record["split"], "scope": scope, "full_prepared_shape_zyx": list(full_shape),
                    "evaluated_shape_zyx": list(crop_shape), "crop_start_zyx": list(origin), "crop_stop_zyx": list(stop),
                    "source_z_crop": record.get("crop_z"), "spacing_zyx": spacing.tolist(),
                    "spacing_unit": spacing_unit, "spacing_source": spacing_source,
                    "partial_annotation": metrics["partial_annotation"],
                    "annotation_protocol": metrics["annotation_protocol"],
                    "inference": window_info, "reconstruction": reconstruction,
                    "feature_crop_protocol": "direct global feature coordinates for normal windows; compact crop frame only for development crops smaller than a patch; no Z interpolation",
                    "source_sha256": record.get("source_sha256"), "prediction_conditioned_on_gt": False,
                    "probabilities_retained": not getattr(args, "instances_only", False)}
        output_path = output / f"{safe_id}.h5"
        print(json.dumps({"event": "stage", "id": record["id"], "stage": "write_outputs",
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
        write_prediction_h5(output_path, probabilities, instances, labels, metadata,
                            retain_probabilities=not getattr(args, "instances_only", False))
        result = {**metadata, "metrics": metrics, "output_h5": str(output_path.resolve()),
                  "output_h5_sha256": sha256(output_path), "seconds": time.monotonic() - started}
        write_json(output / f"{safe_id}.json", result)
        del probabilities
        gc.collect()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--freeze-record", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--instances-only", action="store_true",
                        help="Retain instances/metric mask/metadata, not dense probabilities; checkpoints permit regeneration")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patch", type=int, nargs=3)
    parser.add_argument("--max-shape", type=int, nargs=3, help="Explicit development-only centre crop Z Y X")
    parser.add_argument("--limit-volumes", type=int)
    parser.add_argument("--volume-ids", nargs="+", help="Explicit train/val subset; never allowed for frozen test")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--spacing-zyx", type=float, nargs=3)
    parser.add_argument("--spacing-unit", choices=("um", "voxel"))
    parser.add_argument("--postprocess-unit", choices=("um", "voxel"), default="voxel")
    parser.add_argument("--min-seed-distance", type=float, default=4.0)
    parser.add_argument("--seed-smoothing", type=float, default=1.0)
    parser.add_argument("--min-cell-volume", type=float, default=0.0)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--interior-seed-threshold", type=float, default=0.35)
    parser.add_argument("--wall-seed-threshold", type=float, default=0.5)
    parser.add_argument("--wall-weight", type=float, default=0.7)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--overlap-threshold", type=float, default=0.1)
    parser.add_argument("--measure-morphology", action="store_true")
    parser.add_argument("--morphology-max-instances", type=int, default=200)
    parser.add_argument("--morphology-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if (args.spacing_zyx is None) != (args.spacing_unit is None):
        parser.error("--spacing-zyx and --spacing-unit must be supplied together")
    if args.limit_volumes is not None and args.limit_volumes < 1:
        parser.error("--limit-volumes must be positive")
    if args.max_shape and min(args.max_shape) < 1:
        parser.error("--max-shape must contain positive dimensions")
    if args.split == "test" and args.freeze_record is None:
        parser.error("Test labels remain locked: --freeze-record is required")
    if args.split == "test" and args.volume_ids:
        parser.error("Frozen test does not permit an ad hoc volume subset")
    root, run = args.root.resolve(), args.run.resolve()
    checkpoint_path = run / "best.pt"
    checkpoint_hash = sha256(checkpoint_path)
    # Local checkpoints are created by this project's trainer and contain config.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    patch = tuple(args.patch or config["patch"])
    if min(patch) < 1:
        parser.error("Patch dimensions must be positive")
    manifest_path = args.manifest or root / "cache/prepared_manifest.json"
    manifest = read_json(manifest_path)
    records = sorted((record for record in manifest["records"] if record["split"] == args.split), key=lambda r: r["id"])
    settings = inference_settings(args, patch)
    hashes = code_hashes(root)
    if args.split == "test":
        validate_test_freeze(read_json(args.freeze_record), checkpoint_hash=checkpoint_hash,
                             current_code_hashes=hashes, manifest_hash=sha256(manifest_path),
                             records=records, settings=settings,
                             max_shape=args.max_shape, limit_volumes=args.limit_volumes)
    if args.limit_volumes:
        records = records[:args.limit_volumes]
    if args.volume_ids:
        if len(args.volume_ids) != len(set(args.volume_ids)):
            raise ValueError("Volume IDs must be unique")
        selected = set(args.volume_ids)
        missing = selected - {record["id"] for record in records}
        if missing:
            raise ValueError(f"Requested volumes not in selected split: {sorted(missing)}")
        records = [record for record in records if record["id"] in selected]
    if not records:
        raise ValueError(f"No prepared records for split {args.split}; no alternative split is substituted")
    output = args.output or run / (f"evaluation_{args.split}_devcrop" if args.max_shape else f"evaluation_{args.split}")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation.json").exists():
        raise FileExistsError("Evaluation already recorded; choose a new --output to preserve provenance")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    model = build_model(config["model"], base_channels=config.get("base_channels", 16))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(torch.device(args.device)).eval()
    results = [evaluate_record(model, record, args, patch, output, settings) for record in records]
    rows = []
    for result in results:
        row = {"id": result["id"], "group_id": result["group_id"], "split": result["split"],
               "scope": result["scope"], "spacing_unit": result["spacing_unit"],
               "evaluated_shape_zyx": json.dumps(result["evaluated_shape_zyx"]),
               "crop_start_zyx": json.dumps(result["crop_start_zyx"]),
               "crop_stop_zyx": json.dumps(result["crop_stop_zyx"]), "seconds": result["seconds"]}
        row.update({key: value for key, value in result["metrics"].items()
                    if not isinstance(value, (dict, list))})
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {"status": "completed", "created_utc": utc_now(), "run": str(run),
              "model": config["model"], "checkpoint_sha256": checkpoint_hash,
              "prepared_manifest_sha256": sha256(manifest_path), "code_sha256": hashes,
              "split": args.split, "volume_ids": [result["id"] for result in results],
              "limited_volume_panel": args.limit_volumes is not None or bool(args.volume_ids),
              "max_shape": args.max_shape, "scope": "development_crop_only" if args.max_shape else "complete_prepared_volumes",
              "inference": settings, "prediction_conditioned_on_gt": False,
              "test_freeze_record_sha256": sha256(args.freeze_record) if args.freeze_record else None,
              "prepared_status": manifest.get("status"),
              "results": [{"id": result["id"], "output_h5": result["output_h5"],
                           "output_h5_sha256": result["output_h5_sha256"], "metrics_json": f"{result['id']}.json"}
                          for result in results]}
    write_json(output / "evaluation.json", report)
    print(json.dumps({"status": "completed", "report": str(output / "evaluation.json"), "volumes": len(results)}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    main()
