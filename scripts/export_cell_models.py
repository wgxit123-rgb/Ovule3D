"""Export real paired validation-cell meshes and descriptive morphometry errors.

Only existing validation predictions are accepted. This never predicts from GT,
does not read test arrays, and does not infer physical calibration. All eligible
match counts and exclusions are recorded before a seeded, capped uniform sample.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from scipy import ndimage as ndi

from ovule3d.geometry import boundary_ids, export_mesh, mesh_from_instance
from ovule3d.io import read_json, sha256, utc_now, write_json


def relative_error(value, reference):
    return abs(value - reference) / abs(reference) if reference != 0 else None


def process_volume(result, prepared, output, args):
    meta_path = args.evaluation.parent / result["metrics_json"]
    meta = read_json(meta_path)
    volume_id = meta["id"]
    record = prepared[volume_id]
    prediction = Path(result["output_h5"])
    if sha256(prediction) != result["output_h5_sha256"]:
        raise ValueError("Prediction H5 changed since validation")
    if meta["prediction_conditioned_on_gt"] is not False:
        raise ValueError("GT-conditioned predictions are not accepted")
    if meta["spacing_unit"] == "um":
        if record.get("spacing_zyx_um") is None:
            raise ValueError("No independently recorded calibration in prepared manifest; physical export refused")
        spacing = tuple(record["spacing_zyx_um"])
        if not np.allclose(spacing, meta["spacing_zyx"]):
            raise ValueError("Prediction and manifest calibrations disagree")
        unit = "um"
    else:
        spacing, unit = (1, 1, 1), "voxel"
    bounds = tuple(slice(a, b) for a, b in zip(meta["crop_start_zyx"], meta["crop_stop_zyx"]))
    gt = np.asarray(np.load(record["labels"], mmap_mode="r")[bounds])
    with h5py.File(prediction, "r") as stream:
        pred = stream["instances"][...]
    if gt.shape != pred.shape:
        raise ValueError("GT and prediction coordinate frames differ")
    valid = gt >= 0
    gt_cut, pred_cut = boundary_ids(gt, valid), boundary_ids(pred, valid)
    matches = meta["metrics"]["matches"]
    eligible = [pair for pair in matches if pair["gt_id"] not in gt_cut and pair["pred_id"] not in pred_cut]
    eligible.sort(key=lambda p: (p["gt_id"], p["pred_id"]))
    selected = eligible
    if len(selected) > args.max_pairs:
        indices = np.random.default_rng(args.seed).choice(len(selected), args.max_pairs, replace=False)
        selected = [selected[i] for i in sorted(indices)]
    # Bounded source IDs prevent find_objects from allocating by malicious IDs.
    if max(int(gt.max()), int(pred.max())) > 2_000_000:
        raise ValueError("Source label ID range exceeds safe geometry bound")
    gt_boxes, pred_boxes = ndi.find_objects(gt), ndi.find_objects(pred)
    folder = output / volume_id
    folder.mkdir(parents=True, exist_ok=False)
    rows, excluded = [], []
    for pair in selected:
        gt_id, pred_id = pair["gt_id"], pair["pred_id"]
        gt_box, pred_box = gt_boxes[gt_id-1], pred_boxes[pred_id-1]
        if gt_box is None or pred_box is None:
            raise ValueError("Recorded matched ID absent from real label data")
        sizes = [int(np.prod([sl.stop-sl.start for sl in box])) for box in (gt_box, pred_box)]
        if max(sizes) > args.max_cell_bbox_voxels:
            excluded.append({**pair, "reason": "bounding box exceeds documented per-cell geometry resource cap"})
            continue
        meshes = {
            "gt": mesh_from_instance(gt, gt_id, spacing_zyx=spacing, origin_zyx=meta["crop_start_zyx"], spatial_unit=unit, bounds=gt_box),
            "pred": mesh_from_instance(pred, pred_id, spacing_zyx=spacing, origin_zyx=meta["crop_start_zyx"], spatial_unit=unit, bounds=pred_box)}
        if not all(m.measures["watertight"] and m.measures["six_connected_components"] == 1 for m in meshes.values()):
            excluded.append({**pair, "reason": "non-watertight or disconnected instance; not reported as one intact cell",
                             "diagnostics": {k: m.measures for k, m in meshes.items()}})
            continue
        files = []
        for label, mesh in meshes.items():
            base = folder / f"gt{gt_id}_pred{pred_id}_{label}"
            for fmt in args.formats:
                path = export_mesh(mesh, base.with_suffix(f".{fmt}"))
                files.append({"path": str(path), "sha256": sha256(path), "format": fmt,
                              "role": label, "spatial_unit": unit})
            write_json(base.with_suffix(".json"), mesh.measures)
        a, b = meshes["gt"].measures, meshes["pred"].measures
        row = {"volume_id": volume_id, "gt_id": gt_id, "pred_id": pred_id, "matched_iou": pair["iou"],
               "spatial_unit": unit, "gt_voxel_volume": a["voxel_volume"], "pred_voxel_volume": b["voxel_volume"],
               "volume_absolute_relative_error": relative_error(b["voxel_volume"], a["voxel_volume"]),
               "gt_mesh_surface_area": a["mesh_surface_area"], "pred_mesh_surface_area": b["mesh_surface_area"],
               "surface_absolute_relative_error": relative_error(b["mesh_surface_area"], a["mesh_surface_area"]),
               "gt_mesh_sphericity": a["mesh_sphericity"], "pred_mesh_sphericity": b["mesh_sphericity"],
               "sphericity_absolute_error": abs(b["mesh_sphericity"]-a["mesh_sphericity"])}
        for axis in range(3):
            row[f"axis{axis+1}_absolute_relative_error"] = relative_error(
                b["equivalent_ellipsoid_full_axes"][axis], a["equivalent_ellipsoid_full_axes"][axis])
        rows.append(row)
        write_json(folder / f"gt{gt_id}_pred{pred_id}_pair.json", {"pair": row, "files": files,
                   "GT_used_only_for_pairing_and_error_measurement": True})
    count = {"volume_id": volume_id, "scope": meta["scope"], "shape_zyx": list(pred.shape), "spatial_unit": unit,
             "source_prediction_h5": str(prediction), "source_prediction_sha256": result["output_h5_sha256"],
             "gt_count": meta["metrics"]["gt_count"], "pred_count": meta["metrics"]["pred_count"],
             "matched_pairs": len(matches), "intact_eligible_matched_pairs": len(eligible),
             "seeded_sample_pairs": len(selected), "successfully_measured_pairs": len(rows),
             "geometry_exclusions": excluded, "unmatched_gt_count": meta["metrics"]["false_negatives"],
             "unmatched_pred_count": meta["metrics"]["false_positives"],
             "sampling_seed": args.seed, "cap": args.max_pairs,
             "interpretation": "conditional errors on selected intact matched cells; not errors of every cell or biological inference"}
    write_json(folder / "coverage.json", count)
    return rows, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--max-cell-bbox-voxels", type=int, default=5_000_000)
    parser.add_argument("--formats", nargs="+", choices=("ply", "obj", "stl"), default=["ply", "obj", "stl"])
    args = parser.parse_args()
    if args.max_pairs < 1 or args.max_cell_bbox_voxels < 1 or len(args.formats) != len(set(args.formats)):
        parser.error("Positive geometry limits and unique export formats required")
    evidence = read_json(args.evaluation)
    if evidence["split"] != "val" or evidence["prediction_conditioned_on_gt"] is not False:
        raise ValueError("Only real annotation-independent validation predictions are allowed")
    manifest = read_json(args.root / "cache/prepared_manifest.json")
    prepared = {r["id"]: r for r in manifest["records"] if r["split"] == "val"}
    if not set(evidence["volume_ids"]).issubset(prepared):
        raise ValueError("Evaluation IDs do not belong to prepared validation data")
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "created_utc": utc_now(), "evaluation_sha256": sha256(args.evaluation),
             "source_checkpoint_sha256": evidence["checkpoint_sha256"], "test_labels_read": False,
             "prediction_changed": False, "geometry_source_sha256": sha256(args.root / "ovule3d/geometry.py"),
             "script_sha256": sha256(__file__), "coverage": []}
    write_json(args.output / "manifest.json", state)
    try:
        rows = []
        for result in evidence["results"]:
            records, coverage = process_volume(result, prepared, args.output, args)
            rows.extend(records)
            state["coverage"].append(coverage)
            write_json(args.output / "manifest.json", state)
            print({"event": "exported", "volume": coverage["volume_id"], "pairs": len(records)}, flush=True)
        if rows:
            with (args.output / "paired_morphometry.csv").open("x", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        state.update(status="completed" if rows else "completed_no_eligible_cells", finished_utc=utc_now(),
                     measured_pairs=len(rows), inference_scope="descriptive matched-cell geometry; not a test or calibrated phenotype claim")
        write_json(args.output / "manifest.json", state)
    except Exception as error:
        state.update(status="failed", error=repr(error), finished_utc=utc_now())
        write_json(args.output / "manifest.json", state)
        raise


if __name__ == "__main__":
    main()
