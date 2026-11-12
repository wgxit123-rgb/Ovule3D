"""Validation morphometry primitives; no training, inference, or file I/O.

Keep the reference population separate from the successfully measured subset.
Affine grid scenarios are sensitivity assumptions, never physical calibration.
This module is outside the active training queue's frozen dependencies.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import mesh_surface_area

from ovule3d.geometry import boundary_ids, mesh_from_instance


GRID_SCENARIOS = {
    "voxel_grid": (1.0, 1.0, 1.0),
    "assumed_ds2x_ratio": (0.235 / 0.150, 1.0, 1.0),
    "assumed_ds2x_z_minus20pct": (0.8 * 0.235 / 0.150, 1.0, 1.0),
    "assumed_ds2x_z_plus20pct": (1.2 * 0.235 / 0.150, 1.0, 1.0),
}


def validate_matches(matches, gt_ids, pred_ids, *, threshold=0.5):
    """Require an existing one-to-one assignment; never generate predictions."""
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("IoU threshold must lie in (0,1]")
    by_gt, used_pred = {}, set()
    for pair in matches:
        g, p, iou = int(pair["gt_id"]), int(pair["pred_id"]), float(pair["iou"])
        if g != pair["gt_id"] or p != pair["pred_id"]:
            raise ValueError("Integer matched IDs required")
        if g not in gt_ids or p not in pred_ids:
            raise ValueError("Matched ID absent from annotated evaluation field")
        if g in by_gt or p in used_pred:
            raise ValueError("Matches must be one-to-one")
        if not np.isfinite(iou) or not threshold <= iou <= 1:
            raise ValueError("Invalid matched IoU")
        by_gt[g] = {"gt_id": g, "pred_id": p, "iou": iou}
        used_pred.add(p)
    return by_gt


def _object_state(labels, label_id, box, cut, cap):
    if label_id in cut:
        return "touches_field_or_ignore_boundary"
    if box is None:
        raise ValueError("Positive ID has no bounding box")
    if int(np.prod([sl.stop - sl.start for sl in box])) > cap:
        return "bbox_resource_cap"
    _, count = ndi.label(labels[box] == label_id, ndi.generate_binary_structure(3, 1))
    return "intact" if count == 1 else "six_disconnected"


def population_ledger(gt, pred, matches, *, max_bbox_voxels=5_000_000):
    """Account for every annotated GT ID before any conditional error summary.

    Intact reference targets depend on GT alone: field/ignore contact, resource
    cap, and connectivity. Failed predictions remain in that denominator.
    Prediction IDs/counts use the annotated evaluation field, as metrics do;
    original prediction arrays remain unchanged and full objects define bounds.
    """
    gt, pred = np.asarray(gt), np.asarray(pred)
    if (gt.ndim != 3 or gt.shape != pred.shape
            or not np.issubdtype(gt.dtype, np.integer)
            or not np.issubdtype(pred.dtype, np.integer) or np.any(pred < 0)):
        raise ValueError("Equal 3D integer arrays and nonnegative predictions required")
    if not isinstance(max_bbox_voxels, int) or max_bbox_voxels < 1:
        raise ValueError("Positive integer bounding-box cap required")
    if max(int(gt.max(initial=0)), int(pred.max(initial=0))) > 2_000_000:
        raise ValueError("Label ID exceeds geometry allocation bound")
    valid = gt >= 0
    gt_ids = {int(x) for x in np.unique(gt[valid]) if x > 0}
    pred_ids = {int(x) for x in np.unique(pred[valid]) if x > 0}
    by_gt = validate_matches(matches, gt_ids, pred_ids)
    gt_cut, pred_cut = boundary_ids(gt, valid), boundary_ids(pred, valid)
    gt_boxes, pred_boxes = ndi.find_objects(gt), ndi.find_objects(pred)
    records = []
    for g in sorted(gt_ids):
        gt_state = _object_state(gt, g, gt_boxes[g - 1], gt_cut, max_bbox_voxels)
        row = {"gt_id": g, "gt_state": gt_state, "pred_id": None, "matched_iou": None}
        pair = by_gt.get(g)
        if pair:
            row.update(pred_id=pair["pred_id"], matched_iou=pair["iou"])
        if gt_state != "intact":
            row["status"] = "gt_excluded_" + gt_state
        elif pair is None:
            row["status"] = "unmatched_intact_gt"
        else:
            p = pair["pred_id"]
            pred_state = _object_state(pred, p, pred_boxes[p - 1], pred_cut, max_bbox_voxels)
            row["status"] = "eligible_pair" if pred_state == "intact" else "pred_excluded_" + pred_state
        records.append(row)
    counts = dict(Counter(row["status"] for row in records))
    targets = [r for r in records if r["gt_state"] == "intact"]
    coverage = {"gt_count_in_annotated_field": len(gt_ids),
                "pred_count_in_annotated_field": len(pred_ids),
                "matched_pairs_all_gt": len(by_gt),
                "false_negatives_all_gt": len(gt_ids) - len(by_gt),
                "false_positives_all_pred": len(pred_ids) - len(by_gt),
                "intact_gt_target_count": len(targets),
                "matched_intact_gt_count": sum(r["pred_id"] is not None for r in targets),
                "eligible_pair_count_before_mesh_checks": counts.get("eligible_pair", 0),
                "status_counts": counts,
                "max_bbox_voxels": max_bbox_voxels,
                "prediction_modified": False}
    return records, coverage, gt_boxes, pred_boxes


def grid_measures(binary, *, scales_zyx=GRID_SCENARIOS):
    """Unsmoothed mesh/solid-voxel measures under explicit relative grid scales.

    Only isotropic voxel_grid results use voxel units. Other scenarios have
    arbitrary relative-grid units and cannot be relabelled as micrometres.
    """
    binary = np.asarray(binary)
    if binary.ndim != 3 or not np.isin(binary, [0, 1]).all() or not binary.any():
        raise ValueError("Nonempty 3D binary object required")
    mesh = mesh_from_instance(binary.astype(np.uint8), 1)
    if not mesh.measures["watertight"] or mesh.measures["six_connected_components"] != 1:
        return None
    points = np.argwhere(binary).astype(np.float64)
    centred = points - points.mean(0)
    covariance_zyx = centred.T @ centred / len(points) + np.eye(3) / 12.0
    output = {}
    for name, scale in scales_zyx.items():
        scale = np.asarray(scale, dtype=np.float64)
        if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Finite positive ZYX scale required")
        if name == "voxel_grid" and not np.array_equal(scale, np.ones(3)):
            raise ValueError("voxel_grid must have unit spacing")
        vertices = mesh.vertices_xyz * scale[::-1]
        area = float(mesh_surface_area(vertices, mesh.faces))
        mesh_volume = float(mesh.measures["mesh_volume"] * np.prod(scale))
        eigenvalues = np.linalg.eigvalsh(covariance_zyx * np.outer(scale, scale))[::-1]
        output[name] = {"scales_zyx": scale.tolist(),
                        "length_unit": "voxel" if name == "voxel_grid" else "relative_grid_unit",
                        "physical_calibration_verified": False,
                        "voxel_volume": float(len(points) * np.prod(scale)),
                        "mesh_volume": mesh_volume, "mesh_surface_area": area,
                        "mesh_sphericity": float((36 * np.pi * mesh_volume**2)**(1 / 3) / area),
                        "equivalent_ellipsoid_full_axes": np.sqrt(20 * np.maximum(eigenvalues, 0)).tolist()}
    return output


def descriptor_errors(reference, prediction):
    """Volume uses solid voxels; area/sphericity use the unsmoothed mesh."""
    result = {}
    for key in ("voxel_volume", "mesh_surface_area"):
        denominator = float(reference[key])
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Positive finite reference measure required")
        signed = (float(prediction[key]) - denominator) / denominator
        result[key + "_signed_relative_error"] = signed
        result[key + "_absolute_relative_error"] = abs(signed)
    signed_sphericity = prediction["mesh_sphericity"] - reference["mesh_sphericity"]
    result["mesh_sphericity_signed_error"] = float(signed_sphericity)
    result["mesh_sphericity_absolute_error"] = abs(float(signed_sphericity))
    reference_axes, prediction_axes = (np.asarray(x["equivalent_ellipsoid_full_axes"], dtype=float)
                                      for x in (reference, prediction))
    if reference_axes.shape != (3,) or prediction_axes.shape != (3,):
        raise ValueError("Exactly three equivalent-ellipsoid axes required")
    for k, (ref, value) in enumerate(zip(reference_axes, prediction_axes), 1):
        if ref <= 0 or not np.isfinite(ref) or not np.isfinite(value):
            raise ValueError("Positive finite reference axes and finite predicted axes required")
        signed = (value - ref) / ref
        result[f"axis{k}_signed_relative_error"] = float(signed)
        result[f"axis{k}_absolute_relative_error"] = abs(float(signed))
    if not all(np.isfinite(x) for x in result.values()):
        raise ValueError("Nonfinite descriptor error")
    return result


def describe(values):
    """Cell-level descriptive distribution only; empty values stay undefined."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite values cannot be silently discarded")
    if not len(values):
        return {"count": 0, "mean": None, "q25": None, "median": None, "q75": None, "p90": None}
    return {"count": len(values), "mean": float(values.mean()),
            **dict(zip(("q25", "median", "q75", "p90"),
                       map(float, np.quantile(values, [0.25, 0.5, 0.75, 0.9], method="linear"))))}


def common_measured_ids(measured_ids_by_model):
    """Same-GT intersection for a single volume/seed; retains selection warning."""
    if not measured_ids_by_model:
        return set()
    return set.intersection(*(set(values) for values in measured_ids_by_model.values()))


def measure_population(gt, pred, matches, *, max_bbox_voxels=5_000_000,
                       scales_zyx=GRID_SCENARIOS):
    """Measure every eligible pair; failed detection/geometry retain ledger rows.

    No export sampling/cap on pair count, smoothing, rematching, or GT-based
    modification of predictions. The bounding-box cap is a declared resource
    exclusion. Existing assignment IoUs are checked from eligible pair masks.
    A discrepancy raises rather than silently replacing a source metric.
    """
    ledger, coverage, gt_boxes, pred_boxes = population_ledger(
        gt, pred, matches, max_bbox_voxels=max_bbox_voxels)
    measurements = []
    for row in ledger:
        if row["status"] != "eligible_pair":
            continue
        g, p = row["gt_id"], row["pred_id"]
        gbox, pbox = gt_boxes[g - 1], pred_boxes[p - 1]
        common = tuple(slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(gbox, pbox))
        intersection = (int(np.count_nonzero((gt[common] == g) & (pred[common] == p)))
                        if all(sl.stop > sl.start for sl in common) else 0)
        gmask, pmask = gt[gbox] == g, pred[pbox] == p
        union = int(gmask.sum()) + int(pmask.sum()) - intersection
        recomputed_iou = intersection / union
        if not np.isclose(recomputed_iou, row["matched_iou"], rtol=0, atol=1e-8):
            raise ValueError(f"Recorded IoU disagrees with intact pair masks: gt={g}, pred={p}")
        gm, pm = grid_measures(gmask, scales_zyx=scales_zyx), grid_measures(pmask, scales_zyx=scales_zyx)
        if gm is None or pm is None:
            row["status"] = "mesh_integrity_failed"
            row["failed_role"] = [name for name, value in (("gt", gm), ("pred", pm)) if value is None]
            continue
        row["status"] = "measured_pair"
        row["recomputed_iou"] = recomputed_iou
        for scenario in gm:
            measurements.append({"gt_id": g, "pred_id": p, "matched_iou": recomputed_iou,
                                 "grid_scenario": scenario,
                                 "length_unit": gm[scenario]["length_unit"],
                                 "gt_measures": gm[scenario], "pred_measures": pm[scenario],
                                 "errors": descriptor_errors(gm[scenario], pm[scenario])})
    measured = sum(row["status"] == "measured_pair" for row in ledger)
    denominator = coverage["intact_gt_target_count"]
    coverage.update(status_counts_after_mesh=dict(Counter(r["status"] for r in ledger)),
                    successfully_measured_pairs=measured,
                    measurement_yield_among_intact_gt=measured / denominator if denominator else None,
                    pair_count_cap=None, physical_calibration_verified=False,
                    inference="conditional descriptive geometry; no biological replication or physical calibration")
    return ledger, coverage, measurements
