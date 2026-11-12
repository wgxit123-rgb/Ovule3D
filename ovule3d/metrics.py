"""Sparse 3D instance metrics with explicit background/annotation handling.

GT labels below zero and ``valid_mask=False`` are ignored; GT zero is true
background. All remaining voxels, including true background, enter ARE and VI.
Objects intersecting the annotated region are evaluated on their visible
parts. Predictions entirely outside it are not false positives. Cut objects
are retained in segmentation metrics and counted explicitly; by default they
are excluded only from optional morphology comparisons.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_array, csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from skimage.measure import marching_cubes, mesh_surface_area
from skimage.metrics import adapted_rand_error, variation_of_information


def _validate_spacing(spacing):
    spacing = np.asarray(spacing, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing must contain three positive finite Z,Y,X values")
    return spacing


def _integer_labels(array, name, nonnegative=False):
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a three-dimensional label volume")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError(f"{name} must contain finite integer labels")
        array = array.astype(np.int64)
    if nonnegative and np.any(array < 0):
        raise ValueError(f"{name} labels must be nonnegative")
    return array


def _edge_labels(labels, valid):
    """IDs touching a volume face or a face-adjacent unannotated voxel."""
    edge = valid & ~ndi.binary_erosion(valid, structure=ndi.generate_binary_structure(3, 1), border_value=0)
    ids = np.unique(labels[edge])
    return set(int(x) for x in ids if x > 0)


def _sparse_contingency(pred, gt, valid):
    # Compress IDs in bounded voxel blocks: neither original large IDs nor a
    # whole-volume per-voxel COO allocation may determine memory consumption.
    gt_flat, pred_flat, valid_flat = gt.ravel(), pred.ravel(), valid.ravel()
    block_size = 2_000_000
    gt_ids = np.array([0], dtype=gt.dtype)
    pred_ids = np.array([0], dtype=pred.dtype)
    for start in range(0, gt.size, block_size):
        stop = min(start + block_size, gt.size)
        mask = valid_flat[start:stop]
        gt_ids = np.union1d(gt_ids, np.unique(gt_flat[start:stop][mask]))
        pred_ids = np.union1d(pred_ids, np.unique(pred_flat[start:stop][mask]))
    if len(gt_ids) * len(pred_ids) > np.iinfo(np.int64).max:
        raise ValueError("too many unique labels to encode sparse overlap pairs")
    all_rows, all_columns, all_counts = [], [], []
    for start in range(0, gt.size, block_size):
        stop = min(start + block_size, gt.size)
        mask = valid_flat[start:stop]
        if not mask.any():
            continue
        gt_index = np.searchsorted(gt_ids, gt_flat[start:stop][mask])
        pred_index = np.searchsorted(pred_ids, pred_flat[start:stop][mask])
        pair_ids, counts = np.unique(gt_index * len(pred_ids) + pred_index, return_counts=True)
        all_rows.append(pair_ids // len(pred_ids))
        all_columns.append(pair_ids % len(pred_ids))
        all_counts.append(counts.astype(np.float64))
    rows = np.concatenate(all_rows) if all_rows else np.array([], dtype=np.int64)
    columns = np.concatenate(all_columns) if all_columns else np.array([], dtype=np.int64)
    counts = np.concatenate(all_counts) if all_counts else np.array([], dtype=np.float64)
    table = coo_array((counts, (rows, columns)), shape=(len(gt_ids), len(pred_ids))).tocsr()
    table.sum_duplicates()
    return table, gt_ids, pred_ids


def _finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def _label_morphology(labels, label_id, spacing, valid, compute_surface):
    coordinates = np.argwhere((labels == label_id) & valid)
    if not len(coordinates):
        return None
    voxel_volume = float(np.prod(spacing))
    result = {"voxel_count": int(len(coordinates)), "volume_um3": float(len(coordinates) * voxel_volume)}
    if not compute_surface:
        return result
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0) + 1
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    binary = ((labels[slices] == label_id) & valid[slices]).astype(np.uint8)
    binary = np.pad(binary, 1, mode="constant")
    vertices, faces, _, _ = marching_cubes(binary, level=0.5, spacing=tuple(spacing))
    area = float(mesh_surface_area(vertices, faces))
    triangles = vertices[faces]
    mesh_volume = float(abs(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()) / 6)
    result.update(
        surface_area_um2=area,
        mesh_volume_um3=mesh_volume,
        # Use the closed mesh's own volume for dimensionless mesh sphericity;
        # volume-error endpoints still use physical voxel volume above.
        sphericity=float((36 * np.pi * mesh_volume**2) ** (1 / 3) / area) if area else None,
    )
    return result


def measure_instance_morphology(
    labels,
    spacing,
    *,
    instance_ids=None,
    valid_mask=None,
    compute_surface=False,
    max_instances=200,
    random_seed=0,
    exclude_boundary=True,
):
    """Measure selected instances; expensive mesh extraction is opt-in.

    Volume is voxel count times physical voxel volume. Surface area comes from
    an unsmoothed, padded, spacing-aware marching-cubes mesh. Sphericity uses
    that mesh's enclosed volume to avoid mixing voxel and mesh conventions.
    No mesh smoothing/repair is applied. A reproducible random subset is used
    if the predeclared cap is exceeded; the chosen IDs are returned.
    """
    labels = _integer_labels(labels, "labels", nonnegative=True)
    spacing = _validate_spacing(spacing)
    valid = np.ones(labels.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != labels.shape:
        raise ValueError("valid_mask and labels must have equal shapes")
    if max_instances is not None and max_instances < 1:
        raise ValueError("max_instances must be positive or None")
    ids = np.unique(labels[valid]) if instance_ids is None else np.asarray(list(instance_ids))
    ids = sorted(set(int(x) for x in ids if x > 0))
    if exclude_boundary:
        ids = sorted(set(ids) - _edge_labels(labels, valid))
    eligible = len(ids)
    if max_instances is not None and len(ids) > max_instances:
        ids = sorted(np.random.default_rng(random_seed).choice(ids, max_instances, replace=False).tolist())
    results = {}
    for label_id in ids:
        value = _label_morphology(labels, label_id, spacing, valid, compute_surface)
        if value is not None:
            results[label_id] = value
    return {
        "instances": results,
        "eligible_count": eligible,
        "measured_count": len(results),
        "sampling_seed": int(random_seed),
        "max_instances": max_instances,
        "exclude_boundary": bool(exclude_boundary),
        "surface_method": "unsmoothed spacing-aware marching cubes at 0.5" if compute_surface else None,
        "sphericity_volume_convention": "closed marching-cubes mesh volume" if compute_surface else None,
    }


def summarize_instances(
    pred,
    gt,
    spacing,
    *,
    valid_mask=None,
    iou_threshold=0.5,
    overlap_threshold=0.1,
    measure_morphology=False,
    morphology_max_instances=200,
    morphology_seed=0,
    morphology_exclude_boundary=True,
):
    """Evaluate 3D instances using sparse overlaps (never an N-by-M dense IoU).

    SEG follows the CTC rule: a predicted instance qualifies only when its
    intersection covers strictly more than 50% of a GT object. Its IoU is
    scored; unmatched GT objects contribute zero to the full-GT denominator.
    SEG is undefined (None) when there are no visible GT objects. Object F1
    uses a maximum-cardinality one-to-one matching with IoU >= threshold;
    unmatched predictions count as FP, including predictions in true GT=0.

    A split is a GT object covered by >=2 predicted objects, each with
    intersection/GT-volume > overlap_threshold. A merge is a predicted object
    covered by >=2 GT objects, each with intersection/pred-volume > threshold.
    ARE/VI use all valid voxels including true background. Their invariance to
    label names means they must be read with SEG/F1/foreground Dice.
    """
    pred = _integer_labels(pred, "pred", nonnegative=True)
    gt = _integer_labels(gt, "gt")
    spacing = _validate_spacing(spacing)
    if pred.shape != gt.shape:
        raise ValueError("pred and gt must have equal shapes")
    if not 0 < iou_threshold <= 1 or not 0 <= overlap_threshold < 1:
        raise ValueError("iou_threshold must be in (0,1]; overlap_threshold in [0,1)")
    valid = gt >= 0
    if valid_mask is not None:
        if np.shape(valid_mask) != gt.shape:
            raise ValueError("valid_mask and gt must have equal shapes")
        valid &= np.asarray(valid_mask, dtype=bool)
    table, gt_ids, pred_ids = _sparse_contingency(pred, gt, valid)
    gt_sizes = np.asarray(table.sum(axis=1)).ravel()
    pred_sizes = np.asarray(table.sum(axis=0)).ravel()
    n_gt, n_pred = len(gt_ids) - 1, len(pred_ids) - 1
    foreground_overlaps = table[1:, 1:].tocoo()
    intersections = foreground_overlaps.data
    gt_index, pred_index = foreground_overlaps.row, foreground_overlaps.col
    union = gt_sizes[gt_index + 1] + pred_sizes[pred_index + 1] - intersections
    ious = intersections / union if len(union) else np.array([], dtype=float)
    cover_gt = intersections / gt_sizes[gt_index + 1] if len(intersections) else np.array([], dtype=float)
    cover_pred = intersections / pred_sizes[pred_index + 1] if len(intersections) else np.array([], dtype=float)
    seg_per_gt = np.zeros(n_gt, dtype=float)
    qualifies = cover_gt > 0.5
    if qualifies.any():
        np.maximum.at(seg_per_gt, gt_index[qualifies], ious[qualifies])
    keep = ious >= iou_threshold
    graph = csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.int8), (gt_index[keep], pred_index[keep])),
        shape=(n_gt, n_pred),
    )
    mapping = maximum_bipartite_matching(graph, perm_type="column") if n_gt and n_pred else np.full(n_gt, -1, dtype=int)
    match_gt = np.flatnonzero(mapping >= 0)
    match_pred = mapping[match_gt]
    tp = int(len(match_gt))
    fp, fn = n_pred - tp, n_gt - tp
    precision = tp / n_pred if n_pred else (1.0 if not n_gt else 0.0)
    recall = tp / n_gt if n_gt else (1.0 if not n_pred else 0.0)
    f1 = 2 * tp / (n_gt + n_pred) if n_gt + n_pred else 1.0
    split_parts = np.bincount(gt_index[cover_gt > overlap_threshold], minlength=n_gt)
    merge_parts = np.bincount(pred_index[cover_pred > overlap_threshold], minlength=n_pred)
    fg_gt = float(gt_sizes[1:].sum())
    fg_pred = float(pred_sizes[1:].sum())
    fg_intersection = float(intersections.sum())
    fg_dice = 2 * fg_intersection / (fg_gt + fg_pred) if fg_gt + fg_pred else 1.0
    are = rand_precision = rand_recall = vi_split = vi_merge = None
    if valid.any():
        # Pass a pre-built float sparse table; no source-label-index allocation.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            are_value, rp, rr = adapted_rand_error(table=table, ignore_labels=())
            # skimage 0.26 still checks image shapes when a table is supplied.
            # The image values are not used because the sparse table is given.
            vi = variation_of_information(gt, pred, table=table / float(valid.sum()))
        are, rand_precision, rand_recall = map(_finite_or_none, (are_value, rp, rr))
        vi_split, vi_merge = map(_finite_or_none, vi)
    gt_boundary_ids = _edge_labels(gt, valid)
    pred_boundary_ids = _edge_labels(pred, valid)
    matches = [
        {"gt_id": int(gt_ids[g + 1]), "pred_id": int(pred_ids[p + 1]),
         "iou": float(table[g + 1, p + 1] / (gt_sizes[g + 1] + pred_sizes[p + 1] - table[g + 1, p + 1]))}
        for g, p in zip(match_gt, match_pred)
    ]
    result = {
        "seg": float(seg_per_gt.mean()) if n_gt else None,
        "object_precision": float(precision) if valid.any() else None,
        "object_recall": float(recall) if valid.any() else None,
        "object_f1": float(f1) if valid.any() else None,
        "iou_threshold": float(iou_threshold),
        "true_positives": tp,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "gt_count": n_gt,
        "pred_count": n_pred,
        "count_abs_error": int(abs(n_pred - n_gt)),
        "foreground_dice": float(fg_dice) if valid.any() else None,
        "adapted_rand_error": are,
        "adapted_rand_precision": rand_precision,
        "adapted_rand_recall": rand_recall,
        "vi_split_bits": vi_split,
        "vi_merge_bits": vi_merge,
        "split_gt_count": int(np.sum(split_parts >= 2)),
        "merge_pred_count": int(np.sum(merge_parts >= 2)),
        "split_excess_parts": int(np.maximum(split_parts - 1, 0).sum()),
        "merge_excess_parts": int(np.maximum(merge_parts - 1, 0).sum()),
        "overlap_threshold": float(overlap_threshold),
        "valid_voxels": int(valid.sum()),
        "ignored_voxels": int(valid.size - valid.sum()),
        "partial_annotation": bool(not valid.all()),
        "gt_boundary_or_roi_cut_count": len(gt_boundary_ids),
        "pred_boundary_or_roi_cut_count": len(pred_boundary_ids),
        "boundary_cell_policy": "retain visible clipped parts for segmentation; morphology exclusion configurable",
        "annotation_protocol": "GT<0 or valid_mask=False ignored; GT=0 is true background; count instances visible in valid region",
        "information_metric_protocol": "ARE/VI include true background in valid region; evaluate jointly with SEG/object F1/foreground Dice",
        "seg_protocol": "CTC strict intersection/GT-size >0.5; IoU averaged over all visible GT objects",
        "matching_protocol": "maximum-cardinality one-to-one sparse matching; IoU>=threshold; ties need not maximize total IoU",
        "split_merge_protocol": "split: >=2 predicted fragments each covering >threshold of GT; merge: >=2 GT each covering >threshold of prediction",
        "spacing_zyx_um": spacing.tolist(),
        "matches": matches,
    }
    if measure_morphology:
        eligible_matches = [m for m in matches if not morphology_exclude_boundary or (m["gt_id"] not in gt_boundary_ids and m["pred_id"] not in pred_boundary_ids)]
        eligible_count = len(eligible_matches)
        if morphology_max_instances is not None and morphology_max_instances < 1:
            raise ValueError("morphology_max_instances must be positive or None")
        if morphology_max_instances is not None and len(eligible_matches) > morphology_max_instances:
            indices = np.random.default_rng(morphology_seed).choice(len(eligible_matches), morphology_max_instances, replace=False)
            eligible_matches = [eligible_matches[i] for i in sorted(indices)]
        rows = []
        for match in eligible_matches:
            gm = _label_morphology(gt, match["gt_id"], spacing, valid, True)
            pm = _label_morphology(pred, match["pred_id"], spacing, valid, True)
            row = {**match, "gt": gm, "pred": pm}
            for key in ("volume_um3", "surface_area_um2", "sphericity"):
                row[f"{key}_abs_error"] = abs(pm[key] - gm[key])
            row["volume_relative_error"] = abs(pm["volume_um3"] - gm["volume_um3"]) / gm["volume_um3"]
            rows.append(row)
        result["morphology"] = {
            "eligible_matched_count": eligible_count,
            "measured_matched_count": len(rows),
            "matching_condition": "conditional on IoU-matched objects; misses and false positives are reported separately",
            "boundary_exclusion": bool(morphology_exclude_boundary),
            "sampling_seed": int(morphology_seed),
            "max_instances": morphology_max_instances,
            "volume_method": "visible voxel count times physical voxel volume",
            "surface_method": "unsmoothed, padded, physical-spacing marching cubes at 0.5",
            "sphericity_volume_convention": "closed marching-cubes mesh volume",
            "instances": rows,
        }
    return result
