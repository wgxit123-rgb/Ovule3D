"""Validation-configured 3D topology-to-instance reconstruction.

All arrays and ``spacing`` use Z, Y, X order. Thresholds in this module must be
selected using training/validation volumes only. The defaults are starting
values, not parameters estimated from the Ovules test labels. Watershed uses
the native voxel grid; Euclidean distances, peak suppression and size filters
use physical units. This is reconstruction, not a claim of super-resolution.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def _spacing_zyx(spacing):
    value = np.asarray(spacing, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
        raise ValueError("spacing must contain three positive finite Z,Y,X values")
    return value


def _physical_footprint(radius, spacing):
    extents = np.maximum(1, np.ceil(radius / spacing).astype(int))
    grid = np.ogrid[tuple(slice(-int(r), int(r) + 1) for r in extents)]
    squared_distance = sum((axis * step) ** 2 for axis, step in zip(grid, spacing))
    return squared_distance <= radius**2 + np.finfo(float).eps


def _physical_nonmaximum_suppression(coords, scores, spacing, radius):
    """Greedy deterministic peak selection in physical Euclidean distance."""
    if not len(coords):
        return coords
    # Stable ordering makes equal-score plateaus reproducible.
    order = np.argsort(-scores, kind="stable")
    physical_coords = coords.astype(np.float64) * spacing
    tree = cKDTree(physical_coords)
    suppressed = np.zeros(len(coords), dtype=bool)
    chosen = []
    for index in order:
        if suppressed[index]:
            continue
        chosen.append(index)
        neighbours = tree.query_ball_point(physical_coords[index], radius * (1 - 1e-10))
        suppressed[neighbours] = True
    return coords[np.asarray(chosen, dtype=int)]


def probabilities_to_instances(
    probs,
    spacing,
    *,
    foreground_threshold=0.5,
    interior_seed_threshold=0.35,
    wall_seed_threshold=0.5,
    min_seed_distance_um=0.6,
    seed_smoothing_um=0.1,
    min_cell_volume_um3=0.0,
    wall_weight=0.7,
    connectivity=1,
    valid_mask=None,
    return_diagnostics=False,
):
    """Convert ``[3,Z,Y,X]`` background/interior/wall probabilities to labels.

    ``spacing`` must be in micrometres when using the ``*_um`` parameters.
    Foreground is ``1 - P(bg) >= foreground_threshold``. Seeds are physical
    distance maxima gated by interior/wall probability. Every connected
    foreground component receives a fallback seed if necessary, preventing
    an entire component from being silently lost when the seed gate is empty.
    Small instances are removed only when an explicitly positive physical
    volume threshold is supplied; removed voxels become background.

    Border-cut instances are retained. With a partial-annotation/ROI mask,
    only the supplied region is reconstructed and edge instances are clipped.
    This mask is appropriate for a predeclared inference ROI, not for deriving
    foreground or seeds from held-out ground-truth labels.
    """
    spacing = _spacing_zyx(spacing)
    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim != 4 or probs.shape[0] != 3 or any(s == 0 for s in probs.shape[1:]):
        raise ValueError("probs must have nonempty shape [3,Z,Y,X]")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0) or np.any(probs > 1):
        raise ValueError("probs must be finite probabilities in [0,1], not logits")
    if not np.allclose(probs.sum(axis=0), 1.0, atol=1e-3):
        raise ValueError("background/interior/wall probabilities must sum to one")
    for name, value in (
        ("foreground_threshold", foreground_threshold),
        ("interior_seed_threshold", interior_seed_threshold),
        ("wall_seed_threshold", wall_seed_threshold),
        ("wall_weight", wall_weight),
    ):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    if not np.isfinite(min_seed_distance_um) or min_seed_distance_um <= 0:
        raise ValueError("min_seed_distance_um must be positive")
    if not np.isfinite(seed_smoothing_um) or seed_smoothing_um < 0:
        raise ValueError("seed_smoothing_um must be nonnegative")
    if not np.isfinite(min_cell_volume_um3) or min_cell_volume_um3 < 0:
        raise ValueError("min_cell_volume_um3 must be nonnegative")
    if connectivity not in (1, 2, 3):
        raise ValueError("connectivity must be 1, 2 or 3 in three dimensions")
    valid = np.ones(probs.shape[1:], dtype=bool)
    if valid_mask is not None:
        if np.shape(valid_mask) != valid.shape:
            raise ValueError("valid_mask and probability volume must have equal shapes")
        valid &= np.asarray(valid_mask, dtype=bool)
    foreground = (1 - probs[0] >= foreground_threshold) & valid
    structure = ndi.generate_binary_structure(3, connectivity)
    diagnostics = {
        "spacing_zyx_um": spacing.tolist(),
        "foreground_voxels": int(foreground.sum()),
        "foreground_threshold": float(foreground_threshold),
        "interior_seed_threshold": float(interior_seed_threshold),
        "wall_seed_threshold": float(wall_seed_threshold),
        "min_seed_distance_um": float(min_seed_distance_um),
        "seed_smoothing_um": float(seed_smoothing_um),
        "min_cell_volume_um3": float(min_cell_volume_um3),
        "wall_weight": float(wall_weight),
        "connectivity": connectivity,
        "border_policy": "retain clipped instances; no border-cell removal",
    }
    if not foreground.any():
        labels = np.zeros(foreground.shape, dtype=np.int32)
        diagnostics.update(seeds=0, fallback_seeds=0, removed_small_instances=0, instances=0)
        return (labels, diagnostics) if return_diagnostics else labels

    seed_allowed = foreground & (probs[1] >= interior_seed_threshold) & (probs[2] <= wall_seed_threshold)
    # Distances to the cell-wall-excluded interior, not to the outer tissue
    # foreground: the latter would favour the organ centre over individual
    # cell centres. Explicit zero padding also defines distances at all faces.
    distance = ndi.distance_transform_edt(np.pad(seed_allowed, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    score = distance * probs[1] * (1 - probs[2])
    if seed_smoothing_um > 0:
        score = ndi.gaussian_filter(score, sigma=seed_smoothing_um / spacing)
    score[~seed_allowed] = 0
    coords = peak_local_max(
        score,
        min_distance=1,
        footprint=_physical_footprint(min_seed_distance_um, spacing),
        labels=foreground.astype(np.uint8),
        exclude_border=False,
        threshold_abs=0,
    )
    if len(coords):
        coords = _physical_nonmaximum_suppression(
            coords, score[tuple(coords.T)], spacing, min_seed_distance_um
        )
    markers = np.zeros(foreground.shape, dtype=np.int32)
    if len(coords):
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1, dtype=np.int32)
    components, _ = ndi.label(foreground, structure=structure)
    component_slices = ndi.find_objects(components)
    seeded_components = set(np.unique(components[markers > 0]).tolist())
    seed_count = len(coords)
    fallback_count = 0
    for component_id, slices in enumerate(component_slices, start=1):
        if slices is None or component_id in seeded_components:
            continue
        component = components[slices] == component_id
        # A seed-gate-empty component has no interior distance. Fall back to
        # its own foreground distance, with physical spacing and zero padding.
        component_distance = ndi.distance_transform_edt(np.pad(component, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
        local_distance = np.where(component, component_distance, -1)
        local_coord = np.unravel_index(np.argmax(local_distance), local_distance.shape)
        coord = tuple(int(index + sl.start) for index, sl in zip(local_coord, slices))
        seed_count += 1
        markers[coord] = seed_count
        fallback_count += 1

    elevation = wall_weight * probs[2] + (1 - wall_weight) * (1 - probs[1])
    labels = watershed(elevation, markers=markers, mask=foreground, connectivity=structure)
    counts = np.bincount(labels.ravel())
    small_ids = np.flatnonzero((counts * float(np.prod(spacing)) < min_cell_volume_um3) & (counts > 0))
    small_ids = small_ids[small_ids > 0]
    if len(small_ids):
        labels[np.isin(labels, small_ids)] = 0
    present_ids = np.unique(labels)
    present_ids = present_ids[present_ids > 0]
    remap = np.zeros(seed_count + 1, dtype=np.int32)
    remap[present_ids] = np.arange(1, len(present_ids) + 1, dtype=np.int32)
    labels = remap[labels]
    diagnostics.update(
        seeds=seed_count,
        fallback_seeds=fallback_count,
        removed_small_instances=int(len(small_ids)),
        instances=int(len(present_ids)),
    )
    return (labels, diagnostics) if return_diagnostics else labels
