"""GT-independent global instance-size filtering; units are explicit voxels."""
import numpy as np


def filter_small_instances(labels, minimum_voxels):
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("Nonnegative 3D integer predictions required")
    if minimum_voxels < 0 or int(minimum_voxels) != minimum_voxels:
        raise ValueError("Nonnegative integer voxel-size threshold required")
    if int(labels.max()) > 2_000_000:
        raise ValueError("Prediction label ID range exceeds safe size-filter bound")
    counts = np.bincount(labels.ravel())
    keep = np.flatnonzero((counts >= minimum_voxels) & (counts > 0))
    keep = keep[keep != 0]
    mapping = np.zeros(len(counts), dtype=np.int32)
    mapping[keep] = np.arange(1, len(keep)+1, dtype=np.int32)
    return mapping[labels], {"minimum_voxels": int(minimum_voxels), "original_instances": int(np.count_nonzero(counts[1:])),
                             "retained_instances": len(keep), "removed_instances": int(np.count_nonzero(counts[1:])-len(keep)),
                             "size_domain": "entire predicted volume before any annotation/ROI exclusion",
                             "ground_truth_used": False}
