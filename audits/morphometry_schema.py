"""Bind both existing H5 layouts without modifying any prediction or GT array.

Eight stage2 filtered caches intentionally contain instances only. Their metric
mask must be read from the hash-bound original prediction, never fabricated.
All other cases must carry the mask in their own prediction H5.
"""
from pathlib import Path

import h5py
import numpy as np

from ovule3d.io import read_json, sha256


def _inside(path, parent):
    path = Path(path).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError("Mask provenance outside declared validation runs")
    return path


def check_h5_layout(path, shape, *, require_mask):
    """Read H5 headers only, not dataset arrays."""
    with h5py.File(path, "r") as stream:
        if ("instances" not in stream or list(stream["instances"].shape) != shape
                or not np.issubdtype(stream["instances"].dtype, np.integer)):
            raise ValueError("Invalid prediction instances schema")
        if "prediction_conditioned_on_gt" not in stream.attrs or bool(stream.attrs["prediction_conditioned_on_gt"]):
            raise ValueError("Prediction must explicitly exclude GT conditioning")
        has_mask = "metric_valid_mask" in stream
        if require_mask and not has_mask:
            raise ValueError("Source prediction lacks the original metric valid mask")
        if has_mask and (list(stream["metric_valid_mask"].shape) != shape
                         or stream["metric_valid_mask"].dtype != np.dtype(bool)):
            raise ValueError("Invalid metric mask shape or dtype")
        return has_mask


def bind_metric_mask_sources(root, cases):
    """Return explicit per-case mask sources and extra byte-hash inventory.

Fallback is allowed only for the eight documented validation-derived filtered
artifacts, with matching source checkpoint, run, volume, frame and source hash.
No arbitrary missing-mask case is accepted.
"""
    root = Path(root).resolve()
    inventory, counts = {}, {"embedded_mask": 0, "original_mask_for_filtered_cache": 0}
    for case in cases:
        prediction = _inside(case["prediction_h5"], root / "runs")
        shape = case["labels"]["shape"]
        if check_h5_layout(prediction, shape, require_mask=False):
            source = {"path": str(prediction), "sha256": case["prediction_sha256"],
                      "route": "embedded_mask"}
        else:
            _inside(prediction, root / "runs/full_size_sensitivity_20261019_v1/selected")
            ev, meta = read_json(case["evaluation"]), read_json(case["metrics_json"])
            source_ev = _inside(ev["source_default_evaluation"], root / "runs/full_validation_20261019_v1")
            original = read_json(source_ev)
            if (ev.get("shared_minimum_voxels") != 512
                    or meta.get("global_size_filter", {}).get("minimum_voxels") != 512
                    or meta["global_size_filter"].get("ground_truth_used") is not False
                    or original["status"] != "completed" or original["split"] != "val"
                    or original["scope"] != "complete_prepared_volumes"
                    or original["model"] != case["model"]
                    or Path(original["run"]).resolve() != Path(ev["run"]).resolve()
                    or original["checkpoint_sha256"] != case["checkpoint_sha256"]
                    or original["prediction_conditioned_on_gt"] is not False
                    or original["volume_ids"] != [case["volume_id"]]
                    or len(original["results"]) != 1):
                raise ValueError("Filtered-mask source model/checkpoint/protocol binding differs")
            entry = original["results"][0]
            source_path = _inside(entry["output_h5"], source_ev.parent)
            source_meta_path = _inside(source_ev.parent / entry["metrics_json"], source_ev.parent)
            source_meta = read_json(source_meta_path)
            if (entry["id"] != case["volume_id"]
                    or source_meta["id"] != case["volume_id"]
                    or source_meta["split"] != "val"
                    or source_meta["prediction_conditioned_on_gt"] is not False
                    or source_meta["crop_start_zyx"] != [0, 0, 0]
                    or source_meta["crop_stop_zyx"] != shape
                    or source_meta["evaluated_shape_zyx"] != shape
                    or entry["output_h5_sha256"] != meta["source_default_prediction_sha256"]
                    or sha256(source_path) != entry["output_h5_sha256"]):
                raise ValueError("Original mask source frame/hash binding differs")
            check_h5_layout(source_path, shape, require_mask=True)
            inventory.update({str(p): sha256(p) for p in (source_ev, source_meta_path)})
            source = {"path": str(source_path), "sha256": entry["output_h5_sha256"],
                      "source_evaluation": str(source_ev), "route": "original_mask_for_filtered_cache"}
        case["metric_mask_source"] = source
        inventory[source["path"]] = source["sha256"]
        counts[source["route"]] += 1
    return inventory, counts
