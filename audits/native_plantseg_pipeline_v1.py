"""Full official PlantSeg functional pipeline, with durable prediction-first IO.

This module does not launch or authorize a process. Its production caller must be
the fresh retained native child. No legacy project boundary adapter/watershed,
model-zoo download, CPU inference fallback, annotation mask or score tuning.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

JSON = dict[str, Any]
WEIGHTS_SHA = "0f11cbd0608d12ef161892586408c6ec1a074f1f27ea6afef9d1ceece3dd3c1f"
CONFIG_SHA = "224e7844dc6b2c7155be1f091721a92d2eac4e60298705450df454c3b1199f66"
PREFIX = Path("E:/cell/ovule3d_data/plantseg_environment_install_20261029_v1/env_py311")
MODEL = Path("D:/cell/ovule3d/models/plantseg_ds2x")
MAX_VOXELS = 500_000_000


def require(condition: bool, message: str) -> None:
    """Fail closed without implicit retries."""
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    """Read complete file bytes, not a cached metadata digest."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            result.update(block)
    return result.hexdigest()


def put(path: Path, value: JSON) -> None:
    """Exclusive, flushed, fully read-back control publication."""
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    pending = path.with_suffix(path.suffix + ".pending")
    require(not path.exists(), "Never replace a prior artifact")
    with pending.open("xb") as stream:
        require(stream.write(raw) == len(raw), "Complete control write")
        stream.flush()
        os.fsync(stream.fileno())
    require(pending.read_bytes() == raw, "Control full readback")
    os.rename(pending, path)  # Windows refuses an existing target.


def check_protocol(value: JSON) -> None:
    """Validate all fixed scientific and execution choices, no partial defaults."""
    require(value["schema"] == "native_plantseg_fixed_development_reference_v1", "Protocol schema")
    require(value["order"] == ["N_420_ds2x", "N_464_ds2x"], "Fixed volume order")
    require(value["plantseg_version"] == "1.8.1" and value["weights_sha256"] == WEIGHTS_SHA, "Exact native model")
    require(value["upstream_commit"] == "c6838b6506f65cb9754866ff21224b308a49c7d4", "Exact upstream version")
    require(value["preprocessing"] == {"factor": [1.0, 1.0, 1.0], "filter": False, "output_type": "data_uint8"}, "Fixed voxel preprocessing")
    require(value["prediction"] == {"patch": [80, 160, 160], "patch_halo": [2, 4, 4], "stride_ratio": 0.75, "single_batch_mode": True, "device": "cuda:0", "model_update": False}, "Fixed CUDA0 single batch inference")
    require(value["segmentation"] == {"linkage": "average", "beta": 0.6, "ws_2D": True, "ws_threshold": 0.5, "ws_minsize": 50, "ws_sigma": 2.0, "ws_w_sigma": 0.0, "post_minsize": 50, "n_threads": 4}, "Fixed native graph parameters")
    require(value["postprocessing"] == {"factor": [1.0, 1.0, 1.0], "order": 0}, "Fixed identity postprocessing")
    require(value["policy"] == {
        "weights_exposed_to_both_volumes_as_official_validation": True,
        "equal_training_budget_claimed": False, "unseen_generalization_claimed": False,
        "physical_calibration_claimed": False, "GT_foreground_mask": False,
        "GT_open_only_after_prediction_readback_commit": True,
        "frozen_thresholds_no_score_tuning": True,
        "official_functional_API_not_bitwise_CLI_equivalence": True,
        "installed_ELF_0_9_2_backend_not_historical_backend_equivalence": True,
    }, "Complete honest comparison policy")


def check_array(array: Any, shape: tuple[int, int, int], *, integer: bool) -> None:
    """Validate complete finite native output in bounded planes."""
    import numpy as np

    require(type(array) is np.ndarray and array.shape == shape and array.flags.c_contiguous, "Same full native grid")
    require(0 < math.prod(shape) <= MAX_VOXELS, "Declared full-volume bound")
    require(array.dtype.kind in ("iu" if integer else "f"), "Native numeric dtype")
    for plane in array:
        require(bool(np.isfinite(plane).all()), "No nonfinite native output")
        require(int(plane.min()) >= 0 if integer else float(plane.min()) >= 0.0, "No negative native output")
        if not integer:
            require(float(plane.max()) <= 1.0 + 1e-6, "Normalized boundary range")
        else:
            require(int(plane.max()) < 2**32, "Lossless uint32 labels, no uint16 wrapping")


def write_array(path: Path, key: str, array: Any) -> JSON:
    """Exclusive HDF5, fsync after close, file AND decoded-array full readback."""
    import h5py
    import numpy as np

    shape = tuple(int(n) for n in array.shape)
    require(len(shape) == 3 and array.flags.c_contiguous, "Complete ZYX C array")
    digest = hashlib.sha256()
    with h5py.File(path, "x") as handle:
        data = handle.create_dataset(key, shape=shape, dtype=array.dtype, chunks=None)
        for z in range(shape[0]):
            plane = np.asarray(array[z : z + 1])
            data[z : z + 1] = plane
            digest.update(memoryview(plane).cast("B"))
        handle.flush()
    with path.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    file_sha = sha(path)
    actual = hashlib.sha256()
    with h5py.File(path, "r") as handle:
        data = handle[key]
        require(data.shape == shape and data.dtype == array.dtype and not data.is_virtual and data.external is None, "Exact saved array layout")
        for z in range(shape[0]):
            plane = np.asarray(data[z : z + 1])
            actual.update(memoryview(plane).cast("B"))
    require(actual.digest() == digest.digest() and sha(path) == file_sha, "Complete saved prediction readback/SHA")
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": file_sha, "array_sha256": actual.hexdigest(), "shape": list(shape), "dtype": str(array.dtype), "key": key, "voxels_read_back": math.prod(shape)}


def load_raw(record: JSON) -> Any:
    """Only the original raw dataset and fixed prepared crop; never label/target."""
    import h5py
    import numpy as np

    shape = tuple(record["shape"])
    require(len(shape) == 3 and 0 < math.prod(shape) <= MAX_VOXELS, "Declared native image grid")
    crop = record["crop_z"]
    require((record["id"], crop) in (("N_420_ds2x", [None, -1]), ("N_464_ds2x", [None, None])), "Original explicit prepared crop, not GT-derived")
    with h5py.File(record["source"], "r") as handle:
        require(isinstance(handle.get("raw", getlink=True), h5py.HardLink), "Direct original raw dataset")
        data = handle["raw"]
        require(data.ndim == 3 and data.dtype.kind in "uif" and not data.is_virtual and data.external is None, "Original fixed-width raw")
        raw = np.asarray(data[slice(*crop), :, :], dtype=np.float32, order="C")
    require(raw.shape == shape and bool(np.isfinite(raw).all()) and float(raw.max()) > float(raw.min()), "Complete nonconstant raw grid")
    return raw


def origins() -> list[JSON]:
    """Actual normal-import locations, not a complete loaded-image security proof."""
    names = ("plantseg", "plantseg.predictions.functional.predictions", "plantseg.predictions.functional.array_predictor", "plantseg.segmentation.functional.segmentation", "plantseg.dataprocessing.functional.dataprocessing", "elf.segmentation.gasp", "elf.segmentation.watershed", "bioimage_cpp", "nifty", "vigra", "torch", "numpy", "h5py")
    rows = []
    for name in names:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(PREFIX.resolve()), "Actual native origin outside selected prefix: " + name)
        rows.append({"module": name, "path": path.as_posix(), "sha256": sha(path), "bytes": path.stat().st_size})
    return rows


def run_native(record: JSON, directory: Path, protocol: JSON) -> JSON:
    """Full official functional inference→nativeWS/GASP→identity postprocessing.

    Must run only inside the fresh controlled worker. No GT is opened here.
    Stages are intentionally synchronous to preserve the prediction-before-GT
    ordering and include actual IO costs; no async observer is created.
    """
    check_protocol(protocol)
    require(Path(os.sys.executable).resolve() == (PREFIX / "python.exe").resolve(), "Exact isolated PlantSeg interpreter")
    require(sha(MODEL / "model.pytorch") == WEIGHTS_SHA and sha(MODEL / "config_train.yml") == CONFIG_SHA, "Original official weights/config unchanged")
    import numpy as np
    import torch
    from plantseg.dataprocessing.functional.dataprocessing import image_rescale, normalize_01
    from plantseg.predictions.functional.predictions import unet_predictions
    from plantseg.segmentation.functional.segmentation import dt_watershed, gasp

    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One admitted CUDA0, no CPU fallback")
    torch.cuda.set_device(0)
    shape = tuple(record["shape"])
    stages = []
    started = time.monotonic()
    raw = load_raw(record)
    # Equivalent quantization and input min/max conversion to the CLI pipeline;
    # official functional inference applies its own global Standardize transform.
    raw = (normalize_01(raw) * np.iinfo(np.uint8).max).astype(np.uint8)
    raw = normalize_01(raw.astype(np.float32))
    raw = image_rescale(raw, (1.0, 1.0, 1.0), order=2)
    stages.append({"stage": "raw_and_native_preprocessing", "started": started, "finished": time.monotonic()})
    started = time.monotonic()
    boundary = unet_predictions(raw, model_name=None, model_id=None, config_path=MODEL / "config_train.yml", model_weights_path=MODEL / "model.pytorch", patch=(80, 160, 160), patch_halo=(2, 4, 4), single_batch_mode=True, device="cuda:0", model_update=False, disable_tqdm=True)
    torch.cuda.synchronize(0)
    del raw
    gc.collect()
    boundary = np.ascontiguousarray(normalize_01(boundary), dtype=np.float32)
    check_array(boundary, shape, integer=False)
    stages.append({"stage": "official_UNet_and_native_CNN_postprocessing", "started": started, "finished": time.monotonic()})
    started = time.monotonic()
    boundary_pin = write_array(directory / "boundary.h5", "predictions", boundary)
    stages.append({"stage": "full_boundary_persistence_readback", "started": started, "finished": time.monotonic()})
    started = time.monotonic()
    ws = dt_watershed(boundary, threshold=0.5, sigma_seeds=2.0, stacked=True, sigma_weights=0.0, min_size=50, n_threads=4)
    check_array(np.ascontiguousarray(ws), shape, integer=True)
    stages.append({"stage": "official_native_stacked_watershed", "started": started, "finished": time.monotonic()})
    started = time.monotonic()
    labels = gasp(boundary, superpixels=ws, gasp_linkage_criteria="average", beta=0.6, post_minsize=50, n_threads=4)
    del boundary, ws
    labels = np.ascontiguousarray(image_rescale(labels, (1.0, 1.0, 1.0), order=0))
    check_array(labels, shape, integer=True)
    # Preserve the actual native partition/IDs (including zero) without silently
    # relabeling any region. The scoring-domain/zero convention is a separate gate.
    labels = labels.astype(np.uint32, copy=False)
    zero_voxels = sum(int(np.count_nonzero(plane == 0)) for plane in labels)
    stages.append({"stage": "official_native_GASP_sizefilter_postprocessing", "started": started, "finished": time.monotonic()})
    started = time.monotonic()
    prediction = write_array(directory / "prediction.h5", "instances", labels)
    del labels
    gc.collect()
    stages.append({"stage": "full_prediction_persistence_readback", "started": started, "finished": time.monotonic()})
    result = {"volume_id": record["id"], "prediction": prediction, "boundary": boundary_pin, "native_zero_voxels": zero_voxels, "scoring_GT_opened": False, "stages": stages, "actual_origins": origins(), "finished_monotonic": time.monotonic(), "unit": "voxel", "independent_plant_n_added": 0, "official_validation_exposure": True, "complete_loaded_runtime_security_proof": False, "CLI_bitwise_equivalence_claimed": False}
    put(directory / "prediction_readback.json", result)
    return result
