"""External-only IO primitives and ordered assessment, not an execution CLI.

Readiness tests use synthetic arrays/TIFFs. Real workers must additionally check
live job prerequisites and a source-bound executable freeze before calling this API.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import hashlib
import math
import re

import numpy as np
import tifffile

COHORT_IDS = ('plant1_4hrs', 'plant2_0hrs', 'plant4_0hrs',
              'plant13_0hrs', 'plant15_0hrs', 'plant18_0hrs')
SEEDS = (20261018, 20261019, 20261020)
MODELS = ('dtpa3d_pyramid', 'pyramid3d_raw')
SETTINGS = {
    'normalization': {'percentiles': [0.5, 99.8], 'stride_target': 2_000_000,
                      'arithmetic': 'float32', 'output_dtype': 'float16',
                      'denominator_floor': 1e-6, 'spatial_resampling': False},
    'prediction': {'patch_zyx': [32, 128, 128], 'overlap': 0.5,
                   'weighting': 'gaussian_sigma0.5_floor0.001',
                   'autocast_dtype': 'float16', 'probability_dtype': 'float32'},
    'reconstruction': {'units': 'voxel', 'foreground_threshold': 0.5,
                       'interior_seed_threshold': 0.35, 'wall_seed_threshold': 0.5,
                       'min_seed_distance': 4.0, 'seed_smoothing': 1.0,
                       'min_cell_volume': 512, 'wall_weight': 0.7, 'connectivity': 1},
    'reference': {'background_id': 1, 'unexpected_zero': 'stop',
                  'positive_cell_ids': 'preserve IDs >= 2', 'transpose': False},
    'metrics': {'object_iou_threshold': 0.5, 'overlap_threshold': 0.1,
                'primary': 'object_f1', 'analysis_unit': 'plant',
                'seed_aggregation': 'mean within plant before equal-plant mean'},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def is_sha(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_freeze_structure(freeze: dict, record: dict, checkpoint: dict,
                              runtime_sources: dict[str, str]) -> None:
    """Structure/binding only; caller must hash runtime files and verify job gates."""
    require(freeze.get('status') == 'frozen_external_transfer', 'External execution freeze required')
    require(freeze.get('settings') == SETTINGS, 'Frozen normalization/reconstruction/analysis settings differ')
    sources = freeze.get('source_sha256', {})
    require(bool(runtime_sources) and sources == runtime_sources
            and all(is_sha(v) for v in sources.values()), 'Runtime source inventory mismatch')
    cohort = freeze.get('cohort', [])
    require(len(cohort) == 6 and {r['id'] for r in cohort} == set(COHORT_IDS),
            'Exact six-record prospective cohort required')
    require(sum(r == record for r in cohort) == 1, 'Record differs from frozen metadata')
    validate_record(record)
    checkpoints = freeze.get('checkpoints', [])
    expected = {(m, s) for m in MODELS for s in SEEDS}
    require(len(checkpoints) == 6 and {(r['model'], r['seed']) for r in checkpoints} == expected,
            'All two-model three-seed checkpoints required')
    require(sum(r == checkpoint for r in checkpoints) == 1
            and all(is_sha(r['checkpoint_sha256']) and is_sha(r['config_sha256']) for r in checkpoints),
            'Checkpoint/config identity differs from freeze')


def validate_record(record: dict) -> tuple[int, int, int]:
    require(record['id'] in COHORT_IDS, 'Record outside preselected external cohort')
    require(str(record['qualification_only']).lower() == 'false', 'Qualification pair excluded')
    require(record['role'] == 'external_evaluation_only', 'External data cannot be a development split')
    require(record['cluster_id'] == record['plant'], 'Plant is the cluster, not timepoint or seed')
    require(record['id'] == f"{record['plant']}_{int(record['timepoint_hours'])}hrs",
            'Record/plant/timepoint identity mismatch')
    require(str(record['shape_match']).lower() == 'true', 'Source paired-shape check missing')
    require(record['raw_axes'] == 'ZYX' and record['label_axes'] in ('ZYX', 'IYX'),
            'Only recorded paired ZYX / page-IYX equivalence is supported')
    shape = tuple(int(record[k]) for k in ('raw_z', 'raw_y', 'raw_x'))
    require(all(n > 0 for n in shape), 'Positive native 3D frame required')
    require(is_sha(record['raw_sha256']) and is_sha(record['label_sha256']), 'Source SHA identities required')
    require(record['raw_dtype'] == 'uint8' and record['label_dtype'] == 'uint16',
            'Unexpected source dtype requires explicit review')
    return shape


def corrected_spacing(reported_xyz: list[float], stretch: float,
                       recorded_corrected_zyx: list[float]) -> tuple[float, float, float]:
    require(len(reported_xyz) == len(recorded_corrected_zyx) == 3,
            'Three XYZ source and ZYX corrected coordinates required')
    values = [*reported_xyz, stretch, *recorded_corrected_zyx]
    require(all(math.isfinite(x) and x > 0 for x in values), 'Positive finite source scales required')
    expected = (reported_xyz[2] / stretch, reported_xyz[1], reported_xyz[0])
    require(all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                for a, b in zip(expected, recorded_corrected_zyx)),
            'Spacing correction differs; no fixed or double-corrected Z scale')
    return expected


def bind_pair_metadata(record: dict, pair: dict) -> dict:
    """Bind archived CSV metadata to its source pair/spacing JSON; no pixel IO."""
    validate_record(record)
    require(pair['id'] == record['id'] and pair['plant'] == record['plant']
            and pair['cluster_id'] == record['cluster_id']
            and pair['timepoint_hours'] == int(record['timepoint_hours']),
            'Pair identity differs from preselected inventory')
    require(pair['complete'] is True and pair['qualification_only'] is False
            and pair['training_allowed'] is False and pair['tuning_allowed'] is False
            and pair['role'] == 'external_evaluation_only', 'Source pair is not external-only')
    require(pair['source_background_id'] == 1 and pair['source_zero_is_background'] is False,
            'Reference convention differs from source README')
    for role in ('raw', 'label'):
        require(pair[role]['path'] == record[role + '_path']
                and pair[role]['sha256'] == record[role + '_sha256']
                and pair[role]['size_bytes'] > 0, 'Source pair byte/path identity differs')
    spacing = pair['spacing']
    corrected = corrected_spacing(spacing['reported_xyz_um'], spacing['z_stretch_factor'],
                                  spacing['corrected_zyx_um'])
    csv_spacing = tuple(float(record[k]) for k in ('spacing_z_um', 'spacing_y_um', 'spacing_x_um'))
    require(all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                for a, b in zip(corrected, csv_spacing))
            and math.isclose(float(record['z_stretch_factor']), spacing['z_stretch_factor'],
                             rel_tol=1e-12, abs_tol=1e-12), 'CSV and source spacing differ')
    return {**record, 'source_spacing': spacing, 'reference_background_id': 1,
            'raw_size_bytes': pair['raw']['size_bytes'], 'label_size_bytes': pair['label']['size_bytes'],
            'source_physical_scale_status': 'source-reported corrected spacing; not independent instrument calibration',
            'independent_plant_cluster': record['plant']}


def normalize_raw(raw: np.ndarray) -> tuple[np.ndarray, dict]:
    require(raw.ndim == 3 and min(raw.shape) > 0 and raw.dtype == np.dtype('uint8'),
            'Expected a nonempty native uint8 membrane stack')
    stride = max(1, raw.size // 2_000_000)
    sample = raw.ravel()[::stride]
    low, high = np.percentile(sample, (0.5, 99.8)).astype(float)
    image = np.clip((raw.astype(np.float32) - low) / max(high - low, 1e-6), 0, 1).astype(np.float16)
    return image, {'p0.5': low, 'p99.8': high, 'sample_stride': stride,
                   'sample_count': int(sample.size), 'spatial_resampling': False}


def metric_reference(source: np.ndarray, expected_shape: tuple[int, int, int]) -> np.ndarray:
    require(source.ndim == 3 and tuple(source.shape) == tuple(expected_shape),
            'Reference frame differs; never transpose or crop to fit predictions')
    require(np.issubdtype(source.dtype, np.integer) and source.dtype != np.dtype('bool'),
            'Integer source instance IDs required')
    require(source.size > 0 and int(source.min()) >= 1, 'Unexpected source zero/negative IDs have no defined policy')
    require(int(source.max()) <= np.iinfo(np.int32).max, 'Reference IDs exceed int32')
    labels = source.astype(np.int32, copy=True)
    labels[source == 1] = 0
    return labels


def read_tiff(path: Path, *, expected_sha256: str, shape: tuple[int, int, int],
              axes: str, dtype: str) -> np.ndarray:
    """Concrete IO helper; real worker calls this only after its access gates."""
    require(digest(path) == expected_sha256, 'TIFF byte identity changed')
    with tifffile.TiffFile(path) as handle:
        require(len(handle.series) == 1, 'Multiple TIFF series are not silently selected')
        series = handle.series[0]
        require(tuple(series.shape) == tuple(shape) and series.axes == axes
                and str(series.dtype) == dtype and len(handle.pages) == shape[0],
                'TIFF frame/axis/dtype/page inventory differs')
        array = series.asarray()
    require(tuple(array.shape) == tuple(shape), 'Decoded TIFF frame changed')
    return array


def validate_commit(receipt: dict, *, record: dict, checkpoint_sha256: str,
                    freeze_sha256: str, prediction_sha256: str,
                    prediction_shape: tuple[int, int, int]) -> None:
    require(receipt.get('stage') == 'prediction_committed_before_reference', 'Prediction must be committed before reference')
    expected = {'volume_id': record['id'], 'checkpoint_sha256': checkpoint_sha256,
                'freeze_sha256': freeze_sha256, 'prediction_sha256': prediction_sha256,
                'shape_zyx': list(prediction_shape), 'reference_used': False}
    require(all(receipt.get(k) == v for k, v in expected.items())
            and all(is_sha(x) for x in (checkpoint_sha256, freeze_sha256, prediction_sha256)),
            'Committed prediction/record/freeze binding differs')


def assess_ordered(record: dict, checkpoint: dict, *, freeze: dict,
                   runtime_sources: dict[str, str], freeze_sha256: str,
                   raw_loader: Callable, predictor: Callable,
                   committer: Callable, committed_hasher: Callable,
                   reference_loader: Callable, scorer: Callable):
    """Injectable ordering used to test no-reference prediction and persistence.

    `committed_hasher` must hash the actual saved prediction, not echo a supplied
    expected value. A real worker still needs live-prerequisite/resource guards.
    """
    validate_freeze_structure(freeze, record, checkpoint, runtime_sources)
    shape = validate_record(record)
    require(is_sha(freeze_sha256), 'Actual frozen-record SHA required')
    raw = raw_loader(record)
    require(tuple(raw.shape) == shape, 'Decoded raw frame differs from freeze')
    image, normalization = normalize_raw(raw)
    prediction = predictor(image)
    require(prediction.shape == image.shape and np.issubdtype(prediction.dtype, np.integer)
            and int(prediction.min()) >= 0 and int(prediction.max()) <= np.iinfo(np.int32).max,
            'Complete nonnegative int32-compatible instance prediction required')
    receipt = committer(prediction)
    validate_commit(receipt, record=record, checkpoint_sha256=checkpoint['checkpoint_sha256'],
                    freeze_sha256=freeze_sha256, prediction_sha256=committed_hasher(),
                    prediction_shape=shape)
    reference = metric_reference(reference_loader(record), shape)
    return {'metrics': scorer(prediction, reference), 'normalization': normalization,
            'prediction_commit': receipt, 'source_background_id': 1,
            'reference_access_after_prediction_commit': True}
