"""Audit the already prepared support panels without training or array reads.

Replays the frozen trainer's selector, selected-only class weights and indexed
volume-choice schedule. It does not choose models or launch an experiment.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ovule3d.io import read_json, sha256, utc_now, write_json
from scripts.train import select_records


def selected_class_weights(records):
    if not records or any(r["split"] != "train" for r in records):
        raise ValueError("Only nonempty selected training records are permitted")
    counts = np.asarray([r["target_class_counts"][:3] for r in records], dtype=np.float64)
    if counts.shape != (len(records), 3) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Three nonnegative finite class counts per record required")
    total = counts.sum(axis=0)
    if total.sum() <= 0:
        raise ValueError("No valid supervised voxels")
    weights = np.sqrt(total.sum() / np.maximum(total, 1))
    return total.astype(np.int64).tolist(), (weights / weights.mean()).astype(np.float32).tolist()


def volume_draw_counts(records, steps=5000, optimization_seed=20261018):
    if not records or steps < 1 or len({r["id"] for r in records}) != len(records):
        raise ValueError("Nonempty unique records and positive update count required")
    counts = Counter({r["id"]: 0 for r in records})
    for index in range(steps):
        rng = np.random.default_rng(np.random.SeedSequence([optimization_seed, index]))
        counts[records[int(rng.integers(len(records)))]["id"]] += 1
    return dict(counts)


def replay_panel(manifest, panel, *, steps=5000, optimization_seed=20261018):
    budget, selection_seed = panel["support_group_budget"], panel["selection_seed"]
    training = [r for r in manifest["records"] if r["split"] == "train"]
    group_count = len({r["group_id"] for r in training})
    if type(budget) is not int or budget < 1 or budget > group_count:
        raise ValueError("Support budget must fit actual training groups")
    rows = select_records(manifest, "train", budget, selection_seed)
    if ([r["id"] for r in rows] != panel["file_ids"]
            or sorted({r["group_id"] for r in rows}) != panel["group_ids"]
            or len(rows) != panel["file_count"]):
        raise ValueError("Prepared panel differs from the actual frozen trainer selector")
    selected_ids = {r["id"] for r in rows}
    if selected_ids & {r["id"] for r in manifest["records"] if r["split"] != "train"}:
        raise ValueError("Validation/test file entered support selection")
    for group in {r["group_id"] for r in rows}:
        if {r["id"] for r in training if r["group_id"] == group} - selected_ids:
            raise ValueError("A specimen-number group was split")
    counts, weights = selected_class_weights(rows)
    full_counts, _ = selected_class_weights(training)
    if (counts != panel["topology_class_counts_bg_interior_wall"]
            or sum(counts) != panel["topology_supervised_voxels"]
            or sum(counts[1:]) != panel["topology_foreground_supervised_voxels"]
            or sum(counts) / sum(full_counts) != panel["fraction_of_full_topology_supervised_voxels"]):
        raise ValueError("Prepared support supervision proxies do not replay exactly")
    draws = volume_draw_counts(rows, steps, optimization_seed)
    if sum(draws.values()) != steps:
        raise ValueError("Indexed volume-choice count differs from the update budget")
    return {**panel, "optimization_seed": optimization_seed, "updates": steps,
            "selected_only_class_weights_float32": weights,
            "foreground_fraction_of_full_training_foreground": sum(counts[1:]) / sum(full_counts[1:]),
            "indexed_volume_draw_counts": draws,
            "group_draw_counts": {group: sum(draws[r["id"]] for r in rows if r["group_id"] == group)
                                  for group in panel["group_ids"]},
            "sampling_unit": "volume file, not specimen-number group or voxel",
            "counts_are_available_supervision_not_unique_training_voxels_seen": True}


def main():
    source = ROOT / "reports/low_label_support_plan_20261019_v1/protocol.json"
    original = read_json(source)
    manifest_path = ROOT / "cache/prepared_manifest.json"
    if original["prepared_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("Prepared metadata differs from the existing support plan")
    manifest = read_json(manifest_path)
    protocol = read_json(ROOT / "runs/repeated_ablation_20261019_v2/protocol.json")
    def verify_active_freeze():
        if len(protocol["source_sha256"]) != 44 or any(sha256(ROOT / p) != h for p, h in protocol["source_sha256"].items()):
            raise ValueError("Active frozen experiment dependencies changed")
    verify_active_freeze()
    if original["status"] != "prepared_only_no_training_started" or len(original["panels"]) != 9:
        raise ValueError("Expected the preserved nine-panel preparation, not run results")
    output = ROOT / "reports/low_label_design_audit_20261019_v1"
    output.mkdir(parents=True, exist_ok=False)
    state = {"material_passport": {"origin_skill": "academic-research-suite/experiment-agent", "origin_mode": "validate",
             "origin_date": utc_now(), "verification_status": "ANALYZED", "version_label": "low_label_design_audit_v1"},
             "status": "running", "array_data_read": False, "test_or_external_labels_read": False,
             "training_started": False, "models_selected": False, "network_calls": 0,
             "source_sha256": sha256(__file__), "input_sha256": {str(source.relative_to(ROOT)): sha256(source),
             str(manifest_path.relative_to(ROOT)): sha256(manifest_path)}}
    write_json(output / "audit.json", state)
    try:
        records = [replay_panel(manifest, p) for p in original["panels"]]
        if {(r["selection_seed"], r["support_group_budget"]) for r in records} != {
                (seed, budget) for seed in (20261018, 20261019, 20261020) for budget in (1, 3, 5)}:
            raise ValueError("Support-selection seed/budget panel incomplete")
        for seed in (20261018, 20261019, 20261020):
            same = sorted((r for r in records if r["selection_seed"] == seed), key=lambda r: r["support_group_budget"])
            if not (set(same[0]["group_ids"]) < set(same[1]["group_ids"]) < set(same[2]["group_ids"])):
                raise ValueError("Support groups are not nested")
        full = select_records(manifest, "train", None, 20261018)
        _, expected_weights = selected_class_weights(full)
        full_weight_checks = []
        for run in protocol["panel"]:
            config = read_json(Path(run["run"]) / "config.json")
            if config["class_weights"] != expected_weights:
                raise ValueError("Existing full-training class weights disagree with selected-record formula")
            full_weight_checks.append({"model": run["model"], "seed": run["seed"],
                                       "config_sha256": sha256(Path(run["run"]) / "config.json")})
        state.update({"panels": records, "exact_trainer_selector_replays": 9,
            "nested_seed_panels": 3, "selected_only_class_weight_plans": 9,
            "indexed_schedule_replays": 9, "indexed_volume_choices_per_panel": 5000,
            "full_training_class_weight_config_checks": full_weight_checks,
            "warnings": [
                "Selection seeds repeat support choices; optimization seed is fixed. This is not a full selection-by-optimization-seed factorial study.",
                "A matched full-data reference should retain optimization seed 20261018 rather than treat its three optimization seeds as three support-selection replicates.",
                "The full-data condition is one common selected-all panel, not three independent panels created by duplicating it across selection seeds.",
                "Available supervised voxels are not unique voxels seen, labeling hours or annotated cell counts.",
                "Sampling is uniform by volume file. If 405 is included, A/B stay together but jointly receive two file sampling opportunities; this is not uniform group sampling.",
                "Current nine panels do not include 405; guard tests retain the A/B whole-group rule for any future selection.",
                "Fixed validation, architecture screening and shared filtering remain development choices; no confirmatory label-efficiency or population claim is established.",
                "No low-budget job has been launched here. Choose and bind the model panel only after full repeated validation is reviewed; keep current source freeze unchanged.",
            ],
            "fallacy_scan": {
                "coverage": "11/11 checked",
                "Simpson": "Keep per-selection and per-volume effects; no pooled superiority currently claimed.",
                "ecological": "Group IDs, support seeds and updates do not prove independent plants or labeling hours.",
                "Berkson": "Support selection is training-only and not ranked by outcomes; source corpus itself is not a population sample.",
                "collider": "Do not restrict later budget comparisons to successful models/cells without all failures and denominators.",
                "base_rate": "Preserve per-panel class counts, foreground fractions, group/file counts and all prediction counts later.",
                "regression_to_mean": "Do not select an unusually weak support seed and report its later recovery as an effect.",
                "survivorship": "No training yet; later failures must remain in the planned-run ledger.",
                "look_elsewhere": "All nine fixed support panels retained; no result-dependent panel replacement.",
                "forking_paths": "Panel was fixed earlier; model choice remains pending. Do not call this retrospective preregistration.",
                "causality": "Voxel budget is an annotation proxy; real annotation labor savings are unmeasured.",
                "reverse_causality": "No biological direction or stage inference from these computational plans.",
            },
            "reproducibility": {"metadata_selector_verdict": "EXACT_MATCH", "underlying_training_verdict": "CANNOT_VERIFY_NOT_RUN"},
            "frozen_dependency_hashes_unchanged": 44, "status": "design_metadata_replayed_training_not_started", "finished_utc": utc_now()})
        verify_active_freeze()
        write_json(output / "audit.json", state)
        lines = ["## Material Passport", "", "- Origin Skill: academic-research-suite / experiment-agent",
            "- Origin Mode: validate", f"- Origin Date: {state['finished_utc']}", "- Verification Status: ANALYZED",
            "- Version Label: low_label_design_audit_v1", "", "## 低标注预算设计核查", "",
            "本轮仅重放元数据与采样逻辑，没有读取真实图像/标签数组，没有启动训练，也没有预选主模型。",
            "9个原有面板均与冻结train.py的真实分组选择器一致；3个selection seed均满足1⊂3⊂5嵌套。",
            "为每面板生成仅用所选训练记录计算的class weights，并与21个既有全训练配置的权重公式核对。",
            "5000步indexed volume选择时序可在不读标签的条件下重放；这不代表已经完成5000步训练。", "",
            "| selection seed | group budget | files | available supervised voxel fraction | available foreground fraction |",
            "|---|---:|---:|---:|---:|"]
        for r in records:
            lines.append(f"| {r['selection_seed']} | {r['support_group_budget']} | {r['file_count']} | {r['fraction_of_full_topology_supervised_voxels']:.6f} | {r['foreground_fraction_of_full_training_foreground']:.6f} |")
        lines += ["", "## 必须保留的解释边界", "", *[f"- {w}" for w in state["warnings"]], "",
            "## ARS验证边界", "", "- Coverage: 11/11 fallacy types checked; detailed risk findings are in audit.json.",
            "- Metadata selector: EXACT_MATCH. Underlying training: CANNOT_VERIFY_NOT_RUN.",
            "- No p-values, confidence intervals, human-review claim, efficiency result or submission-ready claim.",
            "- Class weights are expected selected-panel values, not weights recovered from nonexistent low-budget checkpoints.",
            "- Re-read live stage3 and later morphometry-owner state before any new experiment; do not infer execution authority from this readiness report.", ""]
        (output / "DESIGN_AUDIT_CN_EN.txt").write_text("\n".join(lines), encoding="utf-8")
        print(json.dumps({"status": state["status"], "panels": 9, "existing_weight_checks": len(full_weight_checks),
                          "array_data_read": False, "training_started": False, "output": str(output)}))
    except Exception as exc:
        state.update({"status": "failed", "error": repr(exc), "finished_utc": utc_now()})
        write_json(output / "audit.json", state)
        raise


if __name__ == "__main__":
    main()
