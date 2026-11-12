"""Role-defined low-label panel, source guards and completed-output checks.

No image/label arrays, training, downloads or process starts occur here.
"""
from itertools import product
from pathlib import Path
import csv
import math

from audits.audit_low_label_design import replay_panel
from ovule3d.io import read_json, sha256

MODELS = ("dtpa3d_pyramid", "pyramid3d_raw", "segresnet", "plantseg_arch_topology22")
SELECTION_SEEDS = (20261018, 20261019, 20261020)
BUDGETS = (1, 3, 5)
OPTIMIZATION_SEED = 20261018
VOLUMES = ("N_420_ds2x", "N_464_ds2x")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_support_plan(source, manifest):
    require(source["status"] == "prepared_only_no_training_started", "Unexpected support source")
    panels = source["panels"]
    require(len(panels) == 9 and {(p["selection_seed"], p["support_group_budget"]) for p in panels}
            == set(product(SELECTION_SEEDS, BUDGETS)), "Incomplete/duplicate support grid")
    rows = [replay_panel(manifest, p, optimization_seed=OPTIMIZATION_SEED) for p in panels]
    for seed in SELECTION_SEEDS:
        one, three, five = [set(next(p["group_ids"] for p in rows
            if (p["selection_seed"], p["support_group_budget"]) == (seed, b))) for b in BUDGETS]
        require(one < three < five, "Support groups are not strictly nested")
    return sorted(rows, key=lambda p: (p["selection_seed"], p["support_group_budget"]))


def panel_jobs(root, folder, panels):
    jobs = []
    for panel in panels:
        for model in MODELS:
            name = f"{model}_select{panel['selection_seed']}_groups{panel['support_group_budget']}"
            run = folder / "training" / name
            arguments = ["scripts/train.py", "--model", model, "--output", str(run),
                "--device", "cuda:1", "--steps", "5000", "--seed", str(OPTIMIZATION_SEED),
                "--selection-seed", str(panel["selection_seed"]), "--support-groups", str(panel["support_group_budget"]),
                "--val-seed", "95123", "--patch", "32", "128", "128", "--amp-dtype", "bfloat16",
                "--batch-size", "1", "--base-channels", "16", "--lr", "0.001", "--val-interval", "250",
                "--val-patches-per-volume", "8", "--cpu-threads", "4", "--log-interval", "20"]
            jobs.append(dict(name=name, model=model, optimization_seed=OPTIMIZATION_SEED,
                selection_seed=panel["selection_seed"], support_group_budget=panel["support_group_budget"],
                run=str(run), support=panel, train_arguments=arguments))
    require(len(jobs) == 36 and len({j["name"] for j in jobs}) == 36, "Expected 36 unique training jobs")
    return jobs


def check_training(job, manifest_sha):
    run = Path(job["run"])
    result, config = read_json(run / "result.json"), read_json(run / "config.json")
    expected = {"model": job["model"], "seed": OPTIMIZATION_SEED, "steps": 5000,
        "selection_seed": job["selection_seed"], "support_groups": job["support_group_budget"],
        "patch": [32, 128, 128], "amp_dtype": "bfloat16", "batch_size": 1, "lr": .001,
        "val_seed": 95123, "val_interval": 250, "val_patches_per_volume": 8, "base_channels": 16,
        "train_ids": job["support"]["file_ids"], "train_groups": job["support"]["group_ids"],
        "val_ids": list(VOLUMES), "prepared_manifest_sha256": manifest_sha}
    for key, value in expected.items():
        require(config.get(key) == value, f"New training configuration differs: {key}")
    require(result["status"] == "completed" and result["steps"] == 5000
            and result["seed"] == OPTIMIZATION_SEED and result["model"] == job["model"], "Incomplete training")
    require(all(math.isclose(a, b, abs_tol=1e-7, rel_tol=0) for a, b in
        zip(config["class_weights"], job["support"]["selected_only_class_weights_float32"]))
        and len(config["class_weights"]) == 3, "Selected-only class weights differ")
    checkpoint_hash = sha256(run / "best.pt")
    require(result["best_checkpoint_sha256"] == checkpoint_hash, "Best checkpoint bytes differ")
    with (run / "history.csv").open(encoding="utf-8", newline="") as f:
        history = list(csv.DictReader(f))
    val = [r for r in history if r["macro_dice_interior_wall"]]
    require([int(r["step"]) for r in val] == list(range(250, 5001, 250)), "Validation event grid differs")
    scores = [float(r["macro_dice_interior_wall"]) for r in val]
    require(all(math.isfinite(s) for s in scores), "Nonfinite validation scores")
    best = read_json(run / "best_validation.json")
    require(best["step"] == int(val[scores.index(max(scores))]["step"])
            and math.isclose(result["best_validation_macro_dice"], max(scores), abs_tol=1e-12, rel_tol=0),
            "First-maximum checkpoint rule differs")
    return dict(name=job["name"], model=job["model"], selection_seed=job["selection_seed"],
        optimization_seed=OPTIMIZATION_SEED, support_group_budget=job["support_group_budget"],
        run=str(run), checkpoint_sha256=checkpoint_hash, result_sha256=sha256(run / "result.json"),
        config_sha256=sha256(run / "config.json"), best_step=best["step"])


def check_evaluation(path, job, checkpoint_hash, volume, manifest_sha):
    evidence = read_json(path)
    require(evidence["status"] == "completed" and evidence["split"] == "val"
        and evidence["volume_ids"] == [volume] and evidence["scope"] == "complete_prepared_volumes"
        and evidence["model"] == job["model"] and evidence["checkpoint_sha256"] == checkpoint_hash
        and evidence["prepared_manifest_sha256"] == manifest_sha and evidence["max_shape"] is None
        and evidence["prediction_conditioned_on_gt"] is False, "Evaluation scope/checkpoint differs")
    require(evidence["inference"]["postprocess"]["min_cell_volume"] == 512, "Size threshold was retuned")
    require(len(evidence["results"]) == 1 and evidence["results"][0]["id"] == volume, "Evaluation volume differs")
    output = evidence["results"][0]
    require(sha256(output["output_h5"]) == output["output_h5_sha256"], "Prediction byte hash mismatch")
    return dict(name=job["name"], model=job["model"], selection_seed=job["selection_seed"],
        optimization_seed=OPTIMIZATION_SEED, support_group_budget=job["support_group_budget"],
        volume_id=volume, evaluation=str(path), evaluation_sha256=sha256(path),
        checkpoint_sha256=checkpoint_hash)
