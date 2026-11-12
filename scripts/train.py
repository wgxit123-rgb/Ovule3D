"""Recorded single-model training. Selection sees official validation only."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader

from ovule3d.data import VolumePatchDataset
from ovule3d.io import read_json, sha256, utc_now, write_json
from ovule3d.losses import confusion_matrix, confusion_metrics, topology_loss
from ovule3d.models import MODEL_NAMES, build_model, parameter_counts


def select_records(manifest: dict, split: str, budget: int | None, selection_seed: int) -> list[dict]:
    records = sorted((r for r in manifest["records"] if r["split"] == split), key=lambda r: r["id"])
    if split == "train" and budget is not None:
        groups = sorted({r["group_id"] for r in records})
        rng = np.random.default_rng(selection_seed)
        selected = set(np.asarray(groups)[rng.permutation(len(groups))[:budget]].tolist())
        records = [r for r in records if r["group_id"] in selected]
    return records


def source_hashes(root: Path):
    return {str(p.relative_to(root)): sha256(p) for folder in ("scripts", "ovule3d")
            for p in sorted((root / folder).glob("*.py"))}


def train(args):
    root = args.root.resolve()
    manifest_path = root / "cache/prepared_manifest.json"
    manifest = read_json(manifest_path)
    if manifest["status"] != "prepared" and not args.smoke:
        raise RuntimeError("Full-data training requires fully prepared download manifest")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    amp = device.type == "cuda" and args.amp_dtype != "float32"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    if amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested BF16 is not supported by the selected GPU")
    model = build_model(args.model, base_channels=args.base_channels).to(device)
    needs_features = getattr(model, "requires_features", False)
    train_records = select_records(manifest, "train", args.support_groups, args.selection_seed)
    val_records = select_records(manifest, "val", None, args.selection_seed)
    if not val_records:
        raise RuntimeError("No validation volumes available; never substitute training/test as validation")
    train_dataset = VolumePatchDataset(train_records, tuple(args.patch), args.steps * args.batch_size,
                                      args.seed, True, needs_features)
    val_dataset = VolumePatchDataset(val_records, tuple(args.patch), args.val_patches_per_volume * len(val_records),
                                    args.val_seed, False, needs_features)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=amp)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=amp)
    class_counts = np.sum([np.asarray(r["target_class_counts"][:3], dtype=np.float64) for r in train_records], axis=0)
    weights = np.sqrt(class_counts.sum() / np.maximum(class_counts, 1))
    weights = weights / weights.mean()
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    run_dir = args.output.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "config.json").exists():
        raise FileExistsError(f"Run directory already contains a run; choose a new output: {run_dir}")
    config = {**vars(args), "root": str(root), "output": str(run_dir),
              "patch": list(args.patch), "train_ids": [r["id"] for r in train_records],
              "train_groups": sorted({r["group_id"] for r in train_records}),
              "val_ids": [r["id"] for r in val_records], "parameters": parameter_counts(model),
              "class_weights": weights.cpu().tolist(), "source_hashes": source_hashes(root),
              "prepared_manifest_sha256": sha256(manifest_path),
              "torch": torch.__version__, "device_name": torch.cuda.get_device_name(device) if amp else "CPU",
              "created_utc": utc_now(), "selection_metric": "fixed-validation-patch macro Dice(interior,wall)",
              "evaluation_status": "development screening; no test labels read",
              "feature_backbone_frozen": bool(needs_features), "encoder_parameters_not_counted_in_head": bool(needs_features)}
    if hasattr(model, "architecture_spec"):
        config["architecture_at_initialization"] = model.architecture_spec()
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "data_manifest.json", manifest)
    write_json(run_dir / "status.json", {"status": "running", "pid": os.getpid(), "started_utc": utc_now(), "step": 0})
    started = time.monotonic()
    print(json.dumps({"event": "started", "model": args.model, "parameters": config["parameters"],
                      "train_volumes": len(train_records), "val_volumes": len(val_records)}, ensure_ascii=False), flush=True)
    best_score = -1.0
    last_validation = None
    if amp:
        torch.cuda.reset_peak_memory_stats(device)
    with open(run_dir / "history.csv", "w", encoding="utf-8", newline="") as stream:
        fields = ["step", "seconds", "train_loss", "ce", "soft_dice", "lr", "val_loss", "dice_interior", "dice_wall", "macro_dice_interior_wall"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        rolling = []
        for step, batch in enumerate(train_loader, 1):
            model.train()
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            features = batch["features"].to(device, non_blocking=True) if needs_features else None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                logits = model(image, features)
                loss, details = topology_loss(logits, target, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite gradient at step {step}")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            rolling.append(float(loss.detach()))
            validate = step % args.val_interval == 0 or step == args.steps
            if validate:
                model.eval()
                matrix = torch.zeros((3, 3), dtype=torch.float64, device=device)
                losses = []
                with torch.inference_mode():
                    for val_batch in val_loader:
                        val_image = val_batch["image"].to(device)
                        val_target = val_batch["target"].to(device)
                        val_features = val_batch["features"].to(device) if needs_features else None
                        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                            val_logits = model(val_image, val_features)
                            val_loss, _ = topology_loss(val_logits, val_target, weights)
                        losses.append(float(val_loss))
                        matrix += confusion_matrix(val_logits, val_target)
                last_validation = {**confusion_metrics(matrix), "val_loss": float(np.mean(losses)), "step": step}
                write_json(run_dir / "validation_latest.json", last_validation)
                score = last_validation["macro_dice_interior_wall"]
                if score > best_score:
                    best_score = score
                    checkpoint = {"model_state": model.state_dict(), "config": config,
                                  "step": step, "best_validation_score": best_score}
                    torch.save(checkpoint, run_dir / "best.pt")
                    write_json(run_dir / "best_validation.json", last_validation)
            if step % args.log_interval == 0 or validate:
                row = {"step": step, "seconds": time.monotonic() - started,
                       "train_loss": float(np.mean(rolling)), **details, "lr": optimizer.param_groups[0]["lr"]}
                if validate:
                    row.update({k: last_validation[k] for k in fields if k in last_validation})
                writer.writerow(row)
                stream.flush()
                print(json.dumps(row), flush=True)
                write_json(run_dir / "status.json", {"status": "running", "pid": os.getpid(), "step": step,
                           "steps": args.steps, "elapsed_seconds": time.monotonic() - started, "updated_utc": utc_now()})
                rolling.clear()
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                "config": config, "step": args.steps,
                "rng_state": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                              "python": random.getstate(), "numpy": np.random.get_state()}}, run_dir / "last.pt")
    result = {"status": "completed", "steps": args.steps, "seed": args.seed, "model": args.model,
              "seconds": time.monotonic() - started, "best_validation_macro_dice": best_score,
              "last_validation": last_validation, "best_checkpoint_sha256": sha256(run_dir / "best.pt"),
              "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if amp else 0,
              "finished_utc": utc_now(), "result_scope": "development fixed-patch screening, not full-volume test performance"}
    if hasattr(model, "architecture_spec"):
        result["last_checkpoint_architecture"] = model.architecture_spec()
    write_json(run_dir / "result.json", result)
    write_json(run_dir / "status.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--selection-seed", type=int, default=20261018)
    parser.add_argument("--val-seed", type=int, default=95123)
    parser.add_argument("--support-groups", type=int)
    parser.add_argument("--patch", type=int, nargs=3, default=[32, 128, 128])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-interval", type=int, default=100)
    parser.add_argument("--val-patches-per-volume", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, min(args.patch), args.val_interval,
           args.log_interval, args.cpu_threads, args.val_patches_per_volume) < 1:
        parser.error("Steps, batch-size, patch dimensions, intervals and thread/validation counts must be positive")
    if args.support_groups is not None and args.support_groups < 1:
        parser.error("support-groups must be positive")
    try:
        train(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "failure.json", {"status": "failed", "error": repr(error), "utc": utc_now()})
        raise


if __name__ == "__main__":
    main()
