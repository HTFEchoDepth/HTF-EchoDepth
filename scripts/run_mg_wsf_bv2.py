#!/usr/bin/env python3
"""Run MG-WSF (Metric-Guided Weight-Space Fusion) for BV2 HTF-EchoDepth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from htf_echodepth.data.bv2_dataset import BV2Dataset
from htf_echodepth.fusion.candidate_retention import retain_metric_role_candidates
from htf_echodepth.fusion.checkpoint_fusion import mix_state_dicts
from htf_echodepth.fusion.fusion_selection import select_best_fusion_weights
from htf_echodepth.models import build_htf_echodepth
from htf_echodepth.utils.checkpoint import load_checkpoint
from htf_echodepth.utils.config import get_nested, load_config


def load_candidate_registry(registry_path: Path, registry_root: Path | None = None) -> tuple[list[Path], list[dict]]:
    root = registry_root or registry_path.parent
    donors: list[Path] = []
    records: list[dict] = []
    with registry_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row["checkpoint_path"].strip()
            path = Path(rel)
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                continue
            resolved = path.resolve()
            donors.append(resolved)
            records.append(
                {
                    "checkpoint_path": rel,
                    "checkpoint_name": Path(rel).name,
                    "epoch": int(float(row.get("epoch", 0))),
                    "role_from_filename": row.get("role", Path(rel).stem),
                    "metric_name": row.get("metric_name", ""),
                    "RMSE": float(row["RMSE"]),
                    "REL": float(row["REL"]),
                    "LOG10": float(row["LOG10"]),
                    "DELTA1": float(row["DELTA1"]),
                    "DELTA2": float(row["DELTA2"]),
                    "DELTA3": float(row["DELTA3"]),
                    "WBRS": float(row["WBRS"]),
                }
            )
    seen: set[str] = set()
    unique_donors: list[Path] = []
    for p in donors:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique_donors.append(p)
    return unique_donors, records


def select_mrcr_donors(records: list[dict], *, registry_root: Path) -> list[Path]:
    pool_rows, _ = retain_metric_role_candidates(records)
    paths: list[Path] = []
    seen: set[str] = set()
    for row in pool_rows:
        rel = row["checkpoint_path"]
        p = Path(rel) if Path(rel).is_absolute() else registry_root / rel
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            paths.append(p.resolve())
    return paths


@torch.no_grad()
def collect_predictions(model, loader, device, max_depth_m):
    model.eval()
    preds = []
    gts = []
    for batch in loader:
        echo = batch["echo"].to(device)
        depth = batch["depth"].to(device)
        pred = model(echo)
        pred_m = pred.squeeze(1).cpu().numpy() * max_depth_m
        gt_m = depth.squeeze(1).cpu().numpy() * max_depth_m
        for i in range(pred_m.shape[0]):
            preds.append(pred_m[i])
            gts.append(gt_m[i])
    return preds, gts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MG-WSF fusion for HTF-EchoDepth on BV2")
    p.add_argument("--config", default="configs/bv2/mg_wsf_bv2.yaml")
    p.add_argument("--data-root", required=False)
    p.add_argument("--index-dir", default=None, help="Directory with val_index.csv")
    p.add_argument("--candidate-registry", default=None, help="candidate_registry.csv from training")
    p.add_argument("--registry-root", default=None, help="Resolve relative checkpoint paths (default: registry parent)")
    p.add_argument("--donor-checkpoints", nargs="+", default=None, help="Donor .pth paths")
    p.add_argument("--output-dir", default=None, help="Fusion output directory")
    p.add_argument("--output-checkpoint", default=None)
    p.add_argument("--max-val-samples", type=int, default=None, help="Limit validation samples (smoke)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-degenerate-fusion", action="store_true",
                   help="Allow single-donor copy-through (engineering smoke only; not performance fusion)")
    p.add_argument("--save-fused-checkpoint", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.dry_run:
        from htf_echodepth.fusion import candidate_retention, checkpoint_fusion, fusion_selection, wbrs

        _ = (candidate_retention, checkpoint_fusion, fusion_selection, wbrs)
        print("MG-WSF dry-run: config loaded OK; WBRS/MRCR/VGFS/checkpoint_fusion import OK")
        if args.candidate_registry:
            reg = Path(args.candidate_registry)
            root = Path(args.registry_root) if args.registry_root else reg.parent
            donors, _ = load_candidate_registry(reg, root)
            print(f"candidate_registry OK: {len(donors)} donor checkpoint(s) resolved")
        return

    data_root = args.data_root or cfg.get("data_root")
    if not data_root:
        raise SystemExit("--data-root is required unless --dry-run is set")

    registry_root = Path(args.registry_root) if args.registry_root else None
    donors: list[Path] = []
    registry_records: list[dict] = []
    if args.candidate_registry:
        reg_path = Path(args.candidate_registry)
        root = registry_root or reg_path.parent
        all_donors, registry_records = load_candidate_registry(reg_path, root)
        donors = select_mrcr_donors(registry_records, registry_root=root) if registry_records else all_donors
        if not donors:
            donors = all_donors
        print(f"MRCR pool: {len(donors)} donor(s) selected from registry")
    elif args.donor_checkpoints:
        donors = [Path(p) for p in args.donor_checkpoints]
    else:
        cfg_paths = cfg.get("donor_checkpoints", {}).get("paths") or []
        donors = [Path(p) for p in cfg_paths]

    if len(donors) < 1:
        raise SystemExit("Need at least one donor checkpoint via --candidate-registry or --donor-checkpoints")

    device = torch.device(args.device)
    ds_cfg = cfg["dataset"]
    index_dir = Path(args.index_dir).resolve() if args.index_dir else None
    val_index = (index_dir / "val_index.csv").resolve() if index_dir else ds_cfg["val_index"]
    val_ds = BV2Dataset(data_root, val_index, max_depth_m=ds_cfg["max_depth_m"])
    if args.max_val_samples is not None and args.max_val_samples < len(val_ds):
        val_ds = Subset(val_ds, list(range(args.max_val_samples)))
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    max_depth = float(cfg["metrics"]["max_depth_m"])
    valid_min = float(cfg["metrics"]["valid_min_depth_m"])
    wbrs_ref = cfg["wbrs"]["reference_metrics"]
    grid_step = float(get_nested(cfg, "fusion", "vgfs", "grid_step", default=0.05))

    out_dir = Path(args.output_dir or Path(cfg["fusion"]["output_checkpoint"]).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(donors) < 2:
        if len(donors) == 1 and args.allow_degenerate_fusion:
            print("NOTE: single donor — degenerate copy-through (engineering smoke only, not VGFS fusion)")
        else:
            raise SystemExit(
                "MG-WSF requires ≥2 distinct MRCR donors for validation-guided fusion. "
                "MRCR collapsed to fewer than 2 unique checkpoints. "
                "Use --allow-degenerate-fusion only for engineering smoke tests."
            )

    if len(donors) == 1:
        weights = [1.0]
        model = build_htf_echodepth(depth_norm=bool(get_nested(cfg, "model", "depth_normalize", default=True)))
        load_checkpoint(model, donors[0], map_location=device)
        model.to(device)
        preds, gts = collect_predictions(model, loader, device, max_depth)
        per_sample_preds = [preds]
        val_metrics = {"note": "degenerate single-donor copy-through (engineering smoke only)"}
    else:
        per_sample_preds = []
        gts: list[np.ndarray] = []
        for path in donors:
            model = build_htf_echodepth(depth_norm=bool(get_nested(cfg, "model", "depth_normalize", default=True)))
            load_checkpoint(model, path, map_location=device)
            model.to(device)
            preds, batch_gts = collect_predictions(model, loader, device, max_depth)
            if not gts:
                gts = batch_gts
            per_sample_preds.append(preds)

        weights, val_metrics = select_best_fusion_weights(
            per_sample_preds,
            gts,
            valid_min_depth_m=valid_min,
            wbrs_reference=wbrs_ref,
            grid_step=grid_step,
        )

    if args.save_fused_checkpoint:
        out_path = Path(args.output_checkpoint or out_dir / "htf_echodepth_mg_wsf_fused.pth")
        if len(donors) == 1:
            payload = torch.load(donors[0], map_location="cpu", weights_only=False)
            payload["fusion_tag"] = "mg_wsf_degenerate_smoke"
            payload["fusion_note"] = "single-donor copy-through; engineering smoke only — not performance fusion"
            torch.save(payload, out_path)
        else:
            fused = mix_state_dicts(donors, weights)
            torch.save(fused, out_path)
        result = {
            "fusion_weights": weights,
            "validation_metrics": val_metrics,
            "output_checkpoint": str(out_path),
            "donor_checkpoints": [str(p) for p in donors],
        }
        result_path = out_dir / "fusion_result.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps({"fusion_weights": weights, "donors": len(donors), "val_metrics": val_metrics}, indent=2))


if __name__ == "__main__":
    main()
