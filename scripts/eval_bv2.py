#!/usr/bin/env python3
"""Evaluate HTF-EchoDepth on BV2 test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from htf_echodepth.data.bv2_dataset import BV2Dataset
from htf_echodepth.metrics.depth_metrics import aggregate_samples
from htf_echodepth.models import build_htf_echodepth
from htf_echodepth.utils.checkpoint import load_checkpoint
from htf_echodepth.utils.config import get_nested, load_config


@torch.no_grad()
def run_eval(model, loader, device, max_depth_m, valid_min_depth_m, max_samples: int | None = None):
    model.eval()
    samples = []
    for batch in loader:
        echo = batch["echo"].to(device)
        depth = batch["depth"].to(device)
        pred = model(echo)
        pred_m = pred.squeeze(1).cpu().numpy() * max_depth_m
        gt_m = depth.squeeze(1).cpu().numpy() * max_depth_m
        for i in range(pred_m.shape[0]):
            samples.append((gt_m[i], pred_m[i]))
            if max_samples is not None and len(samples) >= max_samples:
                break
        if max_samples is not None and len(samples) >= max_samples:
            break
    return aggregate_samples(samples, valid_min_depth_m=valid_min_depth_m)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate HTF-EchoDepth on BV2")
    p.add_argument("--config", default="configs/bv2/eval_htf_echodepth_bv2.yaml")
    p.add_argument("--data-root", required=False)
    p.add_argument("--checkpoint", required=False)
    p.add_argument("--index-file", default=None, help="Override eval index CSV path")
    p.add_argument("--output-dir", default=None, help="Directory for eval outputs")
    p.add_argument("--max-samples", type=int, default=None, help="Limit eval samples (smoke)")
    p.add_argument("--output-json", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = args.data_root or cfg.get("data_root")
    ckpt = args.checkpoint or get_nested(cfg, "checkpoint", "path")
    if args.dry_run:
        model = build_htf_echodepth()
        x = torch.randn(1, 2, 256, 256)
        print(f"dry-run OK: output shape {tuple(model(x).shape)}")
        return
    if not data_root or not ckpt:
        raise SystemExit("--data-root and --checkpoint are required")

    device = torch.device(args.device)
    model = build_htf_echodepth(depth_norm=bool(get_nested(cfg, "model", "depth_normalize", default=True)))
    load_checkpoint(model, ckpt, map_location=device)
    model.to(device)

    ds_cfg = cfg["dataset"]
    if args.index_file:
        eval_index = Path(args.index_file)
        if not eval_index.is_absolute():
            eval_index = (Path.cwd() / eval_index).resolve()
    else:
        eval_index = ds_cfg["eval_index"]
    ds = BV2Dataset(data_root, eval_index, max_depth_m=ds_cfg["max_depth_m"])
    if args.max_samples is not None and args.max_samples < len(ds):
        ds = Subset(ds, list(range(args.max_samples)))
    loader = DataLoader(ds, batch_size=int(cfg["evaluation"]["batch_size"]), shuffle=False, num_workers=int(cfg["evaluation"]["num_workers"]))
    metrics = run_eval(
        model,
        loader,
        device,
        float(cfg["metrics"]["max_depth_m"]),
        float(cfg["metrics"]["valid_min_depth_m"]),
        max_samples=args.max_samples,
    )
    print(json.dumps(metrics, indent=2))
    out_dir = Path(args.output_dir) if args.output_dir else None
    out_json = Path(args.output_json) if args.output_json else (out_dir / "eval_metrics.json" if out_dir else None)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
