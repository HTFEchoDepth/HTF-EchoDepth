#!/usr/bin/env python3
"""Train HTF-EchoDepth on BV2 with paper-style recipe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from htf_echodepth.data.bv2_dataset import BV2Dataset
from htf_echodepth.fusion.candidate_manager import CandidateManager
from htf_echodepth.fusion.wbrs import compute_wbrs
from htf_echodepth.losses.bv2_loss import BV2Loss
from htf_echodepth.metrics.depth_metrics import aggregate_samples, metrics_to_wbrs_input
from htf_echodepth.models import build_htf_echodepth
from htf_echodepth.utils.checkpoint import save_checkpoint
from htf_echodepth.utils.config import get_nested, load_config
from htf_echodepth.utils.seed import set_seed


def cosine_warmup_lr(epoch: int, cfg: dict) -> float:
    train = cfg["training"]
    sched = cfg["lr_scheduler"]
    base = float(train["learning_rate"])
    warmup = int(sched.get("warmup_epochs", 3))
    min_lr = float(sched.get("min_lr", 5e-5))
    total = int(train["epochs"])
    if epoch < warmup:
        return base * float(epoch + 1) / float(max(warmup, 1))
    progress = (epoch - warmup) / float(max(total - warmup, 1))
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * progress))


def resolve_index_path(index_dir: Path | None, index_name: str) -> str | Path:
    if index_dir is not None:
        return (index_dir / index_name).resolve()
    return index_name


def build_subset_dataset(data_root: str, index_path: str | Path, max_depth_m: float, max_samples: int | None):
    ds = BV2Dataset(data_root, index_path, max_depth_m=max_depth_m)
    if max_samples is not None and max_samples < len(ds):
        return Subset(ds, list(range(max_samples)))
    return ds


@torch.no_grad()
def evaluate(model, loader, device, max_depth_m: float, valid_min_depth_m: float, max_batches: int | None = None):
    model.eval()
    samples = []
    for batch_idx, batch in enumerate(loader):
        echo = batch["echo"].to(device)
        depth = batch["depth"].to(device)
        pred = model(echo)
        pred_m = pred.squeeze(1).cpu().numpy() * max_depth_m
        gt_m = depth.squeeze(1).cpu().numpy() * max_depth_m
        for i in range(pred_m.shape[0]):
            samples.append((gt_m[i], pred_m[i]))
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    metrics = aggregate_samples(samples, valid_min_depth_m=valid_min_depth_m)
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches: int | None = None):
    model.train()
    total_loss = 0.0
    n = 0
    for batch_idx, batch in enumerate(loader):
        echo = batch["echo"].to(device)
        depth = batch["depth"].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(echo)
        out = criterion(pred, depth)
        out.total.backward()
        optimizer.step()
        total_loss += float(out.total.item())
        n += 1
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    return total_loss / max(n, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HTF-EchoDepth on BV2")
    p.add_argument("--config", default="configs/bv2/train_htf_echodepth_bv2.yaml")
    p.add_argument("--data-root", required=False, help="BV2 processed data root")
    p.add_argument("--index-dir", default=None, help="Directory with train/val/test_index.csv")
    p.add_argument("--output-dir", default=None, help="Override checkpoint output dir")
    p.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--max-batches", type=int, default=None, help="Limit training batches per epoch (smoke)")
    p.add_argument("--max-train-samples", type=int, default=None, help="Limit training samples (smoke)")
    p.add_argument("--max-val-samples", type=int, default=None, help="Limit validation samples (smoke)")
    p.add_argument("--save-smoke-checkpoint", action="store_true", help="Always save smoke checkpoint after run")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true", help="Build model/dataloaders only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    data_root = args.data_root or cfg.get("data_root")
    if not data_root and not args.dry_run:
        raise SystemExit("--data-root is required unless --dry-run is set")

    seed = int(get_nested(cfg, "training", "seed", default=42))
    set_seed(seed, deterministic=bool(get_nested(cfg, "training", "deterministic", default=True)))

    model = build_htf_echodepth(
        input_nc=int(get_nested(cfg, "model", "input_channels", default=2)),
        output_nc=int(get_nested(cfg, "model", "output_channels", default=1)),
        depth_norm=bool(get_nested(cfg, "model", "depth_normalize", default=True)),
    )
    device = torch.device(args.device)
    model.to(device)

    if args.dry_run:
        x = torch.randn(1, 2, 256, 256, device=device)
        y = model(x)
        print(f"dry-run OK: input {tuple(x.shape)} -> output {tuple(y.shape)}")
        return

    ds_cfg = cfg["dataset"]
    index_dir = Path(args.index_dir).resolve() if args.index_dir else None
    train_index = resolve_index_path(index_dir, "train_index.csv")
    val_index = resolve_index_path(index_dir, "val_index.csv")
    max_depth = float(ds_cfg["max_depth_m"])

    train_ds = build_subset_dataset(data_root, train_index, max_depth, args.max_train_samples)
    val_ds = build_subset_dataset(data_root, val_index, max_depth, args.max_val_samples)
    batch_size = int(args.batch_size or cfg["training"]["batch_size"])
    num_workers = int(cfg["training"]["num_workers"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    lambda_lrh = float(get_nested(cfg, "loss", "lambda_lrh", default=0.003))
    criterion = BV2Loss(lambda_lrh=lambda_lrh)
    base_lr = float(cfg["training"]["learning_rate"])
    wd = float(cfg["training"].get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)

    def lr_lambda(epoch: int) -> float:
        return cosine_warmup_lr(epoch, cfg) / base_lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    out_dir = Path(args.output_dir or cfg["checkpoint"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cfg = cfg.get("checkpoint", {})
    enabled_roles = {
        "best_rmse": bool(ckpt_cfg.get("save_best_rmse", True)),
        "best_rel": bool(ckpt_cfg.get("save_best_rel", True)),
        "best_log10": bool(ckpt_cfg.get("save_best_log10", True)),
        "best_delta1": bool(ckpt_cfg.get("save_best_delta1", True)),
        "best_delta2": bool(ckpt_cfg.get("save_best_delta2", True)),
        "best_delta3": bool(ckpt_cfg.get("save_best_delta3", True)),
        "best_wbrs": bool(ckpt_cfg.get("save_best_wbrs", True)),
    }
    candidate_mgr = CandidateManager(out_dir, enabled_roles=enabled_roles)

    wbrs_ref = cfg["wbrs"]["reference_metrics"]
    valid_min = float(cfg["metrics"]["valid_min_depth_m"])
    max_depth_m = float(cfg["metrics"]["max_depth_m"])

    total_epochs = int(cfg["training"]["epochs"])
    for epoch in range(total_epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, max_batches=args.max_batches
        )
        val_metrics = evaluate(model, val_loader, device, max_depth_m, valid_min)
        canon = metrics_to_wbrs_input(val_metrics)
        wbrs = compute_wbrs(canon, wbrs_ref)
        scheduler.step()
        saved = candidate_mgr.update(
            epoch + 1,
            model,
            optimizer,
            val_metrics=val_metrics,
            canon=canon,
            wbrs=wbrs,
            save_last=bool(ckpt_cfg.get("save_last", True)),
        )
        print(
            f"epoch {epoch+1}: train_loss={train_loss:.4f} val_rmse={val_metrics['rmse']:.4f} "
            f"WBRS={wbrs:.4f} saved={saved}"
        )

    val_csv, registry_csv = candidate_mgr.finalize()
    print(f"Wrote {val_csv}")
    print(f"Wrote {registry_csv}")

    if args.save_smoke_checkpoint:
        smoke_path = out_dir / "smoke_checkpoint.pth"
        save_checkpoint(smoke_path, model, epoch=total_epochs, optimizer=optimizer, tag="smoke")
        print(f"Wrote smoke checkpoint: {smoke_path}")

    summary = {
        "output_dir": str(out_dir),
        "epochs_completed": total_epochs,
        "val_metrics_csv": str(val_csv),
        "candidate_registry_csv": str(registry_csv),
        "candidates_saved": sorted({r["checkpoint_name"] for r in candidate_mgr.validation_records}),
    }
    summary_path = out_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
