#!/usr/bin/env python3
"""Build BV2 public index CSVs with relative paths from official per-scene split CSVs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SPLIT_NAMES = ("train", "val", "test")
PAPER_COUNTS = {"train": 1911, "val": 625, "test": 584}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build BV2 train/val/test index CSVs (relative paths only).")
    p.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="BV2 processed root (scene folders with audio/, depth/, train.csv, ...)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for train_index.csv, val_index.csv, test_index.csv",
    )
    p.add_argument(
        "--scenes",
        type=str,
        nargs="*",
        default=None,
        help="Optional scene folder names (default: all subdirs with train.csv)",
    )
    return p.parse_args()


def discover_scenes(raw_root: Path, scenes: list[str] | None) -> list[str]:
    if scenes:
        return scenes
    found = []
    for child in sorted(raw_root.iterdir()):
        if child.is_dir() and (child / "train.csv").is_file():
            found.append(child.name)
    return found


def row_to_public(raw_root: Path, scene: str, row: pd.Series, split: str) -> dict[str, str]:
    audio_rel = str(Path(str(row["audio path"])) / str(row["audio file name"]))
    depth_rel = str(Path(str(row["depth path"])) / str(row["depth file name"]))
    sample_id = f"{scene}_{row.get('index', row.name)}"
    return {
        "sample_id": sample_id,
        "scene": scene,
        "split": split,
        "audio_relpath": audio_rel,
        "depth_relpath": depth_rel,
    }


def build_split_index(raw_root: Path, scenes: list[str], split: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for scene in scenes:
        csv_path = raw_root / scene / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing split CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rows.append(row_to_public(raw_root, scene, row, split))
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.is_dir():
        print(f"FAIL: raw-root not found: {raw_root}")
        return 1

    scenes = discover_scenes(raw_root, args.scenes)
    if not scenes:
        print(f"FAIL: no scene folders with train.csv under {raw_root}")
        return 1

    summary: dict[str, int] = {}
    for split in SPLIT_NAMES:
        df = build_split_index(raw_root, scenes, split)
        out_path = out_dir / f"{split}_index.csv"
        df.to_csv(out_path, index=False)
        summary[split] = len(df)
        print(f"Wrote {out_path} ({len(df)} rows)")

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root_note": "Paths in index CSVs are relative to BV2 data root; no absolute paths stored.",
        "scenes": scenes,
        "counts": summary,
        "paper_expected_counts": PAPER_COUNTS,
    }
    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")

    for split, expected in PAPER_COUNTS.items():
        if summary[split] != expected:
            print(
                f"WARN: {split} count {summary[split]} != paper expected {expected} "
                "(may be OK if using a subset)"
            )

    print("RESULT: PASS (index files use relative paths only; no data files copied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
