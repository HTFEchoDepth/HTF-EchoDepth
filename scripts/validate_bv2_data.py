#!/usr/bin/env python3
"""Validate BatVision V2 processed data layout and dataloader output shapes."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from htf_echodepth.data.bv2_dataset import BV2Dataset, BV2_IMAGE_SIZE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate BV2 data root and index file.")
    p.add_argument("--data-root", type=Path, required=True, help="BV2 processed data root")
    p.add_argument("--index-file", type=Path, required=True, help="Split index CSV (relative or absolute)")
    p.add_argument("--num-samples", type=int, default=3, help="Random samples to load")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    errors: list[str] = []

    data_root = args.data_root.resolve()
    index_path = Path(args.index_file)
    if not index_path.is_absolute():
        index_path = (Path.cwd() / index_path).resolve()
    if not index_path.is_file():
        index_path = (data_root / args.index_file).resolve()

    print(f"BV2 validate: data_root={data_root}")
    print(f"BV2 validate: index_file={index_path}")

    if not data_root.is_dir():
        errors.append(f"data-root is not a directory: {data_root}")
    if not index_path.is_file():
        errors.append(f"index-file not found: {index_path}")

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1

    try:
        ds = BV2Dataset(data_root, index_path)
    except Exception as exc:
        print(f"FAIL: could not construct BV2Dataset: {exc}")
        return 1

    print(f"Index rows: {len(ds)}")
    expected_echo = (2, BV2_IMAGE_SIZE, BV2_IMAGE_SIZE)
    expected_depth = (1, BV2_IMAGE_SIZE, BV2_IMAGE_SIZE)

    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    check_n = min(args.num_samples, len(indices))

    for i in indices[:check_n]:
        row = ds.instances.iloc[i]
        if "audio_relpath" in row.index:
            audio_rel = row["audio_relpath"]
            depth_rel = row["depth_relpath"]
        else:
            audio_rel = str(Path(row["audio path"]) / row["audio file name"])
            depth_rel = str(Path(row["depth path"]) / row["depth file name"])

        audio_path = data_root / audio_rel
        depth_path = data_root / depth_rel
        if not audio_path.is_file():
            errors.append(f"missing wav: {audio_rel}")
        if not depth_path.is_file():
            errors.append(f"missing depth: {depth_rel}")

        try:
            sample = ds[i]
        except Exception as exc:
            errors.append(f"__getitem__({i}) failed: {exc}")
            continue

        echo = sample["echo_spectrogram"]
        depth = sample["depth"]
        depth_m = sample["depth_meters"]

        if tuple(echo.shape) != expected_echo:
            errors.append(f"sample {i}: echo shape {tuple(echo.shape)} != {expected_echo}")
        if tuple(depth.shape) != expected_depth:
            errors.append(f"sample {i}: depth shape {tuple(depth.shape)} != {expected_depth}")
        if float(depth_m.max()) > 30.0 + 1e-3:
            errors.append(f"sample {i}: depth_meters max {float(depth_m.max())} > 30m")

    if errors:
        print("RESULT: FAIL")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"RESULT: PASS ({check_n} samples checked, echo {expected_echo}, depth in meters OK)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
