#!/usr/bin/env python
"""Training entry point placeholder for HTF-EchoDepth BV2."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HTF-EchoDepth on BV2")
    parser.add_argument("--config", default="configs/bv2/train_htf_echodepth_bv2.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--index-dir", default="data/bv2_index")
    parser.add_argument("--output-dir", default="outputs/bv2_run")
    parser.parse_args()
    print("HTF-EchoDepth BV2 training code will be released soon.")


if __name__ == "__main__":
    main()
