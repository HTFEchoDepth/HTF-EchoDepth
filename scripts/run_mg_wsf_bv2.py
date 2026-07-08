#!/usr/bin/env python
"""MG-WSF entry point placeholder for HTF-EchoDepth BV2."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MG-WSF for HTF-EchoDepth BV2")
    parser.add_argument("--config", default="configs/bv2/mg_wsf_bv2.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--index-dir", default="data/bv2_index")
    parser.add_argument("--candidate-registry", default=None)
    parser.add_argument("--output-dir", default="outputs/bv2_run/fused")
    parser.add_argument("--save-fused-checkpoint", action="store_true")
    parser.parse_args()
    print("HTF-EchoDepth MG-WSF code will be released soon.")


if __name__ == "__main__":
    main()
