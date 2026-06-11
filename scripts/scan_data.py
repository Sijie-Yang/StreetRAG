#!/usr/bin/env python3
"""Thin wrapper: scan data directory (legacy CLI compatible)."""
import argparse
import sys

from streetrag.cli import main

if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data-dir", default="data")
    args, _ = p.parse_known_args()
    sys.argv = ["streetrag", "scan", "--data-dir", args.data_dir]
    main()
