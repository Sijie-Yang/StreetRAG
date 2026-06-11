#!/usr/bin/env python3
"""Thin wrapper: integrate features (legacy CLI compatible)."""
import sys

from streetrag.cli import main

if __name__ == "__main__":
    registry = sys.argv[1] if len(sys.argv) > 1 else "data/feature_registry.json"
    sys.argv = ["streetrag", "integrate", registry]
    main()
