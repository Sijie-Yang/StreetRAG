#!/usr/bin/env python3
"""Thin wrapper: download city network (legacy CLI compatible)."""
import sys

from streetrag.cli import main

if __name__ == "__main__":
    sys.argv = ["streetrag", "download"] + sys.argv[1:]
    main()
