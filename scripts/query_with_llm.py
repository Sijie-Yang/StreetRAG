#!/usr/bin/env python3
"""Thin wrapper: natural-language query (legacy CLI compatible)."""
import sys

from streetrag.cli import main

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        sys.argv = ["streetrag", "ask", sys.argv[2], "--registry", sys.argv[1]] + sys.argv[3:]
    elif len(sys.argv) == 2:
        sys.argv = ["streetrag", "ask", sys.argv[1]]
    else:
        sys.argv = ["streetrag", "ask", "--help"]
    main()
