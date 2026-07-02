"""Parse space-syntax radii from strings or lists."""

from __future__ import annotations

import re
from typing import List, Optional, Union

from streetrag.core.feature_catalog import FeatureCatalog

RadiiInput = Optional[Union[str, List[int]]]


def parse_radii(value: RadiiInput, *, default: Optional[List[int]] = None) -> List[int]:
    """Parse radii in meters from '500,1500,4500' or [500, 1500, 4500]."""
    fallback = default if default is not None else FeatureCatalog.DEFAULT_RADII
    if value is None or value == "":
        return list(fallback)
    if isinstance(value, list):
        out = [int(r) for r in value if int(r) > 0]
        return out or list(fallback)
    text = str(value).strip()
    if not text:
        return list(fallback)
    parts = re.split(r"[,;\s]+", text)
    out = [int(p) for p in parts if p.strip()]
    if not out:
        raise ValueError(f"Invalid radii: {value!r}")
    if any(r <= 0 for r in out):
        raise ValueError("Radii must be positive integers (meters)")
    return out


def format_radii(radii: List[int]) -> str:
    return ",".join(str(int(r)) for r in radii)
