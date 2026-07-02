"""Run space syntax for a city catalog."""

from __future__ import annotations

from typing import Callable, List, Optional

from streetrag.core.feature_catalog import FeatureCatalog
from streetrag.core.street_network import StreetNetwork
from streetrag.syntax.engine import compute_syntax
from streetrag.utils.radii import RadiiInput, parse_radii

ProgressFn = Optional[Callable[[str, dict], None]]


def _emit(cb: ProgressFn, step: str, **detail) -> None:
    if cb:
        try:
            cb(step, detail)
        except Exception:
            pass


def apply_syntax_radii(catalog: FeatureCatalog, radii: RadiiInput) -> List[int]:
    parsed = parse_radii(radii, default=catalog.syntax_radii or FeatureCatalog.DEFAULT_RADII)
    catalog.set_syntax_radii(parsed)
    catalog.save()
    return parsed


def run_syntax_for_catalog(
    catalog: FeatureCatalog,
    *,
    radii: RadiiInput = None,
    on_progress: ProgressFn = None,
) -> dict:
    """Set radii (optional) and compute space syntax for the active city network."""
    parsed = apply_syntax_radii(catalog, radii)
    _emit(
        on_progress,
        "syntax",
        phase="start",
        message=f"Computing space syntax, radii {parsed} m…",
        radii=parsed,
    )
    net = StreetNetwork.from_catalog(catalog)
    n_before = sum(
        1
        for c in net.edges.columns
        if c.startswith(("integration_R", "angular_", "nain_", "choice_", "nach_"))
    )
    net = compute_syntax(net, radii=parsed)
    net.save()
    catalog.save()
    n_after = sum(
        1
        for c in net.edges.columns
        if c.startswith(("integration_R", "angular_", "nain_", "choice_", "nach_"))
    )
    _emit(
        on_progress,
        "syntax",
        phase="done",
        message=f"Syntax complete, {n_after} columns",
        radii=parsed,
        columns_added=max(0, n_after - n_before),
    )
    return {"radii": parsed, "syntax_columns": n_after, "columns_added": max(0, n_after - n_before)}
