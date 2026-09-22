# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""The ``STAGE_PROGRESS`` line contract and small CLI helpers.

The desktop worker's stage runner reads stdout line by line and takes
``STAGE_PROGRESS <pct>`` (a whole or one-decimal percent, nothing else on
the line) as the stage's progress; every other line is log. Both lane
CLIs print through ``stage_progress`` so the contract lives in one place.
"""

from __future__ import annotations

import sys


def stage_progress(pct: float) -> None:
    pct = min(100.0, max(0.0, float(pct)))
    sys.stdout.write(f"STAGE_PROGRESS {pct:.1f}\n")
    sys.stdout.flush()


def parse_layer_list(spec: str) -> list[int]:
    """``"0,1,2"`` -> ``[0, 1, 2]``; blank -> ``[]``; refuses non-integers."""
    layers: list[int] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as error:
            raise ValueError(f"layer list entries must be integers, got {part!r}") from error
        if value < 0:
            raise ValueError(f"layer indices are non-negative, got {value}")
        layers.append(value)
    return sorted(set(layers))
