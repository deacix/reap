# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""Collect the research suites only where their stack is installed.

The `legwork` branch installs a slim core (torch, transformers,
accelerate, safetensors, huggingface_hub); the upstream suites import
`reap.main` / `reap.data`, which pull vLLM, lm-eval and datasets (the
`research` extra). Without them those modules fail at import, so they
are left out of collection instead of erroring the whole run; the lane's
own suite under tests/legwork/ always runs.
"""

from __future__ import annotations

import importlib.util
import pathlib

_RESEARCH_MODULES = ("vllm", "lm_eval", "datasets", "scipy", "sklearn", "dotenv")
_HERE = pathlib.Path(__file__).parent

collect_ignore: list[str] = []
if any(importlib.util.find_spec(name) is None for name in _RESEARCH_MODULES):
    collect_ignore.extend(
        str(path.relative_to(_HERE)) for path in sorted(_HERE.glob("test_*.py"))
    )
