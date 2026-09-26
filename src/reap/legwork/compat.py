# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""transformers-version shims for a checkpoint's own model code.

MiMo-V2's ``modeling_mimo_v2.py`` (XiaomiMiMo/MiMo-V2.6-*, written for
transformers 5.3) calls ``create_causal_mask`` and
``create_sliding_window_causal_mask`` with ``input_embeds=`` and
``cache_position=``. transformers 5.17 (the lane's pin) names the first
``inputs_embeds`` and takes no ``cache_position``, so the forward raises a
``TypeError`` before the first layer. ``patch_remote_code(model)`` wraps
those two names in each module a trust_remote_code class came from; a
call that already fits the installed signature passes through unchanged.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any, Callable

import torch.nn as nn

#: The names a remote module imports from ``transformers.masking_utils``.
MASK_FUNCTIONS = ("create_causal_mask", "create_sliding_window_causal_mask")
#: Where transformers puts the classes it loads through trust_remote_code.
REMOTE_MODULE_PREFIX = "transformers_modules"


def adapt_mask_function(function: Callable[..., Any]) -> Callable[..., Any]:
    """``function`` taking the 5.3 keywords too: ``input_embeds`` as
    ``inputs_embeds``, and a ``cache_position`` it has no parameter for dropped."""
    parameters = inspect.signature(function).parameters

    def call(*args: Any, **kwargs: Any) -> Any:
        if "input_embeds" in kwargs and "input_embeds" not in parameters:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        if "cache_position" in kwargs and "cache_position" not in parameters:
            kwargs.pop("cache_position")
        return function(*args, **kwargs)

    call.__legwork_adapted__ = True  # type: ignore[attr-defined]
    return call


def patch_remote_code(model: nn.Module) -> list[str]:
    """Adapt the mask functions of every remote module ``model``'s classes
    live in; returns the ``module.function`` names it wrapped."""
    patched: list[str] = []
    seen: set[str] = set()
    for submodule in model.modules():
        name = type(submodule).__module__
        if name in seen or not name.startswith(REMOTE_MODULE_PREFIX):
            continue
        seen.add(name)
        module = sys.modules.get(name)
        if module is None:
            continue
        for function_name in MASK_FUNCTIONS:
            function = getattr(module, function_name, None)
            if function is None or getattr(function, "__legwork_adapted__", False):
                continue
            setattr(module, function_name, adapt_mask_function(function))
            patched.append(f"{name}.{function_name}")
    return patched
