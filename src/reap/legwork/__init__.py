# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""The Legwork prune lane: a slim, importable REAP for transformers' fused
mixture-of-experts families, DeepSeek-V4 first.

Two stages, each a CLI that prints ``STAGE_PROGRESS <pct>`` lines the
desktop worker reads (its ``runStage`` contract) and a final
``REAP_RESULT {json}`` line:

* ``python -m reap.legwork.collect`` (``reap-collect``): run a calibration
  set through the model and record, per MoE layer and expert, the routed
  hit count, the summed router weight and the summed REAP saliency term
  ``weight * ||expert(x)||`` -> a router-stats file.
* ``python -m reap.legwork.prune`` (``reap-prune``): rank each layer's
  experts by that saliency, keep the top ``--keep`` per layer, slice the
  fused expert tensors and the router (remapping a hash-routed layer's
  ``tid2eid`` table onto the kept set), save the checkpoint with a
  ``reap_pruning`` record in its ``config.json``.

The lane imports only torch, transformers, accelerate, safetensors and
huggingface_hub — none of the research stack (vLLM, lm-eval, datasets)
the upstream entry points pull in.
"""

from reap.legwork.arch import MoeLayer, hash_routed_layers, model_attrs, moe_layers
from reap.legwork.observer import RouterStatsObserver, load_router_stats, save_router_stats
from reap.legwork.prune import PruneReport, prune_model, remap_hash_table, saliency, select_kept

__all__ = [
    "MoeLayer",
    "PruneReport",
    "RouterStatsObserver",
    "hash_routed_layers",
    "load_router_stats",
    "model_attrs",
    "moe_layers",
    "prune_model",
    "remap_hash_table",
    "saliency",
    "save_router_stats",
    "select_kept",
]
