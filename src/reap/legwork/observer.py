# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""The router-stats observer: REAP's saliency inputs, per layer and expert.

REAP ranks expert ``e`` of a layer by the mean, over the tokens routed to
it, of the router weight times the norm of the expert's own output:
``mean_{t -> e} g[t, e] * ||f_e(x_t)||`` (arXiv 2510.13999, the
``reap`` metric of ``reap.pruning_metrics``). The research observer
computes every expert on every token to serve the merging methods too;
this lane only prunes, so it hooks the fused experts module, reads the
routing the model actually applied — ``(hidden_states, top_k_index,
top_k_weights)`` — and recomputes each expert on its routed tokens only.
A hash-routed layer (DeepSeek-V4's ``tid2eid`` bootstrap layers) needs
nothing special: its table picks the indices, its gate the weights, and
the hook sees both.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from reap.legwork.arch import MoeLayer, model_attrs, moe_layers

STATS_VERSION = 1
#: The source a sample belongs to when its row names none.
DEFAULT_SOURCE = "unlabeled"


def routed_expert_norms(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> Iterable[tuple[int, torch.Tensor, torch.Tensor]]:
    """Yield ``(expert, weights, output_norms)`` for every expert with a
    routed token, mirroring the fused module's eager loop: the expert's
    gate-up projection through the family's gate, the down projection, the
    L2 norm per routed token, in float32."""
    num_experts = int(experts.num_experts)
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    index = top_k_index.reshape(flat.shape[0], -1)
    weights = top_k_weights.reshape(flat.shape[0], -1)
    for expert in range(num_experts):
        hit = index == expert
        if not bool(hit.any()):
            continue
        token_idx, slot = torch.where(hit)
        x = flat[token_idx]
        gate_up = F.linear(x, experts.gate_up_proj[expert])
        apply_gate = getattr(experts, "_apply_gate", None)
        if apply_gate is None:
            gate, up = gate_up.chunk(2, dim=-1)
            act = F.silu(gate) * up
        else:
            act = apply_gate(gate_up)
        out = F.linear(act, experts.down_proj[expert])
        yield expert, weights[token_idx, slot].float(), out.float().norm(dim=-1)


@dataclass
class _Sample:
    """The sample the next forward pass runs (``begin_sample``)."""

    ids: torch.Tensor
    source: str
    private: bool
    serial: int


class RouterStatsObserver:
    """Hook a model's fused MoE layers and accumulate the saliency terms.

    ``state()`` returns a plain dict (``save_router_stats`` writes it with
    ``torch.save``): per layer ``count`` (routed hits per expert),
    ``weight_sum`` (summed router weight), ``reap_sum`` (summed
    ``weight * ||output||``), ``max_norm`` and the layer's ``tokens``; top
    level ``sources``, every source's ``{id, private, rows, tokens}``.

    Call ``begin_sample(ids, source=..., private=...)`` before each forward
    pass; a pass run without it only counts (``note_sample``).
    """

    def __init__(self, model: nn.Module, layers: Iterable[int] | None = None):
        self.model = model
        self.attrs = model_attrs(model)
        wanted = None if layers is None else set(int(i) for i in layers)
        self.layers: list[MoeLayer] = [
            layer for layer in moe_layers(model) if wanted is None or layer.index in wanted
        ]
        if not self.layers:
            raise ValueError("no MoE layer selected for observation")
        self._state: dict[int, dict[str, Any]] = {}
        self._sources: dict[str, dict[str, Any]] = {}
        self._sample: _Sample | None = None
        self._hooks = [
            layer.experts.register_forward_hook(self._hook(layer)) for layer in self.layers
        ]
        self.samples = 0

    def _layer_state(self, layer: MoeLayer, device: torch.device) -> dict[str, Any]:
        state = self._state.get(layer.index)
        if state is None:
            n = layer.num_experts
            state = {
                "kind": layer.kind,
                "num_experts": n,
                "tokens": 0,
                "count": torch.zeros(n, dtype=torch.long, device=device),
                "weight_sum": torch.zeros(n, dtype=torch.float64, device=device),
                "reap_sum": torch.zeros(n, dtype=torch.float64, device=device),
                "max_norm": torch.zeros(n, dtype=torch.float32, device=device),
            }
            self._state[layer.index] = state
        return state

    def _hook(self, layer: MoeLayer):
        @torch.no_grad()
        def hook(module: nn.Module, args: tuple, output: Any) -> None:
            if len(args) < 3:
                raise ValueError(
                    f"layer {layer.index}: the experts module was called with {len(args)} "
                    "positional arguments; the lane expects (hidden_states, top_k_index, top_k_weights)"
                )
            hidden_states, top_k_index, top_k_weights = args[0], args[1], args[2]
            state = self._layer_state(layer, hidden_states.device)
            state["tokens"] += int(hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0])
            for expert, weights, norms in routed_expert_norms(
                module, hidden_states, top_k_index, top_k_weights
            ):
                state["count"][expert] += norms.numel()
                state["weight_sum"][expert] += weights.sum().double()
                state["reap_sum"][expert] += (weights * norms).sum().double()
                state["max_norm"][expert] = torch.maximum(state["max_norm"][expert], norms.max())

        return hook

    def begin_sample(
        self,
        ids: Sequence[int] | torch.Tensor,
        source: str = DEFAULT_SOURCE,
        private: bool = False,
    ) -> None:
        """Name the sample the next forward pass runs — its token ids, its
        source and whether it is private — and count it. A source is private
        once any of its rows is."""
        ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1).cpu()
        entry = self._sources.setdefault(
            source, {"id": source, "private": False, "rows": 0, "tokens": 0}
        )
        entry["private"] = entry["private"] or bool(private)
        entry["rows"] += 1
        entry["tokens"] += int(ids.numel())
        self._sample = _Sample(ids=ids, source=source, private=bool(private), serial=self.samples)
        self.samples += 1

    def note_sample(self, n: int = 1) -> None:
        """Count ``n`` samples run without ``begin_sample``."""
        self.samples += n

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def state(self) -> dict[str, Any]:
        layers = {
            index: {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in s.items()}
            for index, s in sorted(self._state.items())
        }
        return {
            "version": STATS_VERSION,
            "model_class": self.model.__class__.__name__,
            "model_type": getattr(self.model.config, "model_type", None),
            "samples": self.samples,
            "sources": [dict(entry) for _, entry in sorted(self._sources.items())],
            "layers": layers,
        }


def save_router_stats(state: dict[str, Any], path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path


def load_router_stats(path: str | pathlib.Path) -> dict[str, Any]:
    state = torch.load(pathlib.Path(path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("version") != STATS_VERSION:
        raise ValueError(f"{path} is not a Legwork router-stats file (version {STATS_VERSION})")
    return state
