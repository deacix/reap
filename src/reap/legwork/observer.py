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
the hook sees both. A loop-based family (MiMo-V2, a ``ModuleList`` of
experts) is hooked at its router instead: the router's input is the
hidden states, its ``(topk_idx, topk_weight)`` output the routing, and
each expert module is recomputed on its routed tokens the same way.

Stats version 2 adds, per layer, the saliency inputs split by calibration
source (``by_source``), a bounded sketch of the token ids each expert
serves most (``top_tokens``) and its best routed tokens in context
(``exemplars``); the last two read only non-private rows. Version 1 files
still load; they lack those keys and prune works from ``count`` and
``reap_sum`` alone.
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from reap.legwork.arch import MoeLayer, model_attrs, moe_layers

STATS_VERSION = 2
#: The router-stats versions ``load_router_stats`` reads.
READABLE_STATS_VERSIONS = (1, 2)
#: The source a sample belongs to when its row names none.
DEFAULT_SOURCE = "unlabeled"
#: Token ids the sketch reports per expert.
TOP_TOKENS = 16
#: Candidate token ids the sketch keeps per expert between merges; its
#: counts are exact while no expert sees more distinct tokens than this.
SKETCH_CAPACITY = 256
#: Exemplars kept per expert (at most one per sample).
EXEMPLARS = 3
#: Tokens of context an exemplar keeps on each side of its routed token.
EXEMPLAR_CONTEXT = 24


def routed_expert_norms(
    experts: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> Iterable[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield ``(expert, token_idx, weights, output_norms)`` for every expert
    with a routed token, mirroring the model's own expert computation: for
    fused experts the gate-up projection through the family's gate and the
    down projection; for a ``ModuleList`` the expert module itself. The L2
    norm per routed token is in float32; ``token_idx`` are the routed
    tokens' rows in the flattened batch."""
    loop = isinstance(experts, nn.ModuleList)
    num_experts = len(experts) if loop else int(experts.num_experts)
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    index = top_k_index.reshape(flat.shape[0], -1)
    weights = top_k_weights.reshape(flat.shape[0], -1)
    for expert in range(num_experts):
        hit = index == expert
        if not bool(hit.any()):
            continue
        token_idx, slot = torch.where(hit)
        x = flat[token_idx]
        if loop:
            out = experts[expert](x)
        else:
            gate_up = F.linear(x, experts.gate_up_proj[expert])
            apply_gate = getattr(experts, "_apply_gate", None)
            if apply_gate is None:
                gate, up = gate_up.chunk(2, dim=-1)
                act = F.silu(gate) * up
            else:
                act = apply_gate(gate_up)
            out = F.linear(act, experts.down_proj[expert])
        yield expert, token_idx, weights[token_idx, slot].float(), out.float().norm(dim=-1)


class TopTokenSketch:
    """The token ids routed to each expert most often, bounded and vectorized.

    ``add`` takes one forward pass's routing and runs one ``torch.unique``
    over the ``expert * vocab + token_id`` keys of its routed slots; the
    result queues. Once the queue holds about a table's worth of keys it
    merges into the table (a second ``unique``, the counts summed) and each
    expert's candidates are cut to its ``capacity`` most frequent. The
    counts are exact while no expert sees more than ``capacity`` distinct
    token ids; past that a cut token restarts from zero when it returns, so
    the report is a heavy-hitter estimate whose leaders stay exact unless
    they drop out of an expert's candidates.
    """

    def __init__(
        self, num_experts: int, vocab_size: int, capacity: int = SKETCH_CAPACITY, top: int = TOP_TOKENS
    ):
        if not 1 <= top <= capacity:
            raise ValueError(f"the sketch reports 1..capacity tokens, got top={top}, capacity={capacity}")
        self.num_experts = int(num_experts)
        self.vocab_size = int(vocab_size)
        self.capacity = int(capacity)
        self.top = int(top)
        self._keys: torch.Tensor | None = None
        self._counts: torch.Tensor | None = None
        self._queue: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._queued = 0

    def add(self, expert_index: torch.Tensor, token_ids: torch.Tensor) -> None:
        """Count one pass: ``expert_index`` is ``[tokens, top_k]`` (any id
        outside ``[0, num_experts)`` is skipped), ``token_ids`` ``[tokens]``."""
        experts = expert_index.reshape(token_ids.shape[0], -1).long()
        keys = experts * self.vocab_size + token_ids.to(experts.device).long().unsqueeze(1)
        keys = keys[(experts >= 0) & (experts < self.num_experts)]
        if keys.numel() == 0:
            return
        keys, counts = torch.unique(keys, return_counts=True)
        self._queue.append((keys, counts))
        self._queued += int(keys.numel())
        if self._queued >= self.num_experts * self.capacity:
            self._merge()

    def _merge(self) -> None:
        if not self._queue:
            return
        keys = [k for k, _ in self._queue]
        counts = [c for _, c in self._queue]
        if self._keys is not None:
            keys.insert(0, self._keys)
            counts.insert(0, self._counts)
        merged, inverse = torch.unique(torch.cat(keys), return_inverse=True)
        summed = torch.zeros(merged.numel(), dtype=torch.long, device=merged.device)
        summed.index_add_(0, inverse, torch.cat(counts))
        self._keys, self._counts = self._leaders(merged, summed, self.capacity)
        self._queue = []
        self._queued = 0

    def _leaders(
        self, keys: torch.Tensor, counts: torch.Tensor, limit: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each expert's ``limit`` most frequent keys, grouped by expert,
        most frequent first; equal counts keep the lower token id."""
        order = torch.argsort(keys)
        keys, counts = keys[order], counts[order]
        order = torch.argsort(-counts, stable=True)
        keys, counts = keys[order], counts[order]
        experts = torch.div(keys, self.vocab_size, rounding_mode="floor")
        order = torch.argsort(experts, stable=True)
        keys, counts, experts = keys[order], counts[order], experts[order]
        positions = torch.arange(keys.numel(), device=keys.device)
        starts = torch.ones_like(experts, dtype=torch.bool)
        starts[1:] = experts[1:] != experts[:-1]
        first = torch.cummax(torch.where(starts, positions, torch.zeros_like(positions)), dim=0).values
        keep = (positions - first) < limit
        return keys[keep], counts[keep]

    def top_tokens(self) -> list[list[list[int]]]:
        """Per expert (by id), its ``top`` most frequent ``[token_id, count]``
        pairs, most frequent first; equal counts keep the lower token id."""
        self._merge()
        report: list[list[list[int]]] = [[] for _ in range(self.num_experts)]
        if self._keys is None:
            return report
        keys, counts = self._leaders(self._keys, self._counts, self.top)
        for key, count in zip(keys.tolist(), counts.tolist()):
            expert, token = divmod(key, self.vocab_size)
            report[expert].append([token, count])
        return report


class ExemplarTable:
    """Each expert's best routed tokens in context, vectorized over experts.

    ``offer`` takes one sample's candidates — per expert its routed token
    with the highest ``weight * ||output||`` — and keeps each expert's
    ``keep`` best across samples, so at most one per sample; equal scores
    keep the earlier sample. Every entry carries the ``±context`` window of
    token ids around its routed position (``-1`` pads past the sample's
    edges; the routed token sits at index ``context``).
    """

    def __init__(
        self,
        num_experts: int,
        device: torch.device | str | None = None,
        keep: int = EXEMPLARS,
        context: int = EXEMPLAR_CONTEXT,
    ):
        self.keep = int(keep)
        self.context = int(context)
        shape = (int(num_experts), self.keep)
        self.score = torch.full(shape, -math.inf, dtype=torch.float64, device=device)
        self.sample = torch.full(shape, -1, dtype=torch.long, device=device)
        self.source = torch.full(shape, -1, dtype=torch.long, device=device)
        self.position = torch.full(shape, -1, dtype=torch.long, device=device)
        self.window = torch.full((*shape, 2 * self.context + 1), -1, dtype=torch.long, device=device)

    def offer(
        self,
        scores: torch.Tensor,
        positions: torch.Tensor,
        token_ids: torch.Tensor,
        sample: int,
        source: int,
    ) -> None:
        """One sample's candidates, one per expert: its best ``scores``
        ``[num_experts]`` and their routed ``positions`` ``[num_experts]`` in
        ``token_ids``. A position outside the sample (``-1``) marks an expert
        without a candidate."""
        device = self.score.device
        ids = token_ids.to(device)
        position = positions.to(device=device, dtype=torch.long)
        offered = (position >= 0) & (position < ids.numel())
        position = torch.where(offered, position, torch.full_like(position, -1))
        score = torch.where(
            offered,
            scores.to(device=device, dtype=torch.float64),
            torch.full_like(position, -math.inf, dtype=torch.float64),
        )
        where = position.unsqueeze(1) + torch.arange(-self.context, self.context + 1, device=device)
        inside = (where >= 0) & (where < ids.numel()) & offered.unsqueeze(1)
        window = torch.where(inside, ids[where.clamp(0, ids.numel() - 1)], torch.full_like(where, -1))
        candidates = {
            "sample": torch.where(offered, torch.full_like(position, sample), position),
            "source": torch.where(offered, torch.full_like(position, source), position),
            "position": position,
        }
        scores_now = torch.cat([self.score, score.unsqueeze(1)], dim=1)
        order = torch.argsort(-scores_now, dim=1, stable=True)[:, : self.keep]
        self.score = scores_now.gather(1, order)
        for name, candidate in candidates.items():
            both = torch.cat([getattr(self, name), candidate.unsqueeze(1)], dim=1)
            setattr(self, name, both.gather(1, order))
        both = torch.cat([self.window, window.unsqueeze(1)], dim=1)
        self.window = both.gather(1, order.unsqueeze(-1).expand(-1, -1, both.shape[-1]))

    def entries(self, source_ids: Sequence[str]) -> list[list[dict[str, Any]]]:
        """Per expert (by id), its exemplars best first: ``source``,
        ``sample`` (the sample's serial), ``position`` (the routed token's
        position in the sample), ``window`` (the token ids around it) and
        ``offset`` (the routed token's index in ``window``), ``score``."""
        score, sample, source, position, window = (
            t.cpu().tolist() for t in (self.score, self.sample, self.source, self.position, self.window)
        )
        report: list[list[dict[str, Any]]] = []
        for e in range(len(score)):
            rows = []
            for j in range(self.keep):
                if sample[e][j] < 0:
                    continue
                ids = window[e][j]
                before = [t for t in ids[: self.context] if t >= 0]
                after = [t for t in ids[self.context + 1 :] if t >= 0]
                rows.append(
                    {
                        "source": source_ids[source[e][j]],
                        "sample": sample[e][j],
                        "position": position[e][j],
                        "window": before + [ids[self.context]] + after,
                        "offset": len(before),
                        "score": score[e][j],
                    }
                )
            report.append(rows)
        return report


@dataclass
class _Sample:
    """The sample the next forward pass runs (``begin_sample``)."""

    ids: torch.Tensor
    source: str
    private: bool
    serial: int
    layers: set[int] = field(default_factory=set)
    _on: dict[torch.device, torch.Tensor] = field(default_factory=dict)

    def ids_on(self, device: torch.device) -> torch.Tensor:
        if device not in self._on:
            self._on[device] = self.ids.to(device)
        return self._on[device]


class RouterStatsObserver:
    """Hook a model's fused MoE layers and accumulate the saliency terms.

    ``state()`` returns a plain dict (``save_router_stats`` writes it with
    ``torch.save``). Per layer: ``count`` (routed hits per expert),
    ``weight_sum`` (summed router weight), ``reap_sum`` (summed
    ``weight * ||output||``), ``max_norm``, the layer's ``tokens``;
    ``by_source[source] = {count, reap_sum}`` (the same terms per source,
    summing to the pooled ones); ``top_tokens`` (per expert, at most
    ``TOP_TOKENS`` ``[token_id, count]`` pairs) and ``exemplars`` (per
    expert, at most ``EXEMPLARS`` entries, ``ExemplarTable.entries``), both
    fed by non-private rows only. Top level: ``sources``, every source's
    ``{id, private, rows, tokens}``, plus ``experts``, ``top_k`` and
    ``vocab_size``.

    Call ``begin_sample(ids, source=..., private=...)`` before each forward
    pass; a pass run without it feeds the pooled terms only (count it with
    ``note_sample``).
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
        config = model.config
        self.num_experts = int(getattr(config, self.attrs["num_experts"]))
        self.top_k = int(getattr(config, self.attrs["num_experts_per_tok"]))
        embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        self.vocab_size = int(
            getattr(embeddings, "num_embeddings", None) or getattr(config, "vocab_size")
        )
        self._state: dict[int, dict[str, Any]] = {}
        self._sketches: dict[int, TopTokenSketch] = {}
        self._exemplars: dict[int, ExemplarTable] = {}
        self._sources: dict[str, dict[str, Any]] = {}
        self._source_ids: list[str] = []
        self._sample: _Sample | None = None
        self._hooks = [
            layer.experts.register_forward_hook(self._hook(layer))
            if layer.fused
            else layer.router.register_forward_hook(self._router_hook(layer))
            for layer in self.layers
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
                "by_source": {},
            }
            self._state[layer.index] = state
        return state

    def _hook(self, layer: MoeLayer):
        """A fused experts module's hook: the routing arrives as its arguments."""

        @torch.no_grad()
        def hook(module: nn.Module, args: tuple, output: Any) -> None:
            if len(args) < 3:
                raise ValueError(
                    f"layer {layer.index}: the experts module was called with {len(args)} "
                    "positional arguments; the lane expects (hidden_states, top_k_index, top_k_weights)"
                )
            self._observe(layer, module, args[0], args[1], args[2])

        return hook

    def _router_hook(self, layer: MoeLayer):
        """A loop-based layer's hook on its router (MiMo-V2's ``mlp.gate``):
        the hidden states are its input, the routing its ``(topk_idx,
        topk_weight)`` output, and each expert is recomputed on its tokens."""

        @torch.no_grad()
        def hook(module: nn.Module, args: tuple, output: Any) -> None:
            if not args or not isinstance(output, tuple) or len(output) < 2:
                raise ValueError(
                    f"layer {layer.index}: the router's call carried no hidden states or "
                    "returned no (topk_idx, topk_weight)"
                )
            self._observe(layer, layer.experts, args[0], output[0], output[1])

        return hook

    def _observe(
        self,
        layer: MoeLayer,
        experts: nn.Module,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> None:
        """Accumulate one forward pass's routing of ``layer``."""
        tokens = int(hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0])
        sample = self._sample
        if sample is not None:
            if int(sample.ids.numel()) != tokens:
                raise ValueError(
                    f"layer {layer.index}: the forward pass routed {tokens} tokens, "
                    f"begin_sample named {int(sample.ids.numel())}"
                )
            if layer.index in sample.layers:
                raise ValueError(
                    f"layer {layer.index}: a second forward pass for one sample; "
                    "call begin_sample before each forward"
                )
            sample.layers.add(layer.index)
        state = self._layer_state(layer, hidden_states.device)
        state["tokens"] += tokens
        track = sample is not None and not sample.private
        hit: list[int] = []
        counts: list[int] = []
        weight_sums, reap_sums, norm_parts, row_parts, contribution_parts = [], [], [], [], []
        for expert, token_idx, weights, norms in routed_expert_norms(
            experts, hidden_states, top_k_index, top_k_weights
        ):
            # float32 x float32 is exact in float64; the sums stay per
            # expert so a GPU run adds in a fixed order.
            contribution = weights.double() * norms.double()
            hit.append(expert)
            counts.append(int(norms.numel()))
            weight_sums.append(weights.double().sum())
            reap_sums.append(contribution.sum())
            norm_parts.append(norms)
            if track:
                row_parts.append(token_idx)
                contribution_parts.append(contribution)
        if not hit:
            return
        device = state["count"].device
        n = layer.num_experts
        experts = torch.tensor(hit, dtype=torch.long, device=device)
        slot_expert = torch.repeat_interleave(
            experts, torch.tensor(counts, dtype=torch.long, device=device)
        )
        count = torch.zeros_like(state["count"])
        count[experts] = torch.tensor(counts, dtype=torch.long, device=device)
        reap = torch.zeros_like(state["reap_sum"])
        reap[experts] = torch.stack(reap_sums).to(device)
        state["count"] += count
        state["reap_sum"] += reap
        state["weight_sum"][experts] += torch.stack(weight_sums).to(device)
        max_norm = torch.zeros_like(state["max_norm"]).scatter_reduce_(
            0, slot_expert, torch.cat(norm_parts).to(device), "amax", include_self=False
        )
        state["max_norm"] = torch.maximum(state["max_norm"], max_norm)
        if sample is None:
            return
        per_source = state["by_source"].get(sample.source)
        if per_source is None:
            per_source = {"count": torch.zeros_like(count), "reap_sum": torch.zeros_like(reap)}
            state["by_source"][sample.source] = per_source
        per_source["count"] += count
        per_source["reap_sum"] += reap
        if not track:
            return
        ids = sample.ids_on(device)
        sketch = self._sketches.get(layer.index)
        if sketch is None:
            sketch = self._sketches[layer.index] = TopTokenSketch(layer.num_experts, self.vocab_size)
        sketch.add(top_k_index.to(device), ids)
        # Per expert, the routed token with the highest contribution (the
        # first such row on a tie); max and min are exact in any order.
        contribution = torch.cat(contribution_parts).to(device)
        rows = torch.cat(row_parts).to(device)
        best = torch.full((n,), -math.inf, dtype=torch.float64, device=device)
        best.scatter_reduce_(0, slot_expert, contribution, "amax", include_self=False)
        at_best = contribution == best[slot_expert]
        position = torch.full((n,), tokens, dtype=torch.long, device=device)
        position.scatter_reduce_(0, slot_expert[at_best], rows[at_best], "amin")
        table = self._exemplars.get(layer.index)
        if table is None:
            table = self._exemplars[layer.index] = ExemplarTable(n, device)
        table.offer(
            best,
            torch.where(count > 0, position, torch.full_like(position, -1)),
            ids,
            sample=sample.serial,
            source=self._source_ids.index(sample.source),
        )

    def begin_sample(
        self,
        ids: Sequence[int] | torch.Tensor,
        source: str = DEFAULT_SOURCE,
        private: bool = False,
    ) -> None:
        """Name the sample the next forward pass runs — its token ids, its
        source and whether it is private — and count it. A private sample
        feeds the pooled and per-source terms but never the sketch or the
        exemplars. A source is private once any of its rows is."""
        ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1).cpu()
        entry = self._sources.get(source)
        if entry is None:
            entry = self._sources[source] = {"id": source, "private": False, "rows": 0, "tokens": 0}
            self._source_ids.append(source)
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
        self._sample = None

    def state(self) -> dict[str, Any]:
        layers: dict[int, dict[str, Any]] = {}
        for index, s in sorted(self._state.items()):
            n = int(s["num_experts"])
            entry = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in s.items()}
            entry["by_source"] = {
                source: {k: v.cpu() for k, v in terms.items()}
                for source, terms in sorted(s["by_source"].items())
            }
            sketch = self._sketches.get(index)
            entry["top_tokens"] = sketch.top_tokens() if sketch else [[] for _ in range(n)]
            table = self._exemplars.get(index)
            entry["exemplars"] = table.entries(self._source_ids) if table else [[] for _ in range(n)]
            layers[index] = entry
        return {
            "version": STATS_VERSION,
            "model_class": self.model.__class__.__name__,
            "model_type": getattr(self.model.config, "model_type", None),
            "experts": self.num_experts,
            "top_k": self.top_k,
            "vocab_size": self.vocab_size,
            "samples": self.samples,
            "sketch": {
                "top_tokens": TOP_TOKENS,
                "capacity": SKETCH_CAPACITY,
                "exemplars": EXEMPLARS,
                "context": EXEMPLAR_CONTEXT,
            },
            "sources": [dict(entry) for _, entry in sorted(self._sources.items())],
            "layers": layers,
        }


def save_router_stats(state: dict[str, Any], path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path


def load_router_stats(path: str | pathlib.Path) -> dict[str, Any]:
    """A router-stats file of version 1 or 2 (a v1 file lacks ``sources``,
    ``by_source``, ``top_tokens`` and ``exemplars``)."""
    state = torch.load(pathlib.Path(path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("version") not in READABLE_STATS_VERSIONS:
        versions = " or ".join(str(v) for v in READABLE_STATS_VERSIONS)
        raise ValueError(f"{path} is not a Legwork router-stats file (version {versions})")
    return state
