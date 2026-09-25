# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""Prune a fused-experts checkpoint to ``--keep`` routed experts per layer.

Every MoE layer is pruned to the same width because both stock loaders
build every layer from one ``n_routed_experts`` (transformers'
``DeepseekV4Experts(config)``, vLLM's ``FusedMoE``). A hash-routed layer
(DeepSeek-V4's ``tid2eid`` bootstrap layers) is pruned too: the table
entries of a dropped expert are redirected to the nearest kept expert by
the cosine similarity of the router rows — the method the
``0xSero/DeepSeek-V4-Flash-0731-REAP`` transfer proof used — and the
table is renumbered onto the kept set. ``--skip-layers`` leaves the named
layers at full width; that checkpoint is *ragged*, reloads only through
``reap.legwork.load.load_pruned`` and does not serve on vLLM, so the CLI
says so and records the per-layer widths in ``reap_pruning``.

CLI::

    python -m reap.legwork.prune --model <dir> --stats <router-stats.pt> \
        --out <dir> --keep 192 [--skip-layers 0,1,2] [--method reap]

Prints ``STAGE_PROGRESS <pct>`` lines and a final ``REAP_RESULT {json}``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from reap.legwork.arch import HASH_KIND, MoeLayer, model_attrs, moe_layers, num_routed_experts
from reap.legwork.observer import load_router_stats
from reap.legwork.progress import parse_layer_list, stage_progress

METHODS = ("reap", "frequency", "weighted_frequency")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "added_tokens.json",
)


def saliency(layer_stats: dict[str, Any], method: str = "reap") -> torch.Tensor:
    """Per-expert scores (float64) from one layer's router stats; an expert
    no calibration token reached scores 0 under every method."""
    count = layer_stats["count"].double()
    if method == "reap":
        return layer_stats["reap_sum"].double() / count.clamp(min=1)
    if method == "frequency":
        return count
    if method == "weighted_frequency":
        return layer_stats["weight_sum"].double()
    raise ValueError(f"unknown prune method {method!r}; one of {METHODS}")


def select_kept(scores: torch.Tensor, keep: int) -> torch.Tensor:
    """The ``keep`` highest-scoring expert ids, ascending; ties keep the
    lower id (a stable sort, so two runs agree)."""
    num_experts = int(scores.numel())
    if not 1 <= keep <= num_experts:
        raise ValueError(f"keep must be between 1 and {num_experts}, got {keep}")
    order = torch.argsort(-scores.double().cpu(), stable=True)
    return order[:keep].sort().values


def remap_hash_table(
    tid2eid: torch.Tensor, kept: torch.Tensor, router_weight: torch.Tensor
) -> torch.Tensor:
    """Renumber a ``[vocab, top_k]`` token-id -> expert-id table onto ``kept``.

    A kept expert maps to its position in ``kept``; a dropped expert's
    entries go to the kept expert whose router row is nearest by cosine
    similarity, skipping experts already present in the same row so a
    token never lists one expert twice (the next-nearest fills the slot).
    """
    device = tid2eid.device
    kept = kept.to(device=device, dtype=torch.long)
    rows = F.normalize(router_weight.detach().float().to(device), dim=-1)
    similarity = rows @ rows[kept].T  # [experts, keep]
    # A kept expert ranks itself first (cosine 1 with its own row); the
    # `+ 2` keeps that ahead of any other row even under float rounding.
    own = torch.full_like(similarity, 0.0)
    own[kept, torch.arange(kept.numel(), device=device)] = 2.0
    ranked = torch.argsort(similarity + own, dim=-1, descending=True)  # [experts, keep]
    table = tid2eid.long()
    vocab, top_k = table.shape
    used = torch.zeros((vocab, kept.numel()), dtype=torch.bool, device=device)
    out = torch.empty_like(table)
    for slot in range(top_k):
        candidates = ranked[table[:, slot]]  # [vocab, keep], best first
        available = ~used.gather(1, candidates)
        first = available.to(torch.int8).argmax(dim=1)
        choice = candidates.gather(1, first.unsqueeze(1)).squeeze(1)
        out[:, slot] = choice
        used[torch.arange(vocab, device=device), choice] = True
    return out.to(tid2eid.dtype)


def prune_layer(layer: MoeLayer, kept: torch.Tensor, attrs: dict) -> None:
    """Slice one layer's fused experts and router down to ``kept`` (in place)."""
    experts, router = layer.experts, layer.router
    device = experts.gate_up_proj.device
    kept = kept.to(device)
    hash_attr = attrs.get("hash_table")
    table = getattr(router, hash_attr, None) if hash_attr else None
    remapped = None
    if table is not None:
        remapped = remap_hash_table(table, kept, router.weight)
    experts.gate_up_proj = nn.Parameter(
        experts.gate_up_proj.data[kept].clone(), requires_grad=experts.gate_up_proj.requires_grad
    )
    experts.down_proj = nn.Parameter(
        experts.down_proj.data[kept].clone(), requires_grad=experts.down_proj.requires_grad
    )
    experts.num_experts = int(kept.numel())
    router.weight = nn.Parameter(
        router.weight.data[kept].clone(), requires_grad=router.weight.requires_grad
    )
    if hasattr(router, "bias") and isinstance(router.bias, torch.Tensor):
        router.bias = nn.Parameter(router.bias.data[kept].clone())
    if hasattr(router, "e_score_correction_bias"):
        router.e_score_correction_bias = router.e_score_correction_bias[kept].clone()
    if remapped is not None:
        setattr(router, hash_attr, remapped)
    if hasattr(router, "num_experts"):
        router.num_experts = int(kept.numel())
    if hasattr(router, "out_features"):
        router.out_features = int(kept.numel())


@dataclass
class PrunedLayer:
    index: int
    kind: str
    experts_before: int
    kept: list[int]
    skipped: bool = False


@dataclass
class PruneReport:
    method: str
    keep: int
    experts_before: int
    layers: list[PrunedLayer] = field(default_factory=list)
    skipped_layers: list[int] = field(default_factory=list)

    @property
    def n_routed_experts_per_layer(self) -> list[int]:
        return [len(layer.kept) for layer in self.layers]

    @property
    def ragged(self) -> bool:
        return len(set(self.n_routed_experts_per_layer)) > 1

    def to_record(self) -> dict[str, Any]:
        return {
            "tool": "reap",
            "lane": "legwork",
            "method": self.method,
            "keep": self.keep,
            "experts_before": self.experts_before,
            "skipped_layers": list(self.skipped_layers),
            "hash_routed_layers": [l.index for l in self.layers if l.kind == HASH_KIND],
            "ragged": self.ragged,
            "n_routed_experts_per_layer": self.n_routed_experts_per_layer,
            "layers": [asdict(layer) for layer in self.layers],
        }


def prune_model(
    model: nn.Module,
    stats: dict[str, Any],
    keep: int,
    skip_layers: Any = (),
    method: str = "reap",
    progress: Callable[[float], None] | None = None,
) -> PruneReport:
    """Prune ``model`` in place; returns the report the checkpoint records."""
    attrs = model_attrs(model)
    layers = moe_layers(model)
    experts_before = num_routed_experts(model)
    top_k = int(getattr(model.config, attrs["num_experts_per_tok"]))
    if not 1 <= keep <= experts_before:
        raise ValueError(f"--keep must be between 1 and {experts_before}, got {keep}")
    if keep < top_k:
        raise ValueError(f"--keep {keep} is below the router's top-k {top_k}")
    skipped = sorted(set(int(i) for i in skip_layers))
    known = {layer.index for layer in layers}
    unknown = [i for i in skipped if i not in known]
    if unknown:
        raise ValueError(f"--skip-layers names layers without an MoE block: {unknown}")
    if method not in METHODS:
        raise ValueError(f"unknown prune method {method!r}; one of {METHODS}")
    layer_stats = stats.get("layers", {})
    report = PruneReport(method=method, keep=keep, experts_before=experts_before, skipped_layers=skipped)
    total = len(layers)
    for position, layer in enumerate(layers):
        if layer.index in skipped:
            report.layers.append(
                PrunedLayer(layer.index, layer.kind, layer.num_experts, list(range(layer.num_experts)), True)
            )
        else:
            this = layer_stats.get(layer.index)
            if this is None:
                raise ValueError(f"the router stats carry no layer {layer.index}; re-run collect")
            if int(this["num_experts"]) != layer.num_experts:
                raise ValueError(
                    f"layer {layer.index}: the stats were recorded over {this['num_experts']} experts, "
                    f"the model has {layer.num_experts}"
                )
            kept = select_kept(saliency(this, method), keep)
            prune_layer(layer, kept, attrs)
            report.layers.append(PrunedLayer(layer.index, layer.kind, experts_before, kept.tolist()))
        if progress is not None:
            progress(100.0 * (position + 1) / total)
    setattr(model.config, attrs["num_experts"], keep)
    return report


DRAFT_BLOCK = re.compile(r"^mtp\.(\d+)\.")


def source_draft_blocks(source_dir: str | pathlib.Path | None) -> int:
    """How many DSpark draft blocks (``mtp.<n>.*`` tensors) the source
    checkpoint carries. transformers' DeepSeek-V4 model never builds them and
    ignores them on load, so a pruned checkpoint written from it carries
    none (deacix/legwork#23177): the record says so instead of implying they
    were kept."""
    if source_dir is None:
        return 0
    src = pathlib.Path(source_dir)
    names: list[str] = []
    index = src / "model.safetensors.index.json"
    if index.is_file():
        names = list(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}))
    else:
        from safetensors import safe_open

        for shard in sorted(src.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt") as handle:
                names.extend(handle.keys())
    return len({match.group(1) for name in names if (match := DRAFT_BLOCK.match(name))})


def write_pruning_record(out_dir: str | pathlib.Path, record: dict[str, Any]) -> None:
    config_path = pathlib.Path(out_dir) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["reap_pruning"] = record
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_pruned(
    model: nn.Module,
    out_dir: str | pathlib.Path,
    report: PruneReport,
    source_dir: str | pathlib.Path | None = None,
    extra_record: dict[str, Any] | None = None,
) -> pathlib.Path:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    if source_dir is not None:
        src = pathlib.Path(source_dir)
        for name in TOKENIZER_FILES:
            candidate = src / name
            if candidate.is_file() and not (out / name).exists():
                shutil.copy2(candidate, out / name)
    record = report.to_record()
    record["draft_blocks"] = {"source": source_draft_blocks(source_dir), "carried": 0}
    if extra_record:
        record.update(extra_record)
    write_pruning_record(out, record)
    return out


def _load_model(path: str, dtype: str, device: str):
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {}
    if dtype != "auto":
        kwargs["dtype"] = getattr(torch, dtype)
    else:
        kwargs["dtype"] = "auto"
    if device == "auto":
        kwargs["device_map"] = "auto"
    elif device != "cpu":
        kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    return model.eval()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reap-prune", description="Prune a fused-experts checkpoint with REAP saliency."
    )
    parser.add_argument("--model", required=True, help="the source checkpoint directory")
    parser.add_argument("--stats", required=True, help="the router-stats file reap-collect wrote")
    parser.add_argument("--out", required=True, help="the pruned checkpoint directory")
    parser.add_argument("--keep", required=True, type=int, help="routed experts kept per layer")
    parser.add_argument(
        "--skip-layers",
        default="",
        help="comma-separated layer indices left at full width (a ragged checkpoint; "
        "reloads through reap.legwork.load only, never serves on vLLM)",
    )
    parser.add_argument("--method", default="reap", choices=METHODS)
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "bfloat16", "float16"))
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N or auto (accelerate device_map)")
    return parser


def main(argv: list[str] | None = None, progress: Callable[[float], None] = stage_progress) -> int:
    args = build_parser().parse_args(argv)
    skip = parse_layer_list(args.skip_layers)
    progress(0)
    stats = load_router_stats(args.stats)
    model = _load_model(args.model, args.dtype, args.device)
    progress(10)
    report = prune_model(
        model,
        stats,
        keep=args.keep,
        skip_layers=skip,
        method=args.method,
        progress=lambda pct: progress(10 + pct * 0.7),
    )
    if report.ragged:
        print(
            f"reap-prune: layers {report.skipped_layers} kept their full width — the checkpoint is "
            "ragged: reload it with reap.legwork.load.load_pruned; vLLM builds every layer from one "
            "n_routed_experts and cannot serve it",
            file=sys.stderr,
            flush=True,
        )
    progress(80)
    save_pruned(
        model,
        args.out,
        report,
        source_dir=args.model,
        extra_record={
            "source": str(args.model),
            "stats": str(args.stats),
            "calibration": stats.get("calibration"),
        },
    )
    progress(100)
    result = {
        "out": str(args.out),
        "keep": report.keep,
        "experts_before": report.experts_before,
        "layers": len(report.layers),
        "skipped_layers": report.skipped_layers,
        "ragged": report.ragged,
        "method": report.method,
        "draft_blocks_source": source_draft_blocks(args.model),
        "draft_blocks_carried": 0,
    }
    print("REAP_RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
