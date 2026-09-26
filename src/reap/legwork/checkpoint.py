# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""Stream a sharded safetensors checkpoint tensor by tensor, and write one.

``reap-materialize`` and ``reap-prune --slice-source`` never build a model:
they read the source's tensors one at a time through the index
(``model.safetensors.index.json``, or a lone ``model.safetensors``) and
write shards of bounded size under a fresh index. The top-level files
around the weights (``config.json``, the model code, the tokenizer) are
copied by ``copy_files``; ``config.json`` and the index are the callers'.
"""

from __future__ import annotations

import json
import pathlib
import shutil
from typing import Any, Iterable

import torch

INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"
#: The written shards' size bound (the last one is smaller).
DEFAULT_SHARD_BYTES = 4 << 30


def read_index(source: str | pathlib.Path) -> tuple[dict[str, str], dict[str, Any]]:
    """``(weight_map, metadata)``: every tensor's shard file and the index's
    ``metadata`` (``{}`` for a lone ``model.safetensors``)."""
    source = pathlib.Path(source)
    index = source / INDEX
    if index.is_file():
        data = json.loads(index.read_text(encoding="utf-8"))
        return dict(data["weight_map"]), dict(data.get("metadata") or {})
    single = source / SINGLE
    if not single.is_file():
        raise FileNotFoundError(f"{source} holds neither {INDEX} nor {SINGLE}")
    from safetensors import safe_open

    with safe_open(str(single), framework="pt") as handle:
        return dict.fromkeys(handle.keys(), SINGLE), {}


class TensorReader:
    """Read tensors by name from a checkpoint's shards, one handle per shard."""

    def __init__(self, source: str | pathlib.Path):
        self.source = pathlib.Path(source)
        self.weight_map, self.metadata = read_index(self.source)
        self._handles: dict[str, Any] = {}

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def names(self) -> list[str]:
        """Every tensor name, grouped by shard (shard order, then name order)."""
        return sorted(self.weight_map, key=lambda name: (self.weight_map[name], name))

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"{self.source} has no tensor {name}")
        handle = self._handles.get(shard)
        if handle is None:
            from safetensors import safe_open

            handle = safe_open(str(self.source / shard), framework="pt")
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        self._handles.clear()


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


class ShardWriter:
    """Collect tensors into ``model-NNNNN.safetensors`` shards of at most
    ``shard_bytes`` each (a larger tensor gets a shard of its own), then write
    the index with ``metadata`` and the recomputed ``total_size``."""

    def __init__(self, out: str | pathlib.Path, shard_bytes: int = DEFAULT_SHARD_BYTES):
        self.out = pathlib.Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.shard_bytes = max(1, int(shard_bytes))
        self._pending: dict[str, torch.Tensor] = {}
        self._pending_bytes = 0
        self._shards = 0
        self.weight_map: dict[str, str] = {}
        self.total_size = 0

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if name in self.weight_map or name in self._pending:
            raise ValueError(f"tensor {name} written twice")
        size = tensor_bytes(tensor)
        if self._pending and self._pending_bytes + size > self.shard_bytes:
            self._flush()
        self._pending[name] = tensor.contiguous()
        self._pending_bytes += size
        self.total_size += size

    def _flush(self) -> None:
        if not self._pending:
            return
        from safetensors.torch import save_file

        self._shards += 1
        shard = f"model-{self._shards:05d}.safetensors"
        save_file(self._pending, str(self.out / shard), metadata={"format": "pt"})
        for name in self._pending:
            self.weight_map[name] = shard
        self._pending = {}
        self._pending_bytes = 0

    def close(self, metadata: dict[str, Any] | None = None) -> pathlib.Path:
        self._flush()
        index = {
            "metadata": {**(metadata or {}), "total_size": self.total_size},
            "weight_map": dict(sorted(self.weight_map.items())),
        }
        path = self.out / INDEX
        path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        return path


def copy_files(
    source: str | pathlib.Path,
    out: str | pathlib.Path,
    skip_dirs: Iterable[str] = (),
    include_dirs: bool = True,
) -> list[str]:
    """Copy every top-level file of ``source`` but the weights, the index and
    ``config.json`` (the model code, the tokenizer, the generation config);
    with ``include_dirs`` every subdirectory too, less ``skip_dirs``."""
    source, out = pathlib.Path(source), pathlib.Path(out)
    skipped = set(skip_dirs)
    copied: list[str] = []
    for entry in sorted(source.iterdir()):
        if entry.is_dir():
            if not include_dirs or entry.name in skipped or entry.name.startswith("."):
                continue
            shutil.copytree(entry, out / entry.name, dirs_exist_ok=True)
            copied.append(entry.name + "/")
        elif entry.name.endswith(".safetensors") or entry.name in (INDEX, "config.json"):
            continue
        else:
            shutil.copy2(entry, out / entry.name)
            copied.append(entry.name)
    return copied
