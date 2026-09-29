# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""An offloaded load (``--offload``): a checkpoint the visible GPUs cannot
hold keeps the decoder layers they leave in host memory, and accelerate
streams each one to its GPU on every forward. Never to disk.

``offload_device_map`` builds the model's skeleton on the meta device and
asks accelerate for a device map under an explicit ``max_memory``: each
visible GPU's free memory less a reserve for the activations, then the
host's available memory less a floor. Decoder layers stay whole (the
model's ``_no_split_modules``, else its decoder layer's own class, since
MiMo-V2's remote code declares none), so a spilled layer streams in one
piece. A map that would leave anything on disk is refused before a byte
loads, naming how far the GPUs and host memory fall short.
``offload_folder`` admits the disk for a CPU-only run (the tests and the
worker's smoke), where the host is the execution device and only a folder
can hold what a budget leaves out; the worker's jobs never pass it.

``offload_summary`` is what a stage reports: the bytes the map places on
the GPUs, in host memory and on disk.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import torch

GIB = 1024**3
#: What each GPU keeps free for the forward's activations beside its layers.
GPU_RESERVE_BYTES = 4 * GIB
#: The share of a GPU's free memory kept free when that is more than the
#: reserve (a long calibration sample's attention on a large board).
GPU_RESERVE_SHARE = 0.05
#: What host memory keeps free beside the layers it holds: the process's
#: own tokenizer, calibration set and statistics.
HOST_FLOOR_BYTES = 16 * GIB


class OffloadRefused(ValueError):
    """The GPUs and host memory together cannot hold the model."""


def parse_max_memory(text: str) -> dict[Any, int]:
    """``--max-memory``'s JSON (``{"0": bytes, "cpu": bytes}``): GPU indices
    become ints, ``cpu`` stays; every budget is a non-negative integer."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"--max-memory is not JSON ({error.msg})") from error
    if not isinstance(raw, dict) or not raw:
        raise ValueError('--max-memory names the budgets as {"0": bytes, "cpu": bytes}')
    budgets: dict[Any, int] = {}
    for key, value in raw.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"--max-memory: {key!r} is a byte count, not {value!r}")
        if key == "cpu":
            budgets["cpu"] = value
        elif isinstance(key, str) and key.isdigit():
            budgets[int(key)] = value
        else:
            raise ValueError(f"--max-memory: {key!r} is neither a GPU index nor cpu")
    return budgets


def measured_max_memory() -> dict[Any, int]:
    """The budgets on this machine: every visible GPU's free memory less its
    reserve, then the host's available memory less its floor."""
    import psutil

    budgets: dict[Any, int] = {}
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(index)
            reserve = max(GPU_RESERVE_BYTES, int(free * GPU_RESERVE_SHARE))
            budgets[index] = max(0, free - reserve)
    budgets["cpu"] = max(0, int(psutil.virtual_memory().available) - HOST_FLOOR_BYTES)
    return budgets


def _decoder_layer_classes(skeleton: torch.nn.Module) -> list[str]:
    declared = list(getattr(skeleton, "_no_split_modules", None) or [])
    if declared:
        return declared
    from reap.legwork.arch import _decoder_layers

    try:
        layers = _decoder_layers(skeleton)
    except ValueError:
        return []
    return sorted({type(layer).__name__ for layer in layers})


def _skeleton(path: str | pathlib.Path, dtype: torch.dtype, trust_remote_code: bool):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
    with init_empty_weights():
        return AutoModelForCausalLM.from_config(
            config, trust_remote_code=trust_remote_code, dtype=dtype
        )


def _size(bytes_: int) -> str:
    return f"{bytes_ / GIB:.1f} GiB"


def offload_device_map(
    path: str | pathlib.Path,
    dtype: torch.dtype,
    max_memory: dict[Any, int],
    trust_remote_code: bool = False,
    offload_folder: str | pathlib.Path | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """``(device_map, summary)`` for loading ``path`` at ``dtype`` under
    ``max_memory``: the GPUs first, host memory next, decoder layers whole.
    Raises ``OffloadRefused`` when the map leaves anything on disk and no
    ``offload_folder`` admits it."""
    from accelerate import infer_auto_device_map
    from accelerate.utils import compute_module_sizes

    skeleton = _skeleton(path, dtype, trust_remote_code)
    device_map = infer_auto_device_map(
        skeleton,
        max_memory=dict(max_memory),
        no_split_module_classes=_decoder_layer_classes(skeleton),
        dtype=dtype,
    )
    sizes = compute_module_sizes(skeleton, dtype=dtype)
    summary = offload_summary(device_map, sizes)
    if summary["disk_bytes"] > 0 and offload_folder is None:
        gpus = sum(v for k, v in max_memory.items() if k != "cpu")
        host = max_memory.get("cpu", 0)
        raise OffloadRefused(
            f"the model needs about {_size(sizes[''])} and this machine holds about "
            f"{_size(gpus)} on its GPUs and {_size(host)} in host memory for it — about "
            f"{_size(summary['disk_bytes'])} short"
        )
    return device_map, summary


def offload_summary(device_map: dict[str, Any], sizes: dict[str, int]) -> dict[str, int]:
    """The bytes a device map places on the GPUs, in host memory and on disk."""
    summary = {"gpu_bytes": 0, "host_bytes": 0, "disk_bytes": 0}
    for name, device in device_map.items():
        key = "host_bytes" if device == "cpu" else "disk_bytes" if device == "disk" else "gpu_bytes"
        summary[key] += int(sizes.get(name, 0))
    return summary


def load_offloaded(
    path: str | pathlib.Path,
    device_map: dict[str, Any],
    dtype: torch.dtype,
    trust_remote_code: bool = False,
    offload_folder: str | pathlib.Path | None = None,
):
    """``from_pretrained`` along an ``offload_device_map`` map."""
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"device_map": device_map, "dtype": dtype}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if offload_folder is not None:
        kwargs["offload_folder"] = str(offload_folder)
    return AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()


def config_dtype(path: str | pathlib.Path, requested: str = "auto") -> torch.dtype:
    """The dtype an offloaded load runs at: the one asked for, else the
    checkpoint's own (a working copy's ``bfloat16``), else float32."""
    if requested != "auto":
        return getattr(torch, requested)
    try:
        config = json.loads((pathlib.Path(path) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return torch.float32
    named = config.get("dtype") or config.get("torch_dtype")
    dtype = getattr(torch, named, None) if isinstance(named, str) else None
    return dtype if isinstance(dtype, torch.dtype) else torch.float32


def input_device(model: torch.nn.Module) -> torch.device:
    """Where the forward's input ids go: the embedding's execution device
    (accelerate's hook names it when the embedding itself was offloaded)."""
    embeddings = model.get_input_embeddings()
    hook = getattr(embeddings, "_hf_hook", None)
    device = getattr(hook, "execution_device", None)
    if device is not None:
        return torch.device(device) if not isinstance(device, torch.device) else device
    weight = getattr(embeddings, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.device.type != "meta":
        return weight.device
    return torch.device("cpu")
