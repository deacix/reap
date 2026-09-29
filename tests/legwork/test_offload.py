# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""``--offload`` on a MiMo-V2.6-Pro-shaped working copy (24 routed experts,
the Pro's 384 at the same quarter steps): the device map fills the GPUs
first, keeps decoder layers whole and never leaves a byte on disk unless a
CPU-only run names a folder; a model the budgets cannot hold is refused
before it loads; and a collect whose layers stream through accelerate's
offload hooks records exactly what the resident collect records."""

from __future__ import annotations

import json
import pathlib
import random
import re

import pytest
import torch

from reap.legwork import collect as collect_cli
from reap.legwork.materialize import materialize
from reap.legwork.observer import load_router_stats
from reap.legwork.offload import (
    OffloadRefused,
    _skeleton,
    offload_device_map,
    parse_max_memory,
)
from tests.legwork.tiny_mimo import MOE_LAYERS, TOP_K, build_tiny_mimo, write_hub_checkpoint

PRO_EXPERTS = 24
LAYER_KEY = re.compile(r"^model\.layers\.\d+$")


def _quiet(_pct: float) -> None:
    return None


@pytest.fixture(scope="module")
def pro_copy(tmp_path_factory) -> pathlib.Path:
    hub = write_hub_checkpoint(
        build_tiny_mimo(seed=4, experts=PRO_EXPERTS), tmp_path_factory.mktemp("pro-hub")
    )
    out = tmp_path_factory.mktemp("pro-working-copy")
    materialize(hub, out, shard_bytes=64 << 10)
    return out


@pytest.fixture(scope="module")
def calibration(tmp_path_factory) -> pathlib.Path:
    rng = random.Random(23)
    path = tmp_path_factory.mktemp("pro-calib") / "calibration.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for sample in range(6):
            words = " ".join(f"t{rng.randrange(4, 256)}" for _ in range(16))
            source = "code" if sample % 2 else "chat"
            handle.write(json.dumps({"text": words, "source": source}) + "\n")
    return path


def _model_bytes(path: pathlib.Path) -> int:
    from accelerate.utils import compute_module_sizes

    return int(compute_module_sizes(_skeleton(path, torch.bfloat16, True), dtype=torch.bfloat16)[""])


def test_offload_budgets_parse_as_gpu_indices_and_the_host():
    assert parse_max_memory('{"0": 10, "3": 0, "cpu": 20}') == {0: 10, 3: 0, "cpu": 20}
    for broken in ("", "[]", "{}", '{"0": -1}', '{"0": 1.5}', '{"0": true}', '{"gpu": 1}'):
        with pytest.raises(ValueError):
            parse_max_memory(broken)


def test_offload_fills_the_gpus_first_and_keeps_decoder_layers_whole(pro_copy):
    total = _model_bytes(pro_copy)
    device_map, placement = offload_device_map(
        pro_copy, torch.bfloat16, {0: total // 2, "cpu": 4 * total}, trust_remote_code=True
    )
    devices = set(device_map.values())
    assert devices == {0, "cpu"}, device_map
    assert placement["disk_bytes"] == 0
    assert placement["gpu_bytes"] > 0 and placement["host_bytes"] > 0
    assert placement["gpu_bytes"] + placement["host_bytes"] == total
    # A decoder layer is one unit: no key names anything inside one.
    for name in device_map:
        assert not name.startswith("model.layers.") or LAYER_KEY.match(name), name
    # GPUs that hold it all leave nothing to the host.
    device_map, placement = offload_device_map(
        pro_copy, torch.bfloat16, {0: 4 * total, "cpu": 4 * total}, trust_remote_code=True
    )
    assert set(device_map.values()) == {0}
    assert placement["host_bytes"] == 0


def test_offload_refuses_a_model_the_gpus_and_host_cannot_hold(pro_copy):
    total = _model_bytes(pro_copy)
    with pytest.raises(OffloadRefused, match=r"short$"):
        offload_device_map(
            pro_copy, torch.bfloat16, {0: total // 4, "cpu": total // 4}, trust_remote_code=True
        )


def _collect(pro_copy, calibration, folder: pathlib.Path, extra: list[str]):
    out = folder / "router-stats.pt"
    argv = ["--model", str(pro_copy), "--calib", str(calibration), "--out", str(out)]
    argv += ["--trust-remote-code", "--seq-len", "32", *extra]
    assert collect_cli.main(argv, progress=_quiet) == 0
    return load_router_stats(out)


def test_an_offloaded_collect_records_what_the_resident_one_does(
    pro_copy, calibration, tmp_path, capsys
):
    resident = _collect(pro_copy, calibration, tmp_path / "resident", ["--device", "cpu"])
    capsys.readouterr()
    # A CPU-only run: a small host budget sends the later layers to a
    # folder, which exercises accelerate's offload hooks (weights on the
    # meta device outside their layer's forward) the way host memory does
    # beside a GPU.
    budget = _model_bytes(pro_copy) // 3
    offloaded = _collect(
        pro_copy,
        calibration,
        tmp_path / "offloaded",
        [
            "--offload",
            "--max-memory",
            json.dumps({"cpu": budget}),
            "--offload-folder",
            str(tmp_path / "offload-folder"),
        ],
    )
    out = capsys.readouterr().out
    # The admitted code loads the tokenizer too: nothing asks on stdout,
    # where a prompt would glue itself to the result line.
    assert "Do you wish to run the custom code" not in out
    result = json.loads(out.splitlines()[-1].removeprefix("REAP_RESULT "))
    assert result["offload"]["disk_bytes"] > 0, result
    assert result["offload"]["gpu_bytes"] == 0
    assert sorted(offloaded["layers"]) == sorted(resident["layers"]) == list(MOE_LAYERS)
    for index in MOE_LAYERS:
        got, want = offloaded["layers"][index], resident["layers"][index]
        assert got["num_experts"] == PRO_EXPERTS
        assert torch.equal(got["count"], want["count"])
        assert int(got["count"].sum()) == got["tokens"] * TOP_K
        assert torch.allclose(got["reap_sum"], want["reap_sum"], rtol=1e-9, atol=0.0)
        assert sorted(got["by_source"]) == ["chat", "code"]


def test_an_offload_the_budgets_cannot_hold_exits_before_loading(pro_copy, calibration, tmp_path):
    argv = ["--model", str(pro_copy), "--calib", str(calibration), "--out", str(tmp_path / "s.pt")]
    argv += ["--trust-remote-code", "--offload", "--max-memory", '{"cpu": 1024}']
    with pytest.raises(SystemExit, match=r"reap-collect: the model needs about .* short"):
        collect_cli.main(argv, progress=_quiet)
    assert not (tmp_path / "s.pt").exists()


def test_the_offload_budget_flags_ride_offload_only(pro_copy, calibration, tmp_path):
    argv = ["--model", str(pro_copy), "--calib", str(calibration), "--out", str(tmp_path / "s.pt")]
    with pytest.raises(SystemExit, match="go with --offload"):
        collect_cli.main([*argv, "--max-memory", '{"cpu": 1}'], progress=_quiet)
    with pytest.raises(SystemExit, match="drop --dequantize"):
        collect_cli.main([*argv, "--offload", "--dequantize"], progress=_quiet)
    with pytest.raises(SystemExit, match="--max-memory"):
        collect_cli.main([*argv, "--offload", "--max-memory", "{}"], progress=_quiet)
