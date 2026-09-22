# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""Session fixtures: the tiny V4 saved beside its tokenizer, and a
calibration set (text rows and pre-tokenized rows) over its vocabulary."""

from __future__ import annotations

import json
import pathlib
import random

import pytest

from tests.legwork.tiny_v4 import VOCAB, build_tiny_tokenizer, build_tiny_v4


@pytest.fixture(scope="session")
def tiny_v4_dir(tmp_path_factory) -> pathlib.Path:
    out = tmp_path_factory.mktemp("tiny-v4")
    model = build_tiny_v4()
    model.save_pretrained(out)
    build_tiny_tokenizer().save_pretrained(out)
    return out


@pytest.fixture(scope="session")
def calibration_jsonl(tmp_path_factory) -> pathlib.Path:
    path = tmp_path_factory.mktemp("calib") / "calibration.jsonl"
    rng = random.Random(7)
    with path.open("w", encoding="utf-8") as handle:
        for sample in range(12):
            if sample % 3 == 0:
                ids = [rng.randrange(4, VOCAB) for _ in range(24)]
                handle.write(json.dumps({"input_ids": ids}) + "\n")
            else:
                words = " ".join(f"t{rng.randrange(4, VOCAB)}" for _ in range(24))
                handle.write(json.dumps({"text": words}) + "\n")
    return path
