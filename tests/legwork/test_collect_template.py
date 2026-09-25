# Copyright 2026 the Legwork authors (deacix/reap, the `legwork` branch).
# Modifications to REAP (Copyright 2025 Cerebras Systems), Apache-2.0.
"""collect v2 on the tiny V4: a ``messages`` row's tools reach the chat
template, a template that raises (or none at all) falls back to plain
text and is counted, and every row's source reaches the router stats."""

from __future__ import annotations

import json
import shutil

import pytest

from reap.legwork import collect
from reap.legwork.collect import SampleEncoder, render_plain, row_source
from reap.legwork.observer import load_router_stats

# Renders the tools list, every turn and its tool calls; an unknown role
# raises through transformers' ``raise_exception``.
TEMPLATE = (
    "{%- if tools %}{%- for tool in tools %}"
    "tool {{ tool.function.name }} {{ tool.function.parameters | tojson }}\n"
    "{% endfor %}{%- endif %}"
    "{%- for message in messages %}"
    "{%- if message.role not in ['system', 'user', 'assistant', 'tool'] %}"
    "{{ raise_exception('unknown role ' + message.role) }}{%- endif %}"
    "{{ message.role }} {{ message.content or '' }}"
    "{%- if message.tool_calls %}{%- for call in message.tool_calls %}"
    " call {{ call.function.name }} {{ call.function.arguments | tojson }}"
    "{%- endfor %}{%- endif %}\n"
    "{% endfor %}"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "the weather in a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]

TOOL_TURNS = [
    {"role": "system", "content": "t10 t11"},
    {"role": "user", "content": "t12 t13 t14"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "t15"}}}
        ],
    },
    {"role": "tool", "content": "t16 t17"},
    {"role": "assistant", "content": "t18 t19 t20"},
]


@pytest.fixture(scope="module")
def templated_dir(tmp_path_factory, tiny_v4_dir):
    """The tiny V4 beside a tokenizer whose chat template renders tools."""
    from transformers import AutoTokenizer

    out = tmp_path_factory.mktemp("tiny-v4-chat")
    shutil.copytree(tiny_v4_dir, out, dirs_exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tiny_v4_dir)
    tokenizer.chat_template = TEMPLATE
    tokenizer.save_pretrained(out)
    return out


class _SpyTokenizer:
    """Records the kwargs ``apply_chat_template`` receives."""

    chat_template = "spy"

    def __init__(self):
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return "t5 t6"

    def __call__(self, text, **kwargs):
        return {"input_ids": [5, 6]}


def test_tools_reach_the_template_and_are_omitted_without_them():
    encoder = SampleEncoder("unused", seq_len=16)
    spy = _SpyTokenizer()
    encoder._tokenizer = spy
    encoder.encode({"messages": TOOL_TURNS, "tools": TOOLS}, "row 1")
    encoder.encode({"messages": TOOL_TURNS}, "row 2")
    encoder.encode({"messages": TOOL_TURNS, "tools": []}, "row 3")
    assert spy.calls[0] == {"tools": TOOLS, "tokenize": False, "add_generation_prompt": False}
    assert spy.calls[1] == {"tokenize": False, "add_generation_prompt": False}
    assert spy.calls[2] == {"tokenize": False, "add_generation_prompt": False}
    assert encoder.template_fallbacks == 0


def test_the_chat_template_renders_the_tools_and_the_tool_calls(templated_dir):
    encoder = SampleEncoder(str(templated_dir), seq_len=64)
    text, templated = encoder.render_messages({"messages": TOOL_TURNS, "tools": TOOLS}, "row")
    assert templated and encoder.template_fallbacks == 0
    assert text.startswith('tool get_weather {"type": "object"')
    assert 'call get_weather {"city": "t15"}' in text
    assert "tool t16 t17" in text
    # No tools on the row: the template sees none.
    bare, _ = encoder.render_messages({"messages": TOOL_TURNS}, "row")
    assert bare.startswith("system t10 t11")


def test_a_failing_template_falls_back_to_plain_text_and_is_counted(templated_dir, capsys):
    encoder = SampleEncoder(str(templated_dir), seq_len=64)
    turns = [{"role": "narrator", "content": "t30 t31"}, {"role": "user", "content": "t32"}]
    text, templated = encoder.render_messages({"messages": turns, "tools": TOOLS}, "calib.jsonl:4")
    assert not templated and encoder.template_fallbacks == 1
    assert text == render_plain(turns, TOOLS)
    assert "calib.jsonl:4: the chat template failed (TemplateError: unknown role narrator)" in (
        capsys.readouterr().err
    )
    ids = encoder.encode({"messages": turns}, "calib.jsonl:5")
    assert encoder.template_fallbacks == 2
    assert 30 in ids and 32 in ids


def test_a_tokenizer_without_a_template_falls_back(tiny_v4_dir):
    encoder = SampleEncoder(str(tiny_v4_dir), seq_len=64)
    assert not getattr(encoder.tokenizer, "chat_template", None)
    text, templated = encoder.render_messages({"messages": TOOL_TURNS, "tools": TOOLS}, "row")
    assert not templated and encoder.template_fallbacks == 1
    assert text == render_plain(TOOL_TURNS, TOOLS)


def test_plain_rendering_names_roles_contents_tool_calls_and_tools():
    text = render_plain(TOOL_TURNS, TOOLS)
    blocks = text.split("\n\n")
    assert blocks[0] == "tools:\n" + json.dumps(TOOLS)
    assert blocks[1] == "system:\nt10 t11"
    assert blocks[3] == "assistant:\n" + json.dumps(TOOL_TURNS[2]["tool_calls"])
    assert blocks[4] == "tool:\nt16 t17"
    parts = [{"role": "user", "content": [{"type": "text", "text": "t40"}, {"type": "image"}]}]
    assert render_plain(parts) == 'user:\nt40\n{"type": "image"}'


def test_row_source_and_private():
    assert row_source({"text": "t4"}, "r") == ("unlabeled", False)
    assert row_source({"text": "t4", "source": None, "private": None}, "r") == ("unlabeled", False)
    assert row_source({"text": "t4", "source": "code", "private": True}, "r") == ("code", True)
    with pytest.raises(ValueError, match="source is a non-empty string"):
        row_source({"source": 3}, "r")
    with pytest.raises(ValueError, match="source is a non-empty string"):
        row_source({"source": " "}, "r")
    with pytest.raises(ValueError, match="private is true or false"):
        row_source({"private": "yes"}, "r")


def _result(stdout: str) -> dict:
    return json.loads([l for l in stdout.splitlines() if l.startswith("REAP_RESULT ")][-1][12:])


def test_collect_counts_fallbacks_and_records_sources(tmp_path, templated_dir, capsys):
    calib = tmp_path / "calib.jsonl"
    rows = [
        {"messages": TOOL_TURNS, "tools": TOOLS, "source": "agentic"},
        {"messages": TOOL_TURNS, "source": "agentic"},
        {"messages": [{"role": "narrator", "content": "t50 t51"}], "source": "agentic"},
        {"text": "t60 t61 t62 t63", "source": "code"},
        {"input_ids": [70, 71, 72, 73, 74], "source": "mine", "private": True},
        {"prompt": "t80 t81", "completion": " t82"},
    ]
    calib.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    out = tmp_path / "stats.pt"
    argv = ["--model", str(templated_dir), "--calib", str(calib), "--out", str(out), "--seq-len", "32"]
    assert collect.main(argv, progress=lambda pct: None) == 0
    result = _result(capsys.readouterr().out)
    assert result["samples"] == 6 and result["template_fallbacks"] == 1
    sources = {entry["id"]: entry for entry in result["sources"]}
    assert [entry["id"] for entry in result["sources"]] == ["agentic", "code", "mine", "unlabeled"]
    assert sources["agentic"]["rows"] == 3 and sources["agentic"]["private"] is False
    assert sources["code"] == {"id": "code", "private": False, "rows": 1, "tokens": 4}
    assert sources["mine"] == {"id": "mine", "private": True, "rows": 1, "tokens": 5}
    assert sources["unlabeled"]["rows"] == 1
    assert sum(entry["tokens"] for entry in result["sources"]) == result["tokens"]
    stats = load_router_stats(out)
    assert stats["calibration"]["template_fallbacks"] == 1
    assert stats["sources"] == result["sources"]
