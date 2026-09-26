# MiMo-V2 model code for the CPU suite

These three files are Xiaomi's, copied verbatim from the Hugging Face
repository `XiaomiMiMo/MiMo-V2.6-Flash-RL` at revision
`5711b268169967567844e1e560e8a3966da959b1`. `tests/legwork/tiny_mimo.py`
builds a tiny random MiMo-V2 from them, so the suite runs the same model
code a MiMo working copy runs.

| File | sha256 | Licence |
|---|---|---|
| `configuration_mimo_v2.py` | `773062ac9850b908eb54751b3e4dbe00e653c0e80595599409e97d0c1af2ce3e` | Apache-2.0 (its header) |
| `modeling_mimo_v2.py` | `a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad` | Apache-2.0 (its header) |
| `chat_template.jinja` | `853650bee57bf95020373e4c928bd5a4b41b9915adf964a77711d2b49a291887` | MIT (the repository's licence) |

The two Python files carry their own Apache-2.0 headers
(Copyright 2026 Xiaomi Corporation and The HuggingFace Inc. team). The
repository's model card declares the MIT licence for the release.
