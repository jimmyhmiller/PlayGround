"""Markdown -> audiobook-ready script via local LLM (llama.cpp server)."""
from __future__ import annotations

import httpx

SYSTEM_PROMPT = """You rewrite academic papers into scripts for an audiobook narrator.

Rules:
- Output ONLY the spoken script. No stage directions, no headers, no markdown.
- Preserve the author's argument faithfully. Do not summarize or skip content.
- Expand acronyms on first use, then use the acronym.
- Inline citations: replace "(Smith 1999)" style with natural phrasing like "as Smith argued in 1999" only when the citation matters to the argument; otherwise drop them.
- Footnotes: if a footnote is substantive, weave it into the main text with a phrase like "It is worth noting that..."; if purely bibliographic, drop it.
- Equations: verbalize them in plain English (e.g. "x squared plus y squared equals r squared").
- Figures/tables: give a one-sentence description of what they show.
- Use natural transitions between sections rather than reading section headers.
- Spell out symbols and unusual notation.
- Keep the author's voice; do not editorialize.
"""


def rewrite_for_audio(
    markdown: str,
    *,
    base_url: str = "http://127.0.0.1:8080",
    model: str = "qwen3",
    max_tokens: int = 32768,
    temperature: float = 0.3,
) -> str:
    """Call a local OpenAI-compatible endpoint (llama.cpp server) to rewrite."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": markdown},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with httpx.Client(timeout=httpx.Timeout(60 * 30)) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"].strip()
