"""Markdown -> audiobook-ready script via local LLM (llama.cpp server)."""
from __future__ import annotations

import re

import click
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


def _post_chat(
    user_content: str,
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with httpx.Client(timeout=httpx.Timeout(60 * 60)) as client:
        r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def _is_context_overflow(exc: httpx.HTTPStatusError) -> bool:
    if exc.response.status_code != 400:
        return False
    try:
        body = exc.response.text.lower()
    except Exception:
        return False
    return "exceeds the available context" in body or "context size" in body


def _split_for_chunking(text: str, n_parts: int) -> list[str]:
    """Split text into n_parts on paragraph boundaries, balanced by length."""
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) < n_parts:
        # Fall back to character-based slicing if too few paragraphs.
        size = max(1, len(text) // n_parts)
        return [text[i : i + size] for i in range(0, len(text), size)]
    target = sum(len(p) for p in paragraphs) // n_parts
    chunks: list[list[str]] = [[]]
    running = 0
    for para in paragraphs:
        if running >= target and len(chunks) < n_parts:
            chunks.append([])
            running = 0
        chunks[-1].append(para)
        running += len(para)
    return ["\n\n".join(c).strip() for c in chunks if c]


def rewrite_for_audio(
    markdown: str,
    *,
    base_url: str = "http://127.0.0.1:8080",
    model: str = "qwen3",
    max_tokens: int = 32768,
    temperature: float = 0.3,
) -> str:
    """Call a local OpenAI-compatible endpoint (llama.cpp server) to rewrite.

    If the chapter is too large for the server's context window, recursively
    split it on paragraph boundaries and rewrite each piece, then concatenate.
    """
    try:
        return _post_chat(
            markdown, base_url=base_url, model=model,
            max_tokens=max_tokens, temperature=temperature,
        )
    except httpx.HTTPStatusError as exc:
        if not _is_context_overflow(exc):
            raise
        # Chunk and recurse. Start with 2-way split; recursion handles deeper.
        chunks = _split_for_chunking(markdown, n_parts=2)
        if len(chunks) < 2:
            raise
        click.echo(f"[rewrite] chapter exceeds ctx; splitting into {len(chunks)} chunk(s)")
        out_parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            click.echo(f"[rewrite]   chunk {i}/{len(chunks)} ({len(chunk)} chars)")
            out_parts.append(rewrite_for_audio(
                chunk, base_url=base_url, model=model,
                max_tokens=max_tokens, temperature=temperature,
            ))
        return "\n\n".join(out_parts)
