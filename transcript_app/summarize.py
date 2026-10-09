"""Transcript summaries via a local LLM (shared `localllm` package -> Ollama)."""

from __future__ import annotations

from typing import Callable

from localllm import LLM

from .document import TranscriptDoc
from .exporters import fmt_time

STYLES = {
    "brainstorm": (
        "Solo brainstorm",
        "This is a solo brainstorm: one person thinking out loud. Write organized notes in "
        "Markdown with these sections:\n"
        "## Overview (2-3 sentences)\n## Key ideas (bullets, grouped by theme)\n"
        "## Decisions or conclusions\n## Open questions\n## Next steps\n"
        "Preserve specific names, numbers and examples. Omit a section if nothing fits it.",
    ),
    "discovery": (
        "Client discovery",
        "This is a discovery session between a consultant and a client. Write meeting notes in "
        "Markdown with these sections:\n"
        "## Overview (who, purpose, 2-3 sentences)\n## Client background & context\n"
        "## Goals & desired outcomes\n## Pain points & challenges\n"
        "## Requirements & constraints (budget, timeline, tools, stakeholders)\n"
        "## Decisions made\n## Action items (bullets: owner - task - due date if stated)\n"
        "## Open questions / follow-ups\n"
        "Attribute statements to speakers by name where it matters. Preserve specific names, "
        "numbers and dates. Omit a section if nothing fits it. Do not invent facts.",
    ),
}

CHUNK_CHARS = 24000          # ~6k tokens per map step
SINGLE_PASS_MAX_CHARS = 48000  # ~12k tokens: summarize in one call below this


def transcript_text(doc: TranscriptDoc) -> str:
    lines = []
    for p in doc.paragraphs:
        who = f"{doc.speaker_label(p.speaker)}: " if doc.diarized and p.speaker else ""
        lines.append(f"[{fmt_time(p.start)}] {who}{p.text}")
    return "\n".join(lines)


def _chunks(text: str, size: int) -> list[str]:
    out, cur, n = [], [], 0
    for line in text.splitlines():
        if cur and n + len(line) > size:
            out.append("\n".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += len(line) + 1
    if cur:
        out.append("\n".join(cur))
    return out


def _ctx_for(chars: int) -> int:
    tokens = chars // 4 + 3000  # prompt + output headroom
    for size in (8192, 16384, 24576, 32768):
        if tokens <= size:
            return size
    return 32768


def summarize(doc: TranscriptDoc, model: str, style: str,
              on_progress: Callable[[str], None] = lambda s: None) -> str:
    _, instructions = STYLES[style]
    llm = LLM(model, temperature=0.2, timeout=1800)
    system = ("You are an expert note-taker. You write accurate, well-organized summaries of "
              "transcripts. Use only information present in the transcript.")
    text = transcript_text(doc)
    try:
        if len(text) <= SINGLE_PASS_MAX_CHARS:
            on_progress("Summarizing…")
            return llm.ask(f"{instructions}\n\nTRANSCRIPT:\n{text}", system=system, think=False,
                           num_ctx=_ctx_for(len(text)))
        # Long recording: notes per chunk, then a combined summary.
        parts = _chunks(text, CHUNK_CHARS)
        notes = []
        for i, part in enumerate(parts, 1):
            on_progress(f"Reading part {i} of {len(parts)}…")
            notes.append(llm.ask(
                "Take detailed bullet-point notes on this part of a longer transcript. Keep "
                "timestamps for important points, speaker attributions, names, numbers, decisions "
                f"and action items.\n\nTRANSCRIPT PART {i}/{len(parts)}:\n{part}",
                system=system, think=False, num_ctx=_ctx_for(len(part))))
        combined = "\n\n".join(f"### Notes for part {i}\n{n}" for i, n in enumerate(notes, 1))
        on_progress("Combining notes…")
        return llm.ask(f"{instructions}\n\nBelow are sequential notes taken from one long "
                       f"recording. Merge them into a single summary.\n\n{combined}",
                       system=system, think=False, num_ctx=_ctx_for(len(combined)))
    finally:
        llm.unload()  # give the VRAM back for the next transcription
