"""Custom-vocabulary lists: plain .txt files kept outside the repo (they hold client names)."""

from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_NAME = "default.txt"
MAX_PROMPT_CHARS = 800  # Whisper keeps only ~223 prompt tokens; beyond this, earlier terms get dropped
TEMPLATE = """\
# Custom vocabulary for Transcriber.
# Names, companies, products and jargon Whisper should spell correctly.
# One term per line (or comma-separated). Lines starting with # are ignored.
# Keep lists focused: Whisper only reads roughly the last 200 words of the combined vocabulary.
"""


def vocab_dir() -> Path:
    d = Path(os.environ.get("APPDATA", Path.home())) / "Transcriber" / "vocabulary"
    d.mkdir(parents=True, exist_ok=True)
    default = d / DEFAULT_NAME
    if not default.exists():
        default.write_text(TEMPLATE, encoding="utf-8")
    return d


def default_file() -> Path:
    return vocab_dir() / DEFAULT_NAME


def list_files() -> list[Path]:
    """Every list except the default, sorted by name."""
    return sorted((p for p in vocab_dir().glob("*.txt") if p.name.lower() != DEFAULT_NAME),
                  key=lambda p: p.name.lower())


def create(name: str) -> Path:
    name = re.sub(r'[<>:"/\\|?*]', "", name).strip()
    if not name:
        raise ValueError("Enter a name for the list.")
    path = vocab_dir() / (name if name.lower().endswith(".txt") else f"{name}.txt")
    if not path.exists():
        path.write_text(TEMPLATE, encoding="utf-8")
    return path


def read_terms(paths: list[Path | str]) -> list[str]:
    """Terms from the given files, in order, de-duplicated (case-insensitive). Missing files are skipped."""
    terms, seen = [], set()
    for p in paths:
        try:
            text = Path(p).read_text(encoding="utf-8-sig")
        except OSError:
            continue
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue
            for term in line.split(","):
                term = term.strip()
                if term and term.lower() not in seen:
                    seen.add(term.lower())
                    terms.append(term)
    return terms
