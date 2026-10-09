"""Transcript data model: words -> speaker-smoothed paragraphs, plus JSON persistence."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SENTENCE_END = re.compile(r"[.?!…][\"'”)\]]*$")
PARAGRAPH_GAP_SEC = 2.0     # a pause this long after a sentence starts a new paragraph
PARAGRAPH_MAX_SEC = 45.0    # long monologues break at the next sentence end after this
SMOOTH_MAX_WORDS = 2        # mid-sentence speaker flips this short are treated as diarization noise


def ends_sentence(text: str) -> bool:
    return bool(SENTENCE_END.search(text))


def default_speaker_name(speaker_id: str) -> str:
    m = re.search(r"(\d+)$", speaker_id or "")
    return f"Speaker {int(m.group(1)) + 1}" if m else (speaker_id or "")


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str | None = None


@dataclass
class Paragraph:
    start: float
    end: float
    speaker: str | None
    text: str
    words: list[Word] = field(default_factory=list)
    edited: bool = False


@dataclass
class TranscriptDoc:
    audio_path: str
    duration: float
    model: str
    created: str
    diarized: bool
    paragraphs: list[Paragraph]
    speaker_names: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    summary_model: str = ""
    vocabulary: str = ""
    version: int = 1

    # ----- speakers -----

    def speaker_ids(self) -> list[str]:
        seen: list[str] = []
        for p in self.paragraphs:
            if p.speaker and p.speaker not in seen:
                seen.append(p.speaker)
        for sid in self.speaker_names:
            if sid not in seen:
                seen.append(sid)
        return seen

    def speaker_label(self, speaker_id: str | None) -> str:
        if not speaker_id:
            return ""
        return self.speaker_names.get(speaker_id) or default_speaker_name(speaker_id)

    # ----- persistence -----

    def to_dict(self) -> dict:
        return {
            "version": self.version, "audio_path": self.audio_path, "duration": self.duration,
            "model": self.model, "created": self.created, "diarized": self.diarized,
            "speaker_names": self.speaker_names, "summary": self.summary,
            "summary_model": self.summary_model, "vocabulary": self.vocabulary,
            "paragraphs": [
                {"start": round(p.start, 3), "end": round(p.end, 3), "speaker": p.speaker,
                 "text": p.text, "edited": p.edited,
                 # compact: [text, start, end, speaker]
                 "words": [[w.text, round(w.start, 3), round(w.end, 3), w.speaker] for w in p.words]}
                for p in self.paragraphs
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TranscriptDoc":
        paras = [Paragraph(start=p["start"], end=p["end"], speaker=p.get("speaker"), text=p["text"],
                           edited=p.get("edited", False),
                           words=[Word(*w) for w in p.get("words", [])])
                 for p in d.get("paragraphs", [])]
        return cls(audio_path=d["audio_path"], duration=d.get("duration", 0.0), model=d.get("model", ""),
                   created=d.get("created", ""), diarized=d.get("diarized", False), paragraphs=paras,
                   speaker_names=d.get("speaker_names", {}), summary=d.get("summary", ""),
                   summary_model=d.get("summary_model", ""), vocabulary=d.get("vocabulary", ""),
                   version=d.get("version", 1))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "TranscriptDoc":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Building paragraphs from whisperX output
# ---------------------------------------------------------------------------

def _flatten_words(segments: list[dict]) -> list[Word]:
    words: list[Word] = []
    for seg in segments:
        seg_spk = seg.get("speaker")
        seg_words = [w for w in (seg.get("words") or []) if (w.get("word") or "").strip()]
        if not seg_words:
            text = (seg.get("text") or "").strip()
            if text:
                words.append(Word(text, seg.get("start"), seg.get("end"), seg_spk))
            continue
        for w in seg_words:
            words.append(Word(w["word"].strip(), w.get("start"), w.get("end"), w.get("speaker", seg_spk)))
    # Words wav2vec2 couldn't align (often numerals) have no times: borrow from neighbours.
    for i, w in enumerate(words):
        if w.start is None:
            w.start = words[i - 1].end if i else 0.0
        if w.end is None:
            nxt = next((x.start for x in words[i + 1:] if x.start is not None), None)
            w.end = max(w.start, nxt if nxt is not None else w.start)
    return words


def _fill_and_smooth_speakers(words: list[Word]) -> None:
    last = None
    for w in words:
        if w.speaker is None:
            w.speaker = last
        last = w.speaker
    first_known = next((w.speaker for w in words if w.speaker), None)
    for w in words:
        if w.speaker is None:
            w.speaker = first_known
        else:
            break

    # Collapse short mid-sentence speaker flips (A A B A A) — a common diarization artifact.
    runs: list[tuple[int, int]] = []  # (start index, end index exclusive)
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and words[j].speaker == words[i].speaker:
            j += 1
        runs.append((i, j))
        i = j
    for k in range(1, len(runs) - 1):
        s, e = runs[k]
        before, after = words[s - 1], words[e]
        if (e - s <= SMOOTH_MAX_WORDS and before.speaker == after.speaker
                and not ends_sentence(before.text) and not ends_sentence(words[e - 1].text)):
            for w in words[s:e]:
                w.speaker = before.speaker


def build_paragraphs(segments: list[dict], diarized: bool) -> list[Paragraph]:
    words = _flatten_words(segments)
    if diarized:
        _fill_and_smooth_speakers(words)
        # pyannote's labels are arbitrary; renumber so whoever talks first is Speaker 1
        order: dict[str, str] = {}
        for w in words:
            if w.speaker is not None and w.speaker not in order:
                order[w.speaker] = f"SPEAKER_{len(order):02d}"
        for w in words:
            w.speaker = order.get(w.speaker, w.speaker)
    else:
        for w in words:
            w.speaker = None

    paragraphs: list[Paragraph] = []
    cur: list[Word] = []

    def flush():
        if cur:
            paragraphs.append(Paragraph(start=cur[0].start, end=cur[-1].end, speaker=cur[0].speaker,
                                        text=" ".join(w.text for w in cur), words=list(cur)))
            cur.clear()

    for w in words:
        if cur:
            prev = cur[-1]
            if (w.speaker != prev.speaker
                    or (ends_sentence(prev.text) and w.start - prev.end >= PARAGRAPH_GAP_SEC)
                    or (ends_sentence(prev.text) and prev.end - cur[0].start >= PARAGRAPH_MAX_SEC)):
                flush()
        cur.append(w)
    flush()
    return paragraphs
