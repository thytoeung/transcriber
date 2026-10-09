"""GPU transcription worker. Runs in its own process so the GUI stays responsive, Cancel can
simply kill it, and all VRAM is released when it exits.

    python -m transcript_app.engine job.json

Protocol: one JSON object per line on stdout ({"type": "progress"|"log"|"done"|"error", ...}).
Library chatter is redirected to stderr.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import traceback
import warnings
from pathlib import Path

SAMPLE_RATE = 16000
DIARIZATION_MODEL_URL = "https://huggingface.co/pyannote/speaker-diarization-community-1"

_proto = sys.stdout


def emit(**msg) -> None:
    _proto.write(json.dumps(msg) + "\n")
    _proto.flush()


def progress(stage: str, pct: float) -> None:
    emit(type="progress", stage=stage, pct=round(max(0.0, min(100.0, pct)), 1))


class FriendlyError(Exception):
    pass


def _free_gpu() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_audio(path: str):
    """16 kHz mono float32. Uses the ffmpeg CLI when it's on PATH (whisperX's own loader);
    otherwise decodes in-process with PyAV."""
    import shutil

    import numpy as np

    if shutil.which("ffmpeg"):
        from whisperx.audio import load_audio as ffmpeg_load
        return ffmpeg_load(path, SAMPLE_RATE)
    import av
    chunks = []
    with av.open(path) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise FriendlyError("No audio stream found in this file.")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _vocab_terms(vocab: str) -> list[str]:
    return [t.strip() for t in vocab.replace("\n", ",").split(",") if t.strip()]


def run(job: dict) -> Path:
    audio_path = job["audio"]
    out_json = Path(job["output_json"])
    diarize = bool(job.get("diarize"))
    # overall progress ranges per stage
    ranges = ({"transcribe": (2, 60), "align": (60, 70), "diarize": (70, 100)} if diarize
              else {"transcribe": (2, 85), "align": (85, 100)})

    def stage_cb(stage: str, label: str):
        lo, hi = ranges[stage]
        return lambda p: progress(label, lo + (hi - lo) * p / 100.0)

    progress("Loading audio", 0)
    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE
    emit(type="log", message=f"Audio: {duration / 60:.1f} min")

    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    if device == "cpu":
        emit(type="log", message="WARNING: CUDA not available - running on CPU (slow).")

    # ---- 1. transcribe ----
    model_name = job.get("model", "large-v3")
    asr_options = {}
    terms = _vocab_terms(job.get("vocabulary", ""))
    if terms:
        asr_options["initial_prompt"] = "Glossary: " + ", ".join(terms) + "."
    progress(f"Loading {model_name} (first use downloads it)", ranges["transcribe"][0])
    model = whisperx.load_model(model_name, device, compute_type=compute_type, language="en",
                                asr_options=asr_options)
    result = None
    for batch_size in (16, 8, 4, 1):
        try:
            result = model.transcribe(audio, batch_size=batch_size, language="en",
                                      progress_callback=stage_cb("transcribe", "Transcribing"))
            break
        except RuntimeError as e:  # torch.OutOfMemoryError subclasses RuntimeError; ctranslate2 raises plain RuntimeError
            if "out of memory" not in str(e).lower():
                raise
            _free_gpu()
            emit(type="log", message=f"GPU out of memory at batch size {batch_size}; retrying smaller.")
    if result is None:
        raise FriendlyError("GPU ran out of memory even at batch size 1. Close other GPU apps "
                            "(e.g. an Ollama model) or pick a smaller Whisper model.")
    del model
    _free_gpu()
    if not result["segments"]:
        raise FriendlyError("No speech was detected in this file.")

    # ---- 2. align words ----
    progress("Aligning words", ranges["align"][0])
    align_model, meta = whisperx.load_align_model(language_code="en", device=device)
    result = whisperx.align(result["segments"], align_model, meta, audio, device,
                            return_char_alignments=False,
                            progress_callback=stage_cb("align", "Aligning words"))
    del align_model
    _free_gpu()

    # ---- 3. diarize ----
    if diarize:
        from huggingface_hub import get_token
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers

        token = get_token()
        if not token:
            raise FriendlyError("Speaker identification needs a Hugging Face token. Open Settings "
                                f"and add one, and accept the model terms at {DIARIZATION_MODEL_URL}")
        progress("Identifying speakers", ranges["diarize"][0])
        try:
            pipeline = DiarizationPipeline(token=token, device=device)
        except Exception as e:
            if any(k in str(e).lower() for k in ("401", "403", "gated", "access", "unauthorized")):
                raise FriendlyError("Hugging Face refused access to the speaker model. Accept its "
                                    f"terms (while logged in) at {DIARIZATION_MODEL_URL} and check "
                                    "your token in Settings.") from e
            raise
        n = int(job.get("num_speakers") or 0)
        kwargs = {"num_speakers": n} if n > 0 else {
            "min_speakers": job.get("min_speakers") or None,
            "max_speakers": job.get("max_speakers") or None}
        diarize_df = pipeline(audio, progress_callback=stage_cb("diarize", "Identifying speakers"),
                              **kwargs)
        result = assign_word_speakers(diarize_df, result)
        del pipeline
        _free_gpu()

    # ---- 4. build + save document ----
    from .document import TranscriptDoc, build_paragraphs
    from .exporters import now_iso

    doc = TranscriptDoc(audio_path=str(Path(audio_path).resolve()), duration=duration,
                        model=model_name, created=now_iso(), diarized=diarize,
                        paragraphs=build_paragraphs(result["segments"], diarize),
                        vocabulary=job.get("vocabulary", ""))
    if diarize:
        doc.speaker_names = {sid: "" for sid in doc.speaker_ids()}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_json)
    progress("Done", 100)
    return out_json


def main() -> int:
    global _proto
    _proto = sys.stdout
    sys.stdout = sys.stderr  # keep stdout clean for the protocol
    warnings.filterwarnings("ignore")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if len(sys.argv) != 2:
        print("usage: python -m transcript_app.engine job.json", file=sys.stderr)
        return 2
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
    try:
        out = run(job)
        emit(type="done", path=str(out))
        return 0
    except FriendlyError as e:
        emit(type="error", message=str(e))
    except Exception as e:
        traceback.print_exc()
        msg = f"{type(e).__name__}: {e}"
        if "out of memory" in msg.lower():
            msg = "GPU out of memory. Close other GPU apps or choose a smaller model.\n\n" + msg
        emit(type="error", message=msg)
    return 1


if __name__ == "__main__":
    sys.exit(main())
