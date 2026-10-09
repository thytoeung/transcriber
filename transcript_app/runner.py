"""Background workers: GPU transcription (subprocess) and LLM summaries (thread)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from PySide6.QtCore import QThread, Signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _python_exe() -> str:
    exe = Path(sys.executable)
    if exe.name.lower().startswith("pythonw"):
        console = exe.with_name(exe.name.lower().replace("pythonw", "python"))
        if console.exists():
            return str(console)
    return str(exe)


class TranscribeWorker(QThread):
    progress = Signal(str, float)   # stage, overall percent
    log = Signal(str)
    succeeded = Signal(str)         # path to .transcript.json
    failed = Signal(str)

    def __init__(self, job: dict, parent=None):
        super().__init__(parent)
        self.job = job
        self._proc: subprocess.Popen | None = None
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        if self._proc and self._proc.poll() is None:
            self._proc.kill()

    def run(self) -> None:
        fd, job_file = tempfile.mkstemp(prefix="transcribe_job_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.job, f)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                   PYTHONPATH=str(PROJECT_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        error = None
        done_path = None
        try:
            self._proc = subprocess.Popen(
                [_python_exe(), "-m", "transcript_app.engine", job_file],
                cwd=str(PROJECT_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            stderr_tail: list[str] = []
            t = threading.Thread(target=self._pump_stderr, args=(stderr_tail,), daemon=True)
            t.start()
            for line in self._proc.stdout:
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    self.log.emit(line.rstrip())
                    continue
                kind = msg.get("type")
                if kind == "progress":
                    self.progress.emit(msg["stage"], msg["pct"])
                elif kind == "log":
                    self.log.emit(msg["message"])
                elif kind == "done":
                    done_path = msg["path"]
                elif kind == "error":
                    error = msg["message"]
            code = self._proc.wait()
            t.join(timeout=2)
            if self._cancelled:
                error = "Cancelled"
            elif done_path is None and error is None:
                error = f"Transcription process exited with code {code}.\n" + "\n".join(stderr_tail[-15:])
        except Exception as e:
            error = f"Could not start transcription: {e}"
        finally:
            try:
                os.remove(job_file)
            except OSError:
                pass
        if error:
            self.failed.emit(error)
        else:
            self.succeeded.emit(done_path)

    def _pump_stderr(self, tail: list[str]) -> None:
        for raw in self._proc.stderr:
            # tqdm redraws with \r; keep only the final state of each line
            line = raw.rstrip("\n").split("\r")[-1].strip()
            if not line:
                continue
            tail.append(line)
            del tail[:-50]
            if "%|" not in line:  # skip progress-bar noise
                self.log.emit(line)


class SummaryWorker(QThread):
    progress = Signal(str)
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, doc, model: str, style: str, parent=None):
        super().__init__(parent)
        self.doc, self.model, self.style = doc, model, style

    def run(self) -> None:
        try:
            from localllm import ensure_server
            from .summarize import summarize
            self.progress.emit("Connecting to Ollama…")
            if not ensure_server():
                raise RuntimeError("Ollama isn't running and couldn't be started. "
                                   "Launch the Ollama app and try again.")
            self.succeeded.emit(summarize(self.doc, self.model, self.style, self.progress.emit))
        except Exception as e:
            self.failed.emit(f"Summary failed: {e}")
