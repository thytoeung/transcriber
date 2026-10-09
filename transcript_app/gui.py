"""Main window."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QElapsedTimer, QSettings, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
                               QInputDialog, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox,
                               QSplitter, QStackedWidget, QTabWidget, QTextBrowser, QVBoxLayout,
                               QWidget)

from . import vocab
from .document import TranscriptDoc
from .exporters import FORMATS, export, fmt_time
from .player import AudioPlayer
from .runner import SummaryWorker, TranscribeWorker
from .summarize import STYLES
from .viewer import TranscriptView

MODELS = [
    ("large-v3", "Large v3: best accuracy (default)"),
    ("large-v3-turbo", "Large v3 Turbo: ~3× faster, near-same accuracy"),
    ("distil-large-v3", "Distil Large v3: fast"),
    ("medium.en", "Medium (English)"),
    ("small.en", "Small (English): fastest, rough"),
]
PRESETS = {
    "Solo brainstorm": {"diarize": False, "speakers": 1, "style": "brainstorm"},
    "Client session": {"diarize": True, "speakers": 0, "style": "discovery"},
}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".aiff",
              ".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
DIARIZATION_URL = "https://huggingface.co/pyannote/speaker-diarization-community-1"
STATUS_ICON = {"queued": "⏳", "transcribing": "▶", "summarizing": "✎", "done": "✓", "error": "✗",
               "cancelled": "■"}


@dataclass
class QueueItem:
    audio: str
    preset: str
    model: str
    diarize: bool
    num_speakers: int
    vocab_files: list[str]
    style: str
    out_dir: str
    status: str = "queued"
    detail: str = ""
    json_path: str = ""
    list_item: QListWidgetItem | None = field(default=None, repr=False)

    def target_json(self) -> Path:
        a = Path(self.audio)
        folder = Path(self.out_dir) if self.out_dir else a.parent
        return folder / f"{a.stem}.transcript.json"


# ---------------------------------------------------------------------------

class VocabCombo(QComboBox):
    """An optional extra vocabulary list. Re-scans the folder each time it's opened."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.refresh()

    def refresh(self, keep: str | None = None) -> None:
        keep = self.currentData() if keep is None else keep
        self.blockSignals(True)
        self.clear()
        self.addItem("(none)", "")
        for p in vocab.list_files():
            self.addItem(p.stem, p.name)
        self.setCurrentIndex(max(0, self.findData(keep or "")))
        self.blockSignals(False)

    def showPopup(self) -> None:
        self.refresh()
        super().showPopup()

    def path(self) -> Path | None:
        return vocab.vocab_dir() / self.currentData() if self.currentData() else None


# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(560)
        from huggingface_hub import get_token
        self.token = QLineEdit(get_token() or "")
        self.token.setEchoMode(QLineEdit.Password)
        self.token.setPlaceholderText("hf_…")
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setOpenExternalLinks(True)
        test = QPushButton("Save && test access")
        test.clicked.connect(self._save_and_test)

        help_lbl = QLabel(
            "Speaker identification uses pyannote's free model. One-time setup:<ol>"
            "<li>Create a free account at <a href='https://huggingface.co/join'>huggingface.co</a></li>"
            f"<li>Open <a href='{DIARIZATION_URL}'>pyannote/speaker-diarization-community-1</a> and "
            "accept the terms</li>"
            "<li>Create a <b>Read</b> token at <a href='https://huggingface.co/settings/tokens'>"
            "huggingface.co/settings/tokens</a> and paste it below</li></ol>"
            "After the first download, the model runs offline; no audio leaves your laptop.")
        help_lbl.setOpenExternalLinks(True)
        help_lbl.setWordWrap(True)

        row = QHBoxLayout()
        row.addWidget(self.token, 1)
        row.addWidget(test)
        lay = QVBoxLayout(self)
        lay.addWidget(help_lbl)
        lay.addLayout(row)
        lay.addWidget(self.status)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _save_and_test(self) -> None:
        from huggingface_hub import auth_check, login, whoami
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
        tok = self.token.text().strip()
        if not tok:
            self.status.setText("Enter a token first.")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            user = whoami(token=tok)["name"]
            login(token=tok, add_to_git_credential=False)
            try:
                auth_check("pyannote/speaker-diarization-community-1", token=tok)
                self.status.setText(f"✓ Saved. Signed in as <b>{user}</b>; speaker model access OK.")
            except GatedRepoError:
                self.status.setText(f"Token saved (user <b>{user}</b>), but Hugging Face says this "
                                    f"account hasn't accepted the terms at <a href='{DIARIZATION_URL}'>"
                                    f"{DIARIZATION_URL}</a>. Accept them while signed in as {user}; "
                                    "if you just did, wait a minute and test again.")
            except RepositoryNotFoundError as e:
                self.status.setText(f"Token saved, but the model repo wasn't found: {e}")
            except Exception as e:  # network etc. - show the real error instead of guessing
                self.status.setText(f"Token saved (user <b>{user}</b>), but the access check failed: "
                                    f"{type(e).__name__}: {e}")
        except Exception as e:
            self.status.setText(f"✗ Token rejected: {e}")
        finally:
            QApplication.restoreOverrideCursor()


# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("ThyTools", "Transcriber")
        self.queue: list[QueueItem] = []
        self.current: QueueItem | None = None
        self.worker: TranscribeWorker | None = None
        self.summary_worker: SummaryWorker | None = None
        self.manual_summary_worker: SummaryWorker | None = None
        self.running = False
        self.doc: TranscriptDoc | None = None
        self.doc_path: Path | None = None
        self.dirty = False
        self.elapsed = QElapsedTimer()

        self.setAcceptDrops(True)
        self.resize(1400, 860)
        self._build_ui()
        self._load_settings()
        self._refresh_llm_models()
        self._update_title()

    # ================================================================== UI

    def _build_ui(self) -> None:
        # ---------- left: queue + settings ----------
        self.queue_list = QListWidget()
        self.queue_list.setMinimumHeight(150)
        self.queue_list.itemDoubleClicked.connect(self._open_queue_item)
        add_btn = QPushButton("Add files…")
        add_btn.clicked.connect(self._add_files_dialog)
        rm_btn = QPushButton("Remove")
        rm_btn.clicked.connect(self._remove_selected)
        clr_btn = QPushButton("Clear finished")
        clr_btn.clicked.connect(self._clear_finished)
        qb = QHBoxLayout()
        for b in (add_btn, rm_btn, clr_btn):
            qb.addWidget(b)
        files_box = QGroupBox("Queue (drop audio/video files here)")
        fl = QVBoxLayout(files_box)
        fl.addWidget(self.queue_list)
        fl.addLayout(qb)
        hint = QLabel("Double-click a finished file to open it.")
        hint.setStyleSheet("color: gray;")
        fl.addWidget(hint)

        self.preset = QComboBox()
        self.preset.addItems(list(PRESETS))
        self.preset.currentTextChanged.connect(self._apply_preset)
        self.model = QComboBox()
        for key, label in MODELS:
            self.model.addItem(label, key)
        self.diarize = QCheckBox("Identify speakers")
        self.diarize.setToolTip("Speaker recognition (pyannote). Needs a Hugging Face token; see Settings.")
        self.num_speakers = QSpinBox()
        self.num_speakers.setRange(0, 10)
        self.num_speakers.setSpecialValueText("Auto-detect")
        self.num_speakers.setToolTip("Setting the exact number of people noticeably improves accuracy.")
        self.diarize.toggled.connect(self.num_speakers.setEnabled)
        vocab_box = QWidget()
        vg = QGridLayout(vocab_box)
        vg.setContentsMargins(0, 0, 0, 0)
        default_cb = QCheckBox(f"{vocab.default_file().stem} (always on)")
        default_cb.setChecked(True)
        default_cb.setEnabled(False)
        default_open = QPushButton("Open")
        default_open.setToolTip("Edit in your default text editor")
        default_open.clicked.connect(lambda: self._open_path(vocab.default_file()))
        vg.addWidget(default_cb, 0, 0)
        vg.addWidget(default_open, 0, 1)
        self.vocab_slots: list[VocabCombo] = []
        for row in (1, 2):
            combo = VocabCombo()
            combo.setToolTip("Optional extra list, e.g. one per client")
            btn = QPushButton("Open")
            btn.setToolTip("Edit in your default text editor")
            btn.setEnabled(False)
            btn.clicked.connect(lambda _=False, c=combo: c.path() and self._open_path(c.path()))
            combo.currentIndexChanged.connect(
                lambda _i, c=combo, b=btn: (b.setEnabled(c.path() is not None), self._update_vocab_count()))
            vg.addWidget(combo, row, 0)
            vg.addWidget(btn, row, 1)
            self.vocab_slots.append(combo)
        new_btn = QPushButton("New list…")
        new_btn.clicked.connect(self._new_vocab_list)
        folder_btn = QPushButton("Folder")
        folder_btn.setToolTip(str(vocab.vocab_dir()))
        folder_btn.clicked.connect(lambda: self._open_path(vocab.vocab_dir()))
        self.vocab_count = QLabel()
        self.vocab_count.setStyleSheet("color: gray;")
        vrow = QHBoxLayout()
        vrow.addWidget(new_btn)
        vrow.addWidget(folder_btn)
        vrow.addStretch(1)
        vrow.addWidget(self.vocab_count)
        vg.addLayout(vrow, 3, 0, 1, 2)
        vg.setColumnStretch(0, 1)

        proc_box = QGroupBox("Transcription (applies to files you add)")
        pf = QFormLayout(proc_box)
        pf.addRow("Preset", self.preset)
        pf.addRow("Model", self.model)
        pf.addRow(self.diarize)
        pf.addRow("Speakers", self.num_speakers)
        pf.addRow(QLabel("Custom vocabulary (up to 3 lists)"))
        pf.addRow(vocab_box)

        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("Same folder as the audio")
        self.out_dir.setClearButtonEnabled(True)
        out_btn = QPushButton("…")
        out_btn.setFixedWidth(32)
        out_btn.clicked.connect(self._choose_out_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_dir, 1)
        out_row.addWidget(out_btn)
        self.fmt_checks: dict[str, QCheckBox] = {}
        fmt_row = QHBoxLayout()
        for key in FORMATS:
            cb = QCheckBox(key.upper())
            cb.setToolTip(FORMATS[key])
            self.fmt_checks[key] = cb
            fmt_row.addWidget(cb)
        self.auto_summary = QCheckBox("Summarize with local LLM")
        self.llm_model = QComboBox()
        self.llm_model.setToolTip("Ollama models installed on this laptop")
        llm_refresh = QPushButton("↻")
        llm_refresh.setFixedWidth(32)
        llm_refresh.setToolTip("Refresh model list")
        llm_refresh.clicked.connect(self._refresh_llm_models)
        llm_row = QHBoxLayout()
        llm_row.addWidget(self.llm_model, 1)
        llm_row.addWidget(llm_refresh)

        after_box = QGroupBox("When finished")
        af = QFormLayout(after_box)
        af.addRow("Save to", out_row)
        af.addRow("Export", fmt_row)
        af.addRow(self.auto_summary)
        af.addRow("LLM", llm_row)

        self.start_btn = QPushButton("Transcribe")
        self.start_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self.start_btn.clicked.connect(self._start_queue)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        run_row = QHBoxLayout()
        run_row.addWidget(self.start_btn, 2)
        run_row.addWidget(self.cancel_btn, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.stage_lbl = QLabel("Idle")
        self.stage_lbl.setWordWrap(True)
        settings_btn = QPushButton("Settings (Hugging Face token)…")
        settings_btn.clicked.connect(lambda: SettingsDialog(self).exec())

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(files_box, 1)
        ll.addWidget(proc_box)
        ll.addWidget(after_box)
        ll.addLayout(run_row)
        ll.addWidget(self.progress)
        ll.addWidget(self.stage_lbl)
        ll.addWidget(settings_btn)
        left_scroll = QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(380)

        # ---------- right: viewer / summary / log ----------
        self.file_lbl = QLabel("No transcript open")
        self.file_lbl.setStyleSheet("font-size: 15px; font-weight: bold;")
        open_btn = QPushButton("Open…")
        open_btn.setToolTip("Open a saved .transcript.json")
        open_btn.clicked.connect(self._open_dialog)
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._save)
        self.export_btn = QPushButton("Export")
        self.export_btn.setToolTip("Write the formats ticked under 'When finished', using the "
                                   "current Show options")
        self.export_btn.clicked.connect(self._export_current)
        folder_btn = QPushButton("Open folder")
        folder_btn.clicked.connect(self._open_folder)
        head = QHBoxLayout()
        head.addWidget(self.file_lbl, 1)
        for b in (open_btn, self.save_btn, self.export_btn, folder_btn):
            head.addWidget(b)

        self.view = TranscriptView()
        self.view.changed.connect(self._mark_dirty)
        self.view.seek_requested.connect(lambda s, play: self.player.seek(max(0, s - 0.3), play))
        self.player = AudioPlayer()
        self.player.position_changed.connect(self.view.highlight_time)
        tx = QWidget()
        tl = QVBoxLayout(tx)
        tl.addWidget(self.view, 1)
        tl.addWidget(self.player)

        # summary tab
        self.sum_style = QComboBox()
        for key, (label, _) in STYLES.items():
            self.sum_style.addItem(label, key)
        self.sum_btn = QPushButton("Generate summary")
        self.sum_btn.clicked.connect(self._summarize_current)
        self.sum_edit_toggle = QCheckBox("Edit")
        self.sum_edit_toggle.toggled.connect(self._toggle_summary_edit)
        self.sum_status = QLabel()
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Style"))
        srow.addWidget(self.sum_style)
        srow.addWidget(self.sum_btn)
        srow.addWidget(self.sum_status, 1)
        srow.addWidget(self.sum_edit_toggle)
        self.sum_view = QTextBrowser()
        self.sum_editor = QPlainTextEdit()
        self.sum_editor.textChanged.connect(self._summary_edited)
        self.sum_stack = QStackedWidget()
        self.sum_stack.addWidget(self.sum_view)
        self.sum_stack.addWidget(self.sum_editor)
        sw = QWidget()
        sl = QVBoxLayout(sw)
        sl.addLayout(srow)
        sl.addWidget(self.sum_stack, 1)
        note = QLabel("Runs on the LLM chosen under 'When finished'. Everything stays on this laptop.")
        note.setStyleSheet("color: gray;")
        sl.addWidget(note)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)

        self.tabs = QTabWidget()
        self.tabs.addTab(tx, "Transcript")
        self.tabs.addTab(sw, "Summary")
        self.tabs.addTab(self.log, "Log")
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addLayout(head)
        rl.addWidget(self.tabs, 1)

        split = QSplitter()
        split.addWidget(left_scroll)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([400, 1000])
        self.setCentralWidget(split)

        QShortcut(QKeySequence.Save, self, activated=self._save)
        QShortcut(QKeySequence.Open, self, activated=self._add_files_dialog)
        QShortcut(QKeySequence("Ctrl+Space"), self, activated=self.player.toggle)
        self._set_doc(None, None)

    # ================================================================== settings

    def _load_settings(self) -> None:
        s = self.settings
        self.preset.setCurrentText(s.value("preset", "Client session"))
        self._apply_preset(self.preset.currentText())
        idx = self.model.findData(s.value("model", "large-v3"))
        self.model.setCurrentIndex(max(0, idx))
        self.diarize.setChecked(s.value("diarize", self.diarize.isChecked(), type=bool))
        self.num_speakers.setValue(s.value("num_speakers", self.num_speakers.value(), type=int))
        self.num_speakers.setEnabled(self.diarize.isChecked())
        for combo, name in zip(self.vocab_slots, s.value("vocab_slots", ",").split(",")):
            combo.refresh(keep=name)
            combo.currentIndexChanged.emit(combo.currentIndex())
        self._update_vocab_count()
        self.out_dir.setText(s.value("out_dir", ""))
        fmts = s.value("formats", "txt,docx").split(",")
        for k, cb in self.fmt_checks.items():
            cb.setChecked(k in fmts)
        self.auto_summary.setChecked(s.value("auto_summary", False, type=bool))
        self.view.show_time.setChecked(s.value("show_time", True, type=bool))
        self.view.show_speakers.setChecked(s.value("show_speakers", True, type=bool))
        if geo := s.value("geometry"):
            self.restoreGeometry(geo)

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("preset", self.preset.currentText())
        s.setValue("model", self.model.currentData())
        s.setValue("diarize", self.diarize.isChecked())
        s.setValue("num_speakers", self.num_speakers.value())
        s.setValue("vocab_slots", ",".join(c.currentData() or "" for c in self.vocab_slots))
        s.setValue("out_dir", self.out_dir.text())
        s.setValue("formats", ",".join(k for k, cb in self.fmt_checks.items() if cb.isChecked()))
        s.setValue("auto_summary", self.auto_summary.isChecked())
        s.setValue("llm_model", self.llm_model.currentText())
        s.setValue("show_time", self.view.show_time.isChecked())
        s.setValue("show_speakers", self.view.show_speakers.isChecked())
        s.setValue("geometry", self.saveGeometry())

    def _apply_preset(self, name: str) -> None:
        p = PRESETS.get(name)
        if not p:
            return
        self.diarize.setChecked(p["diarize"])
        self.num_speakers.setValue(p["speakers"])
        self.num_speakers.setEnabled(p["diarize"])
        self.sum_style.setCurrentIndex(max(0, self.sum_style.findData(p["style"])))

    def _refresh_llm_models(self) -> None:
        from localllm import list_models
        current = self.llm_model.currentText() or self.settings.value("llm_model", "qwen3:8b")
        models = list_models(timeout=2)
        self.llm_model.clear()
        self.llm_model.addItems(models or [current])
        self.llm_model.setCurrentText(current if current in models or not models else models[0])
        if not models:
            self._log("Ollama not reachable right now; it will be started when a summary is requested.")

    def _choose_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Save transcripts to", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    def _formats(self) -> list[str]:
        return [k for k, cb in self.fmt_checks.items() if cb.isChecked()]

    # ================================================================== vocabulary

    def _selected_vocab_files(self) -> list[Path]:
        files = [vocab.default_file()]
        for c in self.vocab_slots:
            if c.path() and c.path() not in files:
                files.append(c.path())
        return files

    def _update_vocab_count(self) -> None:
        n = len(vocab.read_terms(self._selected_vocab_files()))
        self.vocab_count.setText(f"{n} term{'s' if n != 1 else ''}")

    def _new_vocab_list(self) -> None:
        name, ok = QInputDialog.getText(self, "New vocabulary list", "Name (e.g. the client):")
        if not (ok and name.strip()):
            return
        try:
            path = vocab.create(name)
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "New vocabulary list", str(e))
            return
        target = next((c for c in self.vocab_slots if not c.currentData()), self.vocab_slots[-1])
        for c in self.vocab_slots:
            c.refresh(keep=path.name if c is target else None)
        target.currentIndexChanged.emit(target.currentIndex())
        self._open_path(path)

    def _open_path(self, path: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def changeEvent(self, e) -> None:
        # refresh the term count when coming back from editing a list in Notepad
        if e.type() == e.Type.ActivationChange and self.isActiveWindow() and hasattr(self, "vocab_count"):
            self._update_vocab_count()
        super().changeEvent(e)

    # ================================================================== queue

    def _add_files_dialog(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(AUDIO_EXTS))
        files, _ = QFileDialog.getOpenFileNames(self, "Add recordings", self.settings.value("last_dir", ""),
                                                f"Audio/video ({exts});;All files (*)")
        if files:
            self.settings.setValue("last_dir", str(Path(files[0]).parent))
            self._enqueue(files)

    def _enqueue(self, paths: list[str]) -> None:
        for path in paths:
            item = QueueItem(audio=path, preset=self.preset.currentText(), model=self.model.currentData(),
                             diarize=self.diarize.isChecked(), num_speakers=self.num_speakers.value(),
                             vocab_files=[str(p) for p in self._selected_vocab_files()], style=PRESETS.get(
                                 self.preset.currentText(), {}).get("style", "discovery"),
                             out_dir=self.out_dir.text().strip())
            item.list_item = QListWidgetItem()
            item.list_item.setToolTip(path)
            item.list_item.setData(Qt.UserRole, id(item))
            self.queue.append(item)
            self.queue_list.addItem(item.list_item)
            self._render_item(item)
        self._save_settings()

    def _render_item(self, item: QueueItem) -> None:
        spk = ("speakers: auto" if item.num_speakers == 0 else f"{item.num_speakers} speakers") \
            if item.diarize else "no speakers"
        detail = f" · {item.detail}" if item.detail else ""
        lists = ", ".join(Path(f).stem for f in item.vocab_files)
        item.list_item.setText(f"{STATUS_ICON[item.status]}  {Path(item.audio).name}\n"
                               f"      {item.preset} · {item.model} · {spk}{detail}\n"
                               f"      vocabulary: {lists}")

    def _item_for(self, li: QListWidgetItem) -> QueueItem | None:
        return next((q for q in self.queue if q.list_item is li), None)

    def _remove_selected(self) -> None:
        for li in self.queue_list.selectedItems():
            item = self._item_for(li)
            if item and item is not self.current:
                self.queue.remove(item)
                self.queue_list.takeItem(self.queue_list.row(li))

    def _clear_finished(self) -> None:
        for item in [q for q in self.queue if q.status in ("done", "error", "cancelled")]:
            self.queue.remove(item)
            self.queue_list.takeItem(self.queue_list.row(item.list_item))

    def _open_queue_item(self, li: QListWidgetItem) -> None:
        item = self._item_for(li)
        if item and item.json_path and Path(item.json_path).exists():
            self._open_doc(Path(item.json_path))
        elif item and item.status == "error":
            QMessageBox.warning(self, "Transcription failed", item.detail)

    def _start_queue(self) -> None:
        if self.running:
            return
        for q in self.queue:
            if q.status in ("error", "cancelled"):
                q.status, q.detail = "queued", ""
                self._render_item(q)
        if not any(q.status == "queued" for q in self.queue):
            self._add_files_dialog()
            if not any(q.status == "queued" for q in self.queue):
                return
        self._save_settings()
        self.running = True
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._next()

    def _next(self) -> None:
        item = next((q for q in self.queue if q.status == "queued"), None) if self.running else None
        if item is None:
            self._finish_queue()
            return
        self.current = item
        item.status, item.detail = "transcribing", "starting"
        self._render_item(item)
        self.elapsed.start()
        self.progress.setValue(0)
        self._log(f"=== {Path(item.audio).name} ===")
        # Lists are read now (not when queued) so edits made in the editor meanwhile are picked up.
        terms = vocab.read_terms(item.vocab_files)
        vocabulary = ", ".join(terms)
        self._log(f"Vocabulary: {len(terms)} terms from {', '.join(Path(f).stem for f in item.vocab_files)}")
        if len(vocabulary) > vocab.MAX_PROMPT_CHARS:
            self._log("Warning: vocabulary is long; Whisper only reads roughly the last 200 words, "
                      "so terms near the top of the default list may be ignored.")
        job = {"audio": item.audio, "output_json": str(item.target_json()), "model": item.model,
               "diarize": item.diarize, "num_speakers": item.num_speakers,
               "vocabulary": vocabulary}
        self.worker = TranscribeWorker(job, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._log)
        self.worker.succeeded.connect(self._on_transcribed)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, stage: str, pct: float) -> None:
        self.progress.setValue(int(pct))
        secs = self.elapsed.elapsed() / 1000
        self.stage_lbl.setText(f"{stage}… {pct:.0f}%  ·  {fmt_time(secs)} elapsed")
        if self.current:
            self.current.detail = f"{stage.lower()} {pct:.0f}%"
            self._render_item(self.current)

    def _on_failed(self, msg: str) -> None:
        item = self.current
        item.status = "cancelled" if msg == "Cancelled" else "error"
        item.detail = msg if item.status == "error" else ""
        self._render_item(item)
        self._log(f"✗ {msg}")
        if item.status == "error":
            self.stage_lbl.setText("Failed. Double-click the file in the queue for details.")
            if "Hugging Face" in msg:
                QMessageBox.warning(self, "Speaker identification", msg)
        self.current = None
        self._next()

    def _on_transcribed(self, path: str) -> None:
        item = self.current
        item.json_path = path
        self._log(f"Transcribed in {fmt_time(self.elapsed.elapsed() / 1000)} → {path}")
        if self.auto_summary.isChecked() and self.llm_model.currentText():
            item.status, item.detail = "summarizing", ""
            self._render_item(item)
            doc = TranscriptDoc.load(path)
            self.summary_worker = SummaryWorker(doc, self.llm_model.currentText(), item.style, self)
            self.summary_worker.progress.connect(lambda s: self.stage_lbl.setText(s))
            self.summary_worker.succeeded.connect(lambda text: self._finish_item(item, doc, text))
            self.summary_worker.failed.connect(lambda err: (self._log(err),
                                                            self._finish_item(item, doc, None)))
            self.summary_worker.start()
        else:
            self._finish_item(item, TranscriptDoc.load(path), None)

    def _finish_item(self, item: QueueItem, doc: TranscriptDoc, summary: str | None) -> None:
        path = Path(item.json_path)
        if summary:
            doc.summary, doc.summary_model = summary, self.llm_model.currentText()
            doc.save(path)
        try:
            written = export(doc, path, self._formats(), self.view.export_options())
            if written:
                self._log("Exported: " + ", ".join(p.name for p in written))
        except Exception as e:
            self._log(f"Export failed: {e}")
        item.status, item.detail = "done", fmt_time(doc.duration)
        self._render_item(item)
        if self.doc is None or (self.doc_path == path and not self.dirty):
            self._open_doc(path, force=True)
        self.current = None
        self._next()

    def _finish_queue(self) -> None:
        was_running = self.running
        self.running = False
        self.current = None
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if was_running:
            self.progress.setValue(100)
            done = sum(q.status == "done" for q in self.queue)
            self.stage_lbl.setText(f"Queue finished: {done} done.")
            QApplication.alert(self)

    def _cancel(self) -> None:
        self.running = False  # stop after the current step
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
        self.stage_lbl.setText("Cancelling…")

    # ================================================================== document

    def _set_doc(self, doc: TranscriptDoc | None, path: Path | None) -> None:
        self.doc, self.doc_path, self.dirty = doc, path, False
        self.view.set_doc(doc)
        self.player.load(doc.audio_path if doc and Path(doc.audio_path).exists() else None)
        if doc:
            self.file_lbl.setText(f"{Path(doc.audio_path).name}  ·  {fmt_time(doc.duration)}  ·  "
                                  f"{len(doc.paragraphs)} paragraphs")
            self.sum_style.setCurrentIndex(self.sum_style.findData(
                "discovery" if doc.diarized else "brainstorm"))
            if not Path(doc.audio_path).exists():
                self._log(f"Audio not found for playback: {doc.audio_path}")
        else:
            self.file_lbl.setText("No transcript open. Add recordings on the left, or Open… a saved one.")
        for w in (self.save_btn, self.export_btn, self.sum_btn):
            w.setEnabled(doc is not None)
        self._show_summary()
        self._update_title()

    def _open_doc(self, path: Path, force: bool = False) -> None:
        if not force and not self._confirm_discard():
            return
        try:
            self._set_doc(TranscriptDoc.load(path), path)
            self.tabs.setCurrentIndex(0)
        except Exception as e:
            QMessageBox.critical(self, "Open failed", f"{path}\n\n{e}")

    def _open_dialog(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Open transcript", self.settings.value("last_dir", ""),
                                           "Transcripts (*.transcript.json *.json)")
        if f:
            self._open_doc(Path(f))

    def _mark_dirty(self) -> None:
        self.dirty = True
        self._update_title()

    def _update_title(self) -> None:
        name = Path(self.doc.audio_path).name if self.doc else ""
        self.setWindowTitle(f"{'*' if self.dirty else ''}{name}{' - ' if name else ''}Transcriber")

    def _save(self) -> bool:
        if not (self.doc and self.doc_path):
            return False
        self.doc.save(self.doc_path)
        self.dirty = False
        self._update_title()
        self.stage_lbl.setText(f"Saved {self.doc_path.name}")
        return True

    def _export_current(self) -> None:
        if not self.doc:
            return
        fmts = self._formats()
        if not fmts:
            QMessageBox.information(self, "Export", "Tick at least one format under 'When finished'.")
            return
        self._save()
        written = export(self.doc, self.doc_path, fmts, self.view.export_options())
        self.stage_lbl.setText("Exported " + ", ".join(p.suffix for p in written))
        self._log("Exported: " + ", ".join(str(p) for p in written))

    def _open_folder(self) -> None:
        target = self.doc_path.parent if self.doc_path else (
            Path(self.out_dir.text()) if self.out_dir.text() else None)
        if target and target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Save changes to the current transcript?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return self._save()
        return r == QMessageBox.Discard

    # ================================================================== summary

    def _show_summary(self) -> None:
        text = self.doc.summary if self.doc else ""
        self.sum_editor.blockSignals(True)
        self.sum_editor.setPlainText(text)
        self.sum_editor.blockSignals(False)
        self.sum_view.setMarkdown(text or "*No summary yet. Choose a style and click Generate summary.*")
        model = f"Generated with {self.doc.summary_model}" if self.doc and self.doc.summary_model else ""
        self.sum_status.setText(model)

    def _toggle_summary_edit(self, editing: bool) -> None:
        if not editing:
            self.sum_view.setMarkdown(self.sum_editor.toPlainText())
        self.sum_stack.setCurrentIndex(1 if editing else 0)

    def _summary_edited(self) -> None:
        if self.doc:
            self.doc.summary = self.sum_editor.toPlainText()
            self._mark_dirty()

    def _summarize_current(self) -> None:
        if not self.doc or (self.manual_summary_worker and self.manual_summary_worker.isRunning()):
            return
        if self.worker and self.worker.isRunning():
            r = QMessageBox.question(self, "GPU busy", "A transcription is running. Summarizing now "
                                     "shares the GPU and may run out of memory. Continue anyway?")
            if r != QMessageBox.Yes:
                return
        model = self.llm_model.currentText()
        doc = self.doc
        self.sum_btn.setEnabled(False)
        self.tabs.setCurrentIndex(1)
        w = SummaryWorker(doc, model, self.sum_style.currentData(), self)
        w.progress.connect(self.sum_status.setText)

        def done(text: str):
            self.sum_btn.setEnabled(True)
            if self.doc is doc:
                doc.summary, doc.summary_model = text, model
                self._show_summary()
                self._mark_dirty()

        def failed(err: str):
            self.sum_btn.setEnabled(True)
            self.sum_status.setText(err)
            self._log(err)

        w.succeeded.connect(done)
        w.failed.connect(failed)
        self.manual_summary_worker = w
        w.start()

    # ================================================================== misc

    def _log(self, msg: str) -> None:
        self.log.appendPlainText(msg)

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        docs = [p for p in paths if p.lower().endswith(".json")]
        audio = [p for p in paths if Path(p).suffix.lower() in AUDIO_EXTS]
        if audio:
            self._enqueue(audio)
        if docs:
            self._open_doc(Path(docs[0]))

    def closeEvent(self, e) -> None:
        if self.running or (self.worker and self.worker.isRunning()):
            r = QMessageBox.question(self, "Transcription running",
                                     "A transcription is in progress. Cancel it and quit?")
            if r != QMessageBox.Yes:
                e.ignore()
                return
            self._cancel()
            self.worker.wait(5000)
        if not self._confirm_discard():
            e.ignore()
            return
        self._save_settings()
        e.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Transcriber")
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    for arg in sys.argv[1:]:  # allow "Open with…" / drag onto the launcher
        if Path(arg).suffix.lower() in AUDIO_EXTS:
            win._enqueue([arg])
        elif arg.lower().endswith(".json"):
            win._open_doc(Path(arg))
    return app.exec()
