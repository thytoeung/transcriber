"""Transcript review/edit widget: paragraph table, speaker renaming, find, playback sync."""

from __future__ import annotations

import bisect
import re

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QMenu, QPlainTextEdit, QScrollArea,
                               QStyledItemDelegate, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from .document import TranscriptDoc, default_speaker_name
from .exporters import ExportOptions, fmt_time

SPEAKER_COLORS = ["#2b7bd6", "#d6452b", "#2a9d4b", "#9b4fd1", "#d68a00", "#1595a8", "#b0457f", "#6d7a1f"]
COL_TIME, COL_SPEAKER, COL_TEXT = range(3)
NEW_SPEAKER = "__new__"


class _TextDelegate(QStyledItemDelegate):
    def __init__(self, view: "TranscriptView"):
        super().__init__(view)
        self.view = view

    def createEditor(self, parent, option, index):
        ed = QPlainTextEdit(parent)
        ed.setTabChangesFocus(True)
        return ed

    def setEditorData(self, editor, index):
        editor.setPlainText(index.data(Qt.EditRole) or "")

    def setModelData(self, editor, model, index):
        self.view.set_paragraph_text(index.row(), editor.toPlainText())

    def updateEditorGeometry(self, editor, option, index):
        r = option.rect
        r.setHeight(max(r.height(), 110))
        editor.setGeometry(r)


class _SpeakerDelegate(QStyledItemDelegate):
    def __init__(self, view: "TranscriptView"):
        super().__init__(view)
        self.view = view

    def createEditor(self, parent, option, index):
        cb = QComboBox(parent)
        doc = self.view.doc
        for sid in doc.speaker_ids():
            cb.addItem(doc.speaker_label(sid), sid)
        cb.addItem("+ New speaker", NEW_SPEAKER)
        return cb

    def setEditorData(self, editor, index):
        i = editor.findData(self.view.doc.paragraphs[index.row()].speaker)
        editor.setCurrentIndex(max(0, i))
        editor.showPopup()

    def setModelData(self, editor, model, index):
        self.view.set_paragraph_speaker(index.row(), editor.currentData())


class TranscriptView(QWidget):
    changed = Signal()                  # document edited
    seek_requested = Signal(float, bool)  # seconds, start playing
    display_changed = Signal()          # timecode/speaker toggles

    def __init__(self, parent=None):
        super().__init__(parent)
        self.doc: TranscriptDoc | None = None
        self._starts: list[float] = []
        self._follow_row = -1

        # --- toolbar ---
        self.show_time = QCheckBox("Timecodes")
        self.show_speakers = QCheckBox("Speaker names")
        for cb in (self.show_time, self.show_speakers):
            cb.setChecked(True)
            cb.setToolTip("Shown here and included in exports")
            cb.toggled.connect(self._apply_column_visibility)
            cb.toggled.connect(self.display_changed)
        self.follow = QCheckBox("Follow playback")
        self.follow.setChecked(True)
        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("Find…  (Enter = next)")
        self.find_edit.setClearButtonEnabled(True)
        self.find_edit.setMaximumWidth(240)
        self.find_edit.returnPressed.connect(self.find_next)
        self.find_status = QLabel()

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Show:"))
        bar.addWidget(self.show_time)
        bar.addWidget(self.show_speakers)
        bar.addSpacing(12)
        bar.addWidget(self.follow)
        bar.addStretch(1)
        bar.addWidget(self.find_status)
        bar.addWidget(self.find_edit)

        # --- speaker names ---
        self.speaker_box = QWidget()
        self.speaker_lay = QHBoxLayout(self.speaker_box)
        self.speaker_lay.setContentsMargins(0, 0, 0, 0)
        self.speaker_scroll = QScrollArea()
        self.speaker_scroll.setWidget(self.speaker_box)
        self.speaker_scroll.setWidgetResizable(True)
        self.speaker_scroll.setFixedHeight(46)
        self.speaker_scroll.setFrameShape(QScrollArea.NoFrame)

        # --- table ---
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Time", "Speaker", "Text"])
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(COL_TIME, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(COL_SPEAKER, QHeaderView.Interactive)
        hh.setSectionResizeMode(COL_TEXT, QHeaderView.Stretch)
        self.table.setColumnWidth(COL_SPEAKER, 130)
        self.table.setItemDelegateForColumn(COL_TEXT, _TextDelegate(self))
        self.table.setItemDelegateForColumn(COL_SPEAKER, _SpeakerDelegate(self))
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self._resize_timer = QTimer(self, singleShot=True, interval=150)
        self._resize_timer.timeout.connect(self.table.resizeRowsToContents)
        hh.sectionResized.connect(lambda *_: self._resize_timer.start())

        self.hint = QLabel("Click a timecode to play from there · double-click text or speaker to "
                           "edit · right-click to merge paragraphs")
        self.hint.setStyleSheet("color: gray;")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.speaker_scroll)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.hint)

        QShortcut(QKeySequence.Find, self, activated=lambda: (self.find_edit.setFocus(),
                                                               self.find_edit.selectAll()))

    # ------------------------------------------------------------------ loading

    def set_doc(self, doc: TranscriptDoc | None) -> None:
        self.doc = doc
        self._follow_row = -1
        self._rebuild_speaker_panel()
        self._rebuild_rows()

    def export_options(self, summary: bool = True) -> ExportOptions:
        return ExportOptions(timecodes=self.show_time.isChecked(),
                             speakers=self.show_speakers.isChecked(), summary=summary)

    def _speaker_color(self, sid: str | None) -> QColor:
        ids = self.doc.speaker_ids() if self.doc else []
        i = ids.index(sid) if sid in ids else 0
        return QColor(SPEAKER_COLORS[i % len(SPEAKER_COLORS)])

    def _rebuild_rows(self) -> None:
        t = self.table
        t.setRowCount(0)
        if not self.doc:
            self._starts = []
            self._apply_column_visibility()
            return
        paras = self.doc.paragraphs
        self._starts = [p.start for p in paras]
        t.setRowCount(len(paras))
        bold = QFont()
        bold.setBold(True)
        for r, p in enumerate(paras):
            ti = QTableWidgetItem(f"▶ {fmt_time(p.start)}")
            ti.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            ti.setForeground(QBrush(QColor("#2b7bd6")))
            ti.setToolTip("Play from here")
            si = QTableWidgetItem(self.doc.speaker_label(p.speaker))
            si.setFont(bold)
            si.setForeground(QBrush(self._speaker_color(p.speaker)))
            if not self.doc.diarized:
                si.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            xi = QTableWidgetItem(p.text)
            if p.edited:
                xi.setToolTip("Edited by hand")
            t.setItem(r, COL_TIME, ti)
            t.setItem(r, COL_SPEAKER, si)
            t.setItem(r, COL_TEXT, xi)
        self._apply_column_visibility()
        t.resizeRowsToContents()

    def _refresh_speaker_cells(self) -> None:
        for r, p in enumerate(self.doc.paragraphs):
            item = self.table.item(r, COL_SPEAKER)
            item.setText(self.doc.speaker_label(p.speaker))
            item.setForeground(QBrush(self._speaker_color(p.speaker)))

    def _apply_column_visibility(self) -> None:
        self.table.setColumnHidden(COL_TIME, not self.show_time.isChecked())
        diarized = bool(self.doc and self.doc.diarized)
        self.table.setColumnHidden(COL_SPEAKER, not (diarized and self.show_speakers.isChecked()))
        self.speaker_scroll.setVisible(diarized)

    def _rebuild_speaker_panel(self) -> None:
        while self.speaker_lay.count():
            w = self.speaker_lay.takeAt(0).widget()
            if w:
                w.deleteLater()
        if not (self.doc and self.doc.diarized):
            return
        self.speaker_lay.addWidget(QLabel("Rename speakers:"))
        for sid in self.doc.speaker_ids():
            swatch = QLabel("●")
            swatch.setStyleSheet(f"color: {self._speaker_color(sid).name()}; font-size: 16px;")
            edit = QLineEdit(self.doc.speaker_names.get(sid, ""))
            edit.setPlaceholderText(default_speaker_name(sid))
            edit.setFixedWidth(130)
            edit.editingFinished.connect(lambda s=sid, e=edit: self._rename_speaker(s, e.text()))
            self.speaker_lay.addWidget(swatch)
            self.speaker_lay.addWidget(edit)
        self.speaker_lay.addStretch(1)

    # ------------------------------------------------------------------ editing

    def _rename_speaker(self, sid: str, name: str) -> None:
        name = name.strip()
        if self.doc.speaker_names.get(sid, "") == name:
            return
        self.doc.speaker_names[sid] = name
        self._refresh_speaker_cells()
        self.changed.emit()

    def set_paragraph_text(self, row: int, text: str) -> None:
        text = re.sub(r"\s+", " ", text).strip()
        p = self.doc.paragraphs[row]
        if text == p.text:
            return
        p.text, p.edited = text, True
        item = self.table.item(row, COL_TEXT)
        item.setText(text)
        item.setToolTip("Edited by hand")
        self.table.resizeRowToContents(row)
        self.changed.emit()

    def set_paragraph_speaker(self, row: int, sid: str) -> None:
        if sid == NEW_SPEAKER:
            nums = [int(m.group(1)) for s in self.doc.speaker_ids() if (m := re.search(r"(\d+)$", s))]
            sid = f"SPEAKER_{(max(nums) + 1) if nums else 0:02d}"
            self.doc.speaker_names[sid] = ""
        p = self.doc.paragraphs[row]
        if p.speaker == sid:
            return
        p.speaker = sid
        for w in p.words:
            w.speaker = sid
        self._rebuild_speaker_panel()
        self._refresh_speaker_cells()
        self.changed.emit()

    def _merge(self, row: int) -> None:
        """Merge paragraph `row` into the one before it."""
        if row <= 0:
            return
        prev, p = self.doc.paragraphs[row - 1], self.doc.paragraphs[row]
        prev.text = f"{prev.text} {p.text}".strip()
        prev.words += p.words
        prev.end = max(prev.end, p.end)
        prev.edited = prev.edited or p.edited or prev.speaker != p.speaker
        del self.doc.paragraphs[row]
        self._rebuild_rows()
        self.table.selectRow(row - 1)
        self.changed.emit()

    def _context_menu(self, pos) -> None:
        row = self.table.rowAt(pos.y())
        if not self.doc or row < 0:
            return
        menu = QMenu(self)
        menu.addAction("Play from here", lambda: self.seek_requested.emit(self.doc.paragraphs[row].start, True))
        menu.addSeparator()
        a = menu.addAction("Merge with previous paragraph", lambda: self._merge(row))
        a.setEnabled(row > 0)
        b = menu.addAction("Merge with next paragraph", lambda: self._merge(row + 1))
        b.setEnabled(row + 1 < len(self.doc.paragraphs))
        menu.exec(self.table.viewport().mapToGlobal(pos))

    # ------------------------------------------------------------------ navigation

    def _on_cell_clicked(self, row: int, col: int) -> None:
        if self.doc and col == COL_TIME:
            self.seek_requested.emit(self.doc.paragraphs[row].start, True)

    def highlight_time(self, sec: float) -> None:
        if not (self.doc and self.follow.isChecked() and self._starts):
            return
        if self.table.state() == QAbstractItemView.EditingState:
            return
        row = max(0, bisect.bisect_right(self._starts, sec + 0.05) - 1)
        if row != self._follow_row:
            self._follow_row = row
            self.table.selectRow(row)
            self.table.scrollTo(self.table.model().index(row, COL_TEXT),
                                QAbstractItemView.PositionAtCenter)

    def find_next(self) -> None:
        q = self.find_edit.text().strip().lower()
        if not (self.doc and q):
            self.find_status.clear()
            return
        paras = self.doc.paragraphs
        hits = [i for i, p in enumerate(paras) if q in p.text.lower()]
        if not hits:
            self.find_status.setText("No matches")
            return
        cur = self.table.currentRow()
        nxt = next((i for i in hits if i > cur), hits[0])
        self.find_status.setText(f"{hits.index(nxt) + 1} of {len(hits)}")
        self.follow.setChecked(False)  # don't let playback yank the selection away
        self.table.setCurrentIndex(self.table.model().index(nxt, COL_TEXT))
        self.table.scrollTo(self.table.model().index(nxt, COL_TEXT), QAbstractItemView.PositionAtCenter)
