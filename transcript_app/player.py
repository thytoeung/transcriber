"""Compact audio player bar (Qt Multimedia - uses Qt's bundled FFmpeg backend)."""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSlider, QStyle, QWidget

from .exporters import fmt_time


class AudioPlayer(QWidget):
    position_changed = Signal(float)  # seconds

    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = QMediaPlayer(self)
        self.audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_out)

        self.play_btn = QPushButton(self.style().standardIcon(QStyle.SP_MediaPlay), "")
        self.play_btn.setToolTip("Play / pause (Ctrl+Space)")
        self.back_btn = QPushButton("−5s")
        self.fwd_btn = QPushButton("+5s")
        for b in (self.back_btn, self.fwd_btn):
            b.setFixedWidth(44)
        self.slider = QSlider(Qt.Horizontal)
        self.time_lbl = QLabel("00:00:00 / 00:00:00")
        self.speed = QComboBox()
        self.speed.addItems(["0.75×", "1×", "1.25×", "1.5×", "1.75×", "2×"])
        self.speed.setCurrentText("1×")
        self.speed.setToolTip("Playback speed")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        for w in (self.play_btn, self.back_btn, self.fwd_btn):
            lay.addWidget(w)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.time_lbl)
        lay.addWidget(self.speed)

        self.play_btn.clicked.connect(self.toggle)
        self.back_btn.clicked.connect(lambda: self.seek(self.position() - 5))
        self.fwd_btn.clicked.connect(lambda: self.seek(self.position() + 5))
        self.slider.sliderMoved.connect(self.player.setPosition)
        self.speed.currentTextChanged.connect(
            lambda t: self.player.setPlaybackRate(float(t.rstrip("×"))))
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(lambda ms: self.slider.setRange(0, ms))
        self.player.playbackStateChanged.connect(self._on_state)
        self.setEnabled(False)

    def load(self, path: str | None) -> None:
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(path) if path else QUrl())
        self.setEnabled(bool(path))

    def position(self) -> float:
        return self.player.position() / 1000.0

    def seek(self, sec: float, play: bool = False) -> None:
        self.player.setPosition(int(max(0.0, sec) * 1000))
        if play:
            self.player.play()

    def toggle(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _on_position(self, ms: int) -> None:
        if not self.slider.isSliderDown():
            self.slider.setValue(ms)
        self.time_lbl.setText(f"{fmt_time(ms / 1000)} / {fmt_time(self.player.duration() / 1000)}")
        self.position_changed.emit(ms / 1000.0)

    def _on_state(self, state) -> None:
        icon = QStyle.SP_MediaPause if state == QMediaPlayer.PlayingState else QStyle.SP_MediaPlay
        self.play_btn.setIcon(self.style().standardIcon(icon))
