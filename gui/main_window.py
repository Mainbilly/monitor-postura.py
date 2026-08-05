"""
gui/main_window.py
==================
Janela principal PyQt5: exibe as câmeras (frontal + lateral), painel de
stats e integra detecção, análise, alertas e logging num loop de UI.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

import config
from alert_manager import AlertManager, SoundAlertPlayer
from angle_calculator import compute_all
from posture_analyzer import evaluate
from utils.camera_manager import CameraManager
from utils.file_handler import PostureLogger

from gui.camera_widget import CameraWidget
from gui.stats_widget import StatsWidget


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.WINDOW_TITLE)
        self.resize(960, 540)

        # Widgets
        self.cam_widgets = {
            "front": CameraWidget("frontal"),
            "side": CameraWidget("lateral"),
        }
        self.stats = StatsWidget()

        # Layout: vídeo lado a lado + stats embaixo
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.cam_widgets["front"], 1)
        top.addWidget(self.cam_widgets["side"], 1)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(self.stats, 1)
        bottom.addStretch(1)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top, 3)
        lay.addLayout(bottom, 1)

        # Subsistemas
        self.alert = AlertManager()
        self.sound = SoundAlertPlayer(config.SOUND_ENABLED)
        self.alert.set_callback(self._on_alert)

        self.logger = PostureLogger(interval=config.LOG_INTERVAL_SECONDS)
        self.camera_mgr = CameraManager(config.CAMERAS)
        self.camera_mgr.start()

        # Loop de UI
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._loop)
        self.timer.start(config.GUI_UPDATE_MS)

    def _on_alert(self, level: str) -> None:
        self.sound.play(level)
        # Flash na borda da janela + label.
        self.setStyleSheet("border:3px solid "
                           + _qss_color(level) + ";")
        QtCore.QTimer.singleShot(2000, lambda: self.setStyleSheet(""))

    def _loop(self) -> None:
        latest, errors = self.camera_mgr.drain(latest_per_role=True)
        for err in errors:
            print("[camera] erro:", err)

        roles = {it["role"]: it for it in latest}
        front = roles.get("front")
        side = roles.get("side")

        # Só marca "sem vídeo" se a câmera estiver verdadeiramente ausente.
        # Ticks ocasionais sem frame (GUI mais rápida que a câmera) NÃO derivam
        # em placeholder — evitando o piscar preto entre "vídeo"/"sem vídeo".
        for role, w in self.cam_widgets.items():
            if role in roles:
                if hasattr(w, "_no_video_since"):
                    w._no_video_since = None
            else:
                w._no_video_since = getattr(w, "_no_video_since", 0) + 1
                # Só mostra placeholder depois de N ticks consecutivos sem frame.
                if getattr(w, "_no_video_since", 0) >= 15:
                    w.set_placeholder(f"[{role}] sem vídeo...")

        if not latest:
            return  # nenhum frame este tick; mantém último quadro desenhado

        front_angles = compute_all(front["landmarks"]) if front else {}
        side_angles = compute_all(side["landmarks"]) if side else {}
        evaluation = evaluate(front_angles, side_angles)

        # Overlay de vídeo (nível de cada papel)
        front_level = evaluation["levels"].get(
            "tilt", evaluation["levels"].get("cva"))
        if front:
            self.cam_widgets["front"].update_frame(front["frame"], front_level)
        if side:
            side_level = first_level(evaluation["levels"],
                                     ["lean", "kyphosis", "lordosis"])
            self.cam_widgets["side"].update_frame(side["frame"], side_level)

        # Alertas e logging
        if evaluation["worst"]:
            self.alert.update(evaluation["worst"])
        self.logger.log(evaluation)

        # Stats
        self.stats.update_stats(evaluation)


def _qss_color(level: str) -> str:
    return {"good": "green", "warning": "orange",
            "bad": "orange", "critical": "red"}.get(level, "white")


def first_level(levels: dict, names: list) -> str | None:
    """Retorna o nível do primeiro ângulo (dado por `names`) presente em levels."""
    for name in names:
        if name in levels:
            return levels[name]
    return None