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
        self.setObjectName("MainWindow")
        self.setStyleSheet(_QSS)

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

        # Suavização (EMA) dos ângulos entre frames — evita o medidor oscilando.
        self._prev_angles: dict | None = None

        # Loop de UI
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._loop)
        self.timer.start(config.GUI_UPDATE_MS)

    def _on_alert(self, level: str) -> None:
        self.sound.play(level)
        # Flash na borda da janela (mantém o tema escuro do app).
        flash = _QSS + f"\nQWidget#MainWindow {{ border: 3px solid {_qss_color(level)}; }}"
        self.setStyleSheet(flash)
        QtCore.QTimer.singleShot(2000, lambda: self.setStyleSheet(_QSS))

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
                w._no_video_since = 0
            else:
                w._no_video_since += 1
                # Só mostra placeholder depois de N ticks consecutivos sem frame.
                if w._no_video_since >= 15:
                    w.set_placeholder(f"[{role}] sem vídeo...")

        if not latest:
            return  # nenhum frame este tick; mantém último quadro desenhado

        front_angles = compute_all(front["landmarks"],
                                   _aspect(front["frame"])) if front else {}
        side_angles = compute_all(side["landmarks"],
                                  _aspect(side["frame"])) if side else {}
        evaluation = evaluate(front_angles, side_angles,
                              prev_angles=self._prev_angles)
        self._prev_angles = {n: info["value"]
                             for n, info in evaluation["angles"].items()}

        # Vídeo exibido SEM tint/overlay: a medição de postura fica apenas
        # no painel de estatísticas (o "medidor").
        if front:
            self.cam_widgets["front"].update_frame(front["frame"])
        if side:
            self.cam_widgets["side"].update_frame(side["frame"])

        # Alertas e logging
        self.alert.update(evaluation["worst"])
        self.logger.log(evaluation)

        # Stats
        self.stats.update_stats(evaluation)


def _qss_color(level: str) -> str:
    return {"good": "#00C800", "warning": "#FFC800",
            "bad": "#FF7800", "critical": "#FF0000"}.get(level, "#ffffff")


_QSS = """
QWidget#MainWindow {
    background-color: #1e1e2e;
    color: #e0e0e0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}
QWidget#statsCard {
    background-color: #26263a;
    border: 1px solid #34344c;
    border-radius: 12px;
    padding: 10px;
}
QLabel#cardTitle {
    color: #9a9ac0;
    font-size: 12px;
    font-weight: 600;
}
"""


def _aspect(frame) -> float:
    """Razão largura/altura do frame (para compensar a distorção de aspecto)."""
    h, w = frame.shape[:2]
    return w / h if h else 1.0