"""
gui/camera_widget.py
====================
Widget PyQt5 que exibe o stream de vídeo de uma câmera, com overlay
do esqueleto e do nível de postura.

Implementado como um QWidget com paintEvent (em vez de QLabel+setPixmap)
para EVITAR o flickering: o widget pinta o pixmap por cima do conteúdo
anterior sem limpar o fundo a cada frame (QLabel fazia clear+repaint,
causando piscada a ~30 FPS).
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


class CameraWidget(QtWidgets.QWidget):
    def __init__(self, role: str, parent=None):
        super().__init__(parent)
        self.role = role
        self.setMinimumSize(320, 240)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)  # sem limpar fundo
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)

        self._frame: QtGui.QImage | None = None
        self._level: str | None = None
        self._placeholder = True
        self._no_video_since = 0

    def update_frame(self, frame_bgr: np.ndarray, level: str | None = None) -> None:
        rgb = cv2_bgr_to_rgb(frame_bgr)
        h, w, ch = rgb.shape

        if level:
            red, green, blue = level_color(level)
            rgb[:, :, :] = rgb[:, :, :].astype(np.float64) * 0.85 + \
                np.array([red, green, blue], dtype=np.float64) * 0.15
            rgb = rgb.astype(np.uint8)

        # Garantir que QImage possua os dados (não só referência).
        rgb = np.ascontiguousarray(rgb)
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self._frame = qimg.copy()  # copia para o QImage ser dono dos bytes
        self._placeholder = False
        self.update()  # agenda repaint (sem clear de fundo)

    def set_placeholder(self, text: str) -> None:
        self._placeholder = True
        self._frame = None
        self._level = None
        self._placeholder_text = text
        self.update()

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#111111"))

        if self._placeholder or self._frame is None:
            painter.setPen(QtGui.QColor("#cccccc"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter,
                             getattr(self, "_placeholder_text", "[aguardando vídeo]"))
            painter.end()
            return

        # Desenha o frame centralizado, mantendo proporção, na borda do widget.
        pix = QtGui.QPixmap.fromImage(self._frame)
        target = pix.size()
        target.scale(self.size(), QtCore.Qt.KeepAspectRatio)
        # (QtGui.QSize já escala em-place; QSize.scale retorna self)
        x = (self.width() - target.width()) // 2
        y = (self.height() - target.height()) // 2
        painter.drawPixmap(x, y, target.width(), target.height(), pix)
        painter.end()


def cv2_bgr_to_rgb(bgr):
    import cv2
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def level_color(level: str) -> tuple[int, int, int]:
    # RGB
    colors = {
        "good":     (0, 200, 0),
        "warning":  (255, 200, 0),
        "bad":      (255, 120, 0),
        "critical": (255, 0, 0),
    }
    return colors.get(level, (255, 255, 255))