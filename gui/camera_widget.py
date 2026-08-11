"""
gui/camera_widget.py
====================
Widget PyQt5 que exibe o stream de vídeo de uma câmera (com o esqueleto
desenhado pelo detector). O vídeo NÃO é tintado pelo nível de postura:
a medição fica apenas no painel de estatísticas.

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
        self._placeholder = True
        self._no_video_since = 0
        self._title = role.capitalize()

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        # O vídeo é exibido SEM tint/overlay de nível: a medição de postura
        # fica por conta apenas do painel de estatísticas (o "medidor").
        rgb = cv2_bgr_to_rgb(frame_bgr)
        h, w, ch = rgb.shape

        # Garantir que QImage possua os dados (não só referência).
        rgb = np.ascontiguousarray(rgb)
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self._frame = qimg.copy()  # copia para o QImage ser dono dos bytes
        self._placeholder = False
        self.update()  # agenda repaint (sem clear de fundo)

    def set_placeholder(self, text: str) -> None:
        self._placeholder = True
        self._frame = None
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
        # (QtCore.QSize escala in-place; QSize.scale retorna None)
        x = (self.width() - target.width()) // 2
        y = (self.height() - target.height()) // 2
        painter.drawPixmap(x, y, target.width(), target.height(), pix)

        # Selo com o nome do papel da câmera (Frontal / Lateral).
        fm = painter.fontMetrics()
        pad = 5
        tw = fm.horizontalAdvance(self._title) if hasattr(fm, "horizontalAdvance") \
            else fm.width(self._title)
        label = QtCore.QRect(10, 10, tw + 2 * pad, fm.height() + 2 * pad)
        painter.fillRect(label, QtGui.QColor(20, 20, 35, 190))
        painter.setPen(QtGui.QColor("#ffffff"))
        painter.drawText(label, QtCore.Qt.AlignCenter, self._title)
        painter.end()


def cv2_bgr_to_rgb(bgr):
    import cv2
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)