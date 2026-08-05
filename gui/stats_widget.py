"""
gui/stats_widget.py
===================
Painel de estatísticas: score global, pior nível e os ângulos monitorados
com sua classificação. Não usa matplotlib para se manter leve.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets


class StatsWidget(QtWidgets.QWidget):
    _LEVEL_LABEL = {
        "good":     "BOM",
        "warning":  "AVISO",
        "bad":      "RUIM",
        "critical": "CRÍTICO",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)

        self.score_label = QtWidgets.QLabel("Score: --")
        self.score_label.setStyleSheet("font-size:26px;font-weight:bold;")

        self.worst_label = QtWidgets.QLabel("Postura: --")
        self.worst_label.setStyleSheet("font-size:16px;")

        self.angles_box = QtWidgets.QVBoxLayout()
        self._angle_labels: dict[str, QtWidgets.QLabel] = {}
        for name in ["cva", "lean", "kyphosis", "lordosis", "tilt"]:
            lbl = QtWidgets.QLabel(f"{name:9s}: --")
            self.angles_box.addWidget(lbl)
            self._angle_labels[name] = lbl

        lay.addWidget(self.score_label)
        lay.addWidget(self.worst_label)
        lay.addLayout(self.angles_box)
        lay.addStretch(1)

    def update_stats(self, evaluation: dict) -> None:
        score = evaluation.get("score", 0.0)
        worst = evaluation.get("worst", "good")
        self.score_label.setText(f"Score: {score:.0f} / 100")
        self.worst_label.setText(f"Postura: {self._LEVEL_LABEL.get(worst, worst)}")

        for name, lbl in self._angle_labels.items():
            if name in evaluation.get("angles", {}):
                info = evaluation["angles"][name]
                lvl = info["level"]
                lbl.setText(
                    f"{name:9s}: {info['value']:5.1f}°  [{self._LEVEL_LABEL.get(lvl, lvl)}]"
                )
                lbl.setStyleSheet("color:" + _color_hex(lvl) + ";")
            else:
                lbl.setText(f"{name:9s}: --")
                lbl.setStyleSheet("color:gray;")


def _color_hex(level: str) -> str:
    colors = {
        "good":     "#00C800",
        "warning":  "#FFC800",
        "bad":      "#FF7800",
        "critical": "#FF0000",
    }
    return colors.get(level, "#ffffff")