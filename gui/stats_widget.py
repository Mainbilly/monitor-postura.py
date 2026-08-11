"""
gui/stats_widget.py
===================
Painel de estatísticas: score global, pior nível e os ângulos monitorados
com sua classificação. Não usa matplotlib para se manter leve.
"""

from __future__ import annotations

from PyQt5 import QtWidgets


class StatsWidget(QtWidgets.QWidget):
    _LEVEL_LABEL = {
        "good":     "BOM",
        "warning":  "AVISO",
        "bad":      "RUIM",
        "critical": "CRÍTICO",
    }

    _ANGLE_LABEL = {
        "cva":      "CVA (cabeça)",
        "nose_fwd": "Cabeça p/ frente",
        "lean":     "Inclinação",
        "kyphosis": "Cifose",
        "lordosis": "Lordose",
        "tilt":     "Ombros",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("statsCard")

        lay = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("POSTURA")
        title.setObjectName("cardTitle")
        lay.addWidget(title)

        self.score_label = QtWidgets.QLabel("Pontuação: --")
        self.score_label.setObjectName("scoreLabel")

        self.worst_label = QtWidgets.QLabel("Postura: --")
        self.worst_label.setObjectName("worstLabel")

        angles_title = QtWidgets.QLabel("ÂNGULOS MONITORADOS")
        angles_title.setObjectName("cardTitle")

        self.angles_box = QtWidgets.QVBoxLayout()
        self._angle_labels: dict[str, QtWidgets.QLabel] = {}
        for name in self._ANGLE_LABEL:
            lbl = QtWidgets.QLabel(f"{self._ANGLE_LABEL[name]}: --")
            self.angles_box.addWidget(lbl)
            self._angle_labels[name] = lbl

        lay.addWidget(self.score_label)
        lay.addWidget(self.worst_label)
        lay.addSpacing(6)
        lay.addWidget(angles_title)
        lay.addLayout(self.angles_box)
        lay.addStretch(1)

    def update_stats(self, evaluation: dict) -> None:
        score = evaluation.get("score", 0.0)
        worst = evaluation.get("worst", "good")
        hexc = _color_hex(worst)

        self.score_label.setText(f"Pontuação: {score:.0f} / 100")
        self.score_label.setStyleSheet(
            f"color:{hexc}; font-size:30px; font-weight:600;")
        self.worst_label.setText(
            f"Postura: {self._LEVEL_LABEL.get(worst, worst)}")
        self.worst_label.setStyleSheet(
            f"color:{hexc}; font-size:17px; font-weight:500;")

        for name, lbl in self._angle_labels.items():
            if name in evaluation.get("angles", {}):
                info = evaluation["angles"][name]
                lvl = info["level"]
                lbl.setText(
                    f"{self._ANGLE_LABEL[name]:12s}: {info['value']:5.1f}°  "
                    f"[{self._LEVEL_LABEL.get(lvl, lvl)}]"
                )
                lbl.setStyleSheet("color:" + _color_hex(lvl) + ";")
            else:
                lbl.setText(f"{self._ANGLE_LABEL[name]:12s}: --")
                lbl.setStyleSheet("color:#6a6a82;")


def _color_hex(level: str) -> str:
    colors = {
        "good":     "#00C800",
        "warning":  "#FFC800",
        "bad":      "#FF7800",
        "critical": "#FF0000",
    }
    return colors.get(level, "#ffffff")
