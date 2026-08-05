"""
main.py
=======
Entry point do Monitor de Postura (dual-camera).

Uso:
    python main.py

Depende de: MediaPipe, OpenCV, NumPy, PyQt5. Veja requirements.txt.
"""

from __future__ import annotations

import sys


def main() -> None:
    try:
        from PyQt5 import QtWidgets
    except ImportError:
        sys.exit(
            "PyQt5 não está instalado.\n"
            "Execute:  pip install PyQt5"
        )

    try:
        from gui.main_window import MainWindow
    except ImportError as exc:
        sys.exit(f"Falha ao carregar módulos do projeto: {exc}")

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # Fecha threads/logger de forma ordenada ao sair.
    def _cleanup():
        window.camera_mgr.stop()
        window.logger.close()
        win = getattr(window.sound, "_pygame", None)
        if win is not None:
            win.quit()

    app.aboutToQuit.connect(_cleanup)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()