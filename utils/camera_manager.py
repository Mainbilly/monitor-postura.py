"""
utils/camera_manager.py
=======================
Captura multi-câmera em threads independentes.

Cada câmera roda uma thread que:
- lê frames do OpenCV;
- roda a detecção de pose (um PoseDetector por câmera);
- publica (landmarks, frame_desenhado) em uma Queue segura;

A thread principal consome as filas de forma não-bloqueante.
"""

from __future__ import annotations

import queue
import threading
import time

import cv2

from pose_detector import PoseDetector

# Intervalo mínimo entre ciclos de inferência por câmera (s).
# Limita a carga nas threads (GIL) e deixa folga para a GUI compor os frames
# sem piscar/atrasar. ~20 FPS já é fluido o bastante para detecção de postura.
WORKER_MIN_INTERVAL = 0.05  # 50 ms -> ~20 FPS por câmera


class CameraWorker(threading.Thread):
    def __init__(self, cam_id: int, role: str, out_queue: queue.Queue,
                 delay: float = 0.0):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.role = role
        self.out_queue = out_queue
        self._delay = delay  # evita que as N threads carreguem modelo juntas
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        cap = cv2.VideoCapture(self.cam_id)
        if not cap.isOpened():
            self.out_queue.put({"role": self.role, "ok": False, "error": "não abriu"})
            return

        # Espaça o carregamento do modelo entre as câmeras: carregar 2× TFLite
        # simultaneamente disputa CPU/GIL no boot e TRAVA a GUI que está abrindo.
        if self._delay > 0:
            time.sleep(self._delay)

        detector = PoseDetector()
        while not self._stop.is_set():
            cycle_start = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                break
            # Timestamp em ms derivado de tempo real (não um contador fixo),
            # para o MediaPipe tratar como tracking e não re-disparar detecção.
            ts = int(time.monotonic() * 1000)
            try:
                landmarks, frame = detector.detect(frame, ts)
            except Exception as exc:
                # Erro por frame não deve derrubar a thread.
                print(f"[{self.role}] detecção falhou: {exc}")
                time.sleep(WORKER_MIN_INTERVAL)
                continue
            self.out_queue.put({"role": self.role, "ok": True,
                                "landmarks": landmarks, "frame": frame})

            # Controla a taxa: dorme o restante do intervalo para não
            # monopolizar o GIL e dar respiro à thread da GUI.
            elapsed = time.monotonic() - cycle_start
            remaining = WORKER_MIN_INTERVAL - elapsed
            if remaining > 0:
                time.sleep(remaining)

        detector.close()
        cap.release()


class CameraManager:
    """Cria e gerencia os workers de cada câmera configurada."""

    def __init__(self, camera_config):
        self._config = camera_config
        self._workers: list[CameraWorker] = []
        self.result_queue: queue.Queue = queue.Queue()

    def start(self):
        # Escalona o início das threads: cada uma espera um pouco mais que a
        # anterior antes de carregar o modelo, para não disputar CPU/GIL no boot.
        for i, cam in enumerate(self._config):
            w = CameraWorker(cam["id"], cam["role"], self.result_queue,
                             delay=i * 1.5)  # 0s, 1.5s, ...
            w.start()
            self._workers.append(w)
        return self

    def drain(self, latest_per_role: bool = True) -> list[dict]:
        """
        Consome frames prontos. Se latest_per_role=True, mantém apenas o
        frame mais recente de cada papel (descarta os intermediários).
        """
        items = []
        while not self.result_queue.empty():
            try:
                items.append(self.result_queue.get_nowait())
            except queue.Empty:
                break

        # Filtra falhas de abertura para diagnosticar câmera ausente.
        errors = [i["error"] for i in items if not i.get("ok")]
        if latest_per_role and items:
            latest = {}
            for it in items:
                if it.get("ok"):
                    latest[it["role"]] = it
            return list(latest.values()), errors
        return items, errors

    def stop(self):
        for w in self._workers:
            w.stop()
        for w in self._workers:
            w.join(timeout=2.0)
        self._workers.clear()