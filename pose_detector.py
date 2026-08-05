"""
pose_detector.py
================
Wrapper para o MediaPipe PoseLandmarker (API "tasks", recomendada).

Responsabilidades:
- Baixar automaticamente o modelo .task (uma única vez, cacheada em models/).
- Rodar a detecção de pose sobre um frame opencv (BGR).
- Expor os 33 keypoints como lista de (x, y, z, visibility).

A normalização dos keypoints é feita em RELAÇÃO À LARGURA do frame ([] do
MediaPipe). Consulte README para o funcionamento; a conversão p/ frame é feita
na camada de exibição.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

import config

# A API nova "tasks" existe no mediapipe >= 0.10.0.
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception as exc:  # pragma: no cover - depende do ambiente
    mp = None
    mp_vision = None
    mp_python = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class PoseDetector:
    """Detecta landmarks corporais em um frame BGR usando MediaPipe."""

    # Índices absolutos dos 33 keypoints do BlazePose (MediaPipe).
    NOSE = 0
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26

    def __init__(self):
        if mp_vision is None:
            raise RuntimeError(
                "\nMediaPipe não está instalado ou é muito antigo.\n"
                "Execute:  pip install mediapipe\n"
                f"Erro de importação: {_IMPORT_ERROR}"
            )
        self._ensure_model()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(config.POSE_MODEL_PATH)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=config.POSE_NUM_POSES,
            min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Gerência do modelo
    # ------------------------------------------------------------------
    def _ensure_model(self):
        """Baixa o modelo .task na primeira execução, se não existir."""
        if config.POSE_MODEL_PATH.exists():
            return
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Baixando modelo de pose... ({config.POSE_MODEL_URL})")
        urllib.request.urlretrieve(config.POSE_MODEL_URL, str(config.POSE_MODEL_PATH))
        print("Modelo salvo em:", config.POSE_MODEL_PATH)

    # ------------------------------------------------------------------
    # Detecção
    # ------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int):
        """
        Retorna (landmarks, image):
          - landmarks: np.ndarray shape (33, 4) com [x, y, z, visibility]
                        normalizado em [0, 1] relativo à LARGURA.
          - image:     frame BGR com os keypoints desenhados.
        Retorna (None, image) se nenhuma pose for detectada.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None, self._draw_skeleton(frame_bgr, np.zeros((0, 4)))

        lm = np.array(
            [[p.x, p.y, p.z, p.visibility] for p in result.pose_landmarks[0]],
            dtype=np.float32,
        )
        return lm, self._draw_skeleton(frame_bgr, lm)

    @staticmethod
    def _draw_skeleton(frame_bgr: np.ndarray, lm: np.ndarray) -> np.ndarray:
        """Desenha keypoints e conexões no frame (vermelho)."""
        h, w = frame_bgr.shape[:2]
        canvas = frame_bgr.copy()

        # Conexões principais (índice do par de keypoints).
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
            (24, 26), (26, 28), (0, 7), (0, 8),
        ]

        n = lm.shape[0]
        for a, b in connections:
            # Proteção contra array vazio (nenhuma pose detectada) ou
            # keypoint ausente: pula a conexão sem quebrar a thread.
            if a >= n or b >= n:
                continue
            pa, pb = lm[a], lm[b]
            if pa[3] > config.KEYPOINT_MIN_CONFIDENCE and pb[3] > config.KEYPOINT_MIN_CONFIDENCE:
                ax = (int(pa[0] * w), int(pa[1] * h))
                bx = (int(pb[0] * w), int(pb[1] * h))
                cv2.line(canvas, ax, bx, (0, 0, 255), 2)

        for p in lm:
            if p[3] > config.KEYPOINT_MIN_CONFIDENCE:
                x, y = int(p[0] * w), int(p[1] * h)
                cv2.circle(canvas, (x, y), 3, (0, 255, 255), -1)

        return canvas

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None