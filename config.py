"""
config.py
=========
Configurações centrais do projeto de monitoramento de postura.

Centraliza todos os parâmetros ajustáveis: thresholds de postura, tempos de
alerta, cooldowns, configuração de câmeras e caminhos de arquivos.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Diretórios base
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SOUNDS_DIR = BASE_DIR / "sounds"
LOG_DB_PATH = BASE_DIR / "posture_history.db"

# URL do modelo .task download automático (segunda versão, com z-scoring)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_full.task"

# ---------------------------------------------------------------------------
# MediaPipe PoseLandmarker
# ---------------------------------------------------------------------------
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5
POSE_NUM_POSES = 1

# Confiança mínima de um keypoint para que ele seja usado nos cálculos.
# Abaixo disso o ponto é considerado "não confiável" e o ângulo correspondente
# é descartado.
KEYPOINT_MIN_CONFIDENCE = 0.5

# ---------------------------------------------------------------------------
# Dimensionamento do vídeo
# ---------------------------------------------------------------------------
CAM_FRAME_WIDTH = 640
CAM_FRAME_HEIGHT = 480
# O modelo espera entrada quadrada; mantém apenas a região central do frame.
INPUT_SIZE = 256

# ---------------------------------------------------------------------------
# Câmeras
# ---------------------------------------------------------------------------
# Identificadores físicos das câmeras (índice do OpenCV) e o papel de cada uma.
#
# Setup do usuário:
#   - webcam do NOTEBOOK  = lateral
#   - webcam USB          = frontal
#
# SE na hora do teste as imagens aparecerem INVERTIDAS (frontal mostrando a
# lateral), troque os "id" abaixo.
CAMERAS = [
    {"id": 1, "role": "front"},   # webcam USB -> frontal
    {"id": 0, "role": "side"},    # webcam do notebook -> lateral
]

# ---------------------------------------------------------------------------
# Ângulos monitorados (todas as comparações usam intervalos [min, max))
# ---------------------------------------------------------------------------
# Cada chave: bandas "good", "warning", "bad", "critical".
# A classificação pega a PRIMEIRA banda que contém o ângulo medido.

POSTURE_THRESHOLDS = {
    # CVA — proxy de cabeça pra frente. Nesta base MAIOR = pior (orelha mais
    # à frente do que o pescoço). ALIGNE OS LIMITES à sua câmera antes de usar
    # em produção; por padrão 0-15° = ereta, >40° = muito projetada.
    "cva": {
        "good":      (0, 15),
        "warning":   (15, 25),
        "bad":       (25, 40),
        "critical":  (40, 91),
    },
    # FLA — Inclinação para frente. Ideal 140-159°.
    "lean": {
        "good":      (140, 180),
        "warning":   (120, 140),
        "bad":       (100, 120),
        "critical":  (0, 100),
    },
    # Thoracic Kyphosis — proxy de curvatura. Nesta base MAIOR = mais curvado
    # (cabeça projetada à frente): ~0° = ereto, valores altos = encurvado.
    "kyphosis": {
        "good":      (0, 20),
        "warning":   (20, 35),
        "bad":       (35, 50),
        "critical":  (50, 180),
    },
    # Lumbar Lordosis — proxy de arqueamento lombar. MAIOR = mais arqueado.
    "lordosis": {
        "good":      (0, 25),
        "warning":   (25, 40),
        "bad":       (40, 55),
        "critical":  (55, 180),
    },
    # Shoulder Tilt — inclinação lateral dos ombros.
    "tilt": {
        "good":      (0, 5),
        "warning":   (5, 10),
        "bad":       (10, 15),
        "critical":  (15, 180),
    },
}

# ---------------------------------------------------------------------------
# Pesos de cada ângulo para compor o SCORE FINAL (0-100).
# A soma não precisa ser 1 — o score é a média ponderada dos pontos.
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = {
    "cva":      1.0,
    "lean":     1.0,
    "kyphosis": 1.0,
    "lordosis": 1.0,
    "tilt":     1.0,
}

# ---------------------------------------------------------------------------
# Estados de alerta
# ---------------------------------------------------------------------------
POSTURE_LEVELS = ["good", "warning", "bad", "critical"]
# O nível por padrão dispara alerta apenas em "bad" e "critical".
ALERT_ON_LEVELS = {"bad", "critical"}

# Para "postura ruim", quanto tempo contínuo antes de disparar o alerta (s).
ALERT_TRIGGER_SECONDS = 5.0
# Cooldown: não repetir o mesmo alerta dentro deste intervalo (s).
ALERT_SILENCE_SECONDS = 30.0

# Cores dos níveis (BGR para OpenCV / RGB para UI).
LEVEL_COLORS = {
    "good":     (0, 200, 0),     # verde
    "warning":  (0, 200, 255),   # amarelo
    "bad":      (0, 120, 255),   # laranja
    "critical": (0, 0, 255),     # vermelho
}

# ---------------------------------------------------------------------------
# Sons
# ---------------------------------------------------------------------------
SOUND_WARNING = SOUNDS_DIR / "alert_warning.wav"
SOUND_CRITICAL = SOUNDS_DIR / "alert_critical.wav"
SOUND_ENABLED = True

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# A cada quantos segundos um registro de postura entra no banco SQLite.
LOG_INTERVAL_SECONDS = 1.0

# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
WINDOW_TITLE = "Monitor de Postura — Dual Camera"
GUI_UPDATE_MS = 33  # ~30 FPS na interface