"""
posture_analyzer.py
===================
Lógica de classificação de postura.

Dado o dict de ângulos medidos (de angle_calculator.compute_all), produz:
- Um nível por ângulo ("good"/"warning"/"bad"/"critical");
- Um SCORE global 0-100 (média ponderada);
- Combinando câmera frontal + lateral, quando disponíveis.

Composição por papel de câmera:
  front -> usa  tilt            (única medida robusta no plano frontal)
  side  -> usa  cva, lean, kyphosis, lordosis  (plano sagital)
Isso evita usar medições degeneradas quando o ângulo depende do outro plano
(ex.: CVA na câmera frontal é geometricamente indefinido).
"""

from __future__ import annotations

import config


def classify_value(value: float, bands) -> str:
    """Classifica um valor em uma das bandas configuradas (primeira que casa).

    Os limites das bandas são INCLUSIVOS (lo <= value <= hi): caso contrário,
    um valor exatamente no topo de uma banda (ex.: lean = 180°, postura
    ereta) não cairia em nenhuma faixa e seria classificado como "bad".
    """
    for level in config.POSTURE_LEVELS:
        lo, hi = bands[level]
        if lo <= value <= hi:
            return level
    # Fora de todas as faixas: considera o pior "bad".
    return "bad"


def _band_for(angle_name: str) -> dict:
    return config.POSTURE_THRESHOLDS[angle_name]


def analyze_angles(angles: dict) -> dict:
    """
    angles: {nome_angulo: valor}  ->  {nome_angulo: nível}.
    Ângulos ausentes não geram entrada.
    """
    return {name: classify_value(val, _band_for(name))
            for name, val in angles.items()}


def score_angles(angles: dict) -> tuple[float, int]:
    """
    Retorna (score 0-100, n_angulos_considerados).
    Cada ângulo vale de 0 a 100 proporcional à posição dentro da sua faixa
    "good" — 100 quando dentro do bom, decaindo até 0 conforme se afasta.
    """
    total_w = 0.0
    acc = 0.0
    n_angles = 0  # contagem REAL dos ângulos que entraram no cálculo
    for name, val in angles.items():
        try:
            w = config.SCORE_WEIGHTS[name]
        except KeyError:
            continue  # sem peso -> NÃO conta (não fez parte do cálculo)
        g_lo, g_hi = _band_for(name)["good"]
        # Medida = distância relativa ao centro da faixa boa.
        center = (g_lo + g_hi) / 2
        half = (g_hi - g_lo) / 2 or 1.0
        dist = abs(val - center) / half
        # 100 no centro, 0 numa distância de 3 "meias-larguras" ou mais.
        points = max(0.0, 100.0 - 33.0 * dist)  # 33 ≈ 100/3
        acc += points * w
        total_w += w
        n_angles += 1
    if total_w == 0:
        return 0.0, 0
    return round(acc / total_w, 1), n_angles


# ---------------------------------------------------------------------------
# Composição dual-camera
# ---------------------------------------------------------------------------
CAMERA_ANGLE_MAP = {
    "front": ["tilt"],
    "side":  ["cva", "lean", "kyphosis", "lordosis"],
}


def merge_dual_camera(front_angles: dict, side_angles: dict) -> dict:
    """
    Junta as medições de duas câmeras, respeitando o papel de cada uma.
    Retorna {nome_angulo: valor}.

    Regras:
    - Ângulos do papel "front" (tilt) vêm de front_angles.
    - Ângulos do papel "side" (cva, lean, kyphosis, lordosis) vêm de
      side_angles.
    - Se a câmera "lateral" for a ÚNICA presente (ex.: 1 câmera), aceita os
      ângulos sagitais dela mesmo sem o ditado frontal — para uso monocular.
    """
    merged = {}
    # Se a frontal existe e tem o ângulo, usa-a; caso contrário, qualquer
    # câmera que tenha medido um ângulo do papel "front" contribui.
    for name in CAMERA_ANGLE_MAP["front"]:
        if name in front_angles:
            merged[name] = front_angles[name]
        elif name in side_angles:
            merged[name] = side_angles[name]

    # Ângulos sagitais: esperado da lateral.
    for name in CAMERA_ANGLE_MAP["side"]:
        if name in side_angles:
            merged[name] = side_angles[name]
        elif name in front_angles:
            # Câmera única não-sagitais: se só há a frontal, usa o que der.
            merged[name] = front_angles[name]
    return merged


def evaluate(front_angles: dict, side_angles: dict | None = None) -> dict:
    """
    Função principal de análise pós-medição.

    Retorna:
        {
          "angles":    {nome: {"value": v, "level": lvl}},
          "levels":    {nome: lvl},
          "score":     float 0-100,
          "worst":     pior nível global ou "good",
          "cameras":   papéis que contribuíram,
        }
    """
    angles = merge_dual_camera(front_angles, side_angles or {})
    levels = analyze_angles(angles)
    score, _ = score_angles(angles)

    order = {"good": 0, "warning": 1, "bad": 2, "critical": 3}
    worst = max(levels.values(), default="good", key=lambda lv: order[lv])

    return {
        "angles": {n: {"value": v, "level": levels[n]}
                   for n, v in angles.items() if n in levels},
        "levels": levels,
        "score": score,
        "worst": worst,
        "cameras": [r for r, d in [("front", front_angles),
                                   ("side", side_angles or {})] if d],
    }