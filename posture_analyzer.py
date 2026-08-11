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
  side  -> usa  cva, nose_fwd, lean, kyphosis, lordosis  (plano sagital)
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


def _angle_score(name: str, val: float) -> float:
    """
    Nota 0-100 de UM ângulo, ancorada nas bandas do config.

    Toda a faixa "good" vale 100 (a postura ideal costuma ficar na BORDA da
    faixa, não no centro — ex.: lean=180°). A partir daí, o valor decai de
    forma contínua conforme penetra nos níveis piores:
        warning  -> 100..66
        bad      ->  66..33
        critical ->  33..0
    Valores fora de todas as faixas são "clampados" na nota do extremo mais
    próximo (0 ou 100).
    """
    bands = _band_for(name)
    g_lo, g_hi = bands["good"]
    # Âncoras (valor, nota) ordenadas: cada borda de nível recebe a nota do
    # nível adjacente, mantendo a curva contínua entre as faixas.
    anchors: list[tuple[float, float]] = [(g_lo, 100.0), (g_hi, 100.0)]
    prev_score = 100.0
    # Borda do nível anterior que "encosta" no próximo nível. Começa na faixa
    # boa: ambas as bordas valem 100.
    prev_edge = (g_lo, g_hi)
    for lvl, far_score in (("warning", 100.0 / 3 * 2),
                           ("bad", 100.0 / 3),
                           ("critical", 0.0)):
        lo, hi = bands[lvl]
        if lo in prev_edge:     # nível está ACIMA do anterior
            near_edge = lo
            far_edge = hi
        elif hi in prev_edge:   # nível está ABAIXO do anterior
            near_edge = hi
            far_edge = lo
        else:                   # faixas não contíguas: ignora (não deve ocorrer)
            continue
        anchors.append((near_edge, prev_score))
        anchors.append((far_edge, far_score))
        prev_score = far_score
        prev_edge = (near_edge, far_edge)

    anchors.sort()
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    if val <= xs[0]:
        return ys[0]
    if val >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x0 <= val <= x1:
            if x1 == x0:
                return ys[i]
            t = (val - x0) / (x1 - x0)
            return ys[i] + t * (ys[i + 1] - ys[i])
    return 0.0


def score_angles(angles: dict) -> tuple[float, int]:
    """
    Retorna (score 0-100, n_angulos_considerados).
    Cada ângulo vale 100 dentro da sua faixa "good" e decai continuamente
    conforme se afasta (ver _angle_score). Resultado é a média ponderada.
    """
    total_w = 0.0
    acc = 0.0
    n_angles = 0  # contagem REAL dos ângulos que entraram no cálculo
    for name, val in angles.items():
        try:
            w = config.SCORE_WEIGHTS[name]
        except KeyError:
            continue  # sem peso -> NÃO conta (não fez parte do cálculo)
        acc += _angle_score(name, val) * w
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
    "side":  ["cva", "nose_fwd", "lean", "kyphosis", "lordosis"],
}


def merge_dual_camera(front_angles: dict, side_angles: dict) -> dict:
    """
    Junta as medições de duas câmeras, respeitando o papel de cada uma.
    Retorna {nome_angulo: valor}.

    Regras:
    - Ângulos do papel "front" (tilt) vêm de front_angles.
    - Ângulos do papel "side" (cva, nose_fwd, lean, kyphosis, lordosis)
      vêm de side_angles.
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


def evaluate(front_angles: dict, side_angles: dict | None = None,
             prev_angles: dict | None = None,
             alpha: float | None = None) -> dict:
    """
    Função principal de análise pós-medição.

    Suavização opcional (EMA): se `prev_angles` for passado, cada ângulo
    medido é misturado com o valor anterior do frame (alpha*novo +
    (1-alpha)*anterior), reduzindo a oscilação do medidor. Sem `prev_angles`
    o cálculo é puro (usado pelos testes). `alpha` padrão: config.

    Retorna:
        {
          "angles":    {nome: {"value": v, "level": lvl}},
          "levels":    {nome: lvl},
          "score":     float 0-100,
          "worst":     pior nível global ou "good",
          "cameras":   papéis que contribuíram,
        }
    """
    if alpha is None:
        alpha = config.SMOOTHING_ALPHA
    angles = merge_dual_camera(front_angles, side_angles or {})
    if prev_angles and 0 < alpha <= 1:
        for name, val in list(angles.items()):
            old = prev_angles.get(name)
            if old is not None:
                angles[name] = alpha * val + (1.0 - alpha) * old
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