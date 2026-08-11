"""
angle_calculator.py
===================
Cálculos geométricos dos ângulos posturais a partir dos 33 keypoints do
MediaPipe BlazePose.

Todos os cálculos assumem imagem de corpo inteiro (ou, no pior caso,
"waist up"). Os índices seguem o mapeamento REAL do MediaPipe:

    0 nose · 7 left_ear · 8 right_ear
    11 left_shoulder · 12 right_shoulder
    23 left_hip · 24 right_hip
    25 left_knee · 26 right_knee

Entradas: `lm` é um np.ndarray de shape (33, 4) com colunas
[x, y, z, visibility], normalizados em [0, 1] — x relativo à LARGURA e y
relativo à ALTURA (MediaPipe). O eixo y cresce para baixo (origem no canto
superior esquerdo). Use compute_all(lm, aspect=w/h) para compensar a
diferença de escala entre os dois eixos.
"""

from __future__ import annotations

import numpy as np

import config


# ---------------------------------------------------------------------------
# Acesso a pontos
# ---------------------------------------------------------------------------
NOSE = 0
LT_EAR = 7
RT_EAR = 8
LT_SHO = 11
RT_SHO = 12
LT_HIP = 23
RT_HIP = 24
LT_KNE = 25
RT_KNE = 26


def _mid(lm, a, b):
    """Ponto médio entre dois keypoints."""
    return (lm[a][:2] + lm[b][:2]) / 2.0


def _angle_between(v1, v2):
    """Ângulo (0-180°) entre dois vetores 2D."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos_a = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _conf(lm, *idxs):
    """Confiança mínima dos pontos envolvidos (para descartar medição)."""
    return min(lm[i][3] for i in idxs)


# ---------------------------------------------------------------------------
# Ângulos
# ---------------------------------------------------------------------------
def craniovertebral_angle(lm):
    """
    CVA (proxy pró-cabeça-pra-frente) — deve ser usado na CÂMERA LATERAL
    (plano sagital), onde a geometria é bem definida.

    Ângulo entre o segmento pescoço→orelha e a vertical:
        ang = arctan2(|deslocamento horizontal|, |deslocamento vertical|)

    Valores MAIORES = cabeça mais projetada para a frente (pior).
    Uma pessoa alinhada (orelha sobre o pescoço) fica ~0°.
    Fórmula adaptada do CVA clássico; os thresholds numéricos precisam de
    calibração para o seu posto/câmera (comentário em config.py).
    """
    neck = _mid(lm, LT_SHO, RT_SHO)
    ear = _mid(lm, LT_EAR, RT_EAR)
    v = ear - neck              # pescoço -> orelha
    h = np.abs(v[0])            # componente horizontal (para frente)
    vert = np.abs(v[1])         # componente vertical
    angle = np.degrees(np.arctan2(h, vert))
    return float(angle), _conf(lm, LT_SHO, RT_SHO, LT_EAR, RT_EAR)


def nose_forward_angle(lm):
    """
    NFA — Cabeça pra frente (proxy usando o NARIZ, câmera LATERAL).
    Ângulo entre a reta pescoço→nariz e a vertical, no mesmo espírito do CVA,
    porém MAIS SENSÍVEL: o nariz projeta mais à frente do rosto do que a
    orelha, então a componente horizontal cresce mais cedo ao inclinar a
    cabeça/parte superior. Valores MAIORES = cabeça mais projetada (pior);
    alinhado ≈ 0°. Usa apenas nariz + ombros (sem precisar de quadril).
    """
    neck = _mid(lm, LT_SHO, RT_SHO)
    nose = lm[NOSE][:2]
    v = nose - neck             # pescoço -> nariz
    h = np.abs(v[0])            # componente horizontal (para frente)
    vert = np.abs(v[1])         # componente vertical
    angle = np.degrees(np.arctan2(h, vert))
    return float(angle), _conf(lm, NOSE, LT_SHO, RT_SHO)


def forward_lean_angle(lm):
    """
    FLA — Inclinação do tronco para a frente (câmera LATERAL).
    Ângulo entre a reta pescoço→quadril e a vertical, definido de modo que
    ERETO ≈ 175-180° e, quanto mais o usuário se inclina para a frente, menor
    o valor (tendendo a ~90°). MAIOR = melhor (semântica do papel):
        upright      ~ 180°
        inclinado    ~ 135°
        tombado      ~ 90°

    Convenção de sinal: o componente horizontal é multiplicado por
    config.LEAN_FORWARD_X_SIGN para que inclinar para FRENTE sempre reduza o
    ângulo. Assim, inclinar para trás (v[0] com sinal contrário) deixa o
    ângulo ~180° e não é contado como inclinação ruim. Se a sua câmera lateral
    estiver espelhada, inverta LEAN_FORWARD_X_SIGN em config.py.
    """
    neck = _mid(lm, LT_SHO, RT_SHO)
    hip = _mid(lm, LT_HIP, RT_HIP)
    v = hip - neck                # pescoço -> quadril (aponta para baixo)
    # Componente horizontal com sinal: positivo = inclinação p/ frente,
    # negativo = inclinação p/ trás (segundo a convenção de config).
    fwd_x = config.LEAN_FORWARD_X_SIGN * v[0]
    # arctan2 deixa o sinal escapar: dev < 0 => inclinado para TRÁS.
    dev = np.degrees(np.arctan2(fwd_x, abs(v[1])))
    # Inclinação p/ trás (dev < 0) NÃO é punida: ângulo permanece ~180°.
    return float(180.0 - max(0.0, dev)), _conf(lm, LT_SHO, RT_SHO, LT_HIP, RT_HIP)


def thoracic_kyphosis_angle(lm):
    """
    TKA — Curvatura das costas superiores (proxy, câmera LATERAL).
    Ângulo entre a linha ombro→orelha e a linha ombro→quadril, tomado como
    COMPLEMENTO do ângulo orientado — assim, ERETO ≈ 0° e, quanto mais a
    cabeça/parte superior projeta para a frente, MAIOR o valor (mais curvado).
    NOTA: o MediaPipe não tem keypoints de coluna; isto é uma aproximação.
    """
    sho = _mid(lm, LT_SHO, RT_SHO)
    ear = _mid(lm, LT_EAR, RT_EAR)
    hip = _mid(lm, LT_HIP, RT_HIP)
    v1 = ear - sho
    v2 = hip - sho
    raw = _angle_between(v1, v2)  # ~180 ereto, <180 curvado
    return float(180.0 - raw), _conf(lm, LT_SHO, RT_SHO, LT_EAR, RT_EAR, LT_HIP, RT_HIP)


def lumbar_lordosis_angle(lm):
    """
    LLA — Curvatura da região lombar (proxy, câmera LATERAL).
    Ângulo entre a linha quadril→ombro e a linha quadril→joelho, como
    complemento do ângulo orientado. ERETO ≈ 0°; MAIOR = postura mais
    arqueada/relaxada. Aproximação (sem keypoints de coluna).
    """
    hip = _mid(lm, LT_HIP, RT_HIP)
    sho = _mid(lm, LT_SHO, RT_SHO)
    kne = _mid(lm, LT_KNE, RT_KNE)
    v1 = sho - hip
    v2 = kne - hip
    raw = _angle_between(v1, v2)  # ~180 ereto, <180 arqueado
    return float(180.0 - raw), _conf(lm, LT_HIP, RT_HIP, LT_SHO, RT_SHO, LT_KNE, RT_KNE)


def shoulder_tilt_angle(lm):
    """
    Inclinação lateral dos ombros (assimetria de altura L/R).
    Ideal <5°; valores maiores indicam desequilíbrio.

    Mede o DESVIO da linha dos ombros em relação à horizontal (0-90°). Um
    `arctan2` puro devolveria 0° OU 180° para ombros nivelados, dependendo de
    qual ombro aparece à esquerda na imagem (normal em coordenadas MediaPipe:
    o ombro direito aparece à esquerda, dx<0). O `min(raw, 180-raw)` dobra o
    ângulo para a menor diferença à horizontal.
    """
    dx = lm[RT_SHO][0] - lm[LT_SHO][0]
    dy = lm[LT_SHO][1] - lm[RT_SHO][1]  # sinal: ombro esquerdo mais alto
    raw = abs(float(np.degrees(np.arctan2(dy, dx))))
    ang = min(raw, 180.0 - raw)
    return ang, _conf(lm, LT_SHO, RT_SHO)


# ---------------------------------------------------------------------------
# API única
# ---------------------------------------------------------------------------
def compute_all(lm, aspect: float = 1.0):
    """
    Retorna dict {nome_do_angulo: valor_float} para todos os ângulos válidos.
    Medições com confiança insuficiente (ou pontos ausentes) são ignoradas.

    aspect: razão largura/altura do frame (w/h). O MediaPipe normaliza x em
    relação à LARGURA e y em relação à ALTURA; num frame não-quadrado (ex.:
    640x480) isso distorce os ângulos. Antes de calcular, x é resscalado para
    a mesma métrica de y (unidades de altura), tornando a geometria correta.
    """
    if lm is None:
        return {}
    # Cópia resscalada: só a componente x muda; y já está na métrica correta.
    if aspect and aspect != 1.0:
        lm = lm.copy()
        lm[:, 0] *= aspect
    funcs = {
        "cva":       craniovertebral_angle,
        "nose_fwd":  nose_forward_angle,
        "lean":      forward_lean_angle,
        "kyphosis":  thoracic_kyphosis_angle,
        "lordosis":  lumbar_lordosis_angle,
        "tilt":      shoulder_tilt_angle,
    }
    out = {}
    for name, fn in funcs.items():
        try:
            val, conf = fn(lm)
            if conf >= config.KEYPOINT_MIN_CONFIDENCE and not np.isnan(val):
                out[name] = float(val)
        except Exception:
            continue  # ponto ausente -> ignorar esta medição
    return out