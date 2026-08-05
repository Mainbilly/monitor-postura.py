# Monitor de Postura

Monitor de postura em tempo real usando **duas câmeras** (frontal + lateral),
MediaPipe PoseLandmarker e uma interface PyQt5. Detecta desvios posturais,
classifica a postura em níveis (`good` / `warning` / `bad` / `critical`),
calcula um score 0–100, dispara alertas sonoros/visuais e grava um histórico
em SQLite.

## Funcionalidades

- **Dual-camera**: cada câmera roda em uma thread própria.
  - *frontal* → inclinação lateral dos ombros (`tilt`);
  - *lateral* → cabeça pra frente (`cva`), inclinação do tronco (`lean`),
    curvatura das costas (`kyphosis`) e arqueamento lombar (`lordosis`).
- **Score 0–100**: média ponderada da distância de cada ângulo à sua faixa
  "boa".
- **Alertas** (som + borda vermelha): só após `ALERT_TRIGGER_SECONDS` de
  postura ruim contínua, com cooldown de `ALERT_SILENCE_SECONDS`.
- **Histórico**: registros periódicos em `posture_history.db` (SQLite, sem
  dependências).
- **Modelo auto-baixado**: o `.task` do MediaPipe é baixado na primeira
  execução e cacheado em `models/`.

## Requisitos

- Python 3.10+ (testado em 3.14)
- Câmeras webcam: uma lateral (ex.: notebook) e uma frontal (ex.: USB)

## Instalação

```bash
pip install -r requirements.txt
```

## Como rodar

```bash
python main.py
```

Na primeira execução o modelo de pose (~14 MB) é baixado
automaticamente de `config.POSE_MODEL_URL`.

## Configuração (`config.py`)

- **Câmeras**: `CAMERAS` mapeia o índice do OpenCV para o papel de cada
  webcam. Se as imagens aparecerem trocadas/invertidas, ajuste os `id`.
- **Limites de postura**: `POSTURE_THRESHOLDS` (bandas por ângulo). Os
  valores atuais são **padrões iniciais** — alinhe-os à sua câmera antes de
  usar em produção.
- **Inclinação pra frente (`lean`)**: `LEAN_FORWARD_X_SIGN` define em que
  direção do frame o usuário se inclina pra frente. Se, ao testar, "inclinar
  pra trás" estiver reduzindo o score, **inverta o sinal** dessa constante.
- **Pesos do score**: `SCORE_WEIGHTS`.
- **Alertas**: `ALERT_TRIGGER_SECONDS`, `ALERT_SILENCE_SECONDS`,
  `ALERT_ON_LEVELS`, `SOUND_ENABLED`.

## Estrutura

```
main.py                 # entry point
config.py               # todas as configurações
pose_detector.py        # wrapper do MediaPipe PoseLandmarker (download + detect)
angle_calculator.py     # ângulos posturais a partir dos 33 keypoints
posture_analyzer.py     # classificação, score e composição dual-camera
alert_manager.py        # lógica de alertas + player de som (winsound/pygame)
utils/camera_manager.py # captura multi-câmera em threads
utils/file_handler.py   # histórico SQLite
gui/                    # janela principal, widget de vídeo, painel de stats
tests/                  # testes unitários (python -m unittest)
```

## Testes

```bash
python -m unittest discover -s tests -v
```

## Notas de implementação

- Os ângulos compensam a **distorção de aspecto** do frame
  (`compute_all(lm, aspect=w/h)`), pois o MediaPipe normaliza `x` pela
  largura e `y` pela altura.
- `kyphosis`/`lordosis` são aproximações: o MediaPipe não fornece keypoints
  de coluna.
- O backend de captura usa DSHOW (e cai para o padrão como fallback) para
  evitar crashes do Media Foundation/Intel MFX em certos notebooks.
