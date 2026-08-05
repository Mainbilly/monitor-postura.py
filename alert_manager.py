"""
alert_manager.py
================
Gerencia alertas visuais e sonoros.

Lógica:
- Só dispara quando o pior nível está em ALERT_ON_LEVELS ("bad"/"critical").
- Exige ALERT_TRIGGER_SECONDS de postura ruim CONTÍNUA antes de alertar
  (evita falsos disparos por movimento momentâneo).
- Depois do disparo, impõe ALERT_SILENCE_SECONDS de silêncio antes de
  repetir o mesmo nível.
"""

from __future__ import annotations

import time

import config


class AlertManager:
    def __init__(self):
        self._bad_since: float | None = None   # instante do início da postura ruim
        # nível -> timestamp do último disparo; ausente = nunca disparou.
        self._last_alert: dict[str, float] = {}
        self._on_alert = None  # callback opcional: on_alert(level)

    def set_callback(self, callback):
        """callback(level) é chamado a cada disparo de alerta."""
        self._on_alert = callback

    def update(self, worst_level: str, now: float | None = None) -> bool:
        """
        Alimenta o estado a cada frame. Retorna True quando um alerta
        é disparado neste frame.
        """
        now = time.monotonic() if now is None else now

        if worst_level not in config.ALERT_ON_LEVELS:
            self._bad_since = None
            return False

        if self._bad_since is None:
            self._bad_since = now

        if now - self._bad_since < config.ALERT_TRIGGER_SECONDS:
            return False  # ainda dentro do período de confirmação

        last = self._last_alert.get(worst_level)
        if last is not None and (now - last) < config.ALERT_SILENCE_SECONDS:
            return False  # em silêncio

        self._last_alert[worst_level] = now
        self._bad_since = None  # reinicia o "período ruim" p/ o próximo ciclo
        if self._on_alert:
            self._on_alert(worst_level)
        return True


class SoundAlertPlayer:
    """Reproduz um som por nível.

    Backend em ordem de preferência:
      1. winsound (Windows built-in) — toca .wav nativo, sem dependências.
      2. pygame — se instalado e os arquivos existirem.
      3. fallback silencioso — se nada estiver disponível.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._backend = None
        self._sounds: dict[str, str | None] = {}
        self._init_backend()

    def _init_backend(self):
        if not self._enabled:
            return
        sound_files = {
            "bad":      config.SOUND_WARNING,
            "critical": config.SOUND_CRITICAL,
        }
        # Backend 1: winsound (nativo do Windows).
        try:
            import winsound
            self._backend = "winsound"
            self._sounds = {lvl: str(path) for lvl, path in sound_files.items()
                            if path.exists()}
            return
        except Exception:
            pass
        # Backend 2: pygame.
        try:
            import pygame
            pygame.mixer.init()
            self._backend = "pygame"
            self._pygame = pygame
            self._sounds = {
                lvl: pygame.mixer.Sound(str(path))
                for lvl, path in sound_files.items() if path.exists()
            }
        except Exception:
            self._backend = None
            self._pygame = None
            self._sounds = {}

    def play(self, level: str):
        if not self._enabled:
            return
        snd = self._sounds.get(level)
        if self._backend == "winsound" and snd:
            try:
                import winsound
                winsound.PlaySound(snd, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception:
                pass
        elif self._backend == "pygame" and snd is not None:
            try:
                snd.play()
            except Exception:
                pass