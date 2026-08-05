"""
utils/file_handler.py
=====================
Persistência do histórico de postura em SQLite (built-in, sem dependências).
"""

from __future__ import annotations

import sqlite3
import time

import config


class PostureLogger:
    """Registra um snapshot de postura a cada intervalo configurado."""

    def __init__(self, db_path=None, interval: float = 1.0):
        self._path = str(db_path or config.LOG_DB_PATH)
        self._interval = interval
        self._conn = sqlite3.connect(self._path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS posture_log (
                ts            REAL,      -- timestamp unix
                score         REAL,
                worst         TEXT,
                cva           REAL,
                lean          REAL,
                kyphosis      REAL,
                lordosis      REAL,
                tilt          REAL
            )"""
        )
        self._conn.commit()
        self._last_log = 0.0

    def log(self, evaluation: dict, now: float | None = None) -> bool:
        """Grava se o intervalo já passou. Retorna True se gravou."""
        now = time.time() if now is None else now
        if now - self._last_log < self._interval:
            return False

        vals = evaluation.get("angles", {})
        self._conn.execute(
            "INSERT INTO posture_log VALUES (?,?,?,?,?,?,?,?)",
            (
                now,
                evaluation.get("score", 0.0),
                evaluation.get("worst", "good"),
                vals.get("cva", {}).get("value"),
                vals.get("lean", {}).get("value"),
                vals.get("kyphosis", {}).get("value"),
                vals.get("lordosis", {}).get("value"),
                vals.get("tilt", {}).get("value"),
            ),
        )
        self._conn.commit()
        self._last_log = now
        return True

    def recent(self, n: int = 100) -> list[dict]:
        """Últimos n registros, do mais recente para o mais antigo."""
        rows = self._conn.execute(
            "SELECT ts, score, worst FROM posture_log ORDER BY ts DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [{"ts": r[0], "score": r[1], "worst": r[2]} for r in rows]

    def close(self):
        self._conn.close()