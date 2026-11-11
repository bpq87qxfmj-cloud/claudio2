"""Aprendizaje a partir de los errores.

El agente observa todas las señales "candidatas" del modelo (el 5 % de barras con predicción
más fuerte), las haya operado o no, y sigue cada una como una operación en la sombra para
saber cómo habría acabado. Con esa experiencia entrena un "crítico": un segundo modelo que,
dado el contexto de una señal (hora, volatilidad, tamaño de la barrera frente al coste,
movimiento reciente...), estima el resultado neto esperado.

El crítico se usa de dos formas:
  * Veto: una operación que el modelo quiere abrir se descarta si el crítico espera pérdida,
    es decir, si se parece a situaciones en las que antes perdió.
  * Entradas aprendidas (opcional): abre candidatas que el modelo base no operaría, pero solo
    mientras las últimas aprobadas por el crítico hayan sido rentables en la sombra.
"""
from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .brokers import Order, PaperBroker

# Variables del predictor que también describen el contexto de una señal.
FROM_FEATURES = ["vol_bps", "vol_10_60", "vol_60_390", "range_rel", "sess_bars", "dow", "ret_1", "ret_5",
                 "ret_30", "pos_60", "rsi_7"]
CONTEXT = ["hour", "side", "abs_pred_rel", "cost_w", *FROM_FEATURES]
EXP_COLS = ["time", "score", "net", "net_r", *CONTEXT]


@dataclass
class LearnConfig:
    min_experiences: int = 1500  # experiencias resueltas antes de que el crítico opine
    max_experiences: int = 30_000  # memoria: solo las más recientes
    refit_days: int = 1  # el crítico se reajusta cada N jornadas
    learn_entries: bool = False  # permitir entradas aprendidas además del veto
    track_trades: int = 300  # nº de aprobadas recientes que deben respaldar las entradas aprendidas
    track_tstat: float = 1.0
    max_gap_min: float = 5.0  # un hueco de datos mayor invalida las operaciones en la sombra abiertas


CRITIC_PARAMS = dict(n_estimators=150, learning_rate=0.03, num_leaves=6, min_child_samples=150, subsample=0.8,
                     subsample_freq=1, colsample_bytree=0.6, reg_lambda=20.0, verbose=-1, n_jobs=2, random_state=3)


class Learner:
    def __init__(self, cost: float, horizon: int, cfg: LearnConfig | None = None, path: Path | None = None):
        self.cost, self.horizon = cost, horizon
        self.cfg = cfg or LearnConfig()
        self.path = Path(path) if path else None
        self.X: list[np.ndarray] = []
        self.y: list[float] = []  # resultado neto en múltiplos de la barrera
        self.critic: lgb.LGBMRegressor | None = None
        self._shadows: list[tuple[PaperBroker, np.ndarray, float | None]] = []
        self._last_t: pd.Timestamp | None = None
        self._days_since_fit = 10**9
        self._fitted_on = 0
        self.approved: deque[float] = deque(maxlen=self.cfg.track_trades)  # netos de las aprobadas ya resueltas
        self.tally = {k: [0, 0.0] for k in ("aprobadas", "rechazadas", "sin_opinion")}  # [n, suma de netos]
        self.vetoes = 0
        self.learned_entries = 0
        if self.path is not None and self.path.exists():
            self._load()

    # ------------------------------------------------------------------ experiencia
    @staticmethod
    def context(t: pd.Timestamp, side: int, pred: float, width: float, cost: float, cand_thr: float,
                feats: np.ndarray) -> np.ndarray:
        hour = t.tz_convert("America/New_York").hour
        return np.concatenate(([hour, side, abs(pred) / cand_thr, cost / width], np.asarray(feats, dtype=float)))

    def observe(self, t: pd.Timestamp, side: int, pred: float, width: float, cand_thr: float,
                feats: np.ndarray) -> float | None:
        """Registra una señal candidata, abre su operación en la sombra y devuelve la opinión del crítico."""
        x = self.context(t, side, pred, width, self.cost, cand_thr, feats)
        score = self.score(x)
        shadow = PaperBroker(0.0, self.cost)
        shadow.submit(Order(side, 1.0, width, self.horizon, t, pred))
        self._shadows.append((shadow, x, score))
        return score

    def on_bar(self, t: pd.Timestamp, o: float, h: float, l: float, c: float) -> None:
        """Avanza las operaciones en la sombra con la barra recién cerrada."""
        if self._last_t is not None and (t - self._last_t).total_seconds() / 60 > self.cfg.max_gap_min:
            self._shadows.clear()  # no se aprende de operaciones que habrían cruzado un hueco
        self._last_t = t
        still = []
        for shadow, x, score in self._shadows:
            closed = shadow.on_bar(t, o, h, l, c)
            if closed:
                self._resolve(t, x, score, closed[0])
            else:
                still.append((shadow, x, score))
        self._shadows = still

    def _resolve(self, t, x: np.ndarray, score: float | None, tr: dict) -> None:
        net, net_r = tr["net"], tr["net"] / tr["width"]
        self.X.append(x)
        self.y.append(net_r)
        key = "sin_opinion" if score is None else ("aprobadas" if score > 0 else "rechazadas")
        self.tally[key][0] += 1
        self.tally[key][1] += net
        if score is not None and score > 0:
            self.approved.append(net)
        if len(self.y) > self.cfg.max_experiences + 2000:
            self.X, self.y = self.X[-self.cfg.max_experiences:], self.y[-self.cfg.max_experiences:]
        if self.path is not None:
            new = not self.path.exists()
            with self.path.open("a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(EXP_COLS)
                w.writerow([str(tr["time"]), "" if score is None else f"{score:.5f}", f"{net:.5f}", f"{net_r:.5f}",
                            *(f"{v:.6g}" for v in x)])

    def _load(self) -> None:
        d = pd.read_csv(self.path).tail(self.cfg.max_experiences)
        self.X = [row for row in d[CONTEXT].to_numpy(float)]
        self.y = d["net_r"].astype(float).tolist()
        ok = d[d["score"] > 0]
        self.approved.extend(ok["net"].astype(float).tolist()[-self.cfg.track_trades:])
        self.refit()

    # ------------------------------------------------------------------ crítico
    def refit_due(self) -> bool:
        """Se llama una vez por jornada; dice si toca reajustar el crítico."""
        self._days_since_fit += 1
        n = len(self.y)
        return n >= self.cfg.min_experiences and n != self._fitted_on and self._days_since_fit >= self.cfg.refit_days

    def refit(self) -> bool:
        n = len(self.y)
        if n < self.cfg.min_experiences or n == self._fitted_on:
            return False
        X = pd.DataFrame(np.vstack(self.X[-self.cfg.max_experiences:]), columns=CONTEXT)
        self.critic = lgb.LGBMRegressor(**CRITIC_PARAMS).fit(X, np.asarray(self.y[-self.cfg.max_experiences:]))
        self._days_since_fit, self._fitted_on = 0, n
        return True

    def score(self, x: np.ndarray) -> float | None:
        """Resultado neto esperado (en múltiplos de la barrera); None si aún no hay experiencia suficiente."""
        if self.critic is None:
            return None
        return float(self.critic.predict(pd.DataFrame([x], columns=CONTEXT))[0])

    def track_ok(self) -> bool:
        """¿Respaldan las últimas aprobadas (ya resueltas) abrir entradas aprendidas?"""
        a = np.asarray(self.approved)
        if len(a) < self.cfg.track_trades:
            return False
        sd = a.std(ddof=1)
        return bool(a.mean() > 0 and sd > 0 and a.mean() / sd * np.sqrt(len(a)) >= self.cfg.track_tstat)

    # ------------------------------------------------------------------ informe
    def lessons(self, top: int = 5) -> list[tuple[str, float]]:
        """Qué variables de contexto pesan más en la opinión del crítico (fracción de la ganancia)."""
        if self.critic is None:
            return []
        gain = self.critic.booster_.feature_importance(importance_type="gain")
        total = gain.sum() or 1.0
        order = np.argsort(gain)[::-1][:top]
        return [(CONTEXT[i], float(gain[i] / total)) for i in order]

    def summary(self) -> dict:
        mean = lambda k: (self.tally[k][1] / self.tally[k][0]) if self.tally[k][0] else None
        return dict(
            experiencias=len(self.y), critico_activo=self.critic is not None, vetos=self.vetoes,
            entradas_aprendidas=self.learned_entries,
            aprobadas=(self.tally["aprobadas"][0], mean("aprobadas")),
            rechazadas=(self.tally["rechazadas"][0], mean("rechazadas")),
            respaldo_entradas=self.track_ok(), lecciones=self.lessons(),
        )
