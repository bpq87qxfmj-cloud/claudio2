"""Agente autónomo de scalping.

Cada minuto: lee las barras cerradas, actualiza la posición, comprueba los límites de riesgo,
pide una predicción al modelo y, si procede, abre una operación con TP, SL y caducidad.
Además se vigila a sí mismo: se detiene si pierde demasiado, se suspende si su resultado
real deja de respaldar la ventaja estimada, y se reentrena periódicamente con datos nuevos
(tras lo cual puede decidir no operar en absoluto si la validación no encuentra ventaja).
"""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest, model
from .brokers import Order, PaperBroker, round_lots
from .config import MARKETS, STATE_DIR, Market, Params
from .features import FEATURE_WARMUP, build_features
from .labels import barrier_width

log = logging.getLogger("scalper.agent")
JOURNAL_COLS = ["time", "entry_time", "exit_time", "side", "units", "entry", "exit", "width", "bars", "pred",
                "gross", "net", "net_money", "reason", "equity"]


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.0025  # fracción del capital arriesgada por operación (distancia al SL)
    max_daily_loss: float = 0.02  # pérdida del día (fracción del capital inicial del día) que detiene la jornada
    max_trades_per_day: int = 60
    max_consecutive_losses: int = 5  # tras N pérdidas seguidas, pausa
    pause_minutes: int = 60
    max_drawdown: float = 0.10  # caída desde el máximo de capital que detiene al agente hasta revisión manual
    monitor_trades: int = 100  # ventana de operaciones reales para vigilar la ventaja
    monitor_tstat: float = -2.0  # si el t-stat de esa ventana cae por debajo, se suspende hasta reentrenar
    max_spread_cost: float = 0.8  # no entrar si el spread actual supera esta fracción del coste asumido
    max_bar_age_min: float = 3.0  # datos más viejos que esto se consideran caducados (en vivo)
    fixed_units: float | None = None  # si se indica, tamaño fijo en lugar de dimensionar por riesgo


@dataclass
class AgentState:
    last_bar: str | None = None
    day: str | None = None
    day_start_equity: float = 0.0
    day_pnl: float = 0.0
    day_trades: int = 0
    consecutive_losses: int = 0
    paused_until: str | None = None
    peak_equity: float = 0.0
    halted: str | None = None  # motivo de parada definitiva (requiere --reset-halt)
    suspended: bool = False  # suspendido por el monitor de ventaja hasta el próximo reentrenamiento
    recent_net: list = field(default_factory=list)  # resultados netos (precio) de las últimas operaciones
    trades: int = 0
    last_train: str | None = None
    broker: dict | None = None


def trading_day(t: pd.Timestamp) -> str:
    """Jornada de negociación: cambia a las 17:00 de Nueva York."""
    return str((t.tz_convert("America/New_York") + pd.Timedelta(hours=7)).date())


class Agent:
    def __init__(self, market: str, broker, predictor: model.Predictor, risk: RiskConfig | None = None,
                 cost: float | None = None, state_dir: Path | None = None, use_reference: bool = False,
                 persist: bool = True):
        self.mk: Market = MARKETS[market]
        self.broker, self.predictor = broker, predictor
        self.risk = risk or RiskConfig()
        self.cost = self.mk.cost if cost is None else cost
        self.use_reference = use_reference
        self.persist = persist
        self.dir = Path(state_dir or STATE_DIR)
        self.state_path = self.dir / f"{market}_state.json"
        self.journal_path = self.dir / f"{market}_trades.csv"
        self.stop_path = self.dir / "STOP"
        self.state = AgentState()
        self.closed: list[dict] = []
        self.skips: dict[str, int] = {}
        if persist:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._load_state()
        if not self.state.peak_equity:
            self.state.peak_equity = self.broker.equity()

    # ------------------------------------------------------------------ persistencia
    def _load_state(self) -> None:
        if self.state_path.exists():
            self.state = AgentState(**json.loads(self.state_path.read_text()))
            if self.state.broker and hasattr(self.broker, "restore"):
                self.broker.restore(self.state.broker)
            log.info("Estado recuperado: %d operaciones previas, capital %.2f", self.state.trades, self.broker.equity())

    def _save_state(self) -> None:
        if not self.persist:
            return
        if hasattr(self.broker, "dump"):
            self.state.broker = self.broker.dump()
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self.state), indent=1))
        tmp.replace(self.state_path)

    def _journal(self, tr: dict) -> None:
        if not self.persist:
            return
        new = not self.journal_path.exists()
        with self.journal_path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=JOURNAL_COLS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(tr)

    # ------------------------------------------------------------------ contabilidad y riesgo
    def _roll_day(self, t: pd.Timestamp) -> None:
        day = trading_day(t)
        if day != self.state.day:
            self.state.day, self.state.day_pnl, self.state.day_trades = day, 0.0, 0
            self.state.day_start_equity = self.broker.equity()

    def _record(self, tr: dict, t: pd.Timestamp) -> None:
        st, r = self.state, self.risk
        eq = self.broker.equity()
        tr["equity"] = eq
        self.closed.append(tr)
        self._journal(tr)
        st.trades += 1
        st.day_trades += 1
        st.day_pnl += tr["net_money"]
        st.recent_net = (st.recent_net + [tr["net"]])[-r.monitor_trades:]
        st.consecutive_losses = st.consecutive_losses + 1 if tr["net_money"] < 0 else 0
        st.peak_equity = max(st.peak_equity, eq)
        log.info("CIERRE %s %s entrada=%.2f salida=%.2f neto=%+.2f (%s) capital=%.2f",
                 "LARGO" if tr["side"] > 0 else "CORTO", tr["exit_time"], tr["entry"], tr["exit"],
                 tr["net_money"], tr["reason"], eq)
        if st.consecutive_losses >= r.max_consecutive_losses:
            st.paused_until = str(t + pd.Timedelta(minutes=r.pause_minutes))
            st.consecutive_losses = 0
            log.warning("Pausa hasta %s tras %d pérdidas seguidas", st.paused_until, r.max_consecutive_losses)
        if st.peak_equity > 0 and eq < st.peak_equity * (1 - r.max_drawdown):
            st.halted = f"caída máxima superada: capital {eq:.2f} frente a máximo {st.peak_equity:.2f}"
            log.error("AGENTE DETENIDO: %s", st.halted)
        if len(st.recent_net) >= r.monitor_trades and not st.suspended:
            x = np.asarray(st.recent_net)
            sd = x.std(ddof=1)
            if sd > 0 and x.mean() / sd * np.sqrt(len(x)) < r.monitor_tstat:
                st.suspended = True
                log.warning("Suspendido: las últimas %d operaciones (media %+.3f) contradicen la ventaja estimada. "
                            "Se reanudará tras reentrenar.", len(x), x.mean())

    def blocked(self, t: pd.Timestamp, now: pd.Timestamp | None = None) -> str | None:
        """Motivo por el que no se puede abrir una operación ahora (None si se puede)."""
        st, r = self.state, self.risk
        if st.halted:
            return "detenido"
        if st.suspended:
            return "suspendido por el monitor de ventaja"
        if self.predictor.threshold is None and not self.use_reference:
            return "sin ventaja validada"
        if st.paused_until and t < pd.Timestamp(st.paused_until):
            return "en pausa"
        if st.day_start_equity > 0 and st.day_pnl <= -r.max_daily_loss * st.day_start_equity:
            return "límite de pérdida diaria"
        if st.day_trades >= r.max_trades_per_day:
            return "máximo de operaciones diarias"
        ny = t.tz_convert("America/New_York")
        to_break = self.mk.break_ny - (ny.hour * 60 + ny.minute)
        if 0 <= to_break <= self.predictor.params.horizon + 1 or (ny.dayofweek == 5):
            return "cierre de sesión próximo"
        if now is not None and (now - t).total_seconds() / 60 > r.max_bar_age_min + 1:
            return "datos caducados"
        sp = self.broker.spread()
        if sp is not None and sp > r.max_spread_cost * self.cost:
            return "spread demasiado alto"
        return None

    def size(self, width: float) -> float:
        """Unidades de subyacente, redondeadas al lote, para arriesgar `risk_per_trade` del capital."""
        if self.risk.fixed_units is not None:
            return self.risk.fixed_units
        units = self.broker.equity() * self.risk.risk_per_trade / width
        return round_lots(units, self.mk) * self.mk.lot_size

    # ------------------------------------------------------------------ ciclo
    def on_bar(self, t: pd.Timestamp, o: float, h: float, l: float, c: float, pred: float, width: float,
               now: pd.Timestamp | None = None) -> str:
        """Procesa una barra cerrada. Devuelve lo que hizo (para el registro)."""
        st = self.state
        if st.last_bar is not None and t <= pd.Timestamp(st.last_bar):
            return "barra ya procesada"
        self._roll_day(t)
        closed = self.broker.on_bar(t, o, h, l, c)
        for tr in closed:
            self._record(tr, t)
        st.last_bar = str(t)
        action = self._decide(t, c, pred, width, now, just_closed=bool(closed))
        self._save_state()
        return action

    def _decide(self, t, c, pred, width, now, just_closed: bool) -> str:
        if self.stop_path.exists():
            for tr in self.broker.close_all(t, c, "parada manual"):
                self._record(tr, t)
            self.state.halted = "archivo STOP presente"
            return "parada manual"
        if self.state.halted and not self.broker.flat():
            for tr in self.broker.close_all(t, c, "agente detenido"):
                self._record(tr, t)
        if not self.broker.flat():
            return "en posición"
        if just_closed:
            return "espera tras cierre"
        reason = self.blocked(t, now)
        if reason:
            self.skips[reason] = self.skips.get(reason, 0) + 1
            return reason
        side = self.predictor.side(pred, width, self.cost, self.use_reference)
        if side == 0:
            return "sin señal"
        units = self.size(width)
        if units <= 0:
            self.skips["tamaño mínimo excede el riesgo"] = self.skips.get("tamaño mínimo excede el riesgo", 0) + 1
            return "tamaño mínimo excede el riesgo"
        if not self.broker.submit(Order(side, units, width, self.predictor.params.horizon, t, pred)):
            return "orden no aceptada"
        log.info("ORDEN %s %.2f uds  cierre=%.2f  TP/SL=±%.2f  pred=%+.4f  máx %d min",
                 "COMPRA" if side > 0 else "VENTA", units, c, width, pred, self.predictor.params.horizon)
        return "COMPRA" if side > 0 else "VENTA"

    def on_bars(self, bars: pd.DataFrame, now: pd.Timestamp | None = None) -> str:
        """Versión en vivo: recibe la ventana de barras cerradas y calcula la predicción."""
        t = bars.index[-1]
        if self.state.last_bar is not None and t <= pd.Timestamp(self.state.last_bar):
            return "barra ya procesada"
        if len(bars) < FEATURE_WARMUP + 200:
            pred, width = np.nan, np.nan
        else:
            pred, width = self.predictor.last(bars)
        b = bars.iloc[-1]
        return self.on_bar(t, b.open, b.high, b.low, b.close, pred, width, now)

    def set_predictor(self, predictor: model.Predictor, when: pd.Timestamp) -> None:
        """Sustituye el modelo (tras reentrenar) y levanta la suspensión del monitor de ventaja."""
        self.predictor = predictor
        self.state.last_train = str(when)
        self.state.suspended = False
        self.state.recent_net = []
        thr = predictor.threshold
        log.info("Modelo nuevo (datos hasta %s): %s", predictor.meta["trained_until"],
                 f"umbral {thr:.4f}" if thr is not None else "sin ventaja validada -> el agente se queda fuera")

    def summary(self) -> dict:
        trades = pd.DataFrame(self.closed, columns=[c for c in JOURNAL_COLS]) if self.closed else pd.DataFrame(columns=JOURNAL_COLS)
        st = backtest.stats(trades)
        st.update(equity=self.broker.equity(), money=float(trades["net_money"].sum()) if len(trades) else 0.0,
                  halted=self.state.halted, suspended=self.state.suspended, skips=dict(self.skips))
        return st


# ---------------------------------------------------------------------- repetición histórica
def replay(market: str, df: pd.DataFrame, start: str, end: str | None = None, equity: float = 10_000.0,
           cost: float | None = None, risk: RiskConfig | None = None, p: Params | None = None,
           retrain: bool = True, use_reference: bool = False, predictor: model.Predictor | None = None) -> Agent:
    """Hace vivir al agente el periodo [start, end) barra a barra sobre el histórico.

    Con `retrain`, el agente se reentrena al empezar cada mes usando solo los datos anteriores,
    igual que hará en vivo. Las variables se precalculan (son causales) para ir rápido.
    """
    from .config import PARAMS

    p = p or PARAMS
    cost = MARKETS[market].cost if cost is None else cost
    months = df.index.tz_localize(None).to_period("M")
    first, last = pd.Period(start, "M"), (pd.Period(end, "M") if end else months.max() + 1)
    X = build_features(df)
    width = barrier_width(df, p)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    agent: Agent | None = None
    for m in pd.period_range(first, last - 1, freq="M"):
        pos = np.flatnonzero(months == m)
        if len(pos) == 0:
            continue
        lo, hi = pos[0], pos[-1] + 1
        if retrain or predictor is None:
            predictor = model.Predictor(*model.fit_final(market, df.iloc[:lo], cost, p))
        if agent is None:
            agent = Agent(market, PaperBroker(equity, cost), predictor, risk, cost, use_reference=use_reference, persist=False)
        if retrain:
            agent.set_predictor(predictor, df.index[lo])
        pred = predictor.booster.predict(X.iloc[lo:hi][predictor.meta["features"]], num_iteration=predictor.meta["trees"])
        idx = df.index[lo:hi]
        for j in range(hi - lo):
            i = lo + j
            agent.on_bar(idx[j], o[i], h[i], l[i], c[i], pred[j], width[i])
    if agent is None:
        raise ValueError("No hay datos en el periodo pedido")
    return agent


# ---------------------------------------------------------------------- bucle en vivo
def run_live(agent: Agent, fetch, retrain_days: float = 7.0, retrainer=None, max_steps: int | None = None,
             poll_offset: float = 3.0, sleep=time.sleep) -> None:
    """Bucle principal. `fetch()` devuelve las barras M1 cerradas más recientes (DataFrame UTC)."""
    steps, errors = 0, 0
    while max_steps is None or steps < max_steps:
        steps += 1
        now = pd.Timestamp.now(tz="UTC")
        try:
            last = agent.state.last_train
            due = retrainer is not None and (last is None or (now - pd.Timestamp(last)).total_seconds() > retrain_days * 86400)
            if due and agent.broker.flat():
                log.info("Reentrenando con datos nuevos...")
                agent.set_predictor(retrainer(), now)
                agent._save_state()
            bars = fetch()
            action = agent.on_bars(bars, now=now)
            if action != "barra ya procesada":
                log.info("[%s] cierre=%.2f capital=%.2f -> %s", bars.index[-1], bars["close"].iloc[-1],
                         agent.broker.equity(), action)
            errors = 0
            if action == "parada manual":
                break
        except KeyboardInterrupt:
            raise
        except Exception as e:  # la red o el bróker fallan: se reintenta, no se muere
            errors += 1
            log.error("Error en el ciclo (%d seguidos): %s", errors, e)
            if errors >= 30:
                log.error("Demasiados errores seguidos; el agente se detiene.")
                break
        if max_steps is None or steps < max_steps:
            sleep(max(1.0, 60 - time.time() % 60 + poll_offset))
