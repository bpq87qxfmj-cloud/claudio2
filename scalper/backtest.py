"""Backtest secuencial: una sola posición a la vez, con costes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import Outcomes


def run(index: pd.DatetimeIndex, signal: np.ndarray, out: Outcomes, cost: float, lo: int = 0, hi: int | None = None) -> pd.DataFrame:
    """Ejecuta las señales (+1 largo, -1 corto, 0 nada) entre las posiciones [lo, hi).

    Devuelve una fila por operación con el resultado bruto y neto en unidades de precio.
    """
    hi = len(signal) if hi is None else hi
    cand = lo + np.flatnonzero(signal[lo:hi])
    rows = []
    free = lo
    for i in cand:
        if i < free:
            continue
        s = signal[i]
        gross = out.long_pnl[i] if s > 0 else out.short_pnl[i]
        if not np.isfinite(gross):
            continue
        k = int(out.long_exit[i] if s > 0 else out.short_exit[i])
        rows.append((index[i], int(s), out.entry[i], out.width[i], k, gross, gross - cost))
        free = i + k + 1  # la siguiente señal válida es la del cierre de la barra de salida
    return pd.DataFrame(rows, columns=["time", "side", "entry", "width", "bars", "gross", "net"])


def stats(trades: pd.DataFrame, cost: float | None = None) -> dict:
    """Métricas de un conjunto de operaciones. Con `cost` recalcula el neto para otro coste."""
    if len(trades) == 0:
        return dict(trades=0, win_rate=np.nan, gross_avg=np.nan, net_avg=np.nan, net_total=0.0, net_R=np.nan,
                    profit_factor=np.nan, tstat=np.nan, sharpe_daily=np.nan, max_dd=0.0)
    net = trades["net"].to_numpy() if cost is None else trades["gross"].to_numpy() - cost
    daily = pd.Series(net, index=pd.DatetimeIndex(trades["time"])).resample("1D").sum()
    daily = daily[daily != 0]
    eq = np.cumsum(net)
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    sd = net.std(ddof=1) if len(net) > 1 else np.nan
    return dict(
        trades=len(net),
        win_rate=float((net > 0).mean()),
        gross_avg=float(trades["gross"].mean()),
        net_avg=float(net.mean()),
        net_total=float(net.sum()),
        net_R=float((net / trades["width"].to_numpy()).mean()),  # resultado medio en múltiplos del riesgo
        profit_factor=float(gains / losses) if losses > 0 else np.inf,
        tstat=float(net.mean() / sd * np.sqrt(len(net))) if sd and sd > 0 else np.nan,
        sharpe_daily=float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if len(daily) > 2 and daily.std(ddof=1) > 0 else np.nan,
        max_dd=float((np.maximum.accumulate(eq) - eq).max()),
    )
