"""Etiquetado de triple barrera y resultado simulado de cada operación posible.

Para cada barra t (señal al cierre) se simula entrar en la apertura de t+1 con TP y SL
simétricos a distancia `width` y un límite de `horizon` barras. Convención conservadora:
si una misma barra toca TP y SL, se asume que saltó el SL.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Params
from .features import ewm_vol, minutes, session_id


@dataclass
class Outcomes:
    width: np.ndarray  # distancia de la barrera en unidades de precio
    entry: np.ndarray  # precio de entrada (apertura de t+1)
    long_pnl: np.ndarray  # resultado bruto de un largo, en precio (NaN si no operable)
    short_pnl: np.ndarray
    long_exit: np.ndarray  # nº de barras hasta la salida (1..horizon)
    short_exit: np.ndarray
    target: np.ndarray  # objetivo de regresión en [-1, 1]: (long - short) / (2 * width)


def barrier_width(df: pd.DataFrame, p: Params) -> np.ndarray:
    vol = ewm_vol(df["close"], p.vol_span)
    return (p.barrier_mult * vol * np.sqrt(p.horizon) * df["close"]).to_numpy()


def _shift(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full(a.shape, np.nan)
    out[:-k] = a[k:]
    return out


def simulate(df: pd.DataFrame, p: Params) -> Outcomes:
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n, H = len(df), p.horizon
    width = barrier_width(df, p)
    entry = _shift(o, 1)
    up, dn = entry + width, entry - width

    long_pnl = np.full(n, np.nan)
    short_pnl = np.full(n, np.nan)
    long_exit = np.full(n, H, dtype=np.int32)
    short_exit = np.full(n, H, dtype=np.int32)
    long_open = np.ones(n, bool)
    short_open = np.ones(n, bool)
    for k in range(1, H + 1):
        hi, lo = _shift(h, k), _shift(l, k)
        hit_up, hit_dn = hi >= up, lo <= dn
        # largo: primero se comprueba el SL (barrera inferior)
        sl = long_open & hit_dn
        tp = long_open & hit_up & ~hit_dn
        long_pnl[sl], long_pnl[tp] = -width[sl], width[tp]
        long_exit[sl | tp] = k
        long_open &= ~(sl | tp)
        # corto: su SL es la barrera superior
        sl = short_open & hit_up
        tp = short_open & hit_dn & ~hit_up
        short_pnl[sl], short_pnl[tp] = -width[sl], width[tp]
        short_exit[sl | tp] = k
        short_open &= ~(sl | tp)
    last = _shift(c, H)
    long_pnl[long_open] = (last - entry)[long_open]
    short_pnl[short_open] = (entry - last)[short_open]

    # operable solo si toda la ventana cae en la misma sesión y sin huecos grandes
    sid = session_id(df.index)
    t = minutes(df.index)
    same = _shift(sid.astype(float), H) == sid
    compact = (_shift(t, H) - t) <= H + 5
    ok = same & compact & np.isfinite(width) & (width > 0) & np.isfinite(entry)
    long_pnl[~ok] = np.nan
    short_pnl[~ok] = np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        target = np.clip((long_pnl - short_pnl) / (2 * width), -1, 1)
    return Outcomes(width, entry, long_pnl, short_pnl, long_exit, short_exit, target)
