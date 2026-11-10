"""Variables predictoras, todas causales (solo usan información hasta el cierre de la barra t).

Son adimensionales (normalizadas por volatilidad) y no dependen del volumen, de modo que el
mismo modelo sirve para el histórico spot, los futuros de Yahoo o un CSV de cualquier bróker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SESSION_GAP_MIN = 15  # un hueco mayor inicia una sesión nueva (cierre diario / fin de semana)


def minutes(index: pd.DatetimeIndex) -> np.ndarray:
    """Marca temporal en minutos (independiente de la resolución interna del índice)."""
    return index.as_unit("ns").asi8 / 60e9


def gap_minutes(index: pd.DatetimeIndex) -> np.ndarray:
    t = minutes(index)
    return np.diff(t, prepend=t[0])


def session_id(index: pd.DatetimeIndex) -> np.ndarray:
    return np.cumsum(gap_minutes(index) > SESSION_GAP_MIN)


def ewm_vol(close: pd.Series, span: int) -> pd.Series:
    """Volatilidad EWM de los rendimientos logarítmicos de 1 min (sin media)."""
    r = np.log(close).diff()
    return np.sqrt((r**2).ewm(span=span, min_periods=span // 2).mean())


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    lc = np.log(c)
    r1 = lc.diff()
    f: dict[str, pd.Series] = {}

    vol60 = ewm_vol(c, 60)
    vol = vol60.clip(lower=1e-6)

    # --- momento / reversión en varios horizontes, en unidades de volatilidad
    for n in (1, 2, 3, 5, 10, 15, 30, 60, 120):
        f[f"ret_{n}"] = (lc - lc.shift(n)) / (vol * np.sqrt(n))

    # --- régimen de volatilidad
    vol10, vol390 = ewm_vol(c, 10), ewm_vol(c, 390)
    f["vol_10_60"] = np.log(vol10 / vol60)
    f["vol_60_390"] = np.log(vol60 / vol390)
    f["vol_bps"] = vol60 * 1e4
    rng = (h - l) / c
    f["range_rel"] = rng / vol
    f["range_10"] = rng.rolling(10).mean() / vol
    park10 = np.sqrt((np.log(h / l) ** 2).rolling(10).mean() / (4 * np.log(2)))
    f["park_10_60"] = np.log(park10.clip(lower=1e-7) / vol)

    # --- forma de la vela
    span = (h - l).replace(0, np.nan)
    f["body"] = ((c - o) / span).fillna(0.0)
    f["wick_up"] = ((h - np.maximum(o, c)) / span).fillna(0.0)
    f["wick_dn"] = ((np.minimum(o, c) - l) / span).fillna(0.0)
    f["close_pos"] = ((c - l) / span).fillna(0.5)

    # --- posición dentro del rango reciente
    for n in (15, 60, 240):
        hh, ll = h.rolling(n).max(), l.rolling(n).min()
        f[f"pos_{n}"] = ((c - ll) / (hh - ll).replace(0, np.nan)).fillna(0.5)
        f[f"width_{n}"] = np.log(hh / ll) / (vol * np.sqrt(n))

    # --- distancia a medias y z-scores
    for n in (10, 30, 120):
        f[f"ema_{n}"] = (lc - lc.ewm(span=n, min_periods=n).mean()) / (vol * np.sqrt(n))
    for n in (20, 60):
        sd = lc.rolling(n).std().replace(0, np.nan)
        f[f"z_{n}"] = ((lc - lc.rolling(n).mean()) / sd).fillna(0.0)
    f["macd"] = (lc.ewm(span=12).mean() - lc.ewm(span=26).mean()) / vol
    f["macd_hist"] = f["macd"] - f["macd"].ewm(span=9).mean()

    # --- osciladores y persistencia
    f["rsi_7"], f["rsi_14"] = _rsi(c, 7), _rsi(c, 14)
    for n in (10, 30):
        path = r1.abs().rolling(n).sum().replace(0, np.nan)
        f[f"eff_{n}"] = ((lc - lc.shift(n)) / path).fillna(0.0)  # ratio de eficiencia de Kaufman
    up = (r1 > 0).astype(float)
    f["up_5"], f["up_15"] = up.rolling(5).mean(), up.rolling(15).mean()
    f["autocorr_30"] = (r1 * r1.shift(1)).rolling(30).mean() / (vol**2)
    f["skew_60"] = (r1**3).rolling(60).mean() / (vol**3)

    # --- contexto de sesión (hora de Nueva York: los tres mercados giran en torno a EE. UU.)
    ny = df.index.tz_convert("America/New_York")
    minute = (ny.hour * 60 + ny.minute).to_numpy()
    f["tod_sin"] = pd.Series(np.sin(2 * np.pi * minute / 1440), index=df.index)
    f["tod_cos"] = pd.Series(np.cos(2 * np.pi * minute / 1440), index=df.index)
    f["minute_ny"] = pd.Series(minute.astype(float), index=df.index)
    f["dow"] = pd.Series(ny.dayofweek.astype(float), index=df.index)
    f["cash"] = pd.Series(((minute >= 570) & (minute < 960)).astype(float), index=df.index)
    f["mins_to_hour"] = pd.Series((minute % 60).astype(float), index=df.index)

    sid = pd.Series(session_id(df.index), index=df.index)
    g = df.groupby(sid.to_numpy())
    bars_in = g.cumcount() + 1
    f["sess_bars"] = np.log1p(bars_in.astype(float))
    sess_open = g["open"].transform("first")
    sess_hi, sess_lo = g["high"].cummax(), g["low"].cummin()
    scale = vol * np.sqrt(bars_in.clip(lower=1))
    f["sess_ret"] = np.log(c / sess_open) / scale
    f["sess_pos"] = ((c - sess_lo) / (sess_hi - sess_lo).replace(0, np.nan)).fillna(0.5)
    f["sess_hi_dist"] = np.log(sess_hi / c) / vol
    f["sess_lo_dist"] = np.log(c / sess_lo) / vol
    twap = g["close"].cumsum() / bars_in  # sin volumen fiable: TWAP anclado a la sesión
    f["twap_dist"] = np.log(c / twap) / scale
    prev_close = df["close"].groupby(sid.to_numpy()).last().shift(1)  # cierre de la sesión anterior
    f["prev_close_dist"] = np.log(c / sid.map(prev_close)) / (vol * np.sqrt(1380))

    # --- huecos de datos
    gap = pd.Series(gap_minutes(df.index), index=df.index)
    f["gap_now"] = np.log1p(gap.clip(upper=5000))
    f["gaps_30"] = (gap > 1.5).astype(float).rolling(30).sum()

    X = pd.DataFrame(f, index=df.index).replace([np.inf, -np.inf], np.nan)
    return X.astype("float32")


FEATURE_WARMUP = 400  # barras que hay que descartar al inicio (ventanas largas sin llenar)
