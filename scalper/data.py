"""Carga de barras de 1 minuto.

Fuentes:
  * histdata.com  -> histórico de varios años (cotización bid, sin volumen, reloj de servidor EET-7h).
  * Yahoo Finance -> últimos ~30 días de futuros (GC=F, NQ=F, ES=F) para la señal en vivo.
  * CSV propio    -> exportación de MT4/MT5 u otro bróker.

Todas devuelven un DataFrame con índice DatetimeIndex UTC y columnas open/high/low/close.
"""
from __future__ import annotations

import io
import re
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR, MARKETS

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
HISTDATA_PAGE = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair}/{period}"
HISTDATA_GET = "https://www.histdata.com/get.php"
COLS = ["open", "high", "low", "close"]


# ----------------------------------------------------------------------------- limpieza
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena, elimina duplicados y barras imposibles."""
    df = df[COLS].astype("float64")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna()
    ok = (df > 0).all(axis=1) & (df["high"] >= df["low"])
    df = df[ok].copy()
    # coherencia OHLC (algunos proveedores redondean mal)
    df["high"] = df[COLS].max(axis=1)
    df["low"] = df[COLS].min(axis=1)
    return df


# ----------------------------------------------------------------------------- histdata
def _histdata_fetch(pair: str, year: int, month: int | None, session: requests.Session) -> bytes | None:
    period = f"{year}" if month is None else f"{year}/{month}"
    page = HISTDATA_PAGE.format(pair=pair.lower(), period=period)
    r = session.get(page, timeout=60)
    r.raise_for_status()
    m = re.search(r'name="tk" id="tk" value="([0-9a-f]+)"', r.text)
    if not m:
        return None
    datemonth = f"{year}" if month is None else f"{year}{month:02d}"
    form = dict(tk=m.group(1), date=str(year), datemonth=datemonth, platform="ASCII", timeframe="M1", fxpair=pair.upper())
    z = session.post(HISTDATA_GET, data=form, headers={"Referer": page}, timeout=300)
    if z.status_code != 200 or not z.content.startswith(b"PK"):
        return None
    return z.content


def _parse_histdata_zip(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = pd.read_csv(zf.open(name), sep=";", header=None, names=["ts", *COLS, "vol"], dtype={"ts": str})
    ts = pd.to_datetime(raw["ts"], format="%Y%m%d %H%M%S")
    # histdata lo documenta como "EST sin horario de verano", pero contrastado con Yahoo (UTC real)
    # el reloj del fichero es el típico de servidor de bróker: hora de Europa oriental menos 7 h,
    # es decir UTC-5 en invierno y UTC-4 en verano *con las fechas de cambio europeas*.
    idx = pd.DatetimeIndex(ts + pd.Timedelta(hours=7)).tz_localize("Europe/Athens", ambiguous="NaT", nonexistent="NaT")
    out = pd.DataFrame(raw[COLS].to_numpy(), index=idx.tz_convert("UTC").rename("time"), columns=COLS)
    return out[out.index.notna()]


def download_histdata(market: str, start_year: int = 2023, force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Descarga (con caché en data/) el histórico M1 de un mercado y lo devuelve limpio."""
    pair = MARKETS[market].histdata
    raw_dir = DATA_DIR / "raw" / pair
    raw_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    jobs: list[tuple[int, int | None]] = [(y, None) for y in range(start_year, now.year)]
    jobs += [(now.year, m) for m in range(1, now.month + 1)]

    sess = requests.Session()
    sess.headers["User-Agent"] = UA
    frames = []
    for year, month in jobs:
        tag = f"{year}" if month is None else f"{year}{month:02d}"
        path = raw_dir / f"{pair}_{tag}.zip"
        current = month is not None and (year, month) == (now.year, now.month)
        if force or current or not path.exists():
            blob = None
            for attempt in range(3):
                try:
                    blob = _histdata_fetch(pair, year, month, sess)
                    break
                except requests.RequestException:
                    time.sleep(2 * (attempt + 1))
            if blob is None:
                if verbose:
                    print(f"  {pair} {tag}: no disponible")
                continue
            path.write_bytes(blob)
            time.sleep(0.5)
        df = _parse_histdata_zip(path.read_bytes())
        if verbose:
            print(f"  {pair} {tag}: {len(df):,} barras")
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No se pudo descargar nada para {pair}")
    out = clean(pd.concat(frames))
    out.to_parquet(DATA_DIR / f"{market}_m1.parquet")
    return out


def load_history(market: str) -> pd.DataFrame:
    path = DATA_DIR / f"{market}_m1.parquet"
    if not path.exists():
        return download_histdata(market)
    return pd.read_parquet(path)


# ----------------------------------------------------------------------------- yahoo
def download_yahoo(ticker: str, days: int = 7, drop_incomplete: bool = True) -> pd.DataFrame:
    """Barras de 1 min de Yahoo (máx. ~30 días hacia atrás, en ventanas de 7 días)."""
    days = min(days, 29)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    frames = []
    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0"  # Yahoo rechaza (429) cabeceras de navegador completas
    t0 = start
    while t0 < end:
        t1 = min(t0 + timedelta(days=7), end)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        q = dict(interval="1m", period1=int(t0.timestamp()), period2=int(t1.timestamp()), includePrePost="true")
        r = sess.get(url, params=q, timeout=30)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        if res.get("timestamp"):
            quote = res["indicators"]["quote"][0]
            idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).rename("time")
            frames.append(pd.DataFrame({c: quote[c] for c in COLS}, index=idx))
        t0 = t1
    if not frames:
        raise RuntimeError(f"Yahoo no devolvió datos para {ticker}")
    df = clean(pd.concat(frames))
    if drop_incomplete and len(df):
        # La barra en formación llega con la hora del último precio (segundos != 0): se descarta.
        if df.index[-1] != df.index[-1].floor("min"):
            df = df.iloc[:-1]
        df = df[df.index < pd.Timestamp.now(tz="UTC").floor("min")]
    return df


# ----------------------------------------------------------------------------- CSV propio
def load_csv(path: str | Path, tz: str = "UTC") -> pd.DataFrame:
    """Lee un CSV de barras M1 (formato MT5 '<DATE> <TIME> <OPEN>...' o genérico con cabecera)."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip("<> ").lower() for c in df.columns]
    if "date" in df and "time" in df:
        ts = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    else:
        tcol = next(c for c in df.columns if c in ("time", "datetime", "timestamp", "date"))
        ts = pd.to_datetime(df[tcol])
    idx = pd.DatetimeIndex(ts)
    idx = idx.tz_localize(tz) if idx.tz is None else idx
    out = pd.DataFrame(df[COLS].to_numpy(dtype=float), index=idx.tz_convert("UTC").rename("time"), columns=COLS)
    return clean(out)


def synthetic(n: int = 5000, seed: int = 0, price: float = 100.0) -> pd.DataFrame:
    """Paseo aleatorio M1 para tests (sin ventaja predecible por construcción)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 00:00", periods=n, freq="1min", tz="UTC", name="time")
    ret = rng.normal(0, 3e-4, n)
    close = price * np.exp(np.cumsum(ret))
    open_ = np.r_[price, close[:-1]]
    spread = np.abs(rng.normal(0, 2e-4, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame(dict(open=open_, high=high, low=low, close=close), index=idx)
