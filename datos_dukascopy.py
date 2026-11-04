"""
Descarga años de velas de 1 minuto de CFDs de índices desde Dukascopy (gratis)

Yahoo Finance sólo da 60 días de velas de 5 minutos: muy poco para probar
scalping. Dukascopy (banco suizo) publica el histórico de sus CFDs de
índices minuto a minuto, que siguen al mismo subyacente que los de
Capital.com:

    us500 -> USA500IDXUSD     us100 -> USATECHIDXUSD     us30 -> USA30IDXUSD

Guarda cada día en una caché (datos/dukascopy/...), así que si se corta la
conexión vuelves a ejecutarlo y sigue donde se quedó. Al final crea un CSV
con velas de N minutos de la sesión regular de EE. UU. (9:30-16:00 Nueva
York), en el formato que leen scalping.py --csv y daytrading.py --csv.

Uso:
  python datos_dukascopy.py --indice us500 --desde 2020-01-01
  python datos_dukascopy.py --indice us100 --desde 2020-01-01 --minutos 5
  python datos_dukascopy.py --indice us500 --desde 2024-01-01 --ask     # + spread de Dukascopy

Los datos son para investigación personal. El precio de Dukascopy no es
idéntico al de Capital.com (cada broker tiene su spread), pero los
movimientos del índice son los mismos.
"""

import argparse
import datetime
import lzma
import os
import struct
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

SIMBOLOS = {"us500": "USA500IDXUSD", "us100": "USATECHIDXUSD", "us30": "USA30IDXUSD"}
URL = "https://datafeed.dukascopy.com/datafeed/{s}/{y}/{m:02d}/{d:02d}/{lado}_candles_min_1.bi5"
ESCALA = 1000.0                       # precios de índices guardados x1000


def bajar_dia(simbolo, dia, lado, cache, espera):
    ruta = os.path.join(cache, simbolo, lado, f"{dia:%Y}", f"{dia:%Y-%m-%d}.bi5")
    if os.path.exists(ruta):
        return ruta, False
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    url = URL.format(s=simbolo, y=dia.year, m=dia.month - 1, d=dia.day, lado=lado)  # mes empieza en 0
    for intento in range(10):
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, headers={"User-Agent": "claudio2-research/1.0"}), timeout=30) as r:
                datos = r.read()
            with open(ruta + ".tmp", "wb") as fh:
                fh.write(datos)
            os.replace(ruta + ".tmp", ruta)
            time.sleep(espera)
            return ruta, True
        except urllib.error.HTTPError as e:
            if e.code == 404:                 # día sin datos
                open(ruta, "wb").close()
                return ruta, True
            pausa = min(60, 2 * (intento + 1)) if e.code == 429 else 3
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pausa = min(60, 3 * (intento + 1))
        time.sleep(pausa)
    raise RuntimeError(f"no se pudo bajar {url} tras 10 intentos")


def leer_dia(ruta, dia):
    raw = open(ruta, "rb").read()
    if not raw:
        return None
    raw = lzma.decompress(raw)
    n = len(raw) // 24
    if n == 0:
        return None
    a = np.array(struct.unpack(">" + "IIIIIf" * n, raw[: n * 24]), dtype=float).reshape(n, 6)
    t0 = pd.Timestamp(dia, tz="UTC")
    idx = t0 + pd.to_timedelta(a[:, 0], unit="s")
    return pd.DataFrame({"Open": a[:, 1] / ESCALA, "Close": a[:, 2] / ESCALA, "Low": a[:, 3] / ESCALA,
                         "High": a[:, 4] / ESCALA, "Volume": a[:, 5]}, index=idx)


def main():
    ap = argparse.ArgumentParser(description="Velas de 1 minuto de Dukascopy -> CSV intradía")
    ap.add_argument("--indice", default="us500", choices=list(SIMBOLOS))
    ap.add_argument("--desde", default="2021-01-01")
    ap.add_argument("--hasta", default=None, help="por defecto, ayer")
    ap.add_argument("--minutos", type=int, default=5, help="tamaño de vela del CSV final")
    ap.add_argument("--ask", action="store_true", help="bajar también ASK para medir el spread (el doble de descargas)")
    ap.add_argument("--cache", default="datos/dukascopy")
    ap.add_argument("--salida", default=None, help="por defecto datos/<indice>_<N>m.csv")
    ap.add_argument("--espera", type=float, default=0.4, help="segundos entre descargas (sé amable)")
    ap.add_argument("--24h", dest="todo", action="store_true", help="no recortar a la sesión regular de EE. UU.")
    a = ap.parse_args()

    simbolo = SIMBOLOS[a.indice]
    desde = datetime.date.fromisoformat(a.desde)
    hasta = datetime.date.fromisoformat(a.hasta) if a.hasta else datetime.date.today() - datetime.timedelta(days=1)
    dias = [desde + datetime.timedelta(days=k) for k in range((hasta - desde).days + 1)]
    dias = [d for d in dias if d.weekday() < 5]
    lados = ["BID", "ASK"] if a.ask else ["BID"]
    total = len(dias) * len(lados)
    print(f"{simbolo}: {len(dias)} días hábiles de {desde} a {hasta} ({total} archivos)")

    hechos, nuevos, t0 = 0, 0, time.time()
    for lado in lados:
        for dia in dias:
            _, nuevo = bajar_dia(simbolo, dia, lado, a.cache, a.espera)
            hechos += 1
            nuevos += nuevo
            if nuevo and nuevos % 50 == 0:
                vel = nuevos / (time.time() - t0)
                print(f"  {hechos}/{total} ({dia}) {vel:.1f} archivos/s, faltan ~"
                      f"{(total - hechos) / max(vel, 0.01) / 60:.0f} min", flush=True)
    print(f"Descarga completa ({nuevos} nuevos). Construyendo velas de {a.minutos} min...")

    partes = []
    for dia in dias:
        bid = leer_dia(os.path.join(a.cache, simbolo, "BID", f"{dia:%Y}", f"{dia:%Y-%m-%d}.bi5"), dia)
        if bid is None:
            continue
        if a.ask:
            ask = leer_dia(os.path.join(a.cache, simbolo, "ASK", f"{dia:%Y}", f"{dia:%Y-%m-%d}.bi5"), dia)
            if ask is not None:
                bid["Spread"] = (ask["Close"] - bid["Close"]).reindex(bid.index)
        partes.append(bid)
    m1 = pd.concat(partes).sort_index()
    m1 = m1[~m1.index.duplicated()]
    ny = m1.index.tz_convert("America/New_York")
    if not a.todo:
        minuto = ny.hour * 60 + ny.minute
        m1 = m1[(minuto >= 9 * 60 + 30) & (minuto < 16 * 60)]
        ny = m1.index.tz_convert("America/New_York")
    # Agrupar en velas de N minutos alineadas a la hora de Nueva York
    m1.index = ny
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    if "Spread" in m1:
        agg["Spread"] = "median"
    velas = m1.resample(f"{a.minutos}min", origin="start_day", offset="30min" if a.minutos >= 30 else "0min",
                        label="left", closed="left").agg(agg).dropna(subset=["Open"])
    # Quitar festivos / días planos (el CFD sigue cotizando aunque la bolsa esté cerrada)
    rango_dia = velas.groupby(velas.index.date).apply(lambda g: (g["High"].max() - g["Low"].min()) / g["Close"].iloc[-1])
    velas_dia = velas.groupby(velas.index.date).size()
    buenos = rango_dia[(rango_dia > 0.001) & (velas_dia.reindex(rango_dia.index) >= 0.8 * velas_dia.median())].index
    velas = velas[pd.Index(velas.index.date).isin(buenos)]
    velas.index = velas.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S")
    velas.index.name = "Datetime"
    salida = a.salida or os.path.join("datos", f"{a.indice}_{a.minutos}m.csv")
    os.makedirs(os.path.dirname(salida) or ".", exist_ok=True)
    velas.to_csv(salida)
    print(f"{len(velas)} velas en {len(buenos)} días -> {salida}")
    if "Spread" in velas:
        print(f"Spread mediano de Dukascopy: {(velas['Spread'] / velas['Close']).median() * 1e4:.2f} bps")


if __name__ == "__main__":
    sys.exit(main())
