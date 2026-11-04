"""
Bot de scalping (intradía) — investigación y backtest

Qué hace:
  1. Descarga velas intradía (1m, 2m o 5m) de yfinance.
       - 1m: sólo los últimos ~7 días
       - 2m / 5m: últimos ~60 días
  2. Calcula indicadores intradía: VWAP, RSI corto, momentum, volatilidad,
     volumen relativo, posición en el rango del día y hora de la sesión.
  3. Entrena un modelo (HistGradientBoosting) que estima la probabilidad
     de que el precio suba en las próximas H velas.
  4. Validación walk-forward por días: entrena con días pasados y opera
     el día siguiente sin haberlo visto.
  5. Simula operaciones realistas:
       - Entrada en la APERTURA de la vela siguiente a la señal
       - Take-profit y stop-loss sobre máximos/mínimos de cada vela
         (si ambos se tocan en la misma vela, se asume el stop: pesimista)
       - Salida por tiempo tras H velas y cierre obligatorio al final del día
       - Comisión + deslizamiento por operación
       - Una sola posición abierta a la vez, largos y cortos

AVISO: En scalping los costes se comen casi toda la ventaja. La mayoría de
estrategias que parecen ganar sin costes pierden con ellos. Proyecto
educativo, NO es asesoría financiera. No lo conectes a dinero real.

Uso:
  python scalping.py                              # SPY, velas de 5m
  python scalping.py --ticker QQQ --intervalo 1m
  python scalping.py --ticker ES=F --coste-bps 0.5 --tp 0.2 --sl 0.15
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

PERIODO_MAX = {"1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d"}


# ----------------------------------------------------------------------------
# 1. Datos
# ----------------------------------------------------------------------------
def descargar(ticker: str, intervalo: str) -> pd.DataFrame:
    df = yf.download(ticker, period=PERIODO_MAX[intervalo], interval=intervalo,
                     auto_adjust=True, prepost=False, progress=False)
    if df.empty:
        raise SystemExit(f"No hay datos intradía para {ticker} ({intervalo})")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[df["High"] > 0]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("America/New_York")
    df["dia"] = df.index.date
    return df


# ----------------------------------------------------------------------------
# 2. Indicadores intradía (sólo usan información hasta el cierre de la vela t)
# ----------------------------------------------------------------------------
def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    g = df.groupby("dia")
    ret = c.pct_change()

    for n in (1, 2, 3, 5, 10, 20):
        f[f"ret_{n}"] = c.pct_change(n)
    for n in (9, 21):
        f[f"dist_ema{n}"] = c / c.ewm(span=n).mean() - 1

    # VWAP del día (se reinicia cada sesión)
    tp = (h + l + c) / 3
    vwap = (tp * v).groupby(df["dia"]).cumsum() / v.groupby(df["dia"]).cumsum().replace(0, np.nan)
    f["dist_vwap"] = c / vwap - 1

    f["rsi_7"] = rsi(c, 7)
    f["rsi_14"] = rsi(c, 14)
    f["vol_10"] = ret.rolling(10).std()
    f["vol_30"] = ret.rolling(30).std()
    f["vol_ratio"] = f["vol_10"] / f["vol_30"]

    f["rango_vela"] = (h - l) / c
    f["cuerpo_vela"] = (c - df["Open"]) / c
    f["cierre_en_vela"] = (c - l) / (h - l).replace(0, np.nan)
    f["volumen_rel"] = v / v.rolling(20).mean().replace(0, np.nan)

    max_dia, min_dia = g["High"].cummax(), g["Low"].cummin()
    f["pos_rango_dia"] = (c - min_dia) / (max_dia - min_dia).replace(0, np.nan)
    f["ret_desde_apertura"] = c / g["Open"].transform("first") - 1

    minutos = df.index.hour * 60 + df.index.minute
    f["minuto_sesion"] = minutos - (9 * 60 + 30)
    return f


# ----------------------------------------------------------------------------
# 3. Objetivo: ¿sube el precio en las próximas H velas (mismo día)?
# ----------------------------------------------------------------------------
def crear_objetivo(df: pd.DataFrame, h: int) -> pd.Series:
    # Desde la apertura de t+1 (entrada real) hasta el cierre de t+h
    entrada = df.groupby("dia")["Open"].shift(-1)
    salida = df.groupby("dia")["Close"].shift(-h)
    fwd = salida / entrada - 1
    return fwd


# ----------------------------------------------------------------------------
# 4. Simulador de operaciones
# ----------------------------------------------------------------------------
def simular(df, señales, h, tp_pct, sl_pct, coste_bps):
    """señales: Serie con +1 (largo), -1 (corto), 0 (nada), indexada como df."""
    o, hi, lo, c = (df[k].to_numpy() for k in ("Open", "High", "Low", "Close"))
    dias = df["dia"].to_numpy()
    s = señales.reindex(df.index).fillna(0).to_numpy()
    coste = 2 * coste_bps / 1e4      # entrada + salida
    ops = []
    i, n = 0, len(df)
    while i < n - 1:
        if s[i] == 0 or dias[i + 1] != dias[i]:
            i += 1
            continue
        lado, j0 = s[i], i + 1
        px_in = o[j0]
        tp = px_in * (1 + lado * tp_pct / 100)
        sl = px_in * (1 - lado * sl_pct / 100)
        motivo, px_out, j = "tiempo", None, j0
        for j in range(j0, min(j0 + h, n)):
            if dias[j] != dias[j0]:
                j -= 1
                motivo, px_out = "fin_dia", c[j]
                break
            toca_sl = lo[j] <= sl if lado > 0 else hi[j] >= sl
            toca_tp = hi[j] >= tp if lado > 0 else lo[j] <= tp
            if toca_sl:                       # pesimista: el stop primero
                motivo, px_out = "stop", sl
                break
            if toca_tp:
                motivo, px_out = "take_profit", tp
                break
        if px_out is None:
            px_out = c[j]
        r = lado * (px_out / px_in - 1) - coste
        ops.append({"entrada": df.index[j0], "lado": "LARGO" if lado > 0 else "CORTO",
                    "px_in": px_in, "px_out": px_out, "ret": r, "motivo": motivo})
        i = j + 1                              # una posición a la vez
    return pd.DataFrame(ops)


def resumen(ops: pd.DataFrame, titulo: str):
    print(f"\n=== {titulo} ===")
    if ops.empty:
        print("Sin operaciones.")
        return
    r = ops["ret"]
    ganadoras, perdedoras = r[r > 0], r[r <= 0]
    pf = ganadoras.sum() / -perdedoras.sum() if perdedoras.sum() < 0 else np.inf
    curva = (1 + r).cumprod()
    dd = (curva / curva.cummax() - 1).min()
    por_dia = ops.groupby(ops["entrada"].dt.date)["ret"].sum()
    print(f"Operaciones:        {len(ops)}  ({len(ops) / max(len(por_dia), 1):.1f} por día)")
    print(f"Tasa de acierto:    {len(ganadoras) / len(r):.1%}")
    print(f"Ganancia media:     {ganadoras.mean() * 100 if len(ganadoras) else 0:+.3f}%   "
          f"Pérdida media: {perdedoras.mean() * 100 if len(perdedoras) else 0:+.3f}%")
    print(f"Retorno por op:     {r.mean() * 100:+.4f}%")
    print(f"Profit factor:      {pf:.2f}   (>1 gana, <1 pierde)")
    print(f"Retorno total:      {curva.iloc[-1] - 1:+.2%}")
    print(f"Máxima caída:       {dd:.2%}")
    print(f"Días positivos:     {(por_dia > 0).mean():.0%} de {len(por_dia)}")
    print("Salidas:            " + ", ".join(f"{k}={v}" for k, v in ops["motivo"].value_counts().items()))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Backtest de scalping con ML")
    p.add_argument("--ticker", default="SPY", help="SPY, QQQ, ES=F, NQ=F, AAPL... (los índices ^ no traen volumen)")
    p.add_argument("--intervalo", default="5m", choices=list(PERIODO_MAX))
    p.add_argument("--horizonte", type=int, default=6, help="Velas máximas por operación")
    p.add_argument("--tp", type=float, default=0.25, help="Take-profit en %%")
    p.add_argument("--sl", type=float, default=0.20, help="Stop-loss en %%")
    p.add_argument("--umbral", type=float, default=0.58, help="Probabilidad mínima para entrar")
    p.add_argument("--coste-bps", type=float, default=1.0,
                   help="Coste por lado (comisión + deslizamiento) en puntos básicos. 1 bps = 0.01%%")
    p.add_argument("--dias-entreno", type=int, default=15, help="Días iniciales sólo de entrenamiento")
    args = p.parse_args()

    print(f"\nDescargando {args.ticker} en velas de {args.intervalo}...")
    df = descargar(args.ticker, args.intervalo)
    dias = sorted(df["dia"].unique())
    print(f"{len(df)} velas en {len(dias)} días ({dias[0]} → {dias[-1]})")
    if df["Volume"].sum() == 0:
        print("Aviso: este ticker no trae volumen; VWAP y volumen relativo no servirán.")

    X = crear_features(df)
    fwd = crear_objetivo(df, args.horizonte)
    y = (fwd > 0).astype(int)
    validos = X.notna().all(axis=1) & fwd.notna()

    dias_entreno = min(args.dias_entreno, len(dias) - 2)
    if dias_entreno < 2:
        raise SystemExit("Muy pocos días de datos. Usa un intervalo de 2m o 5m.")

    # Walk-forward: para cada día de prueba, entrena con todos los días anteriores
    probas = pd.Series(index=df.index, dtype=float)
    for d in dias[dias_entreno:]:
        train = validos & (df["dia"] < d)
        test = df["dia"] == d
        modelo = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200,
            min_samples_leaf=100, l2_regularization=1.0, random_state=42)
        modelo.fit(X[train], y[train])
        probas[test] = modelo.predict_proba(X[test].fillna(X[train].median()))[:, 1]

    eval_mask = probas.notna() & validos
    auc = roc_auc_score(y[eval_mask], probas[eval_mask])
    print(f"Días de prueba: {len(dias) - dias_entreno} | AUC fuera de muestra: {auc:.3f} (0.50 = azar)")

    # Señales: no entrar en las últimas velas del día (no daría tiempo a salir)
    ultimas = df.groupby("dia").cumcount(ascending=False) < args.horizonte
    señal = pd.Series(0, index=df.index)
    señal[(probas > args.umbral) & ~ultimas] = 1
    señal[(probas < 1 - args.umbral) & ~ultimas] = -1

    test_idx = probas.notna()
    params = (args.horizonte, args.tp, args.sl)
    ops_sin = simular(df[test_idx], señal[test_idx], *params, coste_bps=0)
    ops_con = simular(df[test_idx], señal[test_idx], *params, coste_bps=args.coste_bps)

    # Referencia: entradas aleatorias con la misma frecuencia
    rng = np.random.default_rng(0)
    freq = (señal[test_idx] != 0).mean()
    azar = pd.Series(rng.choice([-1, 0, 1], size=test_idx.sum(),
                                p=[freq / 2, 1 - freq, freq / 2]), index=df.index[test_idx])
    azar[ultimas[test_idx]] = 0
    ops_azar = simular(df[test_idx], azar, *params, coste_bps=args.coste_bps)

    resumen(ops_sin, "Modelo SIN costes (irreal)")
    resumen(ops_con, f"Modelo CON costes ({args.coste_bps} bps por lado)")
    resumen(ops_azar, "Entradas AL AZAR con costes (referencia)")

    if not ops_con.empty:
        ops_con.to_csv("operaciones_scalping.csv", index=False)
        print("\nDetalle de operaciones guardado en operaciones_scalping.csv")

    # Señal en vivo con la última vela cerrada
    modelo.fit(X[validos], y[validos])
    p_ult = modelo.predict_proba(X.iloc[[-1]].fillna(X[validos].median()))[0, 1]
    accion = "COMPRAR (largo)" if p_ult > args.umbral else \
             "VENDER (corto)" if p_ult < 1 - args.umbral else "ESPERAR"
    print(f"\nÚltima vela: {df.index[-1]}  cierre = {df['Close'].iloc[-1]:.2f}")
    print(f"P(sube en {args.horizonte} velas) = {p_ult:.1%}  →  {accion}")
    print("(Fuera de horario de mercado esta señal no tiene sentido.)")

    print("\nCómo leer esto:")
    print(" - Si el modelo con costes no supera claramente al azar, no hay ventaja real.")
    print(" - 2 meses de datos son POCOS: los resultados pueden ser pura suerte.")
    print(" - Prueba distintos --tp/--sl/--umbral, pero cuidado con sobreajustar.")
    print(" - Proyecto educativo. NO es asesoría financiera.\n")


if __name__ == "__main__":
    main()
