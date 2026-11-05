"""
Señales de MAYOR probabilidad (medidas sin trampas)

Adivinar si el índice sube o baja MAÑANA no pasa de ~54% (ver predictor.py).
Este programa cambia la pregunta por tres que sí tienen más probabilidad,
y mide cada una con datos reales, costes de Capital.com y fuera de muestra:

  1. HORIZONTE: ¿qué probabilidad hay de que el índice esté más alto dentro
     de 1 día, 1 semana, 1 mes, 1 año? (la probabilidad sube con el tiempo)

  2. REBOTE TRAS CAÍDA (reversión a la media, regla publicada por Larry
     Connors en 2008): comprar cuando el RSI(2) cae por debajo de 10 estando
     el índice por encima de su media de 200 días, y vender cuando cierra
     por encima de su media de 5 días. Se mide el % de operaciones ganadoras
     con spread y financiación nocturna de CFD, y se compara contra la
     probabilidad base de estar comprado los mismos días al azar.

  3. VOLATILIDAD: ¿mañana será un día de movimiento GRANDE o pequeño?
     (el "cuánto" se predice mucho mejor que el "hacia dónde"). Sirve para
     ajustar stops y tamaño, o para no operar en días peligrosos.

AVISO: Más probabilidad de acierto NO significa ganar más dinero: una regla
puede acertar 70% de las veces y perder en total si las pérdidas son grandes.
Por eso se muestran siempre ganancia media, pérdida media y profit factor.
Proyecto educativo, NO es asesoría financiera.

Uso:
  python alta_probabilidad.py                    # S&P 500 (US500 en Capital.com)
  python alta_probabilidad.py --ticker ^NDX      # Nasdaq 100 (US100)
  python alta_probabilidad.py --coste-bps 1.1 --financiacion 8
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PERIODOS = [("1995-1999", "1995", "1999"), ("2000-2009", "2000", "2009"), ("2010-2019", "2010", "2019"), ("2020-hoy", "2020", "2100")]


def descargar(ticker, start):
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit(f"Sin datos para {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def titulo(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ----------------------------------------------------------------------------
# 1. Probabilidad según el horizonte
# ----------------------------------------------------------------------------
def horizonte(df, financiacion):
    titulo("1. ¿Qué probabilidad hay de que esté MÁS ALTO dentro de...?")
    c = df["Close"]
    print(f"{'Plazo':<14}{'Prob. de subir':>15}{'Media':>9}{'Peor':>9}{'Mejor':>9}"
          f"{'Coste CFD*':>12}")
    for nombre, n in [("1 día", 1), ("1 semana", 5), ("1 mes", 21), ("3 meses", 63),
                      ("6 meses", 126), ("1 año", 252), ("3 años", 756)]:
        r = (c.shift(-n) / c - 1).dropna()
        coste = financiacion / 100 * n * 365 / 252 / 365
        print(f"{nombre:<14}{(r > 0).mean():>15.1%}{r.mean():>9.1%}{r.min():>9.1%}"
              f"{r.max():>9.1%}{-coste:>12.2%}")
    print("* Financiación nocturna estimada si lo mantienes como CFD en Capital.com.")
    print("  A largo plazo es mejor un ETF real (sin financiación) que un CFD.")


# ----------------------------------------------------------------------------
# 2. Rebote tras caída (RSI(2) de Connors)
# ----------------------------------------------------------------------------
def operaciones_rebote(df, rsi_n, rsi_max, sma_tend, sma_salida, max_dias, modo,
                       coste_bps, financiacion):
    c, o = df["Close"], df["Open"]
    r = rsi(c, rsi_n)
    entrada = (r < rsi_max) & (c > c.rolling(sma_tend).mean())
    salida = c > c.rolling(sma_salida).mean()
    idx, cv, ov = df.index, c.to_numpy(), o.to_numpy()
    ent, sal = entrada.to_numpy(), salida.to_numpy()
    ops, i, n = [], sma_tend, len(df)
    while i < n - 1:
        if not ent[i]:
            i += 1
            continue
        # señal al cierre del día i
        if modo == "cierre":
            ie, pe = i, cv[i]                # entras justo antes del cierre (CFD lo permite)
        else:
            ie, pe = i + 1, ov[i + 1]        # entras en la apertura siguiente (más conservador)
        j = ie
        while j < n - 1 and not (sal[j] and j >= i + 1) and j - ie < max_dias:
            j += 1
        if modo == "cierre":
            js, ps = j, cv[j]
        else:
            js = min(j + 1, n - 1)
            ps = ov[js]
        noches = (idx[js] - idx[ie]).days
        bruto = ps / pe - 1
        neto = bruto - 2 * coste_bps / 1e4 - financiacion / 100 * noches / 365
        ops.append({"entrada": idx[ie], "salida": idx[js], "dias": js - ie,
                    "bruto": bruto, "ret": neto})
        i = js + 1
    return pd.DataFrame(ops)


def stats(ops, nombre):
    if ops.empty:
        return
    r = ops["ret"]
    g, p = r[r > 0], r[r <= 0]
    pf = g.sum() / -p.sum() if p.sum() < 0 else np.inf
    curva = (1 + r).cumprod()
    dd = (curva / curva.cummax() - 1).min()
    años = max((ops["salida"].iloc[-1] - ops["entrada"].iloc[0]).days / 365.25, 0.1)
    cagr = curva.iloc[-1] ** (1 / años) - 1
    print(f"{nombre:<26}{len(r):>5}{(r > 0).mean():>9.1%}{g.mean():>9.2%}{p.mean():>9.2%}"
          f"{pf:>7.2f}{r.mean():>9.3%}{cagr:>8.1%}{dd:>9.1%}")


def rebote(df, coste_bps, financiacion):
    titulo("2. REBOTE TRAS CAÍDA: comprar con RSI(2) < 10 en tendencia alcista")
    print("Regla publicada en 2008, así que 2010-hoy es prueba fuera de muestra real.")
    print(f"Costes: spread {2 * coste_bps:.1f} bps por operación + financiación "
          f"{financiacion:.1f}%/año por noche.\n")
    cab = (f"{'':<26}{'Ops':>5}{'Acierto':>9}{'Gana':>9}{'Pierde':>9}{'PF':>7}"
           f"{'Media':>9}{'CAGR':>8}{'MaxDD':>9}")
    res = {}
    for modo, texto in [("cierre", "Entrada al cierre"), ("apertura", "Entrada a la apertura sig.")]:
        ops = operaciones_rebote(df, 2, 10, 200, 5, 10, modo, coste_bps, financiacion)
        res[modo] = ops
        print(f"--- {texto} ---")
        print(cab)
        stats(ops, "Todo el periodo")
        for nom, a, b in PERIODOS:
            stats(ops[(ops["entrada"] >= a) & (ops["entrada"] <= b + "-12-31")], "  " + nom)
        print()

    falsas = (df["Open"].round(2) == df["Close"].shift(1).round(2)).mean()
    if falsas > 0.2:
        print(f"Aviso: en {falsas:.0%} de los días la apertura de este ticker en Yahoo es una copia del")
        print("cierre anterior (pasa con ^GSPC antes de ~2010). El modo 'apertura' no es fiable aquí:")
        print("usa --ticker SPY o QQQ para medirlo con aperturas reales.\n")
    # Probabilidad base: entrar cualquier día en tendencia alcista y aguantar lo mismo
    ops = res["cierre"]
    c = df["Close"]
    dur = max(int(round(ops["dias"].mean())), 1)
    tend = c > c.rolling(200).mean()
    fwd = (c.shift(-dur) / c - 1)[tend].dropna()
    fwd = fwd - 2 * coste_bps / 1e4 - financiacion / 100 * dur * 1.4 / 365
    print(f"Referencia (comprar CUALQUIER día en tendencia y aguantar {dur} días, con costes):")
    print(f"  acierto {(fwd > 0).mean():.1%}, media {fwd.mean():.3%} por operación")
    print(f"El rebote está invertido sólo ~{ops['dias'].sum() / len(df):.0%} del tiempo; "
          f"el resto del tiempo el dinero está libre.")
    return res


# ----------------------------------------------------------------------------
# 3. Volatilidad: ¿día grande o pequeño?
# ----------------------------------------------------------------------------
def volatilidad(df, vix):
    titulo("3. VOLATILIDAD: ¿mañana se moverá MÁS de lo normal (en cualquier dirección)?")
    c = df["Close"]
    ret = c.pct_change()
    a = ret.abs()
    X = pd.DataFrame(index=df.index)
    for k in range(1, 6):
        X[f"abs_{k}"] = a.shift(k - 1)
    X["ret_hoy"] = ret
    X["rv5"], X["rv22"], X["rv66"] = ret.rolling(5).std(), ret.rolling(22).std(), ret.rolling(66).std()
    X["rango"] = (df["High"] - df["Low"]) / c
    X["rango5"] = X["rango"].rolling(5).mean()
    X["dia_sem"] = df.index.dayofweek
    if vix is not None:
        v = vix.reindex(df.index).ffill()
        X["vix"] = v
        X["vix_cambio"] = v.pct_change()
        X["vix_vs_rv"] = v / (X["rv22"] * np.sqrt(252) * 100)
    umbral = a.rolling(252).median()                 # "normal" = mediana del último año
    for col in ("abs_1", "rv5", "rv22", "rv66", "rango5"):
        X[col + "_rel"] = np.log(X[col] / umbral + 1e-6)   # comparado con lo "normal"
    if vix is not None:
        X["vix_rel"] = np.log(X["vix"] / (umbral * np.sqrt(252) * 100))
    y = (a.shift(-1) > umbral).astype(int)
    datos = X.join(y.rename("y")).dropna()
    datos = datos[datos.index.isin(a.shift(-1).dropna().index)]
    Xd, yd = datos.drop(columns="y"), datos["y"]

    probas = pd.Series(index=Xd.index, dtype=float)
    años = sorted(set(Xd.index.year))
    for año in años[5:]:                              # reentrena cada año con el pasado
        tr, te = Xd.index.year < año, Xd.index.year == año
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        m.fit(Xd[tr], yd[tr])
        probas[te] = m.predict_proba(Xd[te])[:, 1]
    probas = probas.dropna()
    yt = yd.loc[probas.index]
    pred = (probas > 0.5).astype(int)
    print(f"Prueba fuera de muestra {probas.index[0].year}-{probas.index[-1].year}, "
          f"{len(yt)} días. Base (adivinar al azar): {max(yt.mean(), 1 - yt.mean()):.1%}")
    print(f"Acierto: {accuracy_score(yt, pred):.1%}   AUC: {roc_auc_score(yt, probas):.3f} "
          f"(0.50 = azar, 0.70+ = bueno)")
    print("\nCuando el modelo está MÁS seguro:")
    print(f"{'Prob. día grande':<22}{'Días':>7}{'Acierto':>9}{'Mov. medio real':>17}")
    sig = a.shift(-1).loc[probas.index]
    for lo, hi, txt in [(0.0, 0.25, "< 25% (día tranquilo)"), (0.25, 0.5, "25-50%"),
                        (0.5, 0.75, "50-75%"), (0.75, 1.01, "> 75% (día movido)")]:
        msk = (probas >= lo) & (probas < hi)
        if msk.sum() == 0:
            continue
        ac = (pred[msk] == yt[msk]).mean()
        print(f"{txt:<22}{msk.sum():>7}{ac:>9.1%}{sig[msk].mean():>17.2%}")

    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    m.fit(Xd, yd)
    hoy = X.iloc[[-1]].ffill()
    p_hoy = m.predict_proba(hoy[Xd.columns].fillna(Xd.median()))[0, 1]
    return p_hoy, umbral.iloc[-1]


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Señales de mayor probabilidad")
    ap.add_argument("--ticker", default="^GSPC", help="^GSPC (US500), ^NDX (US100), ^DJI (US30)")
    ap.add_argument("--start", default="1995-01-01")
    ap.add_argument("--coste-bps", type=float, default=1.0,
                    help="Coste por lado en bps (spread/2). US500 ≈ 0.4-1.1")
    ap.add_argument("--financiacion", type=float, default=8.0,
                    help="Financiación nocturna anual de un CFD largo en %% (≈ SOFR + 4%%)")
    a = ap.parse_args()

    print(f"Descargando {a.ticker} y VIX desde {a.start}...")
    df = descargar(a.ticker, a.start)
    try:
        vix = descargar("^VIX", a.start)["Close"]
    except SystemExit:
        vix = None
    print(f"{len(df)} días: {df.index[0].date()} → {df.index[-1].date()}")

    horizonte(df, a.financiacion)
    rebote(df, a.coste_bps, a.financiacion)
    p_vol, umbral = volatilidad(df, vix)

    titulo(f"HOY ({df.index[-1].date()}, cierre {df['Close'].iloc[-1]:,.2f})")
    c = df["Close"]
    r2 = rsi(c, 2).iloc[-1]
    sma200, sma5 = c.rolling(200).mean().iloc[-1], c.rolling(5).mean().iloc[-1]
    tendencia = c.iloc[-1] > sma200
    print(f"Tendencia: {'ALCISTA' if tendencia else 'BAJISTA'} (precio vs media 200 = "
          f"{c.iloc[-1] / sma200 - 1:+.1%})")
    print(f"RSI(2) = {r2:.1f}  →  ", end="")
    if tendencia and r2 < 10:
        print("SEÑAL DE REBOTE ACTIVA: la regla compraría hoy al cierre "
              f"y vendería cuando cierre por encima de {sma5:,.2f} (media 5).")
    else:
        print("sin señal de rebote (necesita RSI(2) < 10 y tendencia alcista).")
    print(f"Probabilidad de que mañana se mueva más de lo normal (> {umbral:.2%}): {p_vol:.0%}")
    if p_vol > 0.65:
        print("  → día probablemente MOVIDO: stops más amplios y posición más pequeña.")
    elif p_vol < 0.35:
        print("  → día probablemente TRANQUILO.")

    print("\nRecuerda: acierto alto ≠ ganar dinero. Mira el profit factor (PF > 1) y la")
    print("máxima caída. Resultados pasados no garantizan los futuros. No es asesoría financiera.\n")


if __name__ == "__main__":
    main()
