"""
Day trading / scalping del US500 y US100 (Capital.com) — plan del día + backtest honesto

Lo que SÍ se puede predecir dentro del día es CUÁNTO se va a mover el
mercado (el rango), no hacia dónde. Este programa usa eso para:

  PLAN DEL DÍA (por defecto)
    - Rango esperado de hoy (máximo - mínimo) con intervalo del 80%,
      en % y en puntos del índice. Día tranquilo / normal / movido.
    - Si el mercado ya abrió: el hueco (gap) contra el cierre de ayer y la
      probabilidad CALIBRADA de que hoy vuelva a tocar ese cierre.
    - Stop recomendado para no salir por ruido, tamaño de posición en
      Capital.com según tu cuenta y tu riesgo por operación, margen
      necesario y límite de pérdida diaria.

  BACKTEST (--backtest)
    Prueba con costes de Capital.com todas las estrategias intradía
    "clásicas" y dice cuáles tienen ventaja real y cuáles no:
      1. Relleno del hueco de apertura (gap fill), 30 años
      2. Gap fill con stop, con el orden real de las velas de 1 hora (3 años)
      3. Momentum intradía (1ª hora predice la última media hora), 3 años
      4. Scalping de reversión RSI(2) en velas de 5 min (60 días) y 1 hora (3 años)
      5. Precisión del pronóstico de rango

Datos: Yahoo Finance. Se usan los ETF SPY/QQQ porque sus aperturas son
reales (la apertura del índice ^GSPC en Yahoo es falsa antes de ~2010).
Los niveles en puntos se calculan sobre el índice (^GSPC / ^NDX), al que
siguen los CFD US500 / US100 de Capital.com.

AVISO: Proyecto educativo, NO es asesoría financiera. Los CFDs son
apalancados y la mayoría de cuentas minoristas pierden dinero.

Uso:
  python daytrading.py                              # plan de hoy, US500
  python daytrading.py --indice us100 --cuenta 2000 --riesgo 0.5
  python daytrading.py --backtest
  python datos_dukascopy.py --indice us500 --desde 2021-01-01   # años de velas de 5 min
  python daytrading.py --backtest --csv datos/us500_5m.csv
"""

import argparse
import datetime
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

INDICES = {
    "us500": {"etf": "SPY", "indice": "^GSPC", "nombre": "US500 (S&P 500)", "inicio": "1995-01-01"},
    "us100": {"etf": "QQQ", "indice": "^NDX", "nombre": "US100 (Nasdaq 100)", "inicio": "1999-04-01"},
}
NY = "America/New_York"


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
def bajar(ticker, **kw):
    df = yf.download(ticker, auto_adjust=True, progress=False, **kw)
    if df.empty:
        raise SystemExit(f"Sin datos para {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def bajar_intradia(ticker, intervalo, periodo):
    df = bajar(ticker, interval=intervalo, period=periodo)
    df.index = df.index.tz_convert(NY)
    return df


def ahora_ny():
    return pd.Timestamp.now(tz=NY)


def sesion_completa(fecha):
    """True si la sesión de esa fecha ya cerró (16:00 Nueva York)."""
    ahora = ahora_ny()
    return fecha < ahora.date() or (fecha == ahora.date() and ahora.hour >= 16)


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def pf(r):
    neg = -r[r < 0].sum()
    return r[r > 0].sum() / neg if neg > 0 else np.inf


def resumen(r):
    r = pd.Series(r).dropna()
    if len(r) == 0:
        return dict(n=0, win=np.nan, avg=np.nan, pf=np.nan)
    return dict(n=len(r), win=(r > 0).mean(), avg=r.mean(), pf=pf(r))


# ----------------------------------------------------------------------------
# Modelo de rango diario (lo que más se puede predecir)
# ----------------------------------------------------------------------------
def features_rango(d, vix):
    """Todo lo conocido en la APERTURA del día t."""
    X = pd.DataFrame(index=d.index)
    rng = np.log((d["High"] - d["Low"]) / d["Open"])
    ret = d["Close"].pct_change()
    X["rng_1"] = rng.shift(1)
    X["rng_5"] = rng.shift(1).rolling(5).mean()
    X["rng_22"] = rng.shift(1).rolling(22).mean()
    X["abs_ret_1"] = np.log(ret.abs().shift(1) + 1e-5)
    X["baja_ayer"] = (ret.shift(1) < 0).astype(int)
    X["gap_abs"] = np.log((d["Open"] / d["Close"].shift(1) - 1).abs() + 1e-4)
    X["lunes"] = (d.index.dayofweek == 0).astype(int)
    v = vix.reindex(d.index).ffill().shift(1)
    X["vix"] = np.log(v)
    X["vix_vs_rng"] = X["vix"] - X["rng_22"]
    return X, rng


def modelo_rango(d, vix, evaluar=True):
    X, y = features_rango(d, vix)
    ok = X.notna().all(axis=1) & y.notna() & np.isfinite(y)
    Xo, yo = X[ok], y[ok]
    res = {}
    if evaluar:
        pred, lo, hi = (pd.Series(index=Xo.index, dtype=float) for _ in range(3))
        for año in sorted(set(Xo.index.year))[5:]:
            tr, te = Xo.index.year < año, Xo.index.year == año
            m = LinearRegression().fit(Xo[tr], yo[tr])
            resid = yo[tr] - m.predict(Xo[tr])
            p = m.predict(Xo[te])
            pred[te], lo[te], hi[te] = p, p + resid.quantile(0.1), p + resid.quantile(0.9)
        msk = pred.notna()
        yt, pt = yo[msk], pred[msk]
        r2 = 1 - ((yt - pt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        dentro = ((yt >= lo[msk]) & (yt <= hi[msk])).mean()
        med = y.shift(1).rolling(252).median().reindex(yt.index)
        grande = (yt > med)
        auc = roc_auc_score(grande[med.notna()], (pt - med)[med.notna()])
        acierto = ((pt > med) == grande)[med.notna()].mean()
        res = dict(r2=r2, cobertura=dentro, auc=auc, acierto=acierto, desde=yt.index[0].year,
                   err_mediano=np.median(np.abs(np.exp(yt) - np.exp(pt)) / np.exp(yt)))
    m = LinearRegression().fit(Xo, yo)
    resid = yo - m.predict(Xo)
    return m, X, y, resid.quantile(0.1), resid.quantile(0.9), res


# ----------------------------------------------------------------------------
# Gap fill: probabilidad calibrada de tocar el cierre de ayer
# ----------------------------------------------------------------------------
def datos_gap(d, rango_pred):
    pc = d["Close"].shift(1)
    gap = d["Open"] / pc - 1
    toca = np.where(gap < 0, d["High"] >= pc, d["Low"] <= pc)
    tendencia = (d["Close"] > d["Close"].rolling(200).mean()).shift(1)
    G = pd.DataFrame({"gap": gap, "toca": toca.astype(int), "pc": pc,
                      "rel": np.log(gap.abs() / np.exp(rango_pred)),
                      "baja": (gap < 0).astype(int), "tend": tendencia.astype(float)})
    G["baja_x_tend"] = G["baja"] * G["tend"]
    # resultado de la operación "ir a por el relleno": entra a la apertura,
    # objetivo = cierre de ayer, si no se llena sale al cierre
    s = np.where(gap < 0, 1, -1)
    G["ret"] = np.where(toca, s * (pc / d["Open"] - 1), s * (d["Close"] / d["Open"] - 1))
    return G[(gap.abs() >= 0.0005) & (gap.abs() <= 0.03)].dropna()


COLS_GAP = ["rel", "baja", "tend", "baja_x_tend"]


def modelo_gap(G, evaluar=True):
    res = {}
    if evaluar:
        p = pd.Series(index=G.index, dtype=float)
        for año in sorted(set(G.index.year))[5:]:
            tr, te = G.index.year < año, G.index.year == año
            m = make_pipeline(StandardScaler(), LogisticRegression()).fit(G.loc[tr, COLS_GAP], G.loc[tr, "toca"])
            p[te] = m.predict_proba(G.loc[te, COLS_GAP])[:, 1]
        res["p"] = p.dropna()
    m = make_pipeline(StandardScaler(), LogisticRegression()).fit(G[COLS_GAP], G["toca"])
    return m, res


# ----------------------------------------------------------------------------
# PLAN DEL DÍA
# ----------------------------------------------------------------------------
def plan(cfg, a):
    print(f"\nDescargando datos de {cfg['nombre']}...")
    d = bajar(cfg["etf"], start=cfg["inicio"])
    idx = bajar(cfg["indice"], start="2015-01-01")
    vix = bajar("^VIX", start=cfg["inicio"])["Close"]

    hoy = ahora_ny().date()
    ultima = d.index[-1].date()
    abierto = ultima == hoy and not sesion_completa(hoy)
    completos = d[[sesion_completa(x.date()) for x in d.index]]
    vix_c = vix[[sesion_completa(x.date()) for x in vix.index]]

    m_rng, _, _, q10, q90, _ = modelo_rango(completos, vix_c, evaluar=False)
    G = datos_gap(completos, features_rango(completos, vix_c)[0].pipe(
        lambda X: pd.Series(m_rng.predict(X.fillna(X.median())), index=X.index)))
    m_gap, _ = modelo_gap(G, evaluar=False)

    # fila para "hoy" (o la próxima sesión si ya cerró)
    if abierto:
        base = d.copy()
        etiqueta = f"HOY {hoy} (mercado ABIERTO, apertura {d['Open'].iloc[-1]:.2f} en {cfg['etf']})"
    else:
        sig = pd.Timestamp(ultima) + pd.offsets.BDay(1)
        fila = completos.iloc[[-1]].copy()
        fila.index = [sig]
        fila[["Open", "High", "Low"]] = np.nan
        base = pd.concat([completos, fila])
        etiqueta = f"PRÓXIMA SESIÓN {sig.date()} (aún no abre: el hueco se sabrá a las 9:30 NY)"
    Xh, _ = features_rango(base, vix_c.reindex(base.index.union(vix_c.index)).ffill())
    xh = Xh.iloc[[-1]].copy()
    if not abierto:
        xh["gap_abs"] = Xh["gap_abs"].median()
    xh = xh.fillna(Xh.median())
    p_log = m_rng.predict(xh)[0]
    rng_med, rng_lo, rng_hi = np.exp(p_log), np.exp(p_log + q10), np.exp(p_log + q90)
    rng_hist = np.exp(features_rango(completos, vix_c)[1]).iloc[-252:].median()

    nivel = idx["Close"].iloc[-1]
    pts = lambda pct: pct * nivel
    print("\n" + "=" * 78)
    print(etiqueta)
    print("=" * 78)
    print(f"Índice {cfg['indice']} ≈ {nivel:,.1f} puntos (el CFD de Capital.com cotiza muy cerca)")
    estado = "TRANQUILO" if rng_med < 0.8 * rng_hist else "MOVIDO" if rng_med > 1.25 * rng_hist else "NORMAL"
    print(f"\nRANGO ESPERADO (máximo - mínimo del día):  día {estado}")
    print(f"  Lo más probable: {rng_med:.2%}  ≈ {pts(rng_med):,.0f} puntos   "
          f"(un día típico del último año: {rng_hist:.2%} ≈ {pts(rng_hist):,.0f} pts)")
    print(f"  80% de probabilidad entre {rng_lo:.2%} y {rng_hi:.2%}  "
          f"(≈ {pts(rng_lo):,.0f} a {pts(rng_hi):,.0f} puntos)")

    restante = None
    if abierto:
        hoy_f = d.iloc[-1]
        hecho = (hoy_f["High"] - hoy_f["Low"]) / hoy_f["Open"]
        restante = max(rng_med - hecho, 0)
        print(f"  Ya recorrido hoy: {hecho:.2%} ≈ {pts(hecho):,.0f} pts ({hecho / rng_med:.0%} de lo esperado)"
              f"  →  quedan ~{pts(restante):,.0f} pts de rango típico")
        if hecho >= rng_hi:
            print("  ⚠️ Ya se movió más que el máximo esperado: día excepcional, mucho cuidado.")
        elif hecho >= rng_med:
            print("  ⚠️ Ya hizo el rango típico del día: quedan pocos puntos, objetivos cortos o no operes.")

    if abierto:
        o, pc = d["Open"].iloc[-1], completos["Close"].iloc[-1]
        gap = o / pc - 1
        print(f"\nHUECO DE APERTURA: {gap:+.2%} ({pts(gap):+,.0f} pts) contra el cierre de ayer")
        if 0.0005 <= abs(gap) <= 0.03:
            tend = float(completos["Close"].iloc[-1] > completos["Close"].rolling(200).mean().iloc[-1])
            fila = pd.DataFrame({"rel": [np.log(abs(gap) / rng_med)], "baja": [int(gap < 0)], "tend": [tend]})
            fila["baja_x_tend"] = fila["baja"] * fila["tend"]
            p_toca = m_gap.predict_proba(fila[COLS_GAP])[0, 1]
            hechos = d.iloc[-1]
            ya = (hechos["High"] >= pc) if gap < 0 else (hechos["Low"] <= pc)
            print(f"  Probabilidad de que hoy vuelva a tocar el cierre de ayer: {p_toca:.0%}"
                  + ("   (¡YA LO TOCÓ!)" if ya else ""))
            sim = G[(np.sign(G["gap"]) == np.sign(gap)) & (G["gap"].abs().between(abs(gap) * 0.6, abs(gap) * 1.5))]
            rec = sim[sim.index >= sim.index[-1] - pd.Timedelta(days=3 * 365)]
            cost = 2 * a.coste_bps / 1e4
            for nom, s in [("30 años", sim), ("últimos 3 años", rec)]:
                r = resumen(s["ret"] - cost)
                print(f"  Ir a por el relleno, días parecidos ({nom}): {r['n']} días, ganó "
                      f"{r['win']:.0%}, media {r['avg']:+.3%}, PF {r['pf']:.2f}")
            print("  → Es información, no una señal: tras el spread la ventaja es pequeña o nula")
            print("    (sobre todo en los últimos años). Ver --backtest.")
        else:
            print("  Hueco fuera del rango estudiado (0.05%-3%).")

    # Gestión de riesgo para Capital.com
    stop_pct = 0.5 * rng_med
    stop_pts = pts(stop_pct)
    riesgo = a.cuenta * a.riesgo / 100
    tam = riesgo / stop_pts if stop_pts > 0 else 0
    margen = tam * nivel * a.margen / 100
    print("\nGESTIÓN DE RIESGO (Capital.com: tamaño 1 = 1 $ por punto; mínimo y margen reales con")
    print("  capital.py market US500)")
    print(f"  Stop recomendado ≥ {stop_pct:.2%} ≈ {stop_pts:,.0f} pts (la mitad del rango esperado;")
    print("    más cerca, el ruido normal del día te saca casi seguro)")
    print(f"  Cuenta {a.cuenta:,.0f} $, riesgo {a.riesgo}% = {riesgo:,.2f} $ por operación")
    print(f"  → Tamaño: {tam:.2f} contratos  (exposición {tam * nivel:,.0f} $, margen ≈ {margen:,.0f} $ "
          f"con {a.margen:.0f}%)")
    if margen > a.cuenta:
        print("  ⚠️ El margen supera tu cuenta: reduce el tamaño o el riesgo.")
    print(f"  Límite diario: si pierdes {3 * riesgo:,.2f} $ (3 operaciones perdidas), deja de operar hoy.")
    print(f"  Objetivo mínimo por operación: ≥ 1.5 × el stop ≈ {1.5 * stop_pts:,.0f} pts, y que quepa en "
          f"el rango que queda del día.")
    if restante is not None and pts(restante) < 1.5 * stop_pts:
        print(f"  ⚠️ Hoy quedan ~{pts(restante):,.0f} pts de rango típico, menos que ese objetivo: "
              "probablemente no compensa abrir operaciones nuevas.")

    print("\nCÓMO USARLO")
    if estado == "MOVIDO":
        print("  Día movido: stops más amplios, MENOS tamaño, nada de scalping de pocos puntos.")
    elif estado == "TRANQUILO":
        print("  Día tranquilo: objetivos cortos; no esperes rupturas grandes.")
    else:
        print("  Día normal: usa el stop y el tamaño de arriba.")
    print("  Ninguna regla de entrada intradía probada tiene ventaja estable tras el spread")
    print("  (python daytrading.py --backtest). Esto te dice CUÁNTO arriesgar, no hacia dónde ir.")
    print("  Proyecto educativo. NO es asesoría financiera.\n")


# ----------------------------------------------------------------------------
# BACKTEST HONESTO
# ----------------------------------------------------------------------------
def linea(nombre, r, extra=""):
    if r["n"] == 0:
        print(f"  {nombre:<38} sin operaciones")
        return
    print(f"  {nombre:<38}{r['n']:>6}{r['win']:>9.1%}{r['avg']:>10.3%}{r['pf']:>7.2f}  {extra}")


def veredicto(serie):
    """✔ sólo si hay ≥ 100 operaciones y PF > 1.1 en CADA mitad del periodo por separado."""
    r = pd.Series(serie).dropna()
    if len(r) < 100:
        return "✘ pocas operaciones"
    mitades = [r.iloc[: len(r) // 2], r.iloc[len(r) // 2:]]
    pfs = [pf(m) for m in mitades]
    txt = f"(mitades PF {pfs[0]:.2f} / {pfs[1]:.2f})"
    return ("✔ ventaja " if all(x > 1.1 for x in pfs) else "✘ sin ventaja estable ") + txt


def bt_gap(d, cost):
    print("\n1) RELLENO DEL HUECO (gap fill): entrar a la apertura hacia el cierre de ayer,")
    print("   objetivo = cierre de ayer, si no se llena salir al cierre. Datos diarios.")
    print(f"  {'':<38}{'Ops':>6}{'Acierto':>9}{'Media':>10}{'PF':>7}")
    pc = d["Close"].shift(1)
    gap = d["Open"] / pc - 1
    s = np.where(gap < 0, 1, -1)
    toca = np.where(gap < 0, d["High"] >= pc, d["Low"] <= pc)
    ret = pd.Series(np.where(toca, s * (pc / d["Open"] - 1), s * (d["Close"] / d["Open"] - 1)) - cost,
                    index=d.index)
    hace3 = d.index[-1] - pd.Timedelta(days=3 * 365)
    for lo, hi in [(0.001, 0.0025), (0.0025, 0.005), (0.005, 0.01), (0.01, 0.02)]:
        for lado, msk in [("baja", (gap <= -lo) & (gap > -hi)), ("sube", (gap >= lo) & (gap < hi))]:
            linea(f"hueco {lado} {lo:.2%}-{hi:.2%}", resumen(ret[msk]), veredicto(ret[msk]))
            linea(f"   … sólo últimos 3 años", resumen(ret[msk & (d.index >= hace3)]))


def bt_gap_stop(barras, cost, etiqueta):
    print(f"\n2) GAP FILL CON Y SIN STOP, viendo el orden real dentro del día ({etiqueta}).")
    print("   Huecos de 0.1% a 0.5%. Si en la misma vela se tocan stop y objetivo, cuenta el stop.")
    print(f"  {'':<38}{'Ops':>6}{'Acierto':>9}{'Media':>10}{'PF':>7}")
    dias = [g for _, g in barras.groupby(barras.index.date)]
    res = {}
    for k in range(1, len(dias)):
        g, pc = dias[k], dias[k - 1]["Close"].iloc[-1]
        o = g["Open"].iloc[0]
        gap = o / pc - 1
        if not 0.001 <= abs(gap) < 0.005:
            continue
        s = 1 if gap < 0 else -1
        lo_, hi_ = g["Low"].to_numpy(), g["High"].to_numpy()
        for stop in (None, 0.005, 0.01):
            out = None
            for lo, hi in zip(lo_, hi_):
                if stop and ((lo <= o * (1 - stop)) if s == 1 else (hi >= o * (1 + stop))):
                    out = -stop
                    break
                if (hi >= pc) if s == 1 else (lo <= pc):
                    out = s * (pc / o - 1)
                    break
            if out is None:
                out = s * (g["Close"].iloc[-1] / o - 1)
            res.setdefault(("baja" if s == 1 else "sube", stop), []).append(out - cost)
    for (lado, stop), v in sorted(res.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        nombre = f"hueco {lado}, " + ("sin stop" if stop is None else f"stop {stop:.1%}")
        linea(nombre, resumen(v), veredicto(v))


def bt_momentum(barras, cost, etiqueta, minutos_inicio):
    print(f"\n3) MOMENTUM INTRADÍA (Gao et al. 2018) ({etiqueta}): si de ayer a las "
          f"{'10:00' if minutos_inicio == 30 else '10:30'} sube,")
    print("   comprar la última media hora (15:30-16:00); si baja, vender.")
    filas, prev = [], None
    for dia, g in barras.groupby(barras.index.date):
        t0 = pd.Timestamp(dia, tz=NY) + pd.Timedelta(hours=9, minutes=30)
        ini = g[g.index < t0 + pd.Timedelta(minutes=minutos_inicio)]
        fin = g[g.index >= t0 + pd.Timedelta(hours=6)]
        if prev is not None and len(ini) and len(fin):
            primera = ini["Close"].iloc[-1] / prev - 1
            filas.append(np.sign(primera) * (fin["Close"].iloc[-1] / fin["Open"].iloc[0] - 1) - cost)
        prev = g["Close"].iloc[-1]
    print(f"  {'':<38}{'Ops':>6}{'Acierto':>9}{'Media':>10}{'PF':>7}")
    linea("todas las sesiones", resumen(filas), veredicto(filas))


def bt_orb(m5, cost, etiqueta):
    print(f"\n5) RUPTURA DEL RANGO DE APERTURA, ORB de 5 min (Zarattini y Aziz 2023) ({etiqueta}):")
    print("   dirección de la 1ª vela de 5 min; entrar en la 2ª; stop en el otro extremo de la")
    print("   1ª vela; objetivo 10 veces el riesgo o salir al cierre. Acierto bajo por diseño.")
    print(f"  {'':<38}{'Ops':>6}{'Acierto':>9}{'Media':>10}{'PF':>7}")
    ops, rs = [], []
    for _, g in m5.groupby(m5.index.date):
        if len(g) < 10:
            continue
        b0 = g.iloc[0]
        if b0["Close"] == b0["Open"]:
            continue
        s = 1 if b0["Close"] > b0["Open"] else -1
        e = g["Open"].iloc[1]
        stop = b0["Low"] if s == 1 else b0["High"]
        riesgo = (e - stop) * s
        if riesgo <= 0:
            continue
        obj = e + s * 10 * riesgo
        out = None
        for _, b in g.iloc[1:].iterrows():
            if (b["Low"] <= stop) if s == 1 else (b["High"] >= stop):
                out = stop
                break
            if (b["High"] >= obj) if s == 1 else (b["Low"] <= obj):
                out = obj
                break
        if out is None:
            out = g["Close"].iloc[-1]
        ops.append(s * (out / e - 1) - cost)
        rs.append((s * (out - e) - cost * e) / riesgo)
    linea("en % del precio", resumen(ops), veredicto(ops))
    if rs:
        r = pd.Series(rs)
        print(f"  {'':<38}en múltiplos del riesgo (R): media {r.mean():+.3f} R por operación, "
              f"ganadora media {r[r > 0].mean():+.2f} R")


def scalp_rsi(df, cost, max_velas, lado):
    df = df.copy()
    df["dia"] = df.index.date
    c = df["Close"]
    r = rsi(c, 2) if lado == 1 else 100 - rsi(c, 2)
    sal = (c > c.rolling(5).mean()) if lado == 1 else (c < c.rolling(5).mean())
    ent, sal = (r < 10).to_numpy(), sal.to_numpy()
    o, cv, dia = df["Open"].to_numpy(), c.to_numpy(), df["dia"].to_numpy()
    ops, i, n = [], 0, len(df)
    while i < n - 2:
        if not ent[i] or dia[i + 1] != dia[i]:
            i += 1
            continue
        e = j = i + 1
        while j < n - 1 and dia[j + 1] == dia[e] and not sal[j] and j - e < max_velas:
            j += 1
        px = o[j + 1] if (j < n - 1 and dia[j + 1] == dia[e] and sal[j]) else cv[j]
        ops.append(lado * (px / o[e] - 1) - cost)
        i = j + 1
    return ops


def bt_scalping(conjuntos, cost):
    print("\n4) SCALPING DE REVERSIÓN: RSI(2) < 10 compra (o > 90 vende), salida al cruzar")
    print("   la media de 5 velas, máximo 12 velas, siempre cerrado al final del día.")
    print(f"  {'':<38}{'Ops':>6}{'Acierto':>9}{'Media':>10}{'PF':>7}")
    for nom, df, mx in conjuntos:
        for lado, txt in [(1, "compras"), (-1, "ventas")]:
            ops = scalp_rsi(df, cost, mx, lado)
            linea(f"{nom}, {txt}", resumen(ops), veredicto(ops))


def leer_csv_5m(ruta):
    df = pd.read_csv(ruta)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("Datetime"), utc=True)).tz_convert(NY)
    m = df.index.hour * 60 + df.index.minute
    df = df[(m >= 9 * 60 + 30) & (m < 16 * 60)].sort_index()
    return df[~df.index.duplicated()].dropna(subset=["Open", "High", "Low", "Close"])


def backtest(cfg, a):
    cost = 2 * a.coste_bps / 1e4
    print(f"\nBACKTEST {cfg['nombre']}, coste {2 * a.coste_bps:.1f} bps por operación (spread de Capital.com)")
    print("Veredicto: ✔ sólo si hay ≥ 100 operaciones y PF > 1.1 en CADA mitad del periodo")
    print("(si sólo gana en una mitad, probablemente fue suerte o la ventaja desapareció).")
    d = bajar(cfg["etf"], start=cfg["inicio"])
    d = d[[sesion_completa(x.date()) for x in d.index]]
    vix = bajar("^VIX", start=cfg["inicio"])["Close"]
    h = bajar_intradia(cfg["etf"], "1h", "730d")
    h = h[[sesion_completa(x) for x in h.index.date]]
    if a.csv:
        m5 = leer_csv_5m(a.csv)
        n_dias = len(set(m5.index.date))
        et5 = f"velas 5 min, {n_dias} días ({m5.index[0].year}-{m5.index[-1].year})"
    else:
        m5 = bajar_intradia(cfg["etf"], "5m", "60d")
        m5 = m5[[sesion_completa(x) for x in m5.index.date]]
        et5 = f"velas 5 min, sólo {len(set(m5.index.date))} días: usa --csv para más"
    print(f"Datos intradía: {et5}; velas 1 hora de {cfg['etf']}, ~3 años")

    bt_gap(d, cost)
    if a.csv:
        bt_gap_stop(m5, cost, et5)
        bt_momentum(m5, cost, et5, 30)
    else:
        bt_gap_stop(h, cost, "velas 1 hora, ~3 años")
        bt_momentum(h, cost, "velas 1 hora, ~3 años", 60)
    bt_scalping([(et5.split(",")[0] + f" ({len(set(m5.index.date))} días)", m5, 12),
                 ("velas 1 hora (~3 años)", h, 6)], cost)
    bt_orb(m5, cost, et5)

    print("\n6) PRONÓSTICO DEL RANGO DEL DÍA (lo que sí funciona)")
    m, X, y, q10, q90, res = modelo_rango(d, vix[[sesion_completa(x.date()) for x in vix.index]])
    print(f"  Fuera de muestra {res['desde']}-hoy:")
    print(f"  - Explica el {res['r2']:.0%} de las diferencias de rango entre días (R²)")
    print(f"  - El rango real cayó dentro del intervalo del 80% el {res['cobertura']:.0%} de los días")
    print(f"  - Acierta si el día será más o menos movido que lo normal el {res['acierto']:.0%} "
          f"de las veces (AUC {res['auc']:.2f})")
    print(f"  - Error típico del pronóstico: {res['err_mediano']:.0%} del rango")

    G = datos_gap(d, pd.Series(m.predict(X.fillna(X.median())), index=X.index))
    _, rg = modelo_gap(G)
    p = rg["p"]
    toca = G.loc[p.index, "toca"]
    print("\n  Probabilidad de que se rellene el hueco (calibración fuera de muestra):")
    print(f"  {'Prob. dicha':<16}{'Días':>7}{'Pasó de verdad':>16}{'Media operando':>17}")
    for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]:
        msk = (p >= lo) & (p < hi)
        if msk.sum() < 20:
            continue
        r = resumen(G.loc[p.index[msk], "ret"] - cost)
        print(f"  {f'{lo:.0%}-{min(hi, 1):.0%}':<16}{msk.sum():>7}{toca[msk].mean():>16.0%}"
              f"{r['avg']:>16.3%}")
    print("\nCONCLUSIÓN: mira la columna de veredicto. El rango y la probabilidad de tocar un")
    print("nivel se predicen bien; las reglas de entrada, casi nunca con ventaja estable.\n")


def main():
    ap = argparse.ArgumentParser(description="Day trading / scalping: plan del día y backtest honesto")
    ap.add_argument("--indice", default="us500", choices=list(INDICES))
    ap.add_argument("--backtest", action="store_true", help="probar todas las estrategias")
    ap.add_argument("--csv", help="velas de 5 min con años de historia (datos_dukascopy.py o capital.py candles)")
    ap.add_argument("--cuenta", type=float, default=1000, help="saldo de tu cuenta en $")
    ap.add_argument("--riesgo", type=float, default=1.0, help="%% de la cuenta que arriesgas por operación")
    ap.add_argument("--margen", type=float, default=5.0, help="margen del CFD en %% (5%% = apalancamiento 1:20)")
    ap.add_argument("--coste-bps", type=float, default=0.55,
                    help="coste por lado en bps (mitad del spread). US500 ≈ 0.4-1.1")
    a = ap.parse_args()
    cfg = INDICES[a.indice]
    if a.backtest:
        backtest(cfg, a)
    else:
        plan(cfg, a)


if __name__ == "__main__":
    main()
