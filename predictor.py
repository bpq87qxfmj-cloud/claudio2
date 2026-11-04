"""
Predictor de dirección del mercado (S&P 500, NASDAQ, etc.)

Qué hace:
  1. Descarga precios diarios históricos con yfinance.
  2. Calcula indicadores técnicos (retornos, medias móviles, RSI, MACD,
     volatilidad, volumen) usando SOLO información disponible hasta el día t.
  3. Entrena modelos para predecir si el día t+1 cierra al alza o a la baja.
  4. Los evalúa con validación "walk-forward" (entrena con el pasado,
     prueba con el futuro) para evitar trampas de "ver el futuro".
  5. Compara contra la línea base ingenua ("siempre sube") y contra
     comprar y mantener, y da la predicción para la próxima sesión.

AVISO: Esto es un proyecto educativo. Los mercados son casi eficientes;
ningún modelo predice de forma fiable. No es asesoría financiera.

Uso:
  python predictor.py                    # S&P 500 (^GSPC)
  python predictor.py --ticker ^IXIC     # NASDAQ Composite
  python predictor.py --ticker QQQ --start 2010-01-01
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------
# 1. Datos
# ----------------------------------------------------------------------------
def descargar(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit(f"No se pudieron descargar datos para {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# ----------------------------------------------------------------------------
# 2. Indicadores (features)
# ----------------------------------------------------------------------------
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    ganancia = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    perdida = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = ganancia / perdida.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c = df["Close"]
    ret = c.pct_change()

    for n in (1, 2, 5, 10, 20):
        f[f"ret_{n}d"] = c.pct_change(n)
    for n in (5, 20, 50, 200):
        f[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1
    f["sma50_vs_200"] = c.rolling(50).mean() / c.rolling(200).mean() - 1

    f["vol_10d"] = ret.rolling(10).std()
    f["vol_20d"] = ret.rolling(20).std()
    f["vol_ratio"] = f["vol_10d"] / ret.rolling(60).std()

    f["rsi_14"] = rsi(c, 14)
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c

    f["rango_dia"] = (df["High"] - df["Low"]) / c
    f["cierre_en_rango"] = (c - df["Low"]) / (df["High"] - df["Low"]).replace(0, np.nan)
    f["gap"] = df["Open"] / c.shift(1) - 1

    vol = df["Volume"].replace(0, np.nan)
    f["volumen_rel"] = vol / vol.rolling(20).mean()

    f["max_20d"] = c / c.rolling(20).max() - 1
    f["min_20d"] = c / c.rolling(20).min() - 1
    f["dia_semana"] = df.index.dayofweek
    return f


# ----------------------------------------------------------------------------
# 3. Modelos
# ----------------------------------------------------------------------------
def modelos():
    return {
        "Regresión logística": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=50,
            n_jobs=-1, random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.03,
            subsample=0.8, random_state=42,
        ),
    }


# ----------------------------------------------------------------------------
# 4. Validación walk-forward
# ----------------------------------------------------------------------------
def walk_forward(modelo, X, y, inicio_test: int, paso: int = 63):
    """Reentrena cada `paso` días (~1 trimestre) usando sólo el pasado."""
    probas = pd.Series(index=X.index, dtype=float)
    for i in range(inicio_test, len(X), paso):
        modelo.fit(X.iloc[:i], y.iloc[:i])
        fin = min(i + paso, len(X))
        probas.iloc[i:fin] = modelo.predict_proba(X.iloc[i:fin])[:, 1]
    return probas.dropna()


def backtest(probas, ret_siguiente, umbral=0.5):
    """Estrategia: invertido al día siguiente si P(sube) > umbral, si no en efectivo."""
    señal = (probas > umbral).astype(int)
    r = señal * ret_siguiente.loc[probas.index]
    bh = ret_siguiente.loc[probas.index]

    def stats(serie):
        total = (1 + serie).prod() - 1
        años = len(serie) / 252
        cagr = (1 + total) ** (1 / años) - 1
        sharpe = np.sqrt(252) * serie.mean() / serie.std() if serie.std() > 0 else 0
        curva = (1 + serie).cumprod()
        dd = (curva / curva.cummax() - 1).min()
        return cagr, sharpe, dd

    return stats(r), stats(bh), señal.mean()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Predictor de dirección del mercado")
    p.add_argument("--ticker", default="^GSPC", help="^GSPC (S&P 500), ^IXIC (NASDAQ), ^NDX, SPY, QQQ...")
    p.add_argument("--start", default="2000-01-01", help="Fecha de inicio de los datos")
    p.add_argument("--test-años", type=float, default=8, help="Años finales usados como prueba")
    args = p.parse_args()

    print(f"\nDescargando {args.ticker} desde {args.start}...")
    df = descargar(args.ticker, args.start)

    X_all = crear_features(df)
    ret_siguiente = df["Close"].pct_change().shift(-1)   # retorno de t a t+1
    y_all = (ret_siguiente > 0).astype(int)

    datos = X_all.join(y_all.rename("y")).join(ret_siguiente.rename("r1"))
    ultima_fila = X_all.iloc[[-1]]          # para predecir mañana (sin etiqueta aún)
    datos = datos.dropna()
    X, y, r1 = datos[X_all.columns], datos["y"], datos["r1"]

    inicio_test = max(500, len(X) - int(args.test_años * 252))
    print(f"Datos: {len(X)} días | entrenamiento inicial: {inicio_test} | prueba: {len(X) - inicio_test}")
    print(f"Periodo de prueba: {X.index[inicio_test].date()} → {X.index[-1].date()}\n")

    y_test = y.iloc[inicio_test:]
    base = y_test.mean()
    print(f"Línea base 'siempre sube': precisión = {base:.1%}\n")

    print(f"{'Modelo':<22}{'Acierto':>9}{'Precisión↑':>12}{'AUC':>7}"
          f"{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'%Invert.':>10}")
    print("-" * 86)

    resultados = {}
    for nombre, modelo in modelos().items():
        probas = walk_forward(modelo, X, y, inicio_test)
        yt = y.loc[probas.index]
        pred = (probas > 0.5).astype(int)
        acc = accuracy_score(yt, pred)
        prec = precision_score(yt, pred, zero_division=0)
        auc = roc_auc_score(yt, probas)
        (cagr, sh, dd), bh, expo = backtest(probas, r1)
        resultados[nombre] = (modelo, auc)
        print(f"{nombre:<22}{acc:>9.1%}{prec:>12.1%}{auc:>7.3f}"
              f"{cagr:>9.1%}{sh:>8.2f}{dd:>9.1%}{expo:>10.0%}")

    print(f"{'Comprar y mantener':<22}{base:>9.1%}{base:>12.1%}{0.5:>7.3f}"
          f"{bh[0]:>9.1%}{bh[1]:>8.2f}{bh[2]:>9.1%}{1:>10.0%}")

    # Predicción para la próxima sesión con el mejor modelo (por AUC)
    mejor = max(resultados, key=lambda k: resultados[k][1])
    modelo = resultados[mejor][0]
    modelo.fit(X, y)
    p_sube = modelo.predict_proba(ultima_fila[X.columns].fillna(0))[0, 1]

    print(f"\nÚltimo cierre: {df.index[-1].date()} = {df['Close'].iloc[-1]:,.2f}")
    print(f"Mejor modelo (AUC): {mejor}")
    print(f"Probabilidad de que la próxima sesión cierre AL ALZA: {p_sube:.1%}")
    print("→", "SUBE" if p_sube > 0.5 else "BAJA",
          "(confianza baja)" if abs(p_sube - 0.5) < 0.05 else "")

    print("\nCómo leer esto:")
    print(" - AUC ≈ 0.50 = equivale a lanzar una moneda. >0.53 sostenido ya sería notable.")
    print(" - Si el acierto no supera a 'siempre sube', el modelo no aporta valor real.")
    print(" - El backtest ignora comisiones, impuestos y deslizamiento.")
    print(" - Proyecto educativo. NO es asesoría financiera.\n")


if __name__ == "__main__":
    main()
