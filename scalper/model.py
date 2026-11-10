"""Modelo (LightGBM) y validación walk-forward con purga.

El modelo estima el resultado esperado, en múltiplos de la barrera, de un largo abierto en
la siguiente barra (negativo => conviene el corto). Solo se opera cuando |predicción| supera
un umbral elegido en la cola de validación del tramo de entrenamiento; si en validación no
hay ventaja neta de costes, el sistema se queda fuera ese mes.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import backtest
from .config import MARKETS, MODEL_DIR, Params
from .features import FEATURE_WARMUP, build_features
from .labels import Outcomes, barrier_width, simulate


def prepare(df: pd.DataFrame, p: Params) -> tuple[pd.DataFrame, Outcomes]:
    return build_features(df), simulate(df, p)


def make_signal(pred: np.ndarray, width: np.ndarray, thr: float, cost: float, p: Params) -> np.ndarray:
    ok = (np.abs(pred) >= thr) & (width >= p.min_width_cost * cost)
    return np.where(ok, np.sign(pred), 0.0)


def pick_threshold(index, pred, out: Outcomes, lo: int, hi: int, cost: float, p: Params) -> tuple[float, dict]:
    """Elige el umbral de |predicción| en validación; inf si no hay ventaja neta."""
    a = np.abs(pred[lo:hi])
    a = a[np.isfinite(a)]
    best, best_stats = np.inf, {}
    for q in p.quantiles:
        thr = float(np.quantile(a, 1 - q))
        st = backtest.stats(backtest.run(index, make_signal(pred, out.width, thr, cost, p), out, cost, lo, hi))
        if st["trades"] >= p.min_val_trades and st["net_avg"] > 0 and st["tstat"] >= p.min_val_tstat:
            if not best_stats or st["tstat"] > best_stats["tstat"]:
                best, best_stats = thr, dict(st, quantile=q)
    return best, best_stats


def fit(X: pd.DataFrame, y: np.ndarray, lo: int, hi: int, p: Params) -> tuple[lgb.LGBMRegressor, int, int]:
    """Ajusta en [lo, hi): el primer (1 - val_frac) para entrenar y la cola para validar.

    Entre ambos tramos se purgan `horizon` barras para que las etiquetas no se solapen.
    Devuelve el modelo y los límites [val_lo, val_hi) de la validación.
    """
    hi_eff = hi - (p.horizon + 1)  # purga frente a lo que venga después
    cut = lo + int((hi_eff - lo) * (1 - p.val_frac))
    val_lo = cut + p.horizon + 1
    tr = np.arange(lo, cut)
    va = np.arange(val_lo, hi_eff)
    tr, va = tr[np.isfinite(y[tr])], va[np.isfinite(y[va])]
    model = lgb.LGBMRegressor(**p.lgbm)
    model.fit(
        X.iloc[tr], y[tr],
        eval_X=X.iloc[va], eval_y=y[va],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model, val_lo, hi_eff


def walk_forward(df: pd.DataFrame, cost: float, p: Params, start: str | None = None, end: str | None = None,
                 verbose: bool = True) -> dict:
    """Reentrena cada mes con los `train_months` anteriores y opera el mes siguiente.

    `start`/`end` (YYYY-MM) acotan los meses de test: [start, end).
    """
    X, out = prepare(df, p)
    idx = df.index
    months = idx.tz_localize(None).to_period("M")
    first = pd.Period(start or p.first_test, "M")
    last = pd.Period(end, "M") if end else months.max() + 1
    all_trades, rows = [], []
    pred_all = np.full(len(df), np.nan)
    for m in pd.period_range(first, last - 1, freq="M"):
        test = np.flatnonzero(months == m)
        if len(test) == 0:
            continue
        t_lo, t_hi = test[0], test[-1] + 1
        tr_lo = max(int(np.searchsorted(months, m - p.train_months)), FEATURE_WARMUP)
        if t_lo - tr_lo < 50_000:
            continue
        model, v_lo, v_hi = fit(X, out.target, tr_lo, t_lo, p)
        pred = np.full(len(df), np.nan)
        pred[v_lo:v_hi] = model.predict(X.iloc[v_lo:v_hi])
        pred[t_lo:t_hi] = model.predict(X.iloc[t_lo:t_hi])
        thr, vst = pick_threshold(idx, pred, out, v_lo, v_hi, cost, p)
        trades = backtest.run(idx, make_signal(pred, out.width, thr, cost, p), out, cost, t_lo, t_hi)
        all_trades.append(trades)
        pred_all[t_lo:t_hi] = pred[t_lo:t_hi]
        y, ph = out.target[t_lo:t_hi], pred[t_lo:t_hi]
        ok = np.isfinite(y) & np.isfinite(ph)
        ic = float(pd.Series(ph[ok]).corr(pd.Series(y[ok]), method="spearman")) if ok.sum() > 100 else np.nan
        st = backtest.stats(trades)
        rows.append(dict(month=str(m), trees=model.best_iteration_, ic=ic, thr=thr, val_q=vst.get("quantile"),
                         val_tstat=vst.get("tstat"), trades=st["trades"], net=st["net_total"],
                         gross=float(trades["gross"].sum()) if len(trades) else 0.0))
        if verbose:
            print(f"  {m}  árboles={model.best_iteration_:4d}  IC={ic:+.4f}  "
                  f"umbral={'--' if not np.isfinite(thr) else f'{thr:.3f}'}  ops={st['trades']:4d}  neto={st['net_total']:+9.2f}")
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else backtest.run(idx, np.zeros(len(df)), out, cost)
    return dict(trades=trades, months=pd.DataFrame(rows), pred=pred_all, target=out.target, width=out.width, index=idx)


# ----------------------------------------------------------------------------- modelo final / en vivo
def fit_final(market: str, df: pd.DataFrame, cost: float, p: Params) -> tuple[lgb.Booster, dict]:
    """Entrena con los últimos `train_months` meses de `df` y elige el umbral en su cola."""
    X, out = prepare(df, p)
    months = df.index.tz_localize(None).to_period("M")
    lo = max(int(np.searchsorted(months, months.max() + 1 - p.train_months)), FEATURE_WARMUP)
    model, v_lo, v_hi = fit(X, out.target, lo, len(df), p)
    pred = np.full(len(df), np.nan)
    pred[v_lo:v_hi] = model.predict(X.iloc[v_lo:v_hi])
    thr, vst = pick_threshold(df.index, pred, out, v_lo, v_hi, cost, p)
    meta = dict(
        market=market, cost=cost, threshold=thr if np.isfinite(thr) else None,
        # umbral de referencia (top 2 %) por si la validación no encontró ventaja neta
        reference_threshold=float(np.nanquantile(np.abs(pred[v_lo:v_hi]), 0.98)),
        val_stats=vst, features=list(X.columns), trained_until=str(df.index[-1]),
        train_from=str(df.index[lo]), trees=int(model.best_iteration_),
        params={k: v for k, v in asdict(p).items() if k != "lgbm"},
    )
    return model.booster_, meta


def train_final(market: str, df: pd.DataFrame, cost: float, p: Params) -> dict:
    """Como fit_final, pero guarda modelo y metadatos en models/."""
    booster, meta = fit_final(market, df, cost, p)
    MODEL_DIR.mkdir(exist_ok=True)
    booster.save_model(MODEL_DIR / f"{market}.txt", num_iteration=meta["trees"])
    (MODEL_DIR / f"{market}.json").write_text(json.dumps(meta, indent=2))
    return meta


class Predictor:
    """Modelo entrenado + metadatos; convierte barras M1 en predicción y señal."""

    def __init__(self, booster: lgb.Booster, meta: dict):
        self.booster, self.meta = booster, meta
        self.market = meta["market"]
        self.params = Params(**{k: tuple(v) if isinstance(v, list) else v for k, v in meta["params"].items()})
        self.threshold = meta["threshold"]  # None => la validación no encontró ventaja neta

    @classmethod
    def load(cls, market: str) -> "Predictor":
        meta = json.loads((MODEL_DIR / f"{market}.json").read_text())
        return cls(lgb.Booster(model_file=str(MODEL_DIR / f"{market}.txt")), meta)

    def batch(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predicción y anchura de barrera para todas las barras de `df`."""
        X = build_features(df)[self.meta["features"]]
        return self.booster.predict(X, num_iteration=self.meta["trees"]), barrier_width(df, self.params)

    def last(self, df: pd.DataFrame) -> tuple[float, float]:
        """Predicción y anchura de barrera de la última barra (necesita >= ~1500 barras previas)."""
        X = build_features(df)[self.meta["features"]]
        pred = float(self.booster.predict(X.iloc[[-1]], num_iteration=self.meta["trees"])[0])
        return pred, float(barrier_width(df, self.params)[-1])

    def side(self, pred: float, width: float, cost: float, use_reference: bool = False) -> int:
        thr = self.threshold if self.threshold is not None else (self.meta["reference_threshold"] if use_reference else None)
        if thr is None or not np.isfinite(pred) or not np.isfinite(width):
            return 0
        if abs(pred) < thr or width < self.params.min_width_cost * cost:
            return 0
        return int(np.sign(pred))


def live_signal(market: str, df: pd.DataFrame, cost: float | None = None) -> dict:
    """Señal para la próxima barra a partir de las barras M1 más recientes."""
    pr = Predictor.load(market)
    cost = MARKETS[market].cost if cost is None else cost
    pred, width = pr.last(df)
    side = pr.side(pred, width, cost)
    last = float(df["close"].iloc[-1])
    ref = pr.meta["reference_threshold"]
    return dict(
        market=market, bar_time=str(df.index[-1]), close=last, prediction=pred, threshold=pr.threshold,
        reference_threshold=ref, side=side,
        action={1: "COMPRAR", -1: "VENDER", 0: "ESPERAR"}[side],
        take_profit=last + side * width if side else None,
        stop_loss=last - side * width if side else None,
        barrier=width, max_minutes=pr.params.horizon, expected_gross=pred * width, cost=cost,
        bias={1: "alcista", -1: "bajista", 0: "neutral"}[int(np.sign(pred)) if abs(pred) >= ref else 0],
    )
