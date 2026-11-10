import dataclasses

import numpy as np
import pandas as pd
import pytest

from scalper import backtest, model
from scalper.config import PARAMS
from scalper.data import synthetic
from scalper.features import build_features, session_id
from scalper.labels import simulate


def test_features_are_causal():
    """Cambiar el futuro no puede alterar las variables del pasado."""
    df = synthetic(3000, seed=1)
    cut = 2500
    df2 = df.copy()
    df2.iloc[cut:] = df2.iloc[cut:] * 1.05
    a, b = build_features(df).iloc[:cut], build_features(df2).iloc[:cut]
    pd.testing.assert_frame_equal(a, b)


def test_features_finite_after_warmup():
    X = build_features(synthetic(2000))
    X = X.drop(columns="prev_close_dist")  # sin sesión anterior en los datos sintéticos: NaN legítimo
    assert np.isfinite(X.iloc[500:].to_numpy()).all()


def test_session_id_handles_any_time_unit():
    idx = pd.DatetimeIndex(["2024-01-02 10:00", "2024-01-02 10:01", "2024-01-02 12:00", "2024-01-02 12:01"], tz="UTC")
    for unit in ("s", "us", "ns"):
        assert session_id(idx.as_unit(unit)).tolist() == [0, 0, 1, 1]


def _bars(rows, start="2024-01-02 10:00"):
    idx = pd.date_range(start, periods=len(rows), freq="1min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def test_triple_barrier_outcomes(monkeypatch):
    p = dataclasses.replace(PARAMS, horizon=3)
    monkeypatch.setattr("scalper.labels.barrier_width", lambda df, p: np.full(len(df), 1.0))
    df = _bars([
        (100, 100, 100, 100),      # señal en t=0, entrada en la apertura de t=1 (100)
        (100, 100.5, 99.8, 100.2),
        (100.2, 101.2, 100.1, 101),  # toca TP del largo (101) / SL del corto
        (101, 101, 97, 98),
        (98, 102, 96, 100),        # barra que toca ambas barreras para la señal t=3 (entrada 98)
        (100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100),
    ])
    out = simulate(df, p)
    assert out.long_pnl[0] == 1.0 and out.long_exit[0] == 2
    assert out.short_pnl[0] == -1.0 and out.short_exit[0] == 2
    assert out.target[0] == 1.0
    # ambas barreras en la misma barra: se asume stop para los dos lados y objetivo neutro
    assert out.long_pnl[3] == -1.0 and out.short_pnl[3] == -1.0 and out.target[3] == 0.0
    # sin barrera tocada: salida por tiempo al cierre de t+3
    assert out.long_pnl[4] == 0.0 and out.long_exit[4] == 3
    # las últimas barras no tienen ventana completa
    assert np.isnan(out.long_pnl[-1])


def test_no_trade_across_session_gap():
    p = dataclasses.replace(PARAMS, horizon=5, vol_span=4)
    df = synthetic(200)
    df.index = df.index[:100].append(df.index[100:] + pd.Timedelta(hours=2))
    out = simulate(df, p)
    assert np.isnan(out.long_pnl[95:100]).all()
    assert np.isfinite(out.long_pnl[60:94]).all()


def test_backtest_non_overlapping_and_costs():
    p = dataclasses.replace(PARAMS, horizon=5, vol_span=10)
    df = synthetic(1000, seed=3)
    out = simulate(df, p)
    sig = np.ones(len(df))
    tr = backtest.run(df.index, sig, out, cost=0.01)
    assert len(tr) > 50
    pos = df.index.get_indexer(pd.DatetimeIndex(tr["time"]))
    assert (np.diff(pos) > tr["bars"].to_numpy()[:-1]).all()  # nunca dos posiciones a la vez
    np.testing.assert_allclose(tr["gross"] - tr["net"], 0.01)
    assert backtest.stats(tr, cost=0.02)["net_avg"] == pytest.approx(tr["gross"].mean() - 0.02)


def test_random_walk_has_no_edge():
    """En un paseo aleatorio el sistema no debe 'descubrir' una ventaja rentable."""
    df = synthetic(150_000, seed=5)
    lg = dict(PARAMS.lgbm, n_estimators=60, n_jobs=2)
    p = dataclasses.replace(PARAMS, train_months=2, first_test="2024-03", lgbm=lg)
    res = model.walk_forward(df, cost=0.02, p=p, verbose=False)
    st = backtest.stats(res["trades"])
    ok = np.isfinite(res["pred"]) & np.isfinite(res["target"])
    ic = pd.Series(res["pred"][ok]).corr(pd.Series(res["target"][ok]), method="spearman")
    assert abs(ic) < 0.03
    assert st["trades"] == 0 or st["net_avg"] < 0.05 * np.nanmedian(res["width"])
