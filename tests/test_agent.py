import dataclasses
import types

import numpy as np
import pandas as pd
import pytest

from scalper import backtest
from scalper.agent import Agent, RiskConfig, run_live, trading_day
from scalper.brokers import MT5Broker, Order, PaperBroker, round_lots
from scalper.config import MARKETS, PARAMS
from scalper.data import synthetic
from scalper.labels import simulate

NO_LIMITS = RiskConfig(max_daily_loss=1e9, max_trades_per_day=10**9, max_consecutive_losses=10**9,
                       max_drawdown=1e9, monitor_tstat=-1e9, fixed_units=1.0)
T0 = pd.Timestamp("2024-01-03 15:00", tz="UTC")


class StubPredictor:
    """Predictor mínimo: opera cuando |pred| >= umbral."""

    def __init__(self, threshold=0.5, horizon=10):
        self.threshold = threshold
        self.params = dataclasses.replace(PARAMS, horizon=horizon)
        self.meta = dict(trained_until="-", reference_threshold=0.5, features=[], trees=0)
        self.candidate_threshold = 0.5

    def side(self, pred, width, cost, use_reference=False):
        thr = self.threshold if self.threshold is not None else (0.5 if use_reference else None)
        return 0 if thr is None or not np.isfinite(pred) or abs(pred) < thr else int(np.sign(pred))


def make_agent(tmp_path=None, risk=None, equity=10_000.0, cost=0.02, predictor=None, market="XAUUSD"):
    ag = Agent(market, PaperBroker(equity, cost), predictor or StubPredictor(), risk or RiskConfig(), cost,
               state_dir=tmp_path, persist=tmp_path is not None)
    ag.mk = dataclasses.replace(ag.mk, break_ny=-10_000)  # sin cierre diario en los datos sintéticos
    return ag


def feed(agent, df, pred, width):
    for i, (t, b) in enumerate(df.iterrows()):
        agent.on_bar(t, b.open, b.high, b.low, b.close, pred[i], width[i])


def test_agent_reproduces_backtest_exactly():
    """El agente en simulación, barra a barra, debe dar las mismas operaciones que el backtest."""
    p = dataclasses.replace(PARAMS, horizon=10)
    df = synthetic(6000, seed=11)
    out = simulate(df, p)
    rng = np.random.default_rng(0)
    pred = rng.normal(0, 0.4, len(df))
    pred[:300] = 0.0
    signal = np.where(np.abs(pred) >= 0.5, np.sign(pred), 0.0)
    expected = backtest.run(df.index, signal, out, cost=0.02)

    agent = make_agent(risk=NO_LIMITS)
    feed(agent, df, pred, out.width)
    got = pd.DataFrame(agent.closed)
    n = len(got)
    assert n > 200 and n >= len(expected) - 1
    exp = expected.iloc[:n]
    assert (pd.DatetimeIndex(got["time"]) == pd.DatetimeIndex(exp["time"])).all()
    assert (got["side"].to_numpy() == exp["side"].to_numpy()).all()
    np.testing.assert_allclose(got["gross"], exp["gross"], atol=1e-9)
    np.testing.assert_allclose(got["net"], exp["net"], atol=1e-9)
    assert agent.broker.equity() == pytest.approx(10_000 + got["net"].sum())


def test_paper_broker_brackets_and_time_exit():
    b = PaperBroker(1000.0, cost=0.1)
    assert b.submit(Order(1, 2.0, 1.0, 3, T0))
    assert not b.submit(Order(1, 2.0, 1.0, 3, T0))  # una sola posición a la vez
    assert b.on_bar(T0, 100, 100.5, 99.5, 100) == []  # se ejecuta en 100; TP 101, SL 99
    (tr,) = b.on_bar(T0, 100, 101.5, 98.5, 100)  # toca ambas: cuenta el stop
    assert tr["reason"] == "sl" and tr["gross"] == -1.0 and tr["net_money"] == pytest.approx(-2.2)
    assert b.equity() == pytest.approx(997.8) and b.flat()
    b.submit(Order(-1, 1.0, 1.0, 2, T0))
    assert b.on_bar(T0, 50, 50.2, 49.8, 50.1) == []
    (tr,) = b.on_bar(T0, 50.1, 50.3, 49.9, 49.7)  # caduca al cierre de la 2ª barra
    assert tr["reason"] == "tiempo" and tr["gross"] == pytest.approx(0.3)


def test_position_sizing_respects_risk_and_lot_step():
    agent = make_agent(risk=RiskConfig(risk_per_trade=0.0025))
    units = agent.size(width=3.4)  # 25 $ de riesgo / 3,4 $ por onza = 7,35 oz -> 0,07 lotes
    assert units == pytest.approx(7.0)
    assert units * 3.4 <= 10_000 * 0.0025
    assert make_agent(equity=100.0).size(width=3.4) == 0.0  # no llega al lote mínimo: no opera
    assert round_lots(149.0, MARKETS["XAUUSD"]) == pytest.approx(1.49)


def test_daily_loss_limit_stops_trading_for_the_day():
    risk = RiskConfig(risk_per_trade=0.01, max_daily_loss=0.02, max_consecutive_losses=99, monitor_tstat=-1e9)
    agent = make_agent(risk=risk, predictor=StubPredictor(horizon=3))
    t, price = T0, 100.0
    for _ in range(400):  # mercado que siempre salta el stop de los largos
        agent.on_bar(t, price, price, price - 5, price - 5, pred=1.0, width=1.0)
        t += pd.Timedelta(minutes=1)
        price -= 5
        if price < 50:
            price = 100.0
    day = pd.DataFrame(agent.closed)
    assert 2 <= len(day) <= 4  # ~1 % por operación: se detiene al superar el 2 %
    assert agent.blocked(t) == "límite de pérdida diaria"
    assert agent.broker.equity() > 10_000 * 0.955
    # al día siguiente vuelve a poder operar
    nxt = t + pd.Timedelta(days=1)
    agent._roll_day(nxt)
    assert trading_day(nxt) != trading_day(t) and agent.blocked(nxt) is None


def test_pause_after_consecutive_losses_and_drawdown_halt():
    risk = RiskConfig(risk_per_trade=0.01, max_daily_loss=1e9, max_consecutive_losses=3, pause_minutes=30,
                      max_drawdown=0.05, monitor_tstat=-1e9)
    agent = make_agent(risk=risk, predictor=StubPredictor(horizon=3))
    t = T0
    paused = False
    for _ in range(3000):
        agent.on_bar(t, 100, 100, 95, 95, pred=1.0, width=1.0)
        paused |= agent.blocked(t) == "en pausa"
        t += pd.Timedelta(minutes=1)
        if agent.state.halted:
            break
    assert paused
    assert agent.state.halted and "caída máxima" in agent.state.halted
    assert agent.broker.equity() >= 10_000 * 0.93
    n = len(agent.closed)
    for _ in range(50):
        agent.on_bar(t, 100, 100, 95, 95, pred=1.0, width=1.0)
        t += pd.Timedelta(minutes=1)
    assert len(agent.closed) == n and agent.blocked(t) == "detenido"


def test_edge_monitor_suspends_until_retrain():
    risk = RiskConfig(max_daily_loss=1e9, max_consecutive_losses=10**9, max_drawdown=1e9, monitor_trades=20,
                      monitor_tstat=-2.0, fixed_units=1.0)
    agent = make_agent(risk=risk, predictor=StubPredictor(horizon=3))
    t = T0
    rng = np.random.default_rng(1)
    for _ in range(400):
        lo = 100 - (1.2 if rng.random() < 0.8 else 0.0)  # pierde el 80 % de las veces
        agent.on_bar(t, 100, 100 + (0 if lo < 100 else 1.2), lo, 100, pred=1.0, width=1.0)
        t += pd.Timedelta(minutes=1)
    assert agent.state.suspended and agent.blocked(t) == "suspendido por el monitor de ventaja"
    agent.set_predictor(StubPredictor(), t)
    assert not agent.state.suspended and agent.blocked(t) is None


def test_stays_out_without_validated_edge_and_near_session_close():
    agent = make_agent(predictor=StubPredictor(threshold=None))
    assert agent.blocked(T0) == "sin ventaja validada"
    agent = Agent("XAUUSD", PaperBroker(10_000, 0.3), StubPredictor(), persist=False)
    close = pd.Timestamp("2024-01-03 16:55", tz="America/New_York").tz_convert("UTC")
    assert agent.blocked(close) == "cierre de sesión próximo"
    assert agent.blocked(close - pd.Timedelta(minutes=30)) is None


def test_stop_file_closes_position_and_halts(tmp_path):
    agent = make_agent(tmp_path, predictor=StubPredictor(horizon=50))
    agent.on_bar(T0, 100, 100, 100, 100, pred=1.0, width=5.0)
    agent.on_bar(T0 + pd.Timedelta(minutes=1), 100, 100.5, 99.5, 100, pred=1.0, width=5.0)
    assert not agent.broker.flat()
    (tmp_path / "STOP").write_text("")
    assert agent.on_bar(T0 + pd.Timedelta(minutes=2), 100, 100.5, 99.5, 100.2, pred=1.0, width=5.0) == "parada manual"
    assert agent.broker.flat() and agent.closed[-1]["reason"] == "parada manual" and agent.state.halted


def test_state_survives_restart(tmp_path):
    a1 = make_agent(tmp_path, predictor=StubPredictor(horizon=50))
    a1.on_bar(T0, 100, 100, 100, 100, pred=1.0, width=5.0)
    a1.on_bar(T0 + pd.Timedelta(minutes=1), 100, 100.5, 99.5, 100, pred=1.0, width=5.0)
    a2 = make_agent(tmp_path, predictor=StubPredictor(horizon=50))  # "reinicio"
    assert a2.broker.position is not None and a2.broker.position.entry == 100
    assert a2.on_bar(T0 + pd.Timedelta(minutes=1), 100, 100.5, 99.5, 100, pred=1.0, width=5.0) == "barra ya procesada"
    a2.on_bar(T0 + pd.Timedelta(minutes=2), 100, 106, 99.5, 105, pred=0.0, width=5.0)
    assert a2.closed[-1]["reason"] == "tp" and a2.state.trades == 1
    assert (tmp_path / "XAUUSD_paper_trades.csv").read_text().count("\n") == 2


def test_live_loop_survives_feed_errors():
    agent = make_agent(risk=NO_LIMITS)
    df = synthetic(700, seed=2)
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConnectionError("sin red")
        return df.iloc[: 650 + calls["n"]]

    agent.predictor.last = lambda bars: (0.0, 1.0)
    run_live(agent, fetch, retrainer=None, max_steps=4, sleep=lambda s: None)
    assert calls["n"] == 4 and agent.state.last_bar == str(df.index[653])


# ------------------------------------------------------------------ MT5 con un doble de pruebas
class FakeMT5:
    ACCOUNT_TRADE_MODE_DEMO, TRADE_ACTION_DEAL, ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1, 0, 1
    ORDER_TIME_GTC, ORDER_FILLING_IOC, TRADE_RETCODE_DONE, POSITION_TYPE_BUY, DEAL_ENTRY_OUT, TIMEFRAME_M1 = 0, 1, 10009, 0, 1, 1

    def __init__(self, demo=True):
        self.demo, self.sent, self.positions, self.deals = demo, [], [], []
        self.bid, self.ask = 4000.00, 4000.20

    def initialize(self): return True
    def last_error(self): return (0, "ok")
    def account_info(self): return types.SimpleNamespace(trade_mode=0 if self.demo else 2, equity=5000.0)
    def symbol_select(self, s, on): return True
    def symbol_info(self, s): return types.SimpleNamespace(trade_contract_size=100.0, volume_step=0.01, volume_min=0.01, digits=2)
    def symbol_info_tick(self, s):
        import time
        return types.SimpleNamespace(bid=self.bid, ask=self.ask, time=time.time() + 7200)
    def positions_get(self, symbol=None): return list(self.positions)
    def history_deals_get(self, position=None): return list(self.deals)

    def order_send(self, req):
        self.sent.append(req)
        if "position" in req:
            self.positions.clear()
            self.deals = [types.SimpleNamespace(entry=1, price=req["price"], profit=12.0, commission=-0.5, swap=0.0)]
        else:
            self.positions.append(types.SimpleNamespace(magic=req["magic"], type=req["type"], volume=req["volume"],
                                                        ticket=77, price_open=req["price"], tp=req["tp"], sl=req["sl"], time=0))
        return types.SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, price=req["price"], order=77, comment="")


def test_mt5_refuses_real_account_by_default():
    with pytest.raises(RuntimeError, match="no es demo"):
        MT5Broker(MARKETS["XAUUSD"], mt5=FakeMT5(demo=False))
    MT5Broker(MARKETS["XAUUSD"], mt5=FakeMT5(demo=False), allow_real=True)


def test_mt5_order_flow():
    fake = FakeMT5()
    b = MT5Broker(MARKETS["XAUUSD"], mt5=fake)
    assert b.flat() and b.equity() == 5000.0 and b.spread() == pytest.approx(0.20)
    assert b._server_offset() == 7200
    assert b.submit(Order(1, 7.35, 3.0, 2, T0))  # 7,35 oz -> 0,07 lotes
    req = fake.sent[0]
    assert req["volume"] == pytest.approx(0.07) and req["price"] == 4000.20
    assert req["tp"] == pytest.approx(4003.20) and req["sl"] == pytest.approx(3997.20) and req["magic"] == MT5Broker.MAGIC
    assert not b.flat() and not b.submit(Order(1, 7.35, 3.0, 2, T0))
    assert b.on_bar(T0, 4000, 4001, 3999, 4000.5) == []
    fake.bid = 4001.0
    (tr,) = b.on_bar(T0, 4000.5, 4001.5, 4000, 4001)  # caducidad: cierra a mercado
    assert fake.sent[-1]["position"] == 77 and fake.sent[-1]["type"] == fake.ORDER_TYPE_SELL
    assert tr["reason"] == "tiempo" and tr["net_money"] == pytest.approx(11.5) and b.flat()
    # cierre hecho por el bróker (TP/SL en el servidor)
    b.submit(Order(-1, 10.0, 3.0, 5, T0))
    fake.positions.clear()
    fake.deals = [types.SimpleNamespace(entry=1, price=3997.0, profit=30.0, commission=-0.7, swap=0.0)]
    (tr,) = b.on_bar(T0, 4000, 4000, 3996, 3997)
    assert tr["reason"] == "tp" and tr["net_money"] == pytest.approx(29.3)
