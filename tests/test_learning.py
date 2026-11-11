import dataclasses

import numpy as np
import pandas as pd
import pytest

from scalper.agent import Agent, RiskConfig
from scalper.brokers import PaperBroker
from scalper.config import PARAMS
from scalper.data import synthetic
from scalper.labels import simulate
from scalper.learning import CONTEXT, FROM_FEATURES, LearnConfig, Learner

from test_agent import NO_LIMITS, StubPredictor

FEATS = np.zeros(len(FROM_FEATURES))


def at_hour(hour: int, day: str = "2024-01-03") -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hour:02d}:10", tz="America/New_York").tz_convert("UTC")


def teach(learner: Learner, bad_hours=(20, 21), n=6000, seed=0) -> None:
    """Experiencia sintética: se pierde de forma sistemática a ciertas horas y se gana en el resto."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        hour = int(rng.integers(0, 24))
        x = Learner.context(at_hour(hour), int(rng.choice([-1, 1])), 0.8, 1.0, learner.cost, 0.5, rng.normal(size=len(FROM_FEATURES)))
        learner.X.append(x)
        learner.y.append((-0.5 if hour in bad_hours else 0.2) + rng.normal(0, 0.8))
    assert learner.refit()


def test_shadow_trades_match_backtest_outcomes():
    """Lo que el agente 'recuerda' de cada señal candidata es exactamente lo que dice el backtest."""
    p = dataclasses.replace(PARAMS, horizon=10)
    df = synthetic(4000, seed=4)
    out = simulate(df, p)
    cost = 0.02
    rng = np.random.default_rng(1)
    cand = np.flatnonzero((rng.random(len(df)) < 0.05) & (np.arange(len(df)) > 300))
    sides = rng.choice([-1, 1], len(df))
    learner = Learner(cost, p.horizon, LearnConfig(min_experiences=10**9))
    for i, (t, b) in enumerate(df.iterrows()):
        learner.on_bar(t, b.open, b.high, b.low, b.close)
        if i in set(cand):
            feats = FEATS.copy()
            feats[0] = i  # marca para identificar la experiencia
            learner.observe(t, int(sides[i]), 1.0 * sides[i], out.width[i], 0.5, feats)
    assert len(learner.y) > 150
    for x, net_r in zip(learner.X, learner.y):
        i = int(x[CONTEXT.index(FROM_FEATURES[0])])
        gross = out.long_pnl[i] if sides[i] > 0 else out.short_pnl[i]
        assert net_r == pytest.approx((gross - cost) / out.width[i], abs=1e-9)


def test_critic_learns_where_it_loses():
    learner = Learner(0.02, 10, LearnConfig())
    assert learner.score(Learner.context(at_hour(21), 1, 0.8, 1.0, 0.02, 0.5, FEATS)) is None  # sin experiencia no opina
    teach(learner)
    bad = learner.score(Learner.context(at_hour(21), 1, 0.8, 1.0, 0.02, 0.5, FEATS))
    good = learner.score(Learner.context(at_hour(10), 1, 0.8, 1.0, 0.02, 0.5, FEATS))
    assert bad < -0.2 and good > 0.05
    assert learner.lessons(1)[0][0] == "hour"


def make(learn, predictor=None, tmp_path=None):
    ag = Agent("XAUUSD", PaperBroker(10_000.0, 0.02), predictor or StubPredictor(), NO_LIMITS, 0.02,
               state_dir=tmp_path, persist=tmp_path is not None, learn=learn)
    ag.mk = dataclasses.replace(ag.mk, break_ny=-10_000)
    return ag


def test_agent_vetoes_trades_that_resemble_past_losses():
    agent = make(LearnConfig())
    teach(agent.learner)
    t = at_hour(21)
    assert agent.on_bar(t, 100, 100, 100, 100, pred=1.0, width=1.0, ctx=FEATS) == "vetada por el crítico"
    assert agent.broker.flat() and agent.learner.vetoes == 1
    t = at_hour(10, "2024-01-04")
    assert agent.on_bar(t, 100, 100, 100, 100, pred=1.0, width=1.0, ctx=FEATS) == "COMPRA"
    # sin aprendizaje, la misma señal de las 21 h se habría operado
    plain = make(None)
    assert plain.on_bar(at_hour(21), 100, 100, 100, 100, pred=1.0, width=1.0) == "COMPRA"


def test_learned_entries_need_a_profitable_track_record():
    cfg = LearnConfig(learn_entries=True, track_trades=50)
    agent = make(cfg, predictor=StubPredictor(threshold=None))  # el modelo base no opera
    teach(agent.learner)
    t = at_hour(10)
    assert agent.on_bar(t, 100, 100, 100, 100, pred=1.0, width=1.0, ctx=FEATS) == "sin ventaja validada"
    agent.learner.approved.extend([0.5, -0.3] * 25)  # las aprobadas recientes han ganado
    assert agent.learner.track_ok()
    t += pd.Timedelta(minutes=1)
    assert agent.on_bar(t, 100, 100, 100, 100, pred=1.0, width=1.0, ctx=FEATS) == "COMPRA"
    assert agent.learner.learned_entries == 1
    agent2 = make(cfg, predictor=StubPredictor(threshold=None))
    teach(agent2.learner)
    agent2.learner.approved.extend([-0.5, 0.3] * 25)  # han perdido: no se fía
    assert agent2.on_bar(at_hour(10), 100, 100, 100, 100, pred=1.0, width=1.0, ctx=FEATS) == "sin ventaja validada"
    # y el veto sigue mandando sobre las entradas aprendidas
    agent.broker.close_all(t, 100, "test")
    assert agent.on_bar(at_hour(21, "2024-01-04"), 100, 100, 100, 100, pred=1.0, width=1.0, ctx=FEATS) == "sin ventaja validada"


def test_experience_survives_restart(tmp_path):
    cfg = LearnConfig(min_experiences=40)
    a1 = make(cfg, predictor=StubPredictor(horizon=3), tmp_path=tmp_path)
    t = at_hour(10)
    for k in range(200):  # mercado que siempre salta el stop de los largos
        a1.on_bar(t + pd.Timedelta(minutes=k), 100, 100, 95, 95, pred=1.0, width=1.0, ctx=FEATS)
    n = len(a1.learner.y)
    assert n >= 100 and (tmp_path / "XAUUSD_paper_experience.csv").exists()
    a2 = make(cfg, predictor=StubPredictor(horizon=3), tmp_path=tmp_path)
    assert len(a2.learner.y) == n and a2.learner.critic is not None
    assert a2.learner.score(Learner.context(t, 1, 1.0, 1.0, 0.02, 0.5, FEATS)) < 0


def test_shadows_dropped_across_data_gap():
    learner = Learner(0.02, 10, LearnConfig())
    t = at_hour(10)
    learner.on_bar(t, 100, 100, 100, 100)
    learner.observe(t, 1, 1.0, 1.0, 0.5, FEATS)
    learner.on_bar(t + pd.Timedelta(hours=2), 100, 105, 100, 105)  # hueco: no cuenta como experiencia
    assert learner.y == [] and learner._shadows == []
