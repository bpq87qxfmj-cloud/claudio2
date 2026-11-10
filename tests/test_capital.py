import types

import pandas as pd
import pytest

from scalper.agent import Agent, RiskConfig
from scalper.brokers import Order
from scalper.capital import DEMO_URL, LIVE_URL, CapitalBroker, CapitalClient, CapitalError
from scalper.config import MARKETS

from test_agent import StubPredictor

NOW = pd.Timestamp("2026-11-10 15:00:30", tz="UTC")


class FakeCapital:
    """Imita la API documentada de Capital.com (https://open-api.capital.com)."""

    def __init__(self):
        self.logins, self.calls, self.valid = 0, [], set()
        self.balance, self.bid, self.spread = 10_000.0, 4000.00, 0.30
        self.positions, self.posted, self.deleted, self.refs = [], [], [], {}
        self.status = "TRADEABLE"
        self.first_bar = pd.Timestamp("2026-11-08 23:05", tz="UTC")  # reapertura tras el fin de semana

    def _resp(self, status, body, headers=None):
        return types.SimpleNamespace(status_code=status, content=b"x", headers=headers or {}, json=lambda: body)

    def expire(self):
        self.valid.clear()

    def request(self, method, url, headers=None, timeout=None, params=None, json=None):
        base = DEMO_URL + "/api/v1"
        assert url.startswith(base), url
        path = url[len(base):]
        self.calls.append((method, path))
        if (method, path) == ("POST", "/session"):
            if headers.get("X-CAP-API-KEY") != "KEY" or json != {"identifier": "yo@example.com", "password": "pw", "encryptedPassword": False}:
                return self._resp(401, {"errorCode": "error.invalid.details"})
            self.logins += 1
            tok = f"cst{self.logins}"
            self.valid.add(tok)
            return self._resp(200, {"currentAccountId": "1"}, {"CST": tok, "X-SECURITY-TOKEN": "sec"})
        if "CST" not in headers:
            return self._resp(400, {"errorCode": "error.null.client.token"})
        if headers["CST"] not in self.valid:
            return self._resp(401, {"errorCode": "error.invalid.session.token"})
        if (method, path) == ("GET", "/accounts"):
            return self._resp(200, {"accounts": [{"accountId": "1", "preferred": True, "balance": {"balance": self.balance}}]})
        if (method, path) == ("GET", "/markets/GOLD"):
            return self._resp(200, {
                "instrument": {"epic": "GOLD", "name": "Gold"},
                "dealingRules": {"minDealSize": {"unit": "POINTS", "value": 0.01}, "minSizeIncrement": {"unit": "POINTS", "value": 0.01},
                                 "minStopOrProfitDistance": {"unit": "PERCENTAGE", "value": 0.01}},
                "snapshot": {"marketStatus": self.status, "bid": self.bid, "offer": self.bid + self.spread, "decimalPlacesFactor": 2}})
        if (method, path) == ("GET", "/prices/GOLD"):
            lo, hi = (pd.Timestamp(params[k], tz="UTC") for k in ("from", "to"))
            assert params["resolution"] == "MINUTE" and (hi - lo) <= pd.Timedelta(minutes=1000)
            times = pd.date_range(max(lo.ceil("min"), self.first_bar), min(hi, NOW), freq="1min")
            if len(times) == 0:
                return self._resp(404, {"errorCode": "error.prices.not-found"})
            px = lambda t, d=0.0: {"bid": 4000 + t.minute * 0.01 + d, "ask": 4000 + t.minute * 0.01 + d + self.spread}
            return self._resp(200, {"prices": [{"snapshotTimeUTC": t.strftime("%Y-%m-%dT%H:%M:%S"), "openPrice": px(t),
                                                "highPrice": px(t, 0.5), "lowPrice": px(t, -0.5), "closePrice": px(t, 0.1)} for t in times]})
        if (method, path) == ("GET", "/positions"):
            return self._resp(200, {"positions": [{"position": p} for p in self.positions]})
        if (method, path) == ("POST", "/positions"):
            self.posted.append(json)
            level = self.bid + self.spread if json["direction"] == "BUY" else self.bid
            self.positions.append({"dealId": "D1", "direction": json["direction"], "size": json["size"], "level": level})
            self.refs["o_1"] = {"dealStatus": "ACCEPTED", "status": "OPEN", "level": level, "affectedDeals": [{"dealId": "D1", "status": "OPENED"}]}
            return self._resp(200, {"dealReference": "o_1"})
        if method == "DELETE" and path.startswith("/positions/"):
            self.deleted.append(path.rsplit("/", 1)[1])
            pos = self.positions.pop()
            level = self.bid if pos["direction"] == "BUY" else self.bid + self.spread
            self.balance += (level - pos["level"]) * pos["size"] * (1 if pos["direction"] == "BUY" else -1)
            self.refs["p_1"] = {"dealStatus": "ACCEPTED", "status": "CLOSED", "level": level}
            return self._resp(200, {"dealReference": "p_1"})
        if method == "GET" and path.startswith("/confirms/"):
            return self._resp(200, self.refs[path.rsplit("/", 1)[1]])
        return self._resp(404, {"errorCode": "error.not-found"})


@pytest.fixture
def fake():
    return FakeCapital()


def make(fake, now=NOW):
    client = CapitalClient("KEY", "yo@example.com", "pw", http=fake, sleep=lambda s: None)
    return CapitalBroker(MARKETS["XAUUSD"], client, now=lambda: now), client


def test_demo_is_the_default_server():
    assert CapitalClient("k", "i", "p").base.startswith(DEMO_URL)
    assert CapitalClient("k", "i", "p", demo=False).base.startswith(LIVE_URL)


def test_login_errors_and_session_renewal(fake):
    bad = CapitalClient("MALA", "yo@example.com", "pw", http=fake, sleep=lambda s: None)
    with pytest.raises(CapitalError, match="error.invalid.details"):
        bad.balance()
    _, client = make(fake)
    assert client.balance() == 10_000.0 and fake.logins == 1
    fake.expire()  # la sesión caduca tras 10 min sin uso
    assert client.balance() == 10_000.0 and fake.logins == 2


def test_bars_paginate_use_bid_and_skip_forming_bar(fake):
    broker, _ = make(fake)
    df = broker.bars(2000)
    assert len(df) == 2000 and df.index.is_monotonic_increasing and df.index.is_unique
    assert df.index[-1] == pd.Timestamp("2026-11-10 14:59", tz="UTC")  # la de las 15:00 está en formación
    assert df["open"].iloc[-1] == pytest.approx(4000.59)  # precio bid, no ask
    assert broker.spread() == pytest.approx(0.30)
    n_calls = sum(1 for c in fake.calls if c[1] == "/prices/GOLD")
    broker._now = lambda: NOW + pd.Timedelta(minutes=2)
    df2 = broker.bars(2000)  # la segunda vez solo pide lo nuevo
    assert sum(1 for c in fake.calls if c[1] == "/prices/GOLD") == n_calls + 1
    assert df2.index[-1] == pd.Timestamp("2026-11-10 15:00", tz="UTC") and df2.index.is_unique


def test_bars_cross_weekend_gap(fake):
    broker, _ = make(fake)
    df = broker.bars(2500)  # solo hay ~2400 barras desde la reapertura: devuelve las que existen
    assert df.index[0] == fake.first_bar and 2300 < len(df) < 2500


def test_long_order_levels_size_and_time_exit(fake):
    broker, _ = make(fake)
    assert broker.submit(Order(1, 7.359, 3.0, 2, NOW))
    req = fake.posted[0]
    assert req == {"epic": "GOLD", "direction": "BUY", "size": 7.35, "stopLevel": 3997.0, "profitLevel": 4003.0}
    assert broker.position.entry == pytest.approx(4000.30) and not broker.flat()
    assert not broker.submit(Order(1, 7.0, 3.0, 2, NOW))  # una sola posición a la vez
    assert broker.on_bar(NOW, 4000, 4001, 3999, 4000.5) == []
    fake.bid = 4001.50
    (tr,) = broker.on_bar(NOW, 4000.5, 4002, 4000, 4001.5)  # caduca: cierre a mercado
    assert fake.deleted == ["D1"] and tr["reason"] == "tiempo"
    assert tr["exit"] == pytest.approx(4001.50) and tr["gross"] == pytest.approx(1.20)
    assert tr["net_money"] == pytest.approx(1.20 * 7.35) and broker.flat()


def test_short_order_levels_account_for_spread(fake):
    broker, _ = make(fake)
    assert broker.submit(Order(-1, 5.0, 3.0, 5, NOW))
    req = fake.posted[0]
    # un corto se cierra comprando en ask: las barreras en bid se desplazan el spread
    assert req["direction"] == "SELL" and req["profitLevel"] == pytest.approx(3997.30) and req["stopLevel"] == pytest.approx(4003.30)
    fake.positions.clear()  # el bróker ejecuta el TP
    fake.balance += (3.0 - 0.30) * 5.0
    (tr,) = broker.on_bar(NOW, 4000, 4000, 3996, 3997)
    assert tr["reason"] == "tp" and tr["net_money"] == pytest.approx(13.5) and fake.deleted == []


def test_order_refused_when_too_small_too_tight_or_market_closed(fake):
    broker, _ = make(fake)
    assert not broker.submit(Order(1, 0.004, 3.0, 2, NOW))  # menos que el tamaño mínimo
    assert not broker.submit(Order(1, 5.0, 0.2, 2, NOW))  # stop más cerca que el 0,01 % mínimo (0,40)
    fake.status = "CLOSED"
    assert not broker.submit(Order(1, 5.0, 3.0, 2, NOW))
    assert fake.posted == [] and broker.flat()


def test_agent_with_capital_broker_restores_open_position(fake, tmp_path):
    broker, _ = make(fake)
    agent = Agent("XAUUSD", broker, StubPredictor(horizon=3), RiskConfig(max_spread_cost=1.25), 0.30, state_dir=tmp_path)
    broker.bars(600)
    t = pd.Timestamp("2026-11-10 14:00", tz="UTC")
    assert agent.on_bar(t, 4000, 4001, 3999, 4000, pred=1.0, width=3.0, now=t) == "COMPRA"
    assert (tmp_path / "XAUUSD_capital_state.json").exists()
    broker2, _ = make(fake)  # reinicio del proceso
    agent2 = Agent("XAUUSD", broker2, StubPredictor(horizon=3), RiskConfig(max_spread_cost=1.25), 0.30, state_dir=tmp_path)
    assert broker2.position is not None and broker2.position.ticket == "D1"
    fake.positions.clear()  # salta el stop en el servidor
    fake.balance -= 3.3 * 6.25
    agent2.on_bar(t + pd.Timedelta(minutes=1), 4000, 4000, 3996, 3997, pred=0.0, width=3.0)
    assert agent2.closed[-1]["reason"] == "sl" and agent2.closed[-1]["net_money"] == pytest.approx(-20.625)
    assert agent2.state.day_pnl == pytest.approx(-20.625)


def test_spread_above_assumed_cost_blocks_entry(fake):
    fake.spread = 0.60
    broker, _ = make(fake)
    broker.bars(600)
    agent = Agent("XAUUSD", broker, StubPredictor(), RiskConfig(max_spread_cost=1.25), 0.30, persist=False)
    t = pd.Timestamp("2026-11-10 14:00", tz="UTC")
    assert agent.on_bar(t, 4000, 4001, 3999, 4000, pred=1.0, width=3.0) == "spread demasiado alto"
    assert fake.posted == []
