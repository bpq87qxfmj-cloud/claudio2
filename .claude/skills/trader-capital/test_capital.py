"""Pruebas sin red de capital.py, con una API de Capital.com simulada.

    python3 -m unittest .claude/skills/trader-capital/test_capital.py -v
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
SKILLS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import capital  # noqa: E402

MARKETS = {
    "AAPL": {"instrument": {"epic": "AAPL", "name": "Apple Inc", "type": "SHARES", "currency": "USD",
                            "marginFactor": 20, "marginFactorUnit": "PERCENTAGE"},
             "dealingRules": {"minDealSize": {"value": 0.1}, "minSizeIncrement": {"value": 0.01}},
             "snapshot": {"bid": 340.9, "offer": 341.3, "marketStatus": "TRADEABLE"}},
    "US500": {"instrument": {"epic": "US500", "name": "US 500", "type": "INDICES", "currency": "USD"},
              "dealingRules": {"minDealSize": {"value": 0.01}, "minSizeIncrement": {"value": 0.01}},
              "snapshot": {"bid": 7568.6, "offer": 7569.2, "marketStatus": "TRADEABLE"}},
}


class Resp:
    def __init__(self, body, headers=None):
        self._b = json.dumps(body).encode()
        self.headers = headers or {}

    def read(self):
        return self._b


class FakeCapital:
    def __init__(self):
        self.calls = []

    def __call__(self, req, timeout=None):
        path = req.full_url.split(".com", 1)[1].split("?")[0]
        body = json.loads(req.data) if req.data else None
        self.calls.append((req.get_method(), path, body))
        if path == "/api/v1/session":
            return Resp({}, {"CST": "c", "X-SECURITY-TOKEN": "t"})
        if path.startswith("/api/v1/markets/"):
            return Resp(MARKETS[path.rsplit("/", 1)[1]])
        if path == "/api/v1/markets":
            return Resp({"markets": [{"epic": "AAPL", "instrumentName": "Apple", "bid": 1, "offer": 1.1}]})
        if path == "/api/v1/positions" and req.get_method() == "POST":
            return Resp({"dealReference": "o_123"})
        if path.startswith("/api/v1/confirms/"):
            return Resp({"dealStatus": "ACCEPTED", "level": 341.3, "affectedDeals": [{"dealId": "D1"}]})
        if path.startswith("/api/v1/prices/"):
            p = lambda v: {"bid": v, "ask": v + 0.6}
            return Resp({"prices": [{"snapshotTimeUTC": "2026-11-03T14:30:00", "openPrice": p(7560),
                                     "highPrice": p(7565), "lowPrice": p(7558), "closePrice": p(7563),
                                     "lastTradedVolume": 900}]})
        raise AssertionError("ruta inesperada " + path)


class CapitalTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.dir)
        os.environ.update(CAPITAL_API_KEY="k", CAPITAL_IDENTIFIER="i", CAPITAL_PASSWORD="p")
        os.environ.pop("CAPITAL_LIVE", None)
        self.fake = FakeCapital()
        self._orig = capital.Capital.from_env.__func__
        fake = self.fake
        capital.Capital.from_env = classmethod(lambda cls, opener=None: self._orig(cls, opener=fake))
        os.makedirs("trader")
        cfg = json.load(open(os.path.join(SKILLS, "trader-budget", "config.example.json")))
        json.dump(cfg, open("trader/config.json", "w"))
        json.dump({"orders": [{"symbol": "AAPL", "side": "buy", "notional": "1250", "type": "market",
                               "time_in_force": "day"}]}, open("trader/orders.json", "w"))

    def tearDown(self):
        capital.Capital.from_env = classmethod(self._orig)
        os.chdir(self.cwd)
        shutil.rmtree(self.dir)

    def run_cli(self, *args):
        out = io.StringIO()
        with redirect_stdout(out):
            code = capital.main(list(args))
        return code, out.getvalue()

    def tool(self, skill, script, *args):
        return subprocess.run([sys.executable, os.path.join(SKILLS, skill, script)] + list(args),
                              capture_output=True, text=True)

    def test_demo_by_default(self):
        self.run_cli("market", "US500")
        self.assertTrue(capital.Capital(live=False).base.startswith("https://demo-api"))

    def test_place_needs_risk_and_yes(self):
        code, _ = self.run_cli("place", "--index", "1")
        self.assertEqual(code, 2)
        self.assertFalse(any(c[0] == "POST" and c[1] == "/api/v1/positions" for c in self.fake.calls))

    def test_full_flow(self):
        r = self.tool("trader-risk", "check.py")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        r = self.tool("trader-broker", "ticket.py", "approve", "--index", "1", "--reply", "yes")
        self.assertIn("Recorded", r.stdout + r.stderr)
        code, out = self.run_cli("place", "--index", "1", "--stop-pct", "1")
        self.assertEqual(code, 0, out)
        post = [c for c in self.fake.calls if c[0] == "POST" and c[1] == "/api/v1/positions"][0][2]
        self.assertEqual(post["epic"], "AAPL")
        self.assertEqual(post["direction"], "BUY")
        self.assertAlmostEqual(post["size"], 3.66)          # 1250 / 341.3, redondeado hacia abajo
        self.assertAlmostEqual(post["stopLevel"], 337.887)
        self.assertIn("D1", out)

    def test_candles_csv(self):
        code, _ = self.run_cli("candles", "US500", "--out", "v.csv")
        self.assertEqual(code, 0)
        rows = open("v.csv").read().splitlines()
        self.assertEqual(rows[0], "Datetime,Open,High,Low,Close,Volume,Spread")
        self.assertIn("7563.3", rows[1])

    def test_spy_maps_to_us500(self):
        code, out = self.run_cli("quote", "SPY", "--write")
        self.assertIn("US500", out)
        self.assertAlmostEqual(json.load(open("trader/prices.json"))["prices"]["SPY"], 7568.9)


if __name__ == "__main__":
    unittest.main()
