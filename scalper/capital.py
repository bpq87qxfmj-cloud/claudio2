"""Conexión con Capital.com (API REST oficial, https://open-api.capital.com).

Credenciales por variables de entorno (nunca en el código ni en la línea de comandos):
    CAPITAL_API_KEY      clave generada en Ajustes > Integraciones API
    CAPITAL_IDENTIFIER   el correo con el que entras en Capital.com
    CAPITAL_PASSWORD     la contraseña *de la clave API* (no la de la cuenta)

Por defecto se usa el servidor demo; el real solo con demo=False (--allow-real).

AVISO: probado contra un doble de pruebas que imita la API documentada y contra las respuestas
de error del servidor demo real, pero no con una sesión autenticada. Valídalo en demo.
"""
from __future__ import annotations

import logging
import math
import os
import time

import pandas as pd
import requests

from .brokers import Order, Position, closed_trade
from .config import Market

log = logging.getLogger("scalper.capital")
DEMO_URL = "https://demo-api-capital.backend-capital.com"
LIVE_URL = "https://api-capital.backend-capital.com"
EPICS = {"XAUUSD": "GOLD", "NAS100": "US100", "SPX500": "US500"}
COLS = ["open", "high", "low", "close"]


class CapitalError(RuntimeError):
    def __init__(self, status: int, code: str):
        super().__init__(f"Capital.com {status}: {code}")
        self.status, self.code = status, code


class CapitalClient:
    """Cliente mínimo: sesión con renovación automática y las llamadas que necesita el agente."""

    def __init__(self, api_key: str, identifier: str, password: str, demo: bool = True, http=None, sleep=time.sleep):
        self.base = (DEMO_URL if demo else LIVE_URL) + "/api/v1"
        self.demo = demo
        self._key, self._id, self._pw = api_key, identifier, password
        self.http = http or requests.Session()
        self._sleep = sleep
        self._tokens: dict[str, str] = {}
        self._last_login = 0.0

    @classmethod
    def from_env(cls, demo: bool = True, **kw) -> "CapitalClient":
        missing = [k for k in ("CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD") if not os.environ.get(k)]
        if missing:
            raise SystemExit("Faltan variables de entorno: " + ", ".join(missing) +
                             ". Genera la clave en Capital.com > Ajustes > Integraciones API.")
        return cls(os.environ["CAPITAL_API_KEY"], os.environ["CAPITAL_IDENTIFIER"], os.environ["CAPITAL_PASSWORD"],
                   demo=demo, **kw)

    def _call(self, method: str, path: str, headers: dict, **kw):
        r = self.http.request(method, self.base + path, headers=headers, timeout=20, **kw)
        try:
            body = r.json() if r.content else {}
        except ValueError:
            body = {}
        if r.status_code >= 400:
            raise CapitalError(r.status_code, str(body.get("errorCode", r.status_code)))
        return r, body

    def login(self) -> dict:
        wait = 1.1 - (time.time() - self._last_login)  # límite: 1 inicio de sesión por segundo
        if wait > 0:
            self._sleep(wait)
        self._last_login = time.time()
        r, body = self._call("POST", "/session", {"X-CAP-API-KEY": self._key},
                             json={"identifier": self._id, "password": self._pw, "encryptedPassword": False})
        self._tokens = {"CST": r.headers["CST"], "X-SECURITY-TOKEN": r.headers["X-SECURITY-TOKEN"]}
        return body

    def request(self, method: str, path: str, **kw) -> dict:
        if not self._tokens:
            self.login()
        try:
            return self._call(method, path, dict(self._tokens), **kw)[1]
        except CapitalError as e:
            if "token" not in e.code:  # la sesión caduca a los 10 min sin uso
                raise
        self.login()
        return self._call(method, path, dict(self._tokens), **kw)[1]

    # --- llamadas
    def market(self, epic: str) -> dict:
        return self.request("GET", f"/markets/{epic}")

    def prices(self, epic: str, start: pd.Timestamp, end: pd.Timestamp, max_: int = 1000) -> list[dict]:
        fmt = "%Y-%m-%dT%H:%M:%S"
        try:
            return self.request("GET", f"/prices/{epic}", params={
                "resolution": "MINUTE", "max": max_, "from": start.strftime(fmt), "to": end.strftime(fmt)}).get("prices", [])
        except CapitalError as e:
            if e.status in (400, 404) and ("not-found" in e.code or "prices" in e.code or "range" in e.code):
                return []  # tramo sin cotización (fin de semana, cierre diario)
            raise

    def balance(self) -> float:
        """Capital de la cuenta activa (saldo más resultado latente)."""
        accounts = self.request("GET", "/accounts")["accounts"]
        acc = next((a for a in accounts if a.get("preferred")), accounts[0])
        return float(acc["balance"]["balance"])

    def positions(self) -> list[dict]:
        return self.request("GET", "/positions").get("positions", [])

    def confirm(self, deal_reference: str, tries: int = 5) -> dict:
        for i in range(tries):
            try:
                return self.request("GET", f"/confirms/{deal_reference}")
            except CapitalError as e:
                if e.status != 404 or i == tries - 1:
                    raise
                self._sleep(0.3)
        raise AssertionError

    def open_position(self, epic: str, direction: str, size: float, stop: float, profit: float) -> dict:
        ref = self.request("POST", "/positions", json={"epic": epic, "direction": direction, "size": size,
                                                       "stopLevel": stop, "profitLevel": profit})["dealReference"]
        return self.confirm(ref)

    def close_position(self, deal_id: str) -> dict:
        ref = self.request("DELETE", f"/positions/{deal_id}")["dealReference"]
        return self.confirm(ref)


def _bars_frame(rows: list[dict]) -> pd.DataFrame:
    """Barras de la API -> OHLC en bid (como el histórico de entrenamiento) más el spread al cierre."""
    if not rows:
        return pd.DataFrame(columns=COLS + ["spread"], index=pd.DatetimeIndex([], tz="UTC", name="time"), dtype=float)
    idx = pd.DatetimeIndex(pd.to_datetime([r["snapshotTimeUTC"] for r in rows]), name="time").tz_localize("UTC")
    data = {c: [r[f"{c}Price"]["bid"] for r in rows] for c in COLS}
    data["spread"] = [r["closePrice"]["ask"] - r["closePrice"]["bid"] for r in rows]
    return pd.DataFrame(data, index=idx, dtype=float)


class CapitalBroker:
    """Datos y ejecución en Capital.com con la misma interfaz que el resto de brókers."""

    name = "capital"

    def __init__(self, market: Market, client: CapitalClient, epic: str | None = None, now=None):
        self.market, self.client = market, client
        self.epic = epic or EPICS.get(market.key, market.key)
        self._now = now or (lambda: pd.Timestamp.now(tz="UTC"))
        self._cache = _bars_frame([])
        self.position: Position | None = None
        self._balance_before: float | None = None
        self._last_spread: float | None = None
        info = client.market(self.epic)  # falla pronto si el instrumento o las credenciales no valen
        log.info("Capital.com %s: %s (%s), estado %s", "DEMO" if client.demo else "REAL", self.epic,
                 info["instrument"].get("name"), info["snapshot"].get("marketStatus"))

    # ------------------------------------------------------------------ datos
    def bars(self, n: int = 2500) -> pd.DataFrame:
        """Últimas `n` barras M1 cerradas (bid). La primera llamada pagina hacia atrás; luego solo actualiza."""
        now = self._now()
        if len(self._cache) >= 50:
            start = self._cache.index[-1] - pd.Timedelta(minutes=5)
            new = _bars_frame(self.client.prices(self.epic, start, now))
            self._cache = pd.concat([self._cache[self._cache.index < start], new])
        end = self._cache.index[0] - pd.Timedelta(minutes=1) if len(self._cache) else now
        pages = 0
        while len(self._cache) < n and pages < 14:  # 14 páginas de 1000 min cubren un fin de semana largo
            start = end - pd.Timedelta(minutes=999)
            page = _bars_frame(self.client.prices(self.epic, start, end))
            self._cache = pd.concat([page, self._cache])
            end, pages = start - pd.Timedelta(minutes=1), pages + 1
        df = self._cache[~self._cache.index.duplicated(keep="last")].sort_index()
        self._cache = df.iloc[-(n + 500):]
        closed = df[df.index + pd.Timedelta(minutes=1) <= now]  # la barra en formación no cuenta
        if len(closed):
            self._last_spread = float(closed["spread"].iloc[-1])
        return closed[COLS].iloc[-n:]

    def spread_stats(self, n: int = 3000) -> pd.DataFrame:
        """Spread real observado (ask - bid al cierre de cada minuto) por hora de Nueva York."""
        self.bars(n)
        s = self._cache["spread"]
        hours = s.index.tz_convert("America/New_York").hour
        g = s.groupby(hours).agg(["count", "median", "mean", "max"])
        g.index.name = "hora_NY"
        return g

    # ------------------------------------------------------------------ interfaz de bróker
    def equity(self) -> float:
        return self.client.balance()

    def flat(self) -> bool:
        return self.position is None

    def spread(self) -> float | None:
        return self._last_spread

    def submit(self, order: Order) -> bool:
        if self.position is not None:
            return False
        m = self.client.market(self.epic)
        snap, rules = m["snapshot"], m["dealingRules"]
        if snap.get("marketStatus") != "TRADEABLE":
            log.warning("Mercado no operable: %s", snap.get("marketStatus"))
            return False
        bid, ask = float(snap["bid"]), float(snap["offer"])
        step = float(rules["minSizeIncrement"]["value"])
        size = round(math.floor(order.units / step + 1e-9) * step, 8)
        if size < float(rules["minDealSize"]["value"]):
            log.warning("Tamaño %.4f por debajo del mínimo %s", size, rules["minDealSize"]["value"])
            return False
        md = rules.get("minStopOrProfitDistance") or {}
        min_dist = float(md.get("value", 0)) * (bid / 100 if md.get("unit") == "PERCENTAGE" else 1)
        if order.width < min_dist:
            log.warning("Barrera %.3f menor que la distancia mínima del bróker %.3f", order.width, min_dist)
            return False
        # Las barreras del modelo están en bid. Un largo se cierra en bid; un corto, en ask.
        shift = 0.0 if order.side > 0 else ask - bid
        digits = int(snap.get("decimalPlacesFactor", 2))
        tp = round(bid + order.side * order.width + shift, digits)
        sl = round(bid - order.side * order.width + shift, digits)
        self._balance_before = self.client.balance()
        conf = self.client.open_position(self.epic, "BUY" if order.side > 0 else "SELL", size, sl, tp)
        if conf.get("dealStatus") != "ACCEPTED":
            log.error("Orden rechazada: %s", conf.get("rejectReason") or conf.get("dealStatus"))
            return False
        deals = conf.get("affectedDeals") or [{"dealId": conf.get("dealId")}]
        entry = float(conf.get("level") or (ask if order.side > 0 else bid))
        self.position = Position(order.side, size, entry, tp, sl, order.width, order.horizon, self._now(),
                                 order.signal_time, order.pred, ticket=deals[0]["dealId"])
        self._last_spread = ask - bid
        return True

    def _is_open(self) -> bool:
        return any(p["position"].get("dealId") == self.position.ticket for p in self.client.positions())

    def _finish(self, t, exit_price: float, reason: str) -> dict:
        pos, self.position = self.position, None
        money = self.client.balance() - (self._balance_before or 0.0)  # incluye spread y comisiones reales
        return closed_trade(pos, t, exit_price, 0.0, reason, net_money=money)

    def on_bar(self, t, o: float, h: float, l: float, c: float) -> list[dict]:
        pos = self.position
        if pos is None:
            return []
        pos.bars_held += 1
        if not self._is_open():  # el bróker la cerró por TP o SL
            won = self.client.balance() > (self._balance_before or 0.0)
            return [self._finish(t, pos.tp if won else pos.sl, "tp" if won else "sl")]
        expired = (self._now() - pos.entry_time).total_seconds() >= max(pos.horizon, 1) * 60
        if pos.bars_held >= max(pos.horizon, 1) or expired:
            return self.close_all(t, c, "tiempo")
        return []

    def close_all(self, t, price: float, reason: str) -> list[dict]:
        if self.position is None:
            return []
        try:
            conf = self.client.close_position(self.position.ticket)
            price = float(conf.get("level") or price)
        except CapitalError as e:
            if e.status != 404:  # 404: ya estaba cerrada
                raise
        return [self._finish(t, price, reason)]

    # ------------------------------------------------------------------ persistencia
    def dump(self) -> dict:
        from dataclasses import asdict

        pos = None
        if self.position is not None:
            pos = {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in asdict(self.position).items()}
        return dict(position=pos, balance_before=self._balance_before)

    def restore(self, d: dict) -> None:
        self._balance_before = d.get("balance_before")
        if d.get("position"):
            p = {k: (pd.Timestamp(v) if k in ("entry_time", "signal_time") else v) for k, v in d["position"].items()}
            self.position = Position(**p)
