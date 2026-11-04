#!/usr/bin/env python3
"""Capital.com para el pack trader. Sólo librería estándar.

Conecta las skills trader-* con la API oficial de Capital.com
(https://open-api.capital.com). Usa la cuenta DEMO por defecto.

    python3 capital.py account                       # saldo, cuenta demo o real
    python3 capital.py search "US 500"               # buscar mercados (epic, spread)
    python3 capital.py market US500                  # detalle: tamaño mínimo, margen, horario
    python3 capital.py quote AAPL US500 --write      # precios -> trader/prices.json
    python3 capital.py candles US500 --resolution MINUTE_5 --days 60 --out velas.csv
    python3 capital.py positions                     # posiciones abiertas y P&L
    python3 capital.py tickets                       # trader/orders.json -> órdenes Capital.com
    python3 capital.py place --index 1 [--stop-pct 0.5] [--tp-pct 1]
    python3 capital.py close DEAL_ID --yes

Credenciales SÓLO por variables de entorno (nunca en el chat):
    CAPITAL_API_KEY      clave API (Ajustes > API integrations en Capital.com)
    CAPITAL_IDENTIFIER   el email / login de tu cuenta
    CAPITAL_PASSWORD     la contraseña personalizada que pusiste al crear la clave
    CAPITAL_LIVE=1       sólo si quieres la cuenta REAL; sin ella siempre es DEMO

`place` sólo envía una orden si trader-broker (ticket.py approve) registró hoy
tu "yes" para ESA orden exacta, y trader-risk la pasó. Nunca "todas a la vez".
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

LIVE_URL = "https://api-capital.backend-capital.com"
DEMO_URL = "https://demo-api-capital.backend-capital.com"
RESOLUTIONS = ("MINUTE", "MINUTE_5", "MINUTE_15", "MINUTE_30", "HOUR", "HOUR_4", "DAY", "WEEK")

# Mismos campos y misma huella que trader-broker/ticket.py y trader-risk/check.py
ALPACA_FIELDS = ("symbol", "side", "qty", "notional", "type", "time_in_force", "limit_price",
                 "stop_price", "trail_price", "trail_percent", "extended_hours", "client_order_id",
                 "order_class", "take_profit_limit_price", "stop_loss_stop_price",
                 "stop_loss_limit_price", "advanced_instructions")

# Tickers del pack -> epic de Capital.com cuando no coinciden
DEFAULT_EPICS = {"SPY": "US500", "VOO": "US500", "IVV": "US500", "^GSPC": "US500",
                 "QQQ": "US100", "^NDX": "US100", "^IXIC": "US100",
                 "DIA": "US30", "^DJI": "US30", "GOOG": "GOOGL", "BRK.B": "BRKB"}


class CapitalError(Exception):
    pass


# ---------------------------------------------------------------- archivos

def read_json(path, default=None):
    if not path or not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path, data):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def fingerprint(order):
    core = {k: order[k] for k in ALPACA_FIELDS if k in order and order[k] is not None}
    blob = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def today():
    return datetime.date.today().isoformat()


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- API

class Capital:
    def __init__(self, live=False, opener=None):
        self.live = live
        self.base = LIVE_URL if live else DEMO_URL
        self.cst = self.token = None
        self._open = opener or urllib.request.urlopen
        self._last = 0.0

    @classmethod
    def from_env(cls, opener=None):
        live = os.environ.get("CAPITAL_LIVE", "").strip().lower() in ("1", "true", "yes", "si", "sí")
        c = cls(live=live, opener=opener)
        missing = [k for k in ("CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD")
                   if not os.environ.get(k)]
        if missing:
            raise CapitalError(
                "faltan variables de entorno: %s. Créalas en tu terminal (no en el chat). "
                "La clave API se crea en Capital.com > Ajustes > API integrations." % ", ".join(missing))
        c.login(os.environ["CAPITAL_API_KEY"], os.environ["CAPITAL_IDENTIFIER"],
                os.environ["CAPITAL_PASSWORD"])
        return c

    def _req(self, method, path, body=None, params=None, headers=None):
        # máx. 10 peticiones/s por usuario: dejamos 0.15 s entre llamadas
        wait = 0.15 - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        url = self.base + path
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.cst:
            h["CST"], h["X-SECURITY-TOKEN"] = self.cst, self.token
        h.update(headers or {})
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=h)
        self._last = time.time()
        try:
            resp = self._open(req, timeout=30)
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            try:
                msg = json.loads(msg).get("errorCode", msg)
            except (ValueError, AttributeError):
                pass
            raise CapitalError("%s %s -> HTTP %s: %s" % (method, path, e.code, msg))
        except urllib.error.URLError as e:
            raise CapitalError("no se pudo conectar con Capital.com: %s" % e.reason)
        raw = resp.read().decode("utf-8") or "{}"
        return json.loads(raw), resp.headers

    def login(self, api_key, identifier, password):
        _, hdr = self._req("POST", "/api/v1/session",
                           {"identifier": identifier, "password": password, "encryptedPassword": False},
                           headers={"X-CAP-API-KEY": api_key})
        self.cst, self.token = hdr.get("CST"), hdr.get("X-SECURITY-TOKEN")
        if not self.cst or not self.token:
            raise CapitalError("Capital.com no devolvió los tokens de sesión")

    def get(self, path, **params):
        return self._req("GET", path, params=params)[0]

    def accounts(self):
        return self.get("/api/v1/accounts").get("accounts", [])

    def search(self, term):
        return self.get("/api/v1/markets", searchTerm=term).get("markets", [])

    def market(self, epic):
        return self.get("/api/v1/markets/" + urllib.parse.quote(epic))

    def prices(self, epic, resolution="MINUTE_5", maximum=1000, start=None, end=None):
        return self.get("/api/v1/prices/" + urllib.parse.quote(epic), resolution=resolution,
                        max=maximum, **{"from": start, "to": end}).get("prices", [])

    def positions(self):
        return self.get("/api/v1/positions").get("positions", [])

    def open_position(self, body):
        return self._req("POST", "/api/v1/positions", body)[0]

    def close_position(self, deal_id):
        return self._req("DELETE", "/api/v1/positions/" + urllib.parse.quote(deal_id))[0]

    def confirm(self, deal_ref):
        for _ in range(5):
            try:
                return self.get("/api/v1/confirms/" + urllib.parse.quote(deal_ref))
            except CapitalError:
                time.sleep(0.5)
        raise CapitalError("sin confirmación para %s: revisa la plataforma" % deal_ref)


# ---------------------------------------------------------------- utilidades de mercado

def epic_for(api, ticker, mapping):
    t = ticker.upper()
    if t in mapping:
        return mapping[t]
    if t in DEFAULT_EPICS:
        return DEFAULT_EPICS[t]
    for m in api.search(t):
        if str(m.get("epic", "")).upper() == t or str(m.get("symbol", "")).upper() == t:
            return m["epic"]
    raise CapitalError("no encuentro el epic de %s en Capital.com. Búscalo con "
                       "`capital.py search %s` y añádelo a trader/capital_epics.json" % (t, t))


def snapshot(details):
    snap = details.get("snapshot", {})
    bid, offer = f(snap.get("bid")), f(snap.get("offer"))
    mid = (bid + offer) / 2 if bid and offer else None
    spread = offer - bid if bid and offer else None
    bps = spread / mid * 1e4 if mid else None
    return bid, offer, mid, spread, bps, snap.get("marketStatus")


def rule(details, name):
    r = details.get("dealingRules", {}).get(name) or {}
    return f(r.get("value"))


def round_down(x, step):
    if not step or step <= 0:
        return x
    n = math.floor(x / step + 1e-9) * step
    decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return round(n, decimals)


def to_capital(api, order, mapping):
    """Convierte una orden del pack (formato Alpaca) en el cuerpo de POST /positions."""
    if order.get("type", "market") != "market":
        raise CapitalError("sólo órdenes a mercado (type market); %s es %s"
                           % (order.get("symbol"), order.get("type")))
    side = str(order.get("side", "")).lower()
    if side not in ("buy", "sell"):
        raise CapitalError("side debe ser buy o sell")
    epic = epic_for(api, order["symbol"], mapping)
    det = api.market(epic)
    bid, offer, mid, spread, bps, status = snapshot(det)
    if status and status != "TRADEABLE":
        raise CapitalError("%s no se puede operar ahora (estado %s)" % (epic, status))
    px = offer if side == "buy" else bid
    if not px:
        raise CapitalError("sin precio para %s" % epic)
    step = rule(det, "minSizeIncrement") or 0.01
    if order.get("qty") is not None:
        size = f(order["qty"])
    else:
        size = round_down(f(order["notional"]) / px, step)
    minimo = rule(det, "minDealSize")
    if minimo and size < minimo:
        raise CapitalError("%s: tamaño %s menor que el mínimo de Capital.com (%s)" % (epic, size, minimo))
    inst = det.get("instrument", {})
    body = {"epic": epic, "direction": side.upper(), "size": size, "guaranteedStop": False}
    info = {"epic": epic, "name": inst.get("name"), "currency": inst.get("currency"),
            "price": px, "spread": spread, "spread_bps": bps, "exposure": size * px,
            "margin_factor": inst.get("marginFactor"), "margin_unit": inst.get("marginFactorUnit")}
    return body, info


def approval(state, fp):
    """Último registro de hoy de ticket.py para esta huella."""
    last = None
    for rec in state.get("orders", []):
        if rec.get("fingerprint") == fp and str(rec.get("date") or rec.get("at", ""))[:10] == today():
            last = rec
    return last


def risk_pass(risk, fp):
    """Veredicto de trader-risk de HOY para esta huella, o None."""
    if not risk or risk.get("date") != today():
        return None
    return next((r for r in risk.get("results", []) if r.get("fingerprint") == fp), None)


# ---------------------------------------------------------------- comandos

def money(x, cur=""):
    return ("{:,.2f} {}".format(x, cur)).strip() if x is not None else "-"


def cmd_account(api, a):
    print("Cuenta %s" % ("REAL ⚠️" if api.live else "DEMO"))
    for acc in api.accounts():
        b = acc.get("balance", {})
        cur = acc.get("currency", "")
        print("  %s %s%s  saldo %s  disponible %s  P&L %s" % (
            acc.get("accountId"), acc.get("accountName", ""), " (preferida)" if acc.get("preferred") else "",
            money(f(b.get("balance")), cur), money(f(b.get("available")), cur), money(f(b.get("profitLoss")), cur)))


def cmd_search(api, a):
    rows = api.search(a.term)
    if not rows:
        print("Nada para '%s'" % a.term)
        return 1
    print("%-14s %-34s %-12s %12s %12s %9s" % ("epic", "nombre", "tipo", "bid", "offer", "spread bps"))
    for m in rows[:a.limit]:
        bid, off = f(m.get("bid")), f(m.get("offer"))
        bps = (off - bid) / ((off + bid) / 2) * 1e4 if bid and off else None
        print("%-14s %-34.34s %-12s %12s %12s %9s" % (
            m.get("epic"), m.get("instrumentName", ""), m.get("instrumentType", ""),
            bid, off, "%.2f" % bps if bps is not None else "-"))
    return 0


def cmd_market(api, a):
    det = api.market(a.epic)
    if a.json:
        print(json.dumps(det, indent=2))
        return 0
    inst = det.get("instrument", {})
    bid, offer, mid, spread, bps, status = snapshot(det)
    print("%s  %s  (%s, %s)" % (inst.get("epic", a.epic), inst.get("name"), inst.get("type"), inst.get("currency")))
    print("  estado %s   bid %s   offer %s" % (status, bid, offer))
    if spread is not None:
        print("  spread %.4f  = %.2f bps (%.4f%%) por operación ida y vuelta" % (spread, bps, bps / 100))
        print("  -> para scalping.py usa --coste-bps %.2f (la mitad, por lado)" % (bps / 2))
    print("  tamaño mínimo %s  incremento %s  máximo %s" % (
        rule(det, "minDealSize"), rule(det, "minSizeIncrement"), rule(det, "maxDealSize")))
    print("  margen %s %s   stop garantizado %s" % (
        inst.get("marginFactor"), inst.get("marginFactorUnit", ""), inst.get("guaranteedStopAllowed")))
    if inst.get("overnightFee"):
        print("  financiación nocturna: %s" % json.dumps(inst["overnightFee"]))
    return 0


def cmd_quote(api, a):
    mapping = read_json(a.epics, {}) or {}
    out = {}
    for t in a.tickers:
        epic = epic_for(api, t, mapping)
        bid, offer, mid, spread, bps, status = snapshot(api.market(epic))
        out[t.upper()] = mid
        print("%-8s %-10s bid %-12s offer %-12s medio %-12s spread %s bps  %s" % (
            t.upper(), epic, bid, offer, "%.4f" % mid if mid else "-",
            "%.2f" % bps if bps is not None else "-", status))
    if a.write:
        data = read_json(a.prices, {}) or {}
        data.setdefault("prices", {}).update({k: v for k, v in out.items() if v})
        data["as_of"] = datetime.datetime.now().isoformat(timespec="seconds")
        data["source"] = "capital.com " + ("live" if api.live else "demo")
        write_json(a.prices, data)
        print("Guardado en %s" % a.prices)
    return 0


RES_MIN = {"MINUTE": 1, "MINUTE_5": 5, "MINUTE_15": 15, "MINUTE_30": 30, "HOUR": 60,
           "HOUR_4": 240, "DAY": 1440, "WEEK": 10080}


def fetch_candles(api, epic, resolution, maximum, start, end, days):
    if not days:
        return api.prices(epic, resolution, maximum, start, end)
    # La API da máx. 1000 velas por llamada: pedimos ventanas hacia atrás
    fmt = "%Y-%m-%dT%H:%M:%S"
    fin = datetime.datetime.utcnow().replace(microsecond=0)
    limite = fin - datetime.timedelta(days=days)
    paso = datetime.timedelta(minutes=RES_MIN[resolution] * 1000)
    rows = {}
    while fin > limite:
        ini = max(limite, fin - paso)
        try:
            lote = api.prices(epic, resolution, 1000, ini.strftime(fmt), fin.strftime(fmt))
        except CapitalError as e:
            if "not found" not in str(e).lower() and "404" not in str(e):
                raise
            lote = []                      # ventana sin datos (fin de semana)
        for r in lote:
            rows[r.get("snapshotTimeUTC") or r.get("snapshotTime")] = r
        fin = ini
    return [rows[k] for k in sorted(rows)]


def cmd_candles(api, a):
    rows = fetch_candles(api, a.epic, a.resolution, a.max, a.start, a.end, a.days)
    if not rows:
        print("Sin velas")
        return 1
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Datetime", "Open", "High", "Low", "Close", "Volume", "Spread"])
        for r in rows:
            def mid(k):
                p = r.get(k) or {}
                b, s = f(p.get("bid")), f(p.get("ask"))
                return (b + s) / 2 if b is not None and s is not None else (b or s)
            cp = r.get("closePrice") or {}
            spr = (f(cp.get("ask")) or 0) - (f(cp.get("bid")) or 0)
            w.writerow([r.get("snapshotTimeUTC") or r.get("snapshotTime"), mid("openPrice"),
                        mid("highPrice"), mid("lowPrice"), mid("closePrice"),
                        r.get("lastTradedVolume", 0), round(spr, 6)])
    print("%d velas %s de %s guardadas en %s (precio medio bid/ask + spread)" % (
        len(rows), a.resolution, a.epic, a.out))
    print("Backtest: python3 scalping.py --csv %s" % a.out)
    return 0


def cmd_positions(api, a):
    pos = api.positions()
    print("Cuenta %s: %d posiciones abiertas" % ("REAL" if api.live else "DEMO", len(pos)))
    for p in pos:
        q, m = p.get("position", {}), p.get("market", {})
        print("  %s %-5s %-10s tamaño %-8s entrada %-10s ahora %s/%s  P&L %s  stop %s  tp %s" % (
            q.get("dealId"), q.get("direction"), m.get("epic"), q.get("size"), q.get("level"),
            m.get("bid"), m.get("offer"), money(f(q.get("upl")), q.get("currency", "")),
            q.get("stopLevel"), q.get("profitLevel")))
    return 0


def _orders(a):
    data = read_json(a.orders)
    if data is None:
        raise CapitalError("no hay %s: créalo con /trader-budget o /trader-rebalance" % a.orders)
    return data.get("orders", data) if isinstance(data, dict) else data


def cmd_tickets(api, a):
    orders = _orders(a)
    mapping = read_json(a.epics, {}) or {}
    state = read_json(a.state, {}) or {}
    risk = read_json(a.risk, {}) or {}
    for i, o in enumerate(orders, 1):
        fp = fingerprint(o)
        rec = approval(state, fp)
        verdict = risk_pass(risk, fp)
        try:
            body, info = to_capital(api, o, mapping)
        except CapitalError as e:
            print("#%d %s %s  NO CONVERTIBLE: %s" % (i, o.get("side"), o.get("symbol"), e))
            continue
        estado = rec.get("status") if rec else "sin decidir"
        print("#%d %s %s -> Capital.com %s %s x %s @ ~%s %s (exposición %s)" % (
            i, o.get("side"), o.get("symbol"), body["direction"], info["epic"], body["size"],
            info["price"], info["currency"] or "", money(info["exposure"])))
        if info["spread_bps"] is not None:
            print("     spread %.2f bps (coste inmediato ≈ %s)   riesgo: %s   tu decisión: %s" % (
                info["spread_bps"], money(info["exposure"] * info["spread_bps"] / 1e4),
                (verdict or {}).get("verdict", "sin revisar"), estado))
        print("     JSON: %s" % json.dumps(body))
    return 0


def cmd_place(api, a):
    orders = _orders(a)
    if not 1 <= a.index <= len(orders):
        raise CapitalError("no existe la orden #%d" % a.index)
    order = orders[a.index - 1]
    fp = fingerprint(order)
    rv = risk_pass(read_json(a.risk, {}) or {}, fp)
    if not rv or rv.get("verdict") != "PASS":
        raise CapitalError("la orden #%d no pasó trader-risk hoy (o cambió). Corre check.py otra vez" % a.index)
    rec = approval(read_json(a.state, {}) or {}, fp)
    if not rec or rec.get("status") != "approved":
        raise CapitalError(
            "la orden #%d no tiene tu 'yes' registrado hoy. Primero: trader-risk (check.py), luego "
            "ticket.py approve --index %d --reply \"<tu respuesta literal>\"" % (a.index, a.index))
    body, info = to_capital(api, order, read_json(a.epics, {}) or {})
    if a.stop_pct:
        sign = -1 if body["direction"] == "BUY" else 1
        body["stopLevel"] = round(info["price"] * (1 + sign * a.stop_pct / 100), 5)
    if a.tp_pct:
        sign = 1 if body["direction"] == "BUY" else -1
        body["profitLevel"] = round(info["price"] * (1 + sign * a.tp_pct / 100), 5)
    print("Cuenta %s. Enviando: %s" % ("REAL ⚠️" if api.live else "DEMO", json.dumps(body)))
    res = api.open_position(body)
    ref = res.get("dealReference")
    conf = api.confirm(ref) if ref else res
    status = conf.get("dealStatus") or conf.get("status")
    deals = [d.get("dealId") for d in conf.get("affectedDeals", []) if d.get("dealId")]
    print("Resultado: %s  dealReference %s  dealId %s  nivel %s  %s" % (
        status, ref, ",".join(deals) or conf.get("dealId"), conf.get("level"), conf.get("reason", "")))
    print("Regístralo: ticket.py result --index %d --status %s --broker-id %s" % (
        a.index, "placed" if status == "ACCEPTED" else "failed", (deals or [ref])[0]))
    return 0 if status == "ACCEPTED" else 1


def cmd_close(api, a):
    if not a.yes:
        raise CapitalError("cerrar una posición necesita --yes, y sólo si el usuario lo pidió para ese dealId")
    res = api.close_position(a.deal_id)
    ref = res.get("dealReference")
    conf = api.confirm(ref) if ref else res
    print("Cierre %s: %s  %s" % (a.deal_id, conf.get("dealStatus"), conf.get("reason", "")))
    return 0 if conf.get("dealStatus") == "ACCEPTED" else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Capital.com para las skills trader")
    ap.add_argument("--epics", default="trader/capital_epics.json", help="mapa ticker -> epic propio")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("account")
    p = sub.add_parser("search"); p.add_argument("term"); p.add_argument("--limit", type=int, default=15)
    p = sub.add_parser("market"); p.add_argument("epic"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("quote"); p.add_argument("tickers", nargs="+")
    p.add_argument("--write", action="store_true"); p.add_argument("--prices", default="trader/prices.json")
    p = sub.add_parser("candles"); p.add_argument("epic")
    p.add_argument("--resolution", default="MINUTE_5", choices=RESOLUTIONS)
    p.add_argument("--max", type=int, default=1000); p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--days", type=int, help="bajar N días hacia atrás (pagina de 1000 en 1000)")
    p.add_argument("--out", default="velas_capital.csv")
    sub.add_parser("positions")
    for name in ("tickets", "place"):
        p = sub.add_parser(name)
        p.add_argument("--orders", default="trader/orders.json")
        p.add_argument("--state", default="trader/state/broker.json")
        p.add_argument("--risk", default="trader/state/risk.json")
        if name == "place":
            p.add_argument("--index", type=int, required=True)
            p.add_argument("--stop-pct", type=float, help="stop-loss a X%% del precio de entrada")
            p.add_argument("--tp-pct", type=float, help="take-profit a X%% del precio de entrada")
    p = sub.add_parser("close"); p.add_argument("deal_id"); p.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    try:
        api = Capital.from_env()
        fn = globals()["cmd_" + a.cmd]
        return fn(api, a) or 0
    except CapitalError as e:
        print("Error: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
