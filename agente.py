"""
Agente automático para Capital.com: rebote RSI(2) en el US500 / US100

Hace solo, cada día de bolsa, la única regla que en los backtests tuvo
ventaja estable después de costes (ver alta_probabilidad.py):

  - A las 15:45 de Nueva York (14:45 en CDMX en invierno, 13:45 en verano)
    mira el índice. Si el RSI(2) está por debajo de 10 y el precio está por
    encima de su media de 200 días, COMPRA un CFD y deja en Capital.com un
    stop de seguridad (-5% por defecto).
  - Los días siguientes, a la misma hora, VENDE cuando el precio queda por
    encima de su media de 5 días, o a los 10 días como máximo.
  - No hace scalping: con 10 meses de velas de 5 minutos todas las reglas
    intradía perdieron después del spread (ver README).

Candados que trae puestos:
  - Usa la cuenta DEMO. Sólo usa la REAL si TÚ pones CAPITAL_LIVE=1 y además
    le pasas --real.
  - Una posición como máximo por índice: nunca promedia ni añade.
  - Sólo cierra las posiciones que abrió él; no toca las que abras a mano, y
    si ya tienes una en ese índice, no abre otra.
  - No abre si el mercado no está operable, si el spread es anormalmente
    ancho, o si la cuenta cayó más de --max-caida desde su máximo.
  - Exposición máxima: 2 veces tu saldo.
  - Apunta todo en la carpeta registro_agente/ (log.csv y agente.log).

Uso:
  python agente.py --backtest            # cómo le habría ido con estas reglas desde 1995
  python agente.py --simular             # qué haría hoy (no necesita claves, no envía nada)
  python agente.py --probar              # entra a tu cuenta DEMO y calcula el tamaño (no envía nada)
  python agente.py                       # se queda encendido y opera solo en la DEMO
  python agente.py --una-vez             # revisa ahora y termina (lo usa el Programador de tareas)
  python agente.py --instalar-windows    # crea las tareas programadas de Windows

Claves de Capital.com sólo en variables de entorno (CAPITAL_API_KEY,
CAPITAL_IDENTIFIER, CAPITAL_PASSWORD), nunca en este archivo.
Avisos al celular (opcional): TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID.

Proyecto educativo, NO es asesoría financiera. Los CFDs son apalancados y la
mayoría de las cuentas minoristas pierden dinero. Resultados pasados no
garantizan los futuros.
"""

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".claude", "skills", "trader-capital"))
from capital import Capital, CapitalError, f, round_down, rule, snapshot  # noqa: E402

NY = "America/New_York"
INDICES = {
    "us500": {"yahoo": "^GSPC", "epic": "US500", "nombre": "US500 (S&P 500)"},
    "us100": {"yahoo": "^NDX", "epic": "US100", "nombre": "US100 (Nasdaq 100)"},
}
RSI_N, RSI_MAX, SMA_TEND, SMA_SALIDA, MAX_DIAS = 2, 10, 200, 5, 10
MAX_EXPOSICION = 2.0
VENTANA = (datetime.time(15, 44), datetime.time(15, 58))          # hora de Nueva York
VENTANA_TEMPRANA = (datetime.time(12, 44), datetime.time(12, 58))  # días que cierra a la 1 pm
HORA_TAREA = datetime.time(15, 46)                                  # Programador de tareas

# Calendario de la Bolsa de Nueva York (mismo que la skill trader-alerts)
FESTIVOS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}
CIERRE_TEMPRANO = {"2026-11-27", "2026-12-24", "2027-11-26"}
ULTIMO_AÑO_CALENDARIO = 2027

DIR = os.path.join(AQUI, "registro_agente")
LOG_CSV = os.path.join(DIR, "log.csv")
LOG_TXT = os.path.join(DIR, "agente.log")
CAMPOS_LOG = ["fecha_ny", "cuenta", "epic", "accion", "precio", "tamano", "rsi2", "sma200",
              "sma5", "saldo", "detalle"]


# ----------------------------------------------------------------------------
# Regla (la misma función se usa en vivo y en el backtest)
# ----------------------------------------------------------------------------
def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def indicadores(closes):
    c = closes.astype(float)
    v = c.to_numpy()
    return {"precio": float(v[-1]),
            "rsi2": float(rsi(c.iloc[-60:], RSI_N).iloc[-1]),
            "sma200": float(v[-SMA_TEND:].mean()) if len(v) >= SMA_TEND else float("nan"),
            "sma5": float(v[-SMA_SALIDA:].mean())}


def decidir(closes, entrada=None, ultimo_cierre=None):
    """closes: cierres diarios (índice = fechas) cuyo ÚLTIMO valor es el precio de hoy.
    entrada: fecha en que el agente compró (None si no tiene posición).
    Devuelve (acción, indicadores, motivo)."""
    ind = indicadores(closes)
    hoy = closes.index[-1]
    if entrada is not None:
        dias = sum(1 for d in closes.index if d > entrada)
        ind["dias"] = dias
        if dias >= 1 and ind["precio"] > ind["sma5"]:
            return "vender", ind, f"quedó por encima de su media de 5 días ({ind['sma5']:,.1f})"
        if dias >= MAX_DIAS:
            return "vender", ind, f"{MAX_DIAS} días sin rebote: salida por tiempo"
        return "mantener", ind, f"día {dias} de {MAX_DIAS}; vende cuando supere {ind['sma5']:,.1f}"
    if ultimo_cierre == hoy:
        return "esperar", ind, "hoy ya se cerró una operación en este índice"
    if np.isnan(ind["sma200"]):
        return "esperar", ind, "faltan datos para la media de 200 días"
    if ind["rsi2"] < RSI_MAX and ind["precio"] > ind["sma200"]:
        return "comprar", ind, f"RSI(2) {ind['rsi2']:.1f} < {RSI_MAX} en tendencia alcista"
    if ind["precio"] <= ind["sma200"]:
        return "esperar", ind, "tendencia bajista (debajo de la media de 200 días): no compra"
    return "esperar", ind, f"RSI(2) {ind['rsi2']:.0f}: no ha caído lo suficiente (necesita < {RSI_MAX})"


# ----------------------------------------------------------------------------
# Calendario
# ----------------------------------------------------------------------------
def es_dia_de_bolsa(d):
    return d.weekday() < 5 and d.isoformat() not in FESTIVOS


def dia_anterior_de_bolsa(d):
    d -= datetime.timedelta(days=1)
    while not es_dia_de_bolsa(d):
        d -= datetime.timedelta(days=1)
    return d


def ventana(d):
    ini, fin = VENTANA_TEMPRANA if d.isoformat() in CIERRE_TEMPRANO else VENTANA
    return (pd.Timestamp(datetime.datetime.combine(d, ini), tz=NY),
            pd.Timestamp(datetime.datetime.combine(d, fin), tz=NY))


def proxima_ventana(ahora):
    d = ahora.date()
    for _ in range(10):
        if es_dia_de_bolsa(d):
            ini, fin = ventana(d)
            if ahora <= fin:
                return ini
        d += datetime.timedelta(days=1)
    return None


def ahora_ny():
    return pd.Timestamp.now(tz=NY)


def hora_local(ts):
    try:
        return ts.tz_convert("America/Mexico_City").strftime("%H:%M")
    except Exception:
        return ts.strftime("%H:%M NY")


# ----------------------------------------------------------------------------
# Datos diarios (Yahoo)
# ----------------------------------------------------------------------------
def descargar_diario(ticker, start=None, period=None):
    import yfinance as yf
    for intento in range(3):
        df = yf.download(ticker, start=start, period=period, interval="1d", auto_adjust=False,
                         progress=False)
        if not df.empty:
            break
        time.sleep(5 * (intento + 1))
    if df.empty:
        raise RuntimeError(f"Yahoo no devolvió datos de {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = [pd.Timestamp(x).date() for x in df.index]
    return df


def historia_yahoo(ticker, hoy):
    """Cierres diarios hasta AYER (el de hoy lo pone el precio en vivo)."""
    c = descargar_diario(ticker, period="2y")["Close"]
    c = c[[d < hoy for d in c.index]]
    esperado = dia_anterior_de_bolsa(hoy)
    if c.empty or c.index[-1] < esperado:
        raise RuntimeError(f"a Yahoo le falta el cierre del {esperado} de {ticker}; no opero con datos viejos")
    return c


# ----------------------------------------------------------------------------
# Registro y avisos
# ----------------------------------------------------------------------------
def leer_json(ruta, defecto):
    try:
        with open(ruta, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return defecto


def escribir_json(ruta, datos):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(datos, fh, indent=2, ensure_ascii=False, default=str)
    os.replace(ruta + ".tmp", ruta)


def avisar(texto, telegram=True):
    linea = f"[{ahora_ny():%Y-%m-%d %H:%M} NY] {texto}"
    print(linea, flush=True)
    os.makedirs(DIR, exist_ok=True)
    with open(LOG_TXT, "a", encoding="utf-8") as fh:
        fh.write(linea + "\n")
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if telegram and token and chat:
        try:
            datos = urllib.parse.urlencode({"chat_id": chat, "text": "🤖 Agente: " + texto}).encode()
            urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", datos, timeout=10)
        except Exception as e:  # el aviso nunca debe tumbar al agente
            print(f"  (no se pudo mandar el aviso por Telegram: {type(e).__name__})")


def registrar(fila):
    os.makedirs(DIR, exist_ok=True)
    nuevo = not os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS_LOG)
        if nuevo:
            w.writeheader()
        w.writerow({k: fila.get(k, "") for k in CAMPOS_LOG})


# ----------------------------------------------------------------------------
# El agente
# ----------------------------------------------------------------------------
class Agente:
    def __init__(self, api, a, historia=historia_yahoo, aviso=avisar, log=registrar):
        self.api, self.a = api, a
        self.historia, self.avisar, self.log = historia, aviso, log
        self.cuenta = "REAL" if api.live else "DEMO"
        self.ruta_estado = os.path.join(DIR, f"estado_{self.cuenta.lower()}.json")
        self.estado = leer_json(self.ruta_estado, {})
        for k in ("posiciones", "decidido", "ultimo_cierre", "pendiente"):
            self.estado.setdefault(k, {})

    def guardar(self):
        escribir_json(self.ruta_estado, self.estado)

    # --- cuenta ---------------------------------------------------------------
    def saldo(self):
        cuentas = self.api.accounts()
        if not cuentas:
            raise CapitalError("Capital.com no devolvió ninguna cuenta")
        c = next((x for x in cuentas if x.get("preferred")), cuentas[0])
        b = c.get("balance", {})
        bal, pl = f(b.get("balance")) or 0.0, f(b.get("profitLoss")) or 0.0
        return {"equity": bal + pl, "disponible": f(b.get("available")) or 0.0,
                "moneda": c.get("currency", "USD")}

    def tipo_cambio(self, de, a):
        """Cuántas unidades de la moneda de la cuenta vale 1 unidad de la moneda del índice."""
        if not de or not a or de == a:
            return 1.0
        for epic, inverso in ((de + a, False), (a + de, True)):
            try:
                mid = snapshot(self.api.market(epic))[2]
            except CapitalError:
                continue
            if mid:
                return 1 / mid if inverso else mid
        raise CapitalError(f"no encuentro el tipo de cambio {de}/{a} en Capital.com")

    # --- posiciones -----------------------------------------------------------
    def reconciliar(self, hoy):
        """Quita del estado las posiciones que ya no existen (stop o cierre manual) y
        devuelve los epics donde tienes posiciones que NO abrió el agente."""
        abiertas = self.api.positions()
        por_id = {p.get("position", {}).get("dealId"): p for p in abiertas}
        for epic, ref in list(self.estado["pendiente"].items()):
            # orden enviada sin confirmación: adoptar la posición si apareció
            for p in abiertas:
                q, m = p.get("position", {}), p.get("market", {})
                if (m.get("epic") == epic and q.get("direction") == "BUY"
                        and abs((f(q.get("size")) or 0) - ref["tamano"]) < 1e-9
                        and q.get("dealId") not in {x["dealId"] for x in self.estado["posiciones"].values()}):
                    self.estado["posiciones"][epic] = {**ref, "dealId": q["dealId"],
                                                       "precio": f(q.get("level")) or ref["precio"]}
                    self.avisar(f"{epic}: encontré la compra que no se confirmó (dealId {q['dealId']}).")
                    break
            else:
                self.avisar(f"{epic}: la compra sin confirmar no aparece en tus posiciones; la doy por no ejecutada.")
            del self.estado["pendiente"][epic]
        for epic, pos in list(self.estado["posiciones"].items()):
            if pos["dealId"] not in por_id:
                self.avisar(f"{epic}: la posición {pos['dealId']} ya no está abierta (saltó el stop de "
                            f"{pos['stop']:,.1f} o la cerraste tú). Revisa el resultado en Capital.com.")
                self.log({"fecha_ny": hoy, "cuenta": self.cuenta, "epic": epic, "accion": "cerrada_fuera",
                          "precio": pos["stop"], "tamano": pos["tamano"], "detalle": "stop o cierre manual"})
                del self.estado["posiciones"][epic]
                self.estado["ultimo_cierre"][epic] = str(hoy)
        mias = {p["dealId"] for p in self.estado["posiciones"].values()}
        self.guardar()
        return {p.get("market", {}).get("epic") for p in abiertas
                if p.get("position", {}).get("dealId") not in mias}

    # --- decisión diaria ------------------------------------------------------
    def revisar(self, clave, hoy, manuales):
        """Devuelve 'hecho' (ya decidió hoy) o 'reintentar'."""
        cfg = INDICES[clave]
        epic = cfg["epic"]
        if self.estado["decidido"].get(epic) == str(hoy):
            return "hecho"
        det = self.api.market(epic)
        bid, offer, mid, spread, bps, status = snapshot(det)
        if status != "TRADEABLE" or not mid:
            self.avisar(f"{epic}: el mercado no está operable ahora ({status}); reintento.", telegram=False)
            return "reintentar"
        hist = self.historia(cfg["yahoo"], hoy)
        closes = pd.concat([hist, pd.Series([mid], index=[hoy])])
        pos = self.estado["posiciones"].get(epic)
        entrada = datetime.date.fromisoformat(pos["fecha"]) if pos else None
        uc = self.estado["ultimo_cierre"].get(epic)
        accion, ind, motivo = decidir(closes, entrada, datetime.date.fromisoformat(uc) if uc else None)
        s = self.saldo()
        base = {"fecha_ny": hoy, "cuenta": self.cuenta, "epic": epic, "precio": round(mid, 2),
                "rsi2": round(ind["rsi2"], 1), "sma200": round(ind["sma200"], 1),
                "sma5": round(ind["sma5"], 1), "saldo": round(s["equity"], 2)}
        pico = max(self.estado.get("pico_saldo", 0.0), s["equity"])
        self.estado["pico_saldo"] = pico

        if accion == "vender":
            return self._vender(epic, pos, motivo, base, hoy, bid)
        if accion == "comprar":
            return self._comprar(epic, det, motivo, base, hoy, s, pico, manuales, offer, bps)
        self.log({**base, "accion": accion, "detalle": motivo})
        self.avisar(f"{cfg['nombre']}: {accion.upper()} — {motivo}. RSI(2) {ind['rsi2']:.0f}, "
                    f"precio {mid:,.1f}, media 200 {ind['sma200']:,.1f}. Saldo {self.cuenta} "
                    f"{s['equity']:,.2f} {s['moneda']}.")
        self._decidido(epic, hoy)
        return "hecho"

    def _decidido(self, epic, hoy):
        self.estado["decidido"][epic] = str(hoy)
        self.guardar()

    def _vender(self, epic, pos, motivo, base, hoy, bid):
        res = self.api.close_position(pos["dealId"])
        ref = res.get("dealReference")
        conf = self.api.confirm(ref) if ref else res
        if conf.get("dealStatus") != "ACCEPTED":
            self.avisar(f"{epic}: Capital.com rechazó el cierre ({conf.get('reason') or conf}). Reintento.")
            return "reintentar"
        salida = f(conf.get("level")) or bid
        ganancia = (salida - pos["precio"]) * pos["tamano"]
        self.log({**base, "accion": "vender", "precio": salida, "tamano": pos["tamano"],
                  "detalle": f"{motivo}; entrada {pos['precio']:,.1f}; resultado ≈ {ganancia:+,.2f} (moneda del índice)"})
        self.avisar(f"{epic}: VENDÍ {pos['tamano']} a {salida:,.1f} ({motivo}). Entrada {pos['precio']:,.1f} "
                    f"→ resultado ≈ {ganancia:+,.2f} USD antes de financiación.")
        del self.estado["posiciones"][epic]
        self.estado["ultimo_cierre"][epic] = str(hoy)
        self._decidido(epic, hoy)
        return "hecho"

    def _comprar(self, epic, det, motivo, base, hoy, s, pico, manuales, offer, bps):
        a = self.a

        def no_compra(razon, reintentar=False):
            self.log({**base, "accion": "señal_sin_compra", "detalle": f"{motivo}; {razon}"})
            self.avisar(f"{epic}: había señal de compra ({motivo}) pero NO compré: {razon}")
            if reintentar:
                return "reintentar"
            self._decidido(epic, hoy)
            return "hecho"

        if epic in manuales:
            return no_compra("ya tienes una posición abierta a mano en este índice")
        if pico > 0 and s["equity"] < pico * (1 - a.max_caida / 100):
            return no_compra(f"la cuenta cayó más de {a.max_caida:.0f}% desde su máximo ({pico:,.2f}). "
                             "Freno de seguridad: revisa qué pasó antes de seguir")
        if bps is not None and bps > a.max_spread_bps:
            return no_compra(f"spread de {bps:.1f} bps, más ancho que el máximo ({a.max_spread_bps}); "
                             "reintento en un minuto", reintentar=True)
        inst = det.get("instrument", {})
        fx = self.tipo_cambio(inst.get("currency", "USD"), s["moneda"])
        expo = min(a.exposicion, MAX_EXPOSICION) / len(a.lista_indices)
        step = rule(det, "minSizeIncrement") or 0.01
        tamano = round_down(s["equity"] * expo / (offer * fx), step)
        minimo = rule(det, "minDealSize")
        if tamano <= 0 or (minimo and tamano < minimo):
            return no_compra(f"con tu saldo el tamaño sería {tamano}, menor que el mínimo de Capital.com "
                             f"({minimo}). Sube --exposicion o el saldo")
        exposicion = tamano * offer * fx
        mf = f(inst.get("marginFactor"))
        if mf and inst.get("marginFactorUnit") == "PERCENTAGE" and exposicion * mf / 100 > 0.9 * s["disponible"]:
            return no_compra(f"el margen ({exposicion * mf / 100:,.2f}) no cabe en tu saldo disponible")
        dec = int(det.get("snapshot", {}).get("decimalPlacesFactor") or 1)
        stop = round(offer * (1 - a.stop / 100), dec)
        cuerpo = {"epic": epic, "direction": "BUY", "size": tamano, "guaranteedStop": False, "stopLevel": stop}
        info = {"fecha": str(hoy), "precio": offer, "tamano": tamano, "stop": stop}
        self.estado["pendiente"][epic] = info
        self.guardar()
        res = self.api.open_position(cuerpo)
        ref = res.get("dealReference")
        conf = self.api.confirm(ref) if ref else res
        if conf.get("dealStatus") != "ACCEPTED":
            del self.estado["pendiente"][epic]
            self.guardar()
            return no_compra(f"Capital.com rechazó la orden: {conf.get('reason') or conf}")
        deal = next((d.get("dealId") for d in conf.get("affectedDeals", []) if d.get("dealId")),
                    conf.get("dealId"))
        info.update(dealId=deal, precio=f(conf.get("level")) or offer)
        self.estado["posiciones"][epic] = info
        del self.estado["pendiente"][epic]
        self.log({**base, "accion": "comprar", "precio": info["precio"], "tamano": tamano,
                  "detalle": f"{motivo}; stop {stop}; exposición {exposicion:,.2f} {s['moneda']}; dealId {deal}"})
        self.avisar(f"{epic}: COMPRÉ {tamano} a {info['precio']:,.1f} ({motivo}). Stop {stop:,.1f} "
                    f"(-{a.stop:g}%). Exposición {exposicion:,.2f} {s['moneda']} en la cuenta {self.cuenta}.")
        self._decidido(epic, hoy)
        return "hecho"

    def ciclo(self, hoy):
        manuales = self.reconciliar(hoy)
        return [self.revisar(k, hoy, manuales) for k in self.a.lista_indices]


# ----------------------------------------------------------------------------
# Conexión con candado de cuenta real
# ----------------------------------------------------------------------------
def conectar(a):
    live = os.environ.get("CAPITAL_LIVE", "").strip().lower() in ("1", "true", "yes", "si", "sí")
    if live and not a.real:
        raise SystemExit("CAPITAL_LIVE=1 está puesto, pero el agente sólo usa la cuenta REAL si además le "
                         "pasas --real. Quita CAPITAL_LIVE para seguir en la DEMO.")
    if a.real and not live:
        raise SystemExit("--real necesita que TÚ pongas CAPITAL_LIVE=1 en tu terminal. Sin eso es la DEMO.")
    try:
        api = Capital.from_env()
    except CapitalError as e:
        raise SystemExit(f"No pude entrar a Capital.com: {e}")
    if api.live:
        print("⚠️  CUENTA REAL: el agente va a operar con dinero de verdad.")
    return api


def una_vez(a, api=None):
    ahora = ahora_ny()
    hoy = ahora.date()
    if hoy.year > ULTIMO_AÑO_CALENDARIO:
        print(f"Aviso: el calendario de festivos sólo llega a {ULTIMO_AÑO_CALENDARIO}; actualiza FESTIVOS.")
    if not es_dia_de_bolsa(hoy):
        print(f"{hoy}: la bolsa de Nueva York no abre hoy. Nada que hacer.")
        return 0
    ini, fin = ventana(hoy)
    if not ini <= ahora <= fin:
        print(f"Son las {ahora:%H:%M} en Nueva York. El agente decide entre {ini:%H:%M} y {fin:%H:%M} NY "
              f"({hora_local(ini)} en CDMX). Ahora no hace nada.")
        return 0
    ag = Agente(api or conectar(a), a)
    for intento in range(3):
        try:
            if all(r == "hecho" for r in ag.ciclo(hoy)):
                return 0
        except Exception as e:  # noqa: BLE001 - se reporta y se reintenta
            ag.avisar(f"error: {type(e).__name__}: {e}")
        if ahora_ny() > fin:
            break
        time.sleep(60)
    return 1


def bucle(a):
    print("Agente encendido. Déjalo así (Ctrl+C para apagarlo). Cuenta:",
          "REAL" if a.real else "DEMO", "| índices:", ", ".join(a.lista_indices))
    aviso_espera = None
    while True:
        ahora = ahora_ny()
        hoy = ahora.date()
        if es_dia_de_bolsa(hoy):
            ini, fin = ventana(hoy)
            if ini <= ahora <= fin:
                try:
                    ag = Agente(conectar(a), a)
                    if all(r == "hecho" for r in ag.ciclo(hoy)):
                        time.sleep(max(60, (fin - ahora_ny()).total_seconds() + 60))
                        continue
                except SystemExit:
                    raise
                except Exception as e:  # noqa: BLE001
                    avisar(f"error: {type(e).__name__}: {e} (reintento en 1 minuto)")
                time.sleep(60)
                continue
        prox = proxima_ventana(ahora)
        if prox is not None and prox != aviso_espera:
            print(f"Esperando hasta {prox:%a %d/%m %H:%M} NY ({hora_local(prox)} en CDMX).", flush=True)
            aviso_espera = prox
        time.sleep(30)


# ----------------------------------------------------------------------------
# Simular, probar y backtest
# ----------------------------------------------------------------------------
def simular(a):
    print("SIMULACIÓN: precios de Yahoo, no entra a Capital.com, no envía nada.\n")
    for k in a.lista_indices:
        cfg = INDICES[k]
        c = descargar_diario(cfg["yahoo"], period="2y")["Close"]
        accion, ind, motivo = decidir(c)
        expo = min(a.exposicion, MAX_EXPOSICION) / len(a.lista_indices)
        print(f"{cfg['nombre']} — último precio {ind['precio']:,.1f} ({c.index[-1]})")
        print(f"  RSI(2) {ind['rsi2']:.1f} | media 200: {ind['sma200']:,.1f} "
              f"({ind['precio'] / ind['sma200'] - 1:+.1%}) | media 5: {ind['sma5']:,.1f}")
        print(f"  → {accion.upper()}: {motivo}")
        if accion == "comprar":
            print(f"  Con {a.cuenta:,.0f} USD compraría ~{a.cuenta * expo / ind['precio']:.3f} contratos "
                  f"(exposición {a.cuenta * expo:,.0f} USD) con stop en {ind['precio'] * (1 - a.stop / 100):,.1f}.")
        print()
    print("La decisión real se toma a las 15:45 de Nueva York con el precio de ese momento.")
    return 0


def probar(a):
    api = conectar(a)
    ag = Agente(api, a)
    s = ag.saldo()
    print(f"Conectado a la cuenta {ag.cuenta}: saldo {s['equity']:,.2f} {s['moneda']} "
          f"(disponible {s['disponible']:,.2f}).")
    manuales = {p.get("market", {}).get("epic") for p in api.positions()}
    hoy = ahora_ny().date()
    for k in a.lista_indices:
        cfg = INDICES[k]
        det = api.market(cfg["epic"])
        bid, offer, mid, spread, bps, status = snapshot(det)
        if not mid:
            print(f"\n{cfg['epic']}: sin precio ahora ({status}). Prueba en horario de mercado.")
            continue
        inst = det.get("instrument", {})
        fx = ag.tipo_cambio(inst.get("currency", "USD"), s["moneda"])
        expo = min(a.exposicion, MAX_EXPOSICION) / len(a.lista_indices)
        step = rule(det, "minSizeIncrement") or 0.01
        tam = round_down(s["equity"] * expo / (offer * fx), step)
        print(f"\n{cfg['epic']}: {status}, compra {offer} / venta {bid}, spread {bps or 0:.2f} bps "
              f"(máximo permitido {a.max_spread_bps})")
        print(f"  Tamaño mínimo {rule(det, 'minDealSize')}, paso {step}, margen {inst.get('marginFactor')}"
              f"{'%' if inst.get('marginFactorUnit') == 'PERCENTAGE' else ''}")
        print(f"  Si hoy hubiera señal compraría {tam} contratos (exposición {tam * (offer or 0) * fx:,.2f} "
              f"{s['moneda']}) con stop en {(offer or 0) * (1 - a.stop / 100):,.1f}")
        if cfg["epic"] in manuales:
            print("  Ya tienes una posición en este índice: el agente no abriría otra.")
        try:
            hist = historia_yahoo(cfg["yahoo"], hoy)
            accion, ind, motivo = decidir(pd.concat([hist, pd.Series([mid], index=[hoy])]))
            print(f"  Con el precio de ahora: {accion.upper()} — {motivo}")
        except Exception as e:  # noqa: BLE001
            print(f"  (sin datos de Yahoo: {e})")
    print("\nNo se envió ninguna orden. Todo listo para dejarlo encendido.")
    return 0


def backtest_indice(df, a):
    """Simula día a día con decidir(), el stop en el mínimo del día y los costes de CFD."""
    c, lo, op = df["Close"].to_numpy(), df["Low"].to_numpy(), df["Open"].to_numpy()
    fechas = list(df.index)
    serie = pd.Series(c, index=fechas)
    ops, pos, ultimo = [], None, None
    for i in range(SMA_TEND + 60, len(df)):
        hoy = fechas[i]
        if pos and lo[i] <= pos["stop"]:
            salida = min(op[i], pos["stop"])          # si abre por debajo del stop, sale a la apertura
            ops.append((pos["fecha"], hoy, pos["precio"], salida, "stop"))
            pos, ultimo = None, hoy
            continue
        accion, _, _ = decidir(serie.iloc[i - 260:i + 1], pos["fecha"] if pos else None, ultimo)
        if accion == "vender":
            ops.append((pos["fecha"], hoy, pos["precio"], c[i], "regla"))
            pos, ultimo = None, hoy
        elif accion == "comprar":
            pos = {"fecha": hoy, "precio": c[i], "stop": c[i] * (1 - a.stop / 100)}
    o = pd.DataFrame(ops, columns=["entrada", "salida", "p_ent", "p_sal", "tipo"])
    noches = np.array([(s - e).days for e, s in zip(o["entrada"], o["salida"])])
    o["ret"] = o["p_sal"] / o["p_ent"] - 1 - 2 * a.coste_bps / 1e4 - a.financiacion / 100 * noches / 365
    return o


def pf(r):
    p = -r[r <= 0].sum()
    return r[r > 0].sum() / p if p > 0 else float("inf")


def backtest(a):
    expo = min(a.exposicion, MAX_EXPOSICION) / len(a.lista_indices)
    print(f"BACKTEST de las reglas del agente: RSI(2) < {RSI_MAX} sobre la media de {SMA_TEND}, salida sobre "
          f"la media de {SMA_SALIDA} o a {MAX_DIAS} días, stop -{a.stop:g}%.")
    print(f"Costes: spread {2 * a.coste_bps:.1f} bps por operación + financiación {a.financiacion:g}%/año. "
          f"Exposición {expo:g} veces el saldo por índice.\n")
    for k in a.lista_indices:
        cfg = INDICES[k]
        df = descargar_diario(cfg["yahoo"], start="1995-01-01")
        o = backtest_indice(df, a)
        r = o["ret"]
        cuenta = (1 + expo * r).cumprod()
        años = (o["salida"].iloc[-1] - o["entrada"].iloc[0]).days / 365.25
        cagr = cuenta.iloc[-1] ** (1 / años) - 1
        dd = (cuenta / cuenta.cummax() - 1).min()
        mitad = len(o) // 2
        bh = df["Close"].loc[o["entrada"].iloc[0]:]
        bh_cagr = (bh.iloc[-1] / bh.iloc[0]) ** (1 / años) - 1
        bh_dd = (bh / bh.cummax() - 1).min()
        print(f"=== {cfg['nombre']} ({o['entrada'].iloc[0]} → {o['salida'].iloc[-1]}) ===")
        print(f"Operaciones: {len(o)} (~{len(o) / años:.0f} al año)   Acierto: {(r > 0).mean():.1%}   "
              f"Profit factor: {pf(r):.2f} (mitades {pf(r[:mitad]):.2f} / {pf(r[mitad:]):.2f})")
        print(f"Media por operación: {r.mean():+.2%}   Gana en promedio {r[r > 0].mean():+.2%}, "
              f"pierde {r[r <= 0].mean():+.2%}   Peor: {r.min():+.2%}   Stops: {(o['tipo'] == 'stop').sum()}")
        print(f"Tu cuenta con exposición {expo:g}: {cagr:+.1%} al año, peor caída {dd:.1%}")
        print(f"Comparación, comprar y mantener el índice: {bh_cagr:+.1%} al año, peor caída {bh_dd:.1%}")
        dias_inv = sum((s - e).days for e, s in zip(o["entrada"], o["salida"]))
        print(f"El agente sólo tiene dinero invertido ~{dias_inv / (años * 365.25):.0%} del tiempo.")
        o["año"] = [e.year for e in o["entrada"]]
        tabla = o[o["año"] >= 2016].groupby("año")["ret"]
        print("Por año (resultado en tu cuenta):  " + "  ".join(
            f"{y}: {(np.prod(1 + expo * g) - 1):+.1%}" for y, g in tabla))
        print()
    print("Cómo leerlo: el agente acierta ~7 de cada 10 veces, pero opera poco y gana poco por operación.")
    print("Gana MENOS que comprar y mantener en años buenos; su ventaja es estar fuera del mercado la")
    print("mayor parte del tiempo. Pruébalo meses en la DEMO y compara con estos números antes de pensar")
    print("en dinero real. Resultados pasados no garantizan los futuros. No es asesoría financiera.")
    return 0


# ----------------------------------------------------------------------------
# Programador de tareas de Windows
# ----------------------------------------------------------------------------
def horas_tarea():
    """15:46 de Nueva York en la hora de esta PC, en invierno y en verano."""
    horas = set()
    for d in (datetime.date(2026, 1, 15), datetime.date(2026, 7, 15)):
        ts = pd.Timestamp(datetime.datetime.combine(d, HORA_TAREA), tz=NY).to_pydatetime().astimezone()
        horas.add(ts.strftime("%H:%M"))
    return sorted(horas)


def nombre_tarea(h):
    return f"Agente RSI2 ({h.replace(':', '-')})"


def instalar_windows(a, desinstalar=False):
    if os.name != "nt":
        print("Esto sólo funciona en Windows. En Mac/Linux usa cron con: python agente.py --una-vez")
        return 1
    bat = os.path.join(AQUI, "agente_windows.bat")
    extra = f"--indices {a.indices} --exposicion {a.exposicion:g} --stop {a.stop:g}" + (" --real" if a.real else "")
    for h in horas_tarea():
        nombre = nombre_tarea(h)
        if desinstalar:
            subprocess.run(["schtasks", "/Delete", "/F", "/TN", nombre], check=False)
            continue
        r = subprocess.run(["schtasks", "/Create", "/F", "/TN", nombre, "/TR", f'"{bat}" {extra}',
                            "/SC", "WEEKLY", "/D", "MON,TUE,WED,THU,FRI", "/ST", h],
                           capture_output=True, text=True)
        print(f"Tarea '{nombre}' a las {h} de lunes a viernes:", "creada" if r.returncode == 0 else r.stderr.strip())
    if not desinstalar:
        print("\nUna de las dos tareas cae a las 15:46 de Nueva York (la otra es para el cambio de horario")
        print("de EE. UU. y no hace nada). La PC debe estar prendida y con tu sesión abierta a esa hora.")
        print("Lo que haga queda en registro_agente\\agente.log. Para quitarlas: python agente.py --desinstalar-windows")
        if a.real:
            print("⚠️  Las tareas usan la cuenta REAL (--real). Necesitan CAPITAL_LIVE=1 en tus variables de entorno.")
    return 0


# ----------------------------------------------------------------------------
def main(argv=None):
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Agente automático de rebote RSI(2) para Capital.com (DEMO)")
    modo = ap.add_mutually_exclusive_group()
    modo.add_argument("--backtest", action="store_true", help="resultados históricos de estas reglas")
    modo.add_argument("--simular", action="store_true", help="qué haría hoy, sin claves ni órdenes")
    modo.add_argument("--probar", action="store_true", help="conecta a Capital.com y calcula, sin órdenes")
    modo.add_argument("--una-vez", action="store_true", help="revisa ahora y termina")
    modo.add_argument("--instalar-windows", action="store_true", help="crea las tareas programadas")
    modo.add_argument("--desinstalar-windows", action="store_true", help="borra las tareas programadas")
    ap.add_argument("--indices", default="us500", help="us500, us100 o us500,us100")
    ap.add_argument("--exposicion", type=float, default=1.0,
                    help="tamaño de la posición en veces tu saldo (1 = sin apalancamiento; máx. 2). "
                         "Con varios índices se reparte entre ellos")
    ap.add_argument("--stop", type=float, default=5.0, help="stop de seguridad en %% bajo la entrada")
    ap.add_argument("--max-spread-bps", type=float, default=5.0, help="no compra si el spread es mayor")
    ap.add_argument("--max-caida", type=float, default=20.0,
                    help="deja de abrir operaciones si la cuenta cae este %% desde su máximo")
    ap.add_argument("--real", action="store_true", help="cuenta REAL (además necesita CAPITAL_LIVE=1)")
    ap.add_argument("--cuenta", type=float, default=1000, help="saldo supuesto para --simular (USD)")
    ap.add_argument("--coste-bps", type=float, default=0.55, help="backtest: coste por lado (spread/2)")
    ap.add_argument("--financiacion", type=float, default=8.0, help="backtest: financiación anual %%")
    a = ap.parse_args(argv)
    a.lista_indices = [x.strip().lower() for x in a.indices.split(",") if x.strip()]
    malos = [x for x in a.lista_indices if x not in INDICES]
    if malos or not a.lista_indices:
        ap.error(f"índices válidos: {', '.join(INDICES)}")
    if a.exposicion > MAX_EXPOSICION:
        print(f"Aviso: exposición limitada a {MAX_EXPOSICION:g} veces el saldo.")
    if not 0.5 <= a.stop <= 20:
        ap.error("--stop debe estar entre 0.5 y 20 (%)")
    os.chdir(AQUI)

    if a.backtest:
        return backtest(a)
    if a.simular:
        return simular(a)
    if a.probar:
        return probar(a)
    if a.instalar_windows or a.desinstalar_windows:
        return instalar_windows(a, desinstalar=a.desinstalar_windows)
    if a.una_vez:
        return una_vez(a)
    try:
        bucle(a)
    except KeyboardInterrupt:
        print("\nAgente apagado. Las posiciones abiertas siguen en Capital.com con su stop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
