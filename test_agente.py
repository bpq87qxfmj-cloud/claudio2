"""Pruebas sin internet del agente (API de Capital.com simulada).

    python test_agente.py
"""

import datetime
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agente  # noqa: E402

HOY = datetime.date(2026, 11, 5)


def dias_habiles(fin, n):
    out, d = [], fin
    while len(out) < n:
        if agente.es_dia_de_bolsa(d):
            out.append(d)
        d -= datetime.timedelta(days=1)
    return out[::-1]


def historia(final):
    """300 cierres en tendencia alcista que terminan con los valores de `final`, hasta AYER."""
    fechas = dias_habiles(agente.dia_anterior_de_bolsa(HOY), 300)
    base = list(np.linspace(5000, 6800, 300 - len(final))) + list(final)
    return pd.Series(base, index=fechas, dtype=float)


CAIDA = [6800, 6790, 6770, 6740, 6700]           # 4 días bajando: RSI(2) muy bajo


class FakeApi:
    def __init__(self, live=False, moneda="USD", bid=6600.0, offer=6600.6, status="TRADEABLE"):
        self.live = live
        self.moneda = moneda
        self.bid, self.offer, self.status = bid, offer, status
        self.abiertas, self.enviadas, self.cerradas = [], [], []
        self.rechazar = False

    def accounts(self):
        return [{"preferred": True, "currency": self.moneda,
                 "balance": {"balance": 1000.0, "profitLoss": 0.0, "available": 1000.0}}]

    def market(self, epic):
        if epic == "USDMXN":
            return {"snapshot": {"bid": 18.0, "offer": 18.0}}
        if epic in ("MXNUSD",):
            raise agente.CapitalError("404")
        return {"instrument": {"currency": "USD", "marginFactor": 5, "marginFactorUnit": "PERCENTAGE"},
                "dealingRules": {"minDealSize": {"value": 0.01}, "minSizeIncrement": {"value": 0.01}},
                "snapshot": {"bid": self.bid, "offer": self.offer, "marketStatus": self.status,
                             "decimalPlacesFactor": 1}}

    def positions(self):
        return self.abiertas

    def open_position(self, body):
        self.enviadas.append(body)
        if self.rechazar:
            return {"dealReference": "R0"}
        self.abiertas.append({"position": {"dealId": "D1", "direction": "BUY", "size": body["size"],
                                           "level": self.offer}, "market": {"epic": body["epic"]}})
        return {"dealReference": "R1"}

    def close_position(self, deal_id):
        self.cerradas.append(deal_id)
        self.abiertas = [p for p in self.abiertas if p["position"]["dealId"] != deal_id]
        return {"dealReference": "R2"}

    def confirm(self, ref):
        if ref == "R0":
            return {"dealStatus": "REJECTED", "reason": "RISK_CHECK"}
        if ref == "R1":
            return {"dealStatus": "ACCEPTED", "level": self.offer, "affectedDeals": [{"dealId": "D1"}]}
        return {"dealStatus": "ACCEPTED", "level": self.bid}


def args(**kw):
    a = agente.argparse.Namespace(exposicion=1.0, stop=5.0, max_spread_bps=5.0, max_caida=20.0,
                                  real=False, lista_indices=["us500"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        agente.DIR = self.tmp
        self.avisos, self.filas = [], []

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def nuevo(self, api, final=CAIDA, **kw):
        return agente.Agente(api, args(**kw), historia=lambda t, h: historia(final),
                             aviso=lambda t, telegram=True: self.avisos.append(t),
                             log=self.filas.append)


class TestRegla(unittest.TestCase):
    def test_compra_con_rsi_bajo_en_tendencia(self):
        c = pd.concat([historia(CAIDA), pd.Series([6600.0], index=[HOY])])
        accion, ind, _ = agente.decidir(c)
        self.assertEqual(accion, "comprar")
        self.assertLess(ind["rsi2"], 10)

    def test_no_compra_en_tendencia_bajista(self):
        c = pd.Series(np.linspace(7000, 5000, 300), index=dias_habiles(HOY, 300))
        self.assertEqual(agente.decidir(c)[0], "esperar")

    def test_vende_sobre_media_5_y_por_tiempo(self):
        h = historia(CAIDA)
        entrada = h.index[-1]
        sube = pd.concat([h, pd.Series([6900.0], index=[HOY])])
        self.assertEqual(agente.decidir(sube, entrada)[0], "vender")
        baja = pd.concat([h, pd.Series([6500.0], index=[HOY])])
        self.assertEqual(agente.decidir(baja, entrada)[0], "mantener")
        vieja = h.index[-11]
        self.assertEqual(agente.decidir(baja, vieja)[0], "vender")

    def test_calendario(self):
        self.assertFalse(agente.es_dia_de_bolsa(datetime.date(2026, 11, 26)))   # Thanksgiving
        self.assertEqual(agente.dia_anterior_de_bolsa(datetime.date(2026, 11, 30)), datetime.date(2026, 11, 27))
        ini, fin = agente.ventana(datetime.date(2026, 11, 27))                   # cierra a la 1 pm
        self.assertEqual((ini.hour, fin.hour), (12, 12))
        ini, _ = agente.ventana(HOY)
        self.assertEqual(ini.tz_convert("America/Mexico_City").strftime("%H:%M"), "14:44")


class TestAgente(Base):
    def test_compra_con_stop_y_tamano(self):
        api = FakeApi()
        ag = self.nuevo(api)
        self.assertEqual(ag.ciclo(HOY), ["hecho"])
        body = api.enviadas[0]
        self.assertEqual(body["direction"], "BUY")
        self.assertAlmostEqual(body["size"], 0.15)                 # 1000 USD / 6600.6
        self.assertAlmostEqual(body["stopLevel"], round(6600.6 * 0.95, 1))
        self.assertEqual(ag.estado["posiciones"]["US500"]["dealId"], "D1")
        self.assertIn("COMPRÉ", self.avisos[-1])

    def test_no_repite_el_mismo_dia_ni_promedia(self):
        api = FakeApi()
        ag = self.nuevo(api)
        ag.ciclo(HOY)
        ag.ciclo(HOY)
        self.assertEqual(len(api.enviadas), 1)
        ag2 = self.nuevo(api)                                       # otro proceso, mismo estado
        ag2.ciclo(HOY)
        self.assertEqual(len(api.enviadas), 1)

    def test_no_toca_posiciones_manuales(self):
        api = FakeApi()
        api.abiertas.append({"position": {"dealId": "MIA", "direction": "BUY", "size": 1},
                             "market": {"epic": "US500"}})
        self.nuevo(api).ciclo(HOY)
        self.assertEqual(api.enviadas, [])
        self.assertEqual(api.cerradas, [])

    def test_vende_lo_suyo_al_rebotar(self):
        api = FakeApi()
        ag = self.nuevo(api)
        ag.ciclo(HOY)
        manana = datetime.date(2026, 11, 6)
        api.bid, api.offer = 6950.0, 6950.6
        ag.historia = lambda t, h: pd.concat([historia(CAIDA), pd.Series([6600.0], index=[HOY])])
        self.assertEqual(ag.ciclo(manana), ["hecho"])
        self.assertEqual(api.cerradas, ["D1"])
        self.assertNotIn("US500", ag.estado["posiciones"])
        self.assertIn("VENDÍ", self.avisos[-1])

    def test_detecta_stop(self):
        api = FakeApi()
        ag = self.nuevo(api)
        ag.ciclo(HOY)
        api.abiertas.clear()                                         # saltó el stop
        ag.reconciliar(datetime.date(2026, 11, 6))
        self.assertEqual(ag.estado["posiciones"], {})
        self.assertIn("ya no está abierta", self.avisos[-1])

    def test_spread_ancho_y_mercado_cerrado(self):
        api = FakeApi(bid=6590.0, offer=6610.0)                      # 30 bps
        self.assertEqual(self.nuevo(api).ciclo(HOY), ["reintentar"])
        api = FakeApi(status="CLOSED")
        self.assertEqual(self.nuevo(api).ciclo(HOY), ["reintentar"])
        self.assertEqual(api.enviadas, [])

    def test_freno_por_caida(self):
        api = FakeApi()
        ag = self.nuevo(api)
        ag.estado["pico_saldo"] = 2000.0                             # la cuenta bajó 50%
        ag.ciclo(HOY)
        self.assertEqual(api.enviadas, [])
        self.assertIn("Freno de seguridad", self.avisos[-1])

    def test_cuenta_en_pesos(self):
        api = FakeApi(moneda="MXN")                                  # 1000 MXN ≈ 55 USD
        self.nuevo(api).ciclo(HOY)
        self.assertEqual(api.enviadas, [])                           # menor que el mínimo
        self.assertIn("menor que el mínimo", self.avisos[-1])

    def test_rechazo(self):
        api = FakeApi()
        api.rechazar = True
        ag = self.nuevo(api)
        ag.ciclo(HOY)
        self.assertEqual(ag.estado["posiciones"], {})
        self.assertEqual(ag.estado["pendiente"], {})
        self.assertIn("rechazó", self.avisos[-1])

    def test_exposicion_maxima(self):
        api = FakeApi()
        self.nuevo(api, exposicion=10).ciclo(HOY)
        self.assertAlmostEqual(api.enviadas[0]["size"], 0.30)        # tope 2x


class TestCandadoReal(unittest.TestCase):
    def test_live_sin_flag(self):
        os.environ["CAPITAL_LIVE"] = "1"
        try:
            with self.assertRaises(SystemExit) as e:
                agente.conectar(args(real=False))
            self.assertIn("--real", str(e.exception))
        finally:
            del os.environ["CAPITAL_LIVE"]

    def test_flag_sin_live(self):
        os.environ.pop("CAPITAL_LIVE", None)
        with self.assertRaises(SystemExit):
            agente.conectar(args(real=True))


class TestBacktest(unittest.TestCase):
    def test_misma_regla_que_en_vivo(self):
        fechas = dias_habiles(HOY, 600)
        rng = np.random.default_rng(0)
        c = 5000 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 600)))
        df = pd.DataFrame({"Open": c, "High": c * 1.003, "Low": c * 0.997, "Close": c}, index=fechas)
        o = agente.backtest_indice(df, args(coste_bps=0.55, financiacion=8.0))
        self.assertGreater(len(o), 3)
        self.assertTrue((o["salida"] > o["entrada"]).all())
        self.assertTrue((o["entrada"].iloc[1:].to_numpy() > o["salida"].iloc[:-1].to_numpy()).all())


if __name__ == "__main__":
    unittest.main(verbosity=1)
