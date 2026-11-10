"""Ejecución de órdenes.

  * PaperBroker -> simulación (por defecto). Reproduce exactamente las reglas del backtest.
  * MT5Broker   -> MetaTrader 5 (solo Windows, requiere el paquete MetaTrader5 y un terminal abierto).
  * CapitalBroker (scalper/capital.py) -> API REST de Capital.com.

Interfaz común que usa el agente:
    equity() -> float                 capital actual de la cuenta
    flat() -> bool                    no hay posición ni orden pendiente
    submit(order) -> bool             envía una orden a mercado con TP/SL y caducidad
    on_bar(t, o, h, l, c) -> list     procesa una barra cerrada; devuelve las operaciones cerradas
    close_all(t, price, reason) -> list
    spread() -> float | None          spread actual en unidades de precio, si se conoce
"""
from __future__ import annotations

import math
import time as _time
from dataclasses import asdict, dataclass

import pandas as pd

from .config import Market


@dataclass
class Order:
    side: int  # +1 largo, -1 corto
    units: float  # cantidad de subyacente (oro: onzas)
    width: float  # distancia de TP y SL respecto al precio de entrada
    horizon: int  # barras máximas en posición
    signal_time: pd.Timestamp
    pred: float = 0.0


@dataclass
class Position:
    side: int
    units: float
    entry: float
    tp: float
    sl: float
    width: float
    horizon: int
    entry_time: pd.Timestamp
    signal_time: pd.Timestamp
    pred: float = 0.0
    bars_held: int = 0
    ticket: int | str | None = None


def closed_trade(pos: Position, t, exit_price: float, cost: float, reason: str, net_money: float | None = None) -> dict:
    gross = pos.side * (exit_price - pos.entry)
    return dict(
        time=pos.signal_time, entry_time=pos.entry_time, exit_time=t, side=pos.side, units=pos.units,
        entry=pos.entry, exit=exit_price, width=pos.width, bars=pos.bars_held, pred=pos.pred,
        gross=gross, net=gross - cost, reason=reason,
        net_money=(gross - cost) * pos.units if net_money is None else net_money,
    )


class PaperBroker:
    """Cuenta simulada. La orden se ejecuta en la apertura de la barra siguiente a la señal."""

    name = "paper"

    def __init__(self, equity: float, cost: float):
        self._equity = float(equity)
        self.cost = cost
        self.pending: Order | None = None
        self.position: Position | None = None

    def equity(self) -> float:
        return self._equity

    def flat(self) -> bool:
        return self.pending is None and self.position is None

    def spread(self) -> float | None:
        return None

    def submit(self, order: Order) -> bool:
        if not self.flat():
            return False
        self.pending = order
        return True

    def _close(self, t, price: float, reason: str) -> dict:
        tr = closed_trade(self.position, t, price, self.cost, reason)
        self._equity += tr["net_money"]
        self.position = None
        return tr

    def on_bar(self, t, o: float, h: float, l: float, c: float) -> list[dict]:
        if self.pending is not None:
            od, self.pending = self.pending, None
            self.position = Position(od.side, od.units, o, o + od.side * od.width, o - od.side * od.width,
                                     od.width, od.horizon, t, od.signal_time, od.pred)
        pos = self.position
        if pos is None:
            return []
        pos.bars_held += 1
        # Convención conservadora del backtest: si la barra toca TP y SL, cuenta el SL.
        hit_sl = l <= pos.sl if pos.side > 0 else h >= pos.sl
        hit_tp = h >= pos.tp if pos.side > 0 else l <= pos.tp
        if hit_sl:
            return [self._close(t, pos.sl, "sl")]
        if hit_tp:
            return [self._close(t, pos.tp, "tp")]
        if pos.bars_held >= pos.horizon:
            return [self._close(t, c, "tiempo")]
        return []

    def close_all(self, t, price: float, reason: str) -> list[dict]:
        self.pending = None
        return [self._close(t, price, reason)] if self.position is not None else []

    # --- persistencia (para reanudar tras un reinicio)
    def dump(self) -> dict:
        ser = lambda d: {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in asdict(d).items()}
        return dict(equity=self._equity, pending=ser(self.pending) if self.pending else None,
                    position=ser(self.position) if self.position else None)

    def restore(self, d: dict) -> None:
        ts = lambda x, keys: {k: (pd.Timestamp(v) if k in keys else v) for k, v in x.items()}
        self._equity = d["equity"]
        self.pending = Order(**ts(d["pending"], {"signal_time"})) if d.get("pending") else None
        self.position = Position(**ts(d["position"], {"entry_time", "signal_time"})) if d.get("position") else None


def round_lots(units: float, mk: Market) -> float:
    """Redondea hacia abajo al incremento de lote; 0 si no llega al lote mínimo."""
    lots = math.floor(units / mk.lot_size / mk.lot_step + 1e-9) * mk.lot_step
    return round(lots, 8) if lots >= mk.min_lot - 1e-12 else 0.0


class MT5Broker:
    """Ejecución real a través de MetaTrader 5.

    AVISO: esta clase solo se ha probado contra un doble de pruebas del módulo MetaTrader5, no
    contra un terminal real. Por defecto se niega a operar si la cuenta no es de demostración.
    """

    MAGIC = 20261110
    name = "mt5"

    def __init__(self, market: Market, symbol: str | None = None, allow_real: bool = False,
                 deviation: int = 20, mt5=None):
        if mt5 is None:
            import MetaTrader5 as mt5  # type: ignore  # solo existe en Windows
        self.mt5, self.market, self.symbol, self.deviation = mt5, market, symbol or market.key, deviation
        if not mt5.initialize():
            raise RuntimeError(f"No se pudo iniciar MetaTrader 5: {mt5.last_error()}")
        acc = mt5.account_info()
        if acc is None:
            raise RuntimeError("MetaTrader 5 no tiene ninguna cuenta conectada")
        if acc.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO and not allow_real:
            raise RuntimeError("La cuenta no es demo. Para operar con dinero real hay que pasar --allow-real.")
        if not mt5.symbol_select(self.symbol, True):
            raise RuntimeError(f"Símbolo {self.symbol} no disponible en el bróker")
        self.position: Position | None = None
        self._adopt_open_position()

    # --- utilidades
    def _info(self):
        return self.mt5.symbol_info(self.symbol)

    def _server_offset(self) -> int:
        """Desfase (s) entre la hora del servidor del bróker y UTC, redondeado a 30 min."""
        tick = self.mt5.symbol_info_tick(self.symbol)
        return int(round((tick.time - _time.time()) / 1800.0)) * 1800

    def _my_positions(self):
        return [p for p in (self.mt5.positions_get(symbol=self.symbol) or []) if p.magic == self.MAGIC]

    def _adopt_open_position(self) -> None:
        """Tras un reinicio, recupera la posición que este agente dejó abierta."""
        mine = self._my_positions()
        if mine:
            p = mine[0]
            side = 1 if p.type == self.mt5.POSITION_TYPE_BUY else -1
            t = pd.Timestamp(p.time - self._server_offset(), unit="s", tz="UTC")
            width = abs(p.tp - p.price_open) if p.tp else 0.0
            self.position = Position(side, p.volume * self._info().trade_contract_size, p.price_open, p.tp, p.sl,
                                     width, 0, t, t, ticket=p.ticket)

    # --- datos
    def bars(self, n: int = 2500) -> pd.DataFrame:
        """Últimas `n` barras M1 cerradas (la barra en formación se omite)."""
        rates = self.mt5.copy_rates_from_pos(self.symbol, self.mt5.TIMEFRAME_M1, 1, n)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Sin barras de {self.symbol}: {self.mt5.last_error()}")
        df = pd.DataFrame(rates)
        idx = pd.to_datetime(df["time"] - self._server_offset(), unit="s", utc=True).rename("time")
        return pd.DataFrame(df[["open", "high", "low", "close"]].to_numpy(float), index=pd.DatetimeIndex(idx),
                            columns=["open", "high", "low", "close"])

    # --- interfaz de bróker
    def equity(self) -> float:
        return float(self.mt5.account_info().equity)

    def flat(self) -> bool:
        return self.position is None and not self._my_positions()

    def spread(self) -> float | None:
        tick = self.mt5.symbol_info_tick(self.symbol)
        return float(tick.ask - tick.bid) if tick else None

    def _send(self, side: int, lots: float, sl: float = 0.0, tp: float = 0.0, ticket: int | None = None):
        mt5 = self.mt5
        tick = mt5.symbol_info_tick(self.symbol)
        req = dict(action=mt5.TRADE_ACTION_DEAL, symbol=self.symbol, volume=lots,
                   type=mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL,
                   price=tick.ask if side > 0 else tick.bid, deviation=self.deviation, magic=self.MAGIC,
                   comment="scalper", type_time=mt5.ORDER_TIME_GTC, type_filling=mt5.ORDER_FILLING_IOC)
        if ticket is not None:
            req["position"] = ticket
        else:
            req.update(sl=sl, tp=tp)
        res = mt5.order_send(req)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Orden rechazada: {getattr(res, 'retcode', None)} {getattr(res, 'comment', mt5.last_error())}")
        return res

    def submit(self, order: Order) -> bool:
        if not self.flat():
            return False
        info = self._info()
        lots = round_lots(order.units, Market(self.market.key, "", "", "", 0.0, info.trade_contract_size,
                                              info.volume_step, info.volume_min))
        if lots <= 0:
            return False
        tick = self.mt5.symbol_info_tick(self.symbol)
        ref = tick.ask if order.side > 0 else tick.bid
        digits = info.digits
        res = self._send(order.side, lots, sl=round(ref - order.side * order.width, digits),
                         tp=round(ref + order.side * order.width, digits))
        entry = float(res.price) or ref
        self.position = Position(order.side, lots * info.trade_contract_size, entry, entry + order.side * order.width,
                                 entry - order.side * order.width, order.width, order.horizon,
                                 pd.Timestamp.now(tz="UTC"), order.signal_time, order.pred, ticket=res.order)
        return True

    def _realized(self, pos: Position) -> tuple[float, float]:
        """Precio de salida y resultado en dinero (con comisión y swap) según el historial del bróker."""
        deals = self.mt5.history_deals_get(position=pos.ticket) or []
        out = [d for d in deals if d.entry == self.mt5.DEAL_ENTRY_OUT]
        money = sum(d.profit + d.commission + d.swap for d in deals)
        price = out[-1].price if out else pos.entry
        return float(price), float(money)

    def _finish(self, t, reason: str) -> dict:
        pos, self.position = self.position, None
        price, money = self._realized(pos)
        return closed_trade(pos, t, price, 0.0, reason, net_money=money)

    def on_bar(self, t, o: float, h: float, l: float, c: float) -> list[dict]:
        pos = self.position
        if pos is None:
            return []
        pos.bars_held += 1
        if not self._my_positions():  # el bróker ya la cerró por TP o SL
            tr = self._finish(t, "tp/sl")
            tr["reason"] = "tp" if tr["gross"] > 0 else "sl"
            return [tr]
        expired = pos.horizon and (pd.Timestamp.now(tz="UTC") - pos.entry_time).total_seconds() >= pos.horizon * 60
        if pos.bars_held >= max(pos.horizon, 1) or expired:
            return self.close_all(t, c, "tiempo")
        return []

    def close_all(self, t, price: float, reason: str) -> list[dict]:
        for p in self._my_positions():
            side = 1 if p.type == self.mt5.POSITION_TYPE_BUY else -1
            self._send(-side, p.volume, ticket=p.ticket)
        return [self._finish(t, reason)] if self.position is not None else []
