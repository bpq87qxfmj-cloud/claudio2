"""CLI:  python -m scalper <comando>

  download                 descarga el histórico M1 de los mercados candidatos
  compare                  walk-forward en los tres mercados (periodo de diseño) y ranking
  lockbox --market M       evaluación final en los meses reservados
  train --market M         entrena el modelo final y lo guarda en models/
  predict --market M       señal para la próxima barra con datos recientes de Yahoo (o --csv)
  replay --market M        el agente autónomo revive un periodo histórico barra a barra
  agent --market M         agente autónomo en vivo (simulado por defecto; --broker mt5 para MetaTrader 5)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
import warnings

import pandas as pd

from . import model, report
from .config import MARKETS, PARAMS, REPORT_DIR, STATE_DIR
from .data import download_histdata, download_yahoo, load_csv, load_history

warnings.filterwarnings("ignore", category=UserWarning)


def cmd_download(a):
    for m in a.markets:
        print(m)
        df = download_histdata(m, start_year=a.start_year, force=a.force)
        print(f"  total {len(df):,} barras  {df.index[0]} -> {df.index[-1]}")


def cmd_compare(a):
    results = {m: report.evaluate(m, PARAMS, None, PARAMS.lockbox_start, verbose=a.verbose) for m in a.markets}
    table = pd.DataFrame([report.summary_row(r) for r in results.values()])
    print(table.to_string(index=False))
    REPORT_DIR.mkdir(exist_ok=True)
    report.plot_equity({f"{m} · coste {r['cost']:g}": r for m, r in results.items()},
                       f"Walk-forward fuera de muestra, {PARAMS.first_test} a {PARAMS.lockbox_start} (excl.)",
                       REPORT_DIR / "design_equity.png")
    report.save_json({m: dict(summary=report.summary_row(r), edge=r["edge"].reset_index().to_dict("records"),
                              months=r["months"].to_dict("records")) for m, r in results.items()},
                     REPORT_DIR / "design.json")
    for m, r in results.items():
        print(f"\n{m}: ventaja bruta por cuantil de |predicción| (coste {r['cost']:g})")
        print(r["edge"].round(3).to_string())


def cmd_lockbox(a):
    out = {}
    for m in a.markets:
        r = report.evaluate(m, PARAMS, PARAMS.lockbox_start, None, cost=a.cost, verbose=True)
        out[m] = r
        print(pd.DataFrame([report.summary_row(r)]).to_string(index=False))
        print(r["edge"].round(3).to_string())
    report.plot_equity({f"{m} · coste {r['cost']:g}": r for m, r in out.items()},
                       f"Periodo reservado (lockbox) desde {PARAMS.lockbox_start}", REPORT_DIR / "lockbox_equity.png")
    report.save_json({m: dict(summary=report.summary_row(r), edge=r["edge"].reset_index().to_dict("records"),
                              months=r["months"].to_dict("records")) for m, r in out.items()},
                     REPORT_DIR / "lockbox.json")


def cmd_train(a):
    for m in a.markets:
        cost = MARKETS[m].cost if a.cost is None else a.cost
        meta = model.train_final(m, load_history(m), cost, PARAMS)
        print(json.dumps({k: meta[k] for k in ("market", "cost", "threshold", "reference_threshold", "trees", "train_from", "trained_until", "val_stats")}, indent=2))


def cmd_predict(a):
    m = a.markets[0]
    while True:
        df = load_csv(a.csv, tz=a.tz) if a.csv else download_yahoo(MARKETS[m].yahoo, days=5)
        sig = model.live_signal(m, df, cost=a.cost)
        if a.json:
            print(json.dumps(sig))
        else:
            print(f"[{sig['bar_time']}] {m} cierre={sig['close']:.2f}  pred={sig['prediction']:+.4f} "
                  f"(umbral {sig['threshold'] if sig['threshold'] is not None else 'sin ventaja validada'})  "
                  f"sesgo={sig['bias']}  => {sig['action']}")
            if sig["side"]:
                print(f"    TP={sig['take_profit']:.2f}  SL={sig['stop_loss']:.2f}  salida por tiempo a los {sig['max_minutes']} min")
        if not a.loop:
            break
        time.sleep(max(1.0, 61 - time.time() % 60))


def _risk(a):
    from .agent import RiskConfig

    return RiskConfig(risk_per_trade=a.risk, max_daily_loss=a.max_daily_loss, max_drawdown=a.max_drawdown,
                      max_trades_per_day=a.max_trades_per_day)


def _print_summary(agent, equity0: float):
    s = agent.summary()
    print(f"\noperaciones={s['trades']}  acierto={s['win_rate']:.1%}  neto/op={s['net_avg']:+.3f}  t={s['tstat']:+.2f}"
          if s["trades"] else "\noperaciones=0 (el agente se quedó fuera)")
    print(f"capital: {equity0:,.2f} -> {s['equity']:,.2f}  ({s['equity'] / equity0 - 1:+.2%})")
    if s["halted"]:
        print(f"DETENIDO: {s['halted']}")
    if s["skips"]:
        print("barras sin operar por motivo:", s["skips"])


def cmd_replay(a):
    from . import agent as ag

    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")
    m = a.markets[0]
    agent = ag.replay(m, load_history(m), a.start, a.end, equity=a.equity, cost=a.cost, risk=_risk(a),
                      retrain=not a.no_retrain, use_reference=a.use_reference_threshold)
    _print_summary(agent, a.equity)
    if agent.closed:
        tr = pd.DataFrame(agent.closed)
        mon = tr.groupby(pd.DatetimeIndex(tr["time"]).tz_localize(None).to_period("M")).agg(
            ops=("net", "size"), neto_precio=("net", "sum"), neto_dinero=("net_money", "sum"))
        print(mon.round(2).to_string())
        REPORT_DIR.mkdir(exist_ok=True)
        tr.to_csv(REPORT_DIR / f"replay_{m}_{a.start}.csv", index=False)


def cmd_agent(a):
    from . import agent as ag
    from .brokers import MT5Broker, PaperBroker

    m = a.markets[0]
    mk = MARKETS[m]
    cost = mk.cost if a.cost is None else a.cost
    state_dir = STATE_DIR if a.state_dir is None else a.state_dir
    import pathlib

    pathlib.Path(state_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(pathlib.Path(state_dir) / f"{m}_agent.log")])
    log = logging.getLogger("scalper.agent")
    risk = _risk(a)
    feed = a.feed or ("mt5" if a.broker == "mt5" else "yahoo")
    if a.broker == "mt5" and feed != "mt5":
        raise SystemExit("Con --broker mt5 los datos deben venir del propio bróker (--feed mt5).")
    if a.broker == "mt5" and a.use_reference_threshold:
        raise SystemExit("--use-reference-threshold solo se permite con el bróker simulado.")

    mt5 = MT5Broker(mk, symbol=a.symbol, allow_real=a.allow_real or a.broker == "paper") if feed == "mt5" else None
    broker = mt5 if a.broker == "mt5" else PaperBroker(a.equity, cost)
    if feed == "mt5":
        fetch = lambda: mt5.bars(2500)
    elif feed == "csv":
        fetch = lambda: load_csv(a.csv, tz=a.tz).iloc[-2500:]
    else:
        fetch = lambda: download_yahoo(mk.yahoo, days=5).iloc[-2500:]
        risk = dataclasses.replace(risk, max_bar_age_min=20.0)
        log.warning("Yahoo sirve los datos con ~10 min de retraso: válido solo para probar el agente en simulación.")

    def retrainer():
        model.train_final(m, download_histdata(m, verbose=False), cost, PARAMS)
        return model.Predictor.load(m)

    agent = ag.Agent(m, broker, model.Predictor.load(m), risk, cost, state_dir=state_dir,
                     use_reference=a.use_reference_threshold)
    if a.reset_halt:
        agent.state.halted, agent.state.suspended = None, False
        agent.state.peak_equity = broker.equity()
    if agent.state.last_train is None:
        agent.state.last_train = agent.predictor.meta["trained_until"]
    log.info("Agente %s | bróker=%s datos=%s | capital=%.2f coste=%g | riesgo/op=%.2f%% | umbral=%s",
             m, a.broker, feed, broker.equity(), cost, 100 * risk.risk_per_trade,
             agent.predictor.threshold if agent.predictor.threshold is not None else "sin ventaja validada")
    try:
        ag.run_live(agent, fetch, retrain_days=a.retrain_days, retrainer=None if a.no_retrain else retrainer,
                    max_steps=a.max_steps)
    except KeyboardInterrupt:
        log.info("Interrumpido por el usuario. Las posiciones abiertas conservan su TP/SL.")
    _print_summary(agent, a.equity if a.broker == "paper" else broker.equity())


def main():
    ap = argparse.ArgumentParser(prog="scalper", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        sp = sub.add_parser(name)
        sp.add_argument("--market", dest="markets", nargs="+", choices=list(MARKETS), default=kw.get("default", list(MARKETS)))
        sp.add_argument("--cost", type=float, default=None, help="coste ida y vuelta en unidades de precio")
        sp.set_defaults(fn=fn)
        return sp

    sp = add("download", cmd_download)
    sp.add_argument("--start-year", type=int, default=2023)
    sp.add_argument("--force", action="store_true")
    add("compare", cmd_compare).add_argument("--verbose", action="store_true")
    add("lockbox", cmd_lockbox)
    add("train", cmd_train)
    sp = add("predict", cmd_predict, default=["XAUUSD"])
    sp.add_argument("--csv", help="CSV propio de barras M1 en lugar de Yahoo")
    sp.add_argument("--tz", default="UTC", help="zona horaria del CSV")
    sp.add_argument("--loop", action="store_true", help="repetir cada minuto")
    sp.add_argument("--json", action="store_true")
    for name, fn in (("replay", cmd_replay), ("agent", cmd_agent)):
        sp = add(name, fn, default=["XAUUSD"])
        sp.add_argument("--equity", type=float, default=10_000.0, help="capital inicial de la cuenta simulada")
        sp.add_argument("--risk", type=float, default=0.0025, help="fracción del capital arriesgada por operación")
        sp.add_argument("--max-daily-loss", type=float, default=0.02)
        sp.add_argument("--max-drawdown", type=float, default=0.10)
        sp.add_argument("--max-trades-per-day", type=int, default=60)
        sp.add_argument("--no-retrain", action="store_true", help="no reentrenar automáticamente")
        sp.add_argument("--use-reference-threshold", action="store_true",
                        help="operar con el umbral de referencia aunque no haya ventaja validada (solo simulación)")
        if name == "replay":
            sp.add_argument("--start", required=True, help="primer mes, YYYY-MM")
            sp.add_argument("--end", help="mes final exclusivo, YYYY-MM")
            sp.add_argument("--verbose", action="store_true")
        else:
            sp.add_argument("--broker", choices=["paper", "mt5"], default="paper")
            sp.add_argument("--feed", choices=["yahoo", "csv", "mt5"], help="origen de las barras (por defecto yahoo, o mt5 con --broker mt5)")
            sp.add_argument("--csv", help="CSV de barras M1 que otro proceso mantiene actualizado")
            sp.add_argument("--tz", default="UTC")
            sp.add_argument("--symbol", help="nombre del símbolo en el bróker si difiere (p. ej. XAUUSD.m)")
            sp.add_argument("--allow-real", action="store_true", help="permitir cuenta real en MT5 (por defecto solo demo)")
            sp.add_argument("--retrain-days", type=float, default=7.0)
            sp.add_argument("--max-steps", type=int, help="nº de ciclos antes de salir (para pruebas)")
            sp.add_argument("--state-dir")
            sp.add_argument("--reset-halt", action="store_true", help="levantar una parada por caída máxima")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
