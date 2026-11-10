"""CLI:  python -m scalper <comando>

  download                 descarga el histórico M1 de los mercados candidatos
  compare                  walk-forward en los tres mercados (periodo de diseño) y ranking
  lockbox --market M       evaluación final en los meses reservados
  train --market M         entrena el modelo final y lo guarda en models/
  predict --market M       señal para la próxima barra con datos recientes de Yahoo (o --csv)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import pandas as pd

from . import model, report
from .config import MARKETS, PARAMS, REPORT_DIR
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
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
