"""Comparación de mercados e informe de resultados fuera de muestra."""
from __future__ import annotations

import dataclasses
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import backtest, model
from .config import MARKETS, REPORT_DIR, Params
from .data import load_history

INK, MUTED, GRID = "#1f2933", "#6b7785", "#d9dee4"
COLORS = {"XAUUSD": "#b7791f", "NAS100": "#2b6cb0", "SPX500": "#2f855a"}


def edge_table(res: dict, cost: float) -> pd.DataFrame:
    """Ventaja bruta media (en precio) al seguir la predicción, por cuantil de |predicción|."""
    ok = np.isfinite(res["pred"]) & np.isfinite(res["target"])
    d = pd.DataFrame(dict(p=res["pred"][ok], y=res["target"][ok], w=res["width"][ok]))
    d["edge"] = np.sign(d.p) * d.y * d.w
    labels = ["resto (80 %)", "top 20-5 %", "top 5-2 %", "top 2-1 %", "top 1 %"]
    d["q"] = pd.cut(d.p.abs().rank(method="first", pct=True), [0, 0.8, 0.95, 0.98, 0.99, 1], labels=labels)
    g = d.groupby("q", observed=True).agg(barras=("edge", "size"), ventaja_bruta=("edge", "mean"))
    g["ventaja/coste"] = g["ventaja_bruta"] / cost
    return g


def information_coefficient(res: dict) -> float:
    ok = np.isfinite(res["pred"]) & np.isfinite(res["target"])
    return float(pd.Series(res["pred"][ok]).corr(pd.Series(res["target"][ok]), method="spearman"))


def evaluate(market: str, p: Params, start: str | None, end: str | None, cost: float | None = None, verbose: bool = False) -> dict:
    mk = MARKETS[market]
    cost = mk.cost if cost is None else cost
    res = model.walk_forward(load_history(market), cost, p, start=start, end=end, verbose=verbose)
    res.update(market=market, cost=cost, stats=backtest.stats(res["trades"]), ic=information_coefficient(res),
               edge=edge_table(res, cost))
    return res


def summary_row(res: dict) -> dict:
    st, mo = res["stats"], res["months"]
    return {
        "mercado": res["market"], "coste": res["cost"], "IC": round(res["ic"], 4),
        "meses IC>0": f"{(mo.ic > 0).mean():.0%}", "meses activos": f"{int((mo.trades > 0).sum())}/{len(mo)}",
        "ops": st["trades"], "acierto": None if st["trades"] == 0 else round(st["win_rate"], 3),
        "bruto/op": None if st["trades"] == 0 else round(st["gross_avg"], 3),
        "neto/op": None if st["trades"] == 0 else round(st["net_avg"], 3),
        "neto total": round(st["net_total"], 1), "t-stat": None if st["trades"] == 0 else round(st["tstat"], 2),
        "max DD": round(st["max_dd"], 1),
    }


def plot_equity(results: dict[str, dict], title: str, path) -> None:
    fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 3.4), squeeze=False)
    for ax, (name, res) in zip(axes[0], results.items()):
        tr = res["trades"]
        color = COLORS.get(res["market"], "#2b6cb0")
        if len(tr):
            t = pd.DatetimeIndex(tr["time"])
            ax.plot(t, tr["gross"].cumsum(), color=MUTED, lw=1.2, label="bruto (sin costes)")
            ax.plot(t, tr["net"].cumsum(), color=color, lw=2, label="neto de costes")
            ax.legend(frameon=False, fontsize=8, loc="best")
        else:
            ax.text(0.5, 0.5, "sin operaciones\n(sin ventaja validada)", ha="center", va="center",
                    transform=ax.transAxes, color=MUTED, fontsize=9)
            ax.set_xticks([])
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title(name, fontsize=10, color=INK, loc="left")
        ax.set_ylabel("resultado acumulado (unidades de precio)", fontsize=8, color=MUTED)
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(labelsize=8, colors=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.autofmt_xdate(rotation=30)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    df = df.reset_index() if df.index.name else df
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns)
    fmt = lambda v: "–" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    return head + "\n" + "\n".join("| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False))


def cost_sensitivity(market: str, p: Params, start: str | None, end: str | None, mults=(0.0, 0.5, 1.0, 2.0)) -> pd.DataFrame:
    """Repite el walk-forward completo (umbral incluido) con distintos costes."""
    rows = []
    for k in mults:
        r = evaluate(market, p, start, end, cost=MARKETS[market].cost * k)
        row = summary_row(r)
        row["coste"] = f"{k:g}x = {MARKETS[market].cost * k:g}"
        rows.append(row)
    return pd.DataFrame(rows).drop(columns=["mercado", "IC", "meses IC>0"])


def save_json(obj: dict, path) -> None:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return str(o)

    REPORT_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=default))
