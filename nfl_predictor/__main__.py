"""Línea de comandos:  python -m nfl_predictor {partidos,jugadores,backtest}"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import data, games as game_model, players as player_model

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 500)


def _target(args: argparse.Namespace, games: pd.DataFrame) -> tuple[int, int]:
    season, week = data.next_week(games)
    return (args.temporada or season, args.semana or week)


def _save(table: pd.DataFrame, path: str | None) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        print(f"\nGuardado en {path}")


def _predict_games(games: pd.DataFrame, pstats: pd.DataFrame, target: tuple[int, int]) -> pd.DataFrame:
    feats = game_model.build_features(games, pstats)
    slate = feats[(feats["season"] == target[0]) & (feats["week"] == target[1])]
    if slate.empty:
        raise SystemExit(f"No hay partidos para la temporada {target[0]}, semana {target[1]}.")
    # Se entrena solo con partidos anteriores a la jornada pedida (sirve también para semanas pasadas).
    train = feats[feats["result"].notna() & (feats["gameday"] < slate["gameday"].min())]
    return game_model.fit(train).predict(slate)


def cmd_partidos(args: argparse.Namespace) -> None:
    games = data.load_games()
    pstats = data.load_player_stats(games)
    target = _target(args, games)
    pred = _predict_games(games, pstats, target)

    favorite_prob = pred["home_win_prob"].where(pred["pred_margin"] >= 0, 1 - pred["home_win_prob"])
    table = pd.DataFrame({
        "fecha": pred["gameday"].dt.strftime("%a %d-%b"),
        "partido": pred["away_team"] + " @ " + pred["home_team"],
        "pick": pred["pick"],
        "prob": (100 * favorite_prob).round(0).astype(int).astype(str) + "%",
        "marcador": (pred["away_team"] + " " + pred["pred_away_score"].round(0).astype(int).astype(str)
                     + " - " + pred["home_team"] + " " + pred["pred_home_score"].round(0).astype(int).astype(str)),
        "margen_local": pred["pred_margin"].round(1),
        "vegas_local": pred["vegas_spread"],
        "total": pred["pred_total"].round(1),
        "vegas_total": pred["vegas_total"],
    })
    print(f"\nNFL {target[0]} - semana {target[1]}  (margen_local > 0 = gana el local)\n")
    print(table.to_string(index=False, na_rep="-"))
    _save(pred, args.csv)


def cmd_jugadores(args: argparse.Namespace) -> None:
    games = data.load_games()
    pstats = data.load_player_stats(games)
    target = _target(args, games)
    lines = _predict_games(games, pstats, target)
    frame = player_model.build_frame(games, pstats, target=target,
                                     injuries=data.load_injuries(target[0]), fallback_lines=lines)
    upcoming = frame[frame["upcoming"]]
    if upcoming.empty:
        raise SystemExit("Esa jornada ya se jugó; las proyecciones de jugadores son solo a futuro.")
    pred = player_model.fit(frame).predict(upcoming)

    if args.equipo:
        pred = pred[pred["team"] == args.equipo.upper()]
    positions = [args.pos.upper()] if args.pos else list(player_model.TARGETS)
    print(f"\nNFL {target[0]} - semana {target[1]}: proyecciones (suponiendo que el jugador juega)")
    exported = []
    for position in positions:
        rows = pred[pred["position_group"] == position].sort_values("pred_fantasy_ppr", ascending=False)
        if args.jugador:
            rows = rows[rows["player_display_name"].str.contains(args.jugador, case=False, na=False)]
        rows = rows.head(args.top)
        if rows.empty:
            continue
        table = pd.DataFrame({"jugador": rows["player_display_name"], "eq": rows["team"],
                              "vs": rows["opponent_team"], "estatus": rows["status"].fillna("")})
        for stat in player_model.TARGETS[position]:
            decimals = 0 if stat.endswith("yards") else 1
            table[player_model.STAT_LABELS[stat]] = rows[f"pred_{stat}"].round(decimals)
        table["PPR"] = rows["pred_fantasy_ppr"].round(1)
        print(f"\n{position}\n{table.to_string(index=False)}")
        exported.append(rows)
    if exported:
        keep = ["player_id", "player_display_name", "position", "team", "opponent_team", "status"]
        full = pd.concat(exported)
        _save(full[keep + [c for c in full.columns if c.startswith("pred_")]], args.csv)


def cmd_backtest(args: argparse.Namespace) -> None:
    games = data.load_games()
    pstats = data.load_player_stats(games)
    feats = game_model.build_features(games, pstats)
    preds, summary = game_model.backtest(feats, args.desde)
    print(f"\nPARTIDOS: {summary.attrs['partidos']} partidos desde {args.desde}, "
          "cada temporada predicha con un modelo entrenado solo en las anteriores\n")
    print(summary.round(4).to_string(na_rep="-"))
    print("\nContra el spread de cierre (52.4% es el punto de equilibrio con momio -110):")
    for edge in (0, 2, 4):
        ats = game_model.against_the_spread(preds, edge)
        print(f"  discrepancia >= {edge} pts: {ats['ganadas']}/{ats['apuestas']} = {100 * ats['pct']:.1f}%")

    last = data.current_season(games)
    seasons = [last - 2, last - 1, last]
    frame = player_model.build_frame(games, pstats)
    table = player_model.backtest(frame, seasons)
    print(f"\nJUGADORES: error absoluto medio (MAE) en {seasons[0]}-{seasons[-1]}, "
          "QB titulares y jugadores con rol\n")
    print(table.round(2).to_string(index=False))
    _save(table, args.csv)


def main() -> None:
    parser = argparse.ArgumentParser(prog="nfl_predictor", description=__doc__)
    sub = parser.add_subparsers(dest="comando", required=True)

    def week_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--temporada", type=int, help="por defecto, la del próximo partido pendiente")
        p.add_argument("--semana", type=int, help="por defecto, la próxima jornada pendiente")
        p.add_argument("--csv", help="ruta donde guardar la tabla completa")

    p = sub.add_parser("partidos", help="ganador, marcador, margen y total de cada partido")
    week_args(p)
    p.set_defaults(func=cmd_partidos)

    p = sub.add_parser("jugadores", help="proyección de estadísticas por jugador")
    week_args(p)
    p.add_argument("--equipo", help="abreviatura, p. ej. KC")
    p.add_argument("--pos", choices=["QB", "RB", "WR", "TE", "qb", "rb", "wr", "te"])
    p.add_argument("--jugador", help="filtra por nombre (texto parcial)")
    p.add_argument("--top", type=int, default=25, help="jugadores por posición (25)")
    p.set_defaults(func=cmd_jugadores)

    p = sub.add_parser("backtest", help="precisión histórica de ambos modelos")
    p.add_argument("--desde", type=int, default=2015, help="primera temporada de prueba (2015)")
    p.add_argument("--csv", help="ruta donde guardar la tabla de jugadores")
    p.set_defaults(func=cmd_backtest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
