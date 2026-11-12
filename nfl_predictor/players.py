"""Modelo de estadísticas de jugadores (QB, RB, WR, TE) para el siguiente partido.

Para cada (posición, estadística) se entrena un gradient boosting con pérdida de
Poisson sobre variables previas al partido: producción reciente del jugador,
volumen de su equipo, lo que la defensa rival le permite a esa posición, el QB
titular y el contexto del partido (localía, total implícito del equipo, spread).
Las proyecciones son condicionales a que el jugador juegue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .games import _ewm_prior, team_game_table

FIRST_SEASON = 2009          # historial que se carga para calcular promedios
MIN_TRAIN_SEASON = 2010
SHORT_HALFLIFE = 2.0         # partidos: forma reciente
LONG_HALFLIFE = 8.0          # partidos: nivel de fondo
DEFENSE_HALFLIFE = 8.0
MIN_OPPORTUNITIES = 1.0      # oportunidades (pases+acarreos+targets) por partido para entrenar
REGULAR_OPPORTUNITIES = 4.0  # umbral de "jugador con rol" para evaluar

PASSING = ["attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions"]
RUSHING = ["carries", "rushing_yards", "rushing_tds"]
RECEIVING = ["targets", "receptions", "receiving_yards", "receiving_tds"]
STATS = PASSING + RUSHING + RECEIVING
ROLLED = STATS + ["target_share", "carry_share", "opportunities"]

TARGETS = {
    "QB": PASSING + RUSHING,
    "RB": RUSHING + RECEIVING,
    "WR": RECEIVING,
    "TE": RECEIVING,
}
# Estadísticas del propio jugador que alimentan a cada posición.
_OWN = {
    "QB": PASSING + RUSHING + ["opportunities"],
    "RB": RUSHING + RECEIVING + ["target_share", "carry_share", "opportunities"],
    "WR": RECEIVING + ["carries", "rushing_yards", "target_share", "opportunities"],
    "TE": RECEIVING + ["target_share", "opportunities"],
}
_CONTEXT = ["is_home", "implied_total", "team_spread", "game_total", "team_qb_val",
            "team_attempts_form", "team_carries_form", "games_prior", "days_since_last", "week"]

STAT_LABELS = {
    "attempts": "Att", "completions": "Cmp", "passing_yards": "PaYd", "passing_tds": "PaTD",
    "passing_interceptions": "Int", "carries": "Car", "rushing_yards": "RuYd",
    "rushing_tds": "RuTD", "targets": "Tgt", "receptions": "Rec", "receiving_yards": "ReYd",
    "receiving_tds": "ReTD",
}


def feature_names(position: str) -> list[str]:
    own = _OWN[position]
    names = [f"{s}_short" for s in own] + [f"{s}_long" for s in own]
    names += [f"{s}_season" for s in TARGETS[position]]
    names += [f"def_{s}" for s in TARGETS[position]]
    return names + _CONTEXT


# --------------------------------------------------------------------------- #
# Construcción de variables
# --------------------------------------------------------------------------- #

def _upcoming_rows(games: pd.DataFrame, pstats: pd.DataFrame, long: pd.DataFrame,
                   target: tuple[int, int], injuries: pd.DataFrame | None) -> pd.DataFrame:
    """Renglones vacíos para los jugadores que se espera que jueguen la jornada objetivo."""
    season, week = target
    slate = long[(long["season"] == season) & (long["week"] == week)]
    slate = slate[slate["game_id"].isin(games.loc[games["result"].isna(), "game_id"])]
    if slate.empty:
        return pd.DataFrame()

    played = long[long["pf"].notna() & (long["gameday"] < slate["gameday"].min())]
    recent = played.groupby("team").tail(2)[["team", "game_id"]]       # 2 últimos partidos del equipo
    last = (pstats.merge(games[["game_id", "gameday"]], on="game_id")
            .sort_values("gameday").groupby("player_id").tail(1))
    active = last.merge(recent, on=["team", "game_id"])                # siguen en ese equipo y activos
    active = active[["player_id", "player_display_name", "position", "position_group", "team"]]

    status = pd.DataFrame(columns=["player_id", "status"])
    if injuries is not None:
        report = injuries[(injuries["season"] == season) & (injuries["week"] == week)]
        status = report.rename(columns={"gsis_id": "player_id", "report_status": "status"})
        status = status[["player_id", "status"]].dropna().drop_duplicates("player_id")
    active = active.merge(status, on="player_id", how="left")
    active = active[~active["status"].isin(["Out", "Doubtful"])]

    # QB: solo el titular anunciado (aunque no tenga historial, p. ej. un novato).
    starters = slate[["team", "qb_id"]].dropna().rename(columns={"qb_id": "player_id"})
    names = pd.concat([
        games[["home_qb_id", "home_qb_name"]].set_axis(["player_id", "player_display_name"], axis=1),
        games[["away_qb_id", "away_qb_name"]].set_axis(["player_id", "player_display_name"], axis=1),
    ]).dropna().drop_duplicates("player_id", keep="last")
    starters = starters.merge(names, on="player_id", how="left").assign(position="QB", position_group="QB")
    starters = starters.merge(status, on="player_id", how="left")
    active = pd.concat([active[active["position_group"] != "QB"], starters], ignore_index=True)

    rows = active.merge(slate[["team", "opp", "game_id", "season", "week", "game_type"]], on="team")
    rows = rows.rename(columns={"opp": "opponent_team"})
    rows["season_type"] = np.where(rows["game_type"] == "REG", "REG", "POST")
    return rows.drop(columns="game_type")


def build_frame(games: pd.DataFrame, pstats: pd.DataFrame, *,
                target: tuple[int, int] | None = None,
                injuries: pd.DataFrame | None = None,
                fallback_lines: pd.DataFrame | None = None) -> pd.DataFrame:
    """Un renglón por (jugador, partido) con variables previas al partido.

    Si se pasa `target=(temporada, semana)` se agregan renglones para esa jornada
    aún no jugada (marcados con `upcoming=True`), con las mismas variables.
    `fallback_lines` (game_id, pred_margin, pred_total) suple las líneas de Las
    Vegas cuando todavía no están publicadas.
    """
    long = team_game_table(games, pstats)
    stats = pstats[pstats["season"] >= FIRST_SEASON].drop_duplicates(["player_id", "game_id"]).copy()
    stats["upcoming"] = False
    if target is not None:
        future = _upcoming_rows(games, pstats, long, target, injuries)
        if not future.empty:
            future["upcoming"] = True
            stats = pd.concat([stats, future], ignore_index=True)
    stats["upcoming"] = stats["upcoming"].astype(bool)

    # --- contexto del partido ------------------------------------------------
    lines = games[["game_id", "spread_line", "total_line"]].copy()
    if fallback_lines is not None:
        fb = fallback_lines.set_index("game_id")
        lines["spread_line"] = lines["spread_line"].fillna(lines["game_id"].map(fb["pred_margin"]))
        lines["total_line"] = lines["total_line"].fillna(lines["game_id"].map(fb["pred_total"]))
    team_ctx = long[["game_id", "team", "gameday", "is_home", "qb_id", "qb_val"]].merge(lines, on="game_id")
    side = np.where(team_ctx["is_home"] == 1, 1.0, -1.0)
    team_ctx["team_spread"] = side * team_ctx["spread_line"]            # positivo = favorito
    team_ctx["game_total"] = team_ctx["total_line"]
    team_ctx["implied_total"] = (team_ctx["total_line"] + team_ctx["team_spread"]) / 2
    team_ctx = team_ctx.rename(columns={"qb_val": "team_qb_val"}).drop(columns=["spread_line", "total_line"])

    # --- volumen ofensivo del equipo ----------------------------------------
    volume = pstats.groupby(["game_id", "team"], as_index=False)[["attempts", "carries"]].sum()
    volume = long[["game_id", "team", "gameday"]].merge(volume, on=["game_id", "team"], how="left")
    volume = volume.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    form = _ewm_prior(volume, "team", ["attempts", "carries"], LONG_HALFLIFE)
    volume["team_attempts_form"], volume["team_carries_form"] = form["attempts"], form["carries"]
    volume = volume.rename(columns={"carries": "team_carries"})

    frame = stats.merge(team_ctx, on=["game_id", "team"], how="inner")
    frame = frame.merge(volume[["game_id", "team", "team_carries", "team_attempts_form",
                                "team_carries_form"]], on=["game_id", "team"], how="left")
    frame = frame.sort_values(["gameday", "game_id", "player_id"]).reset_index(drop=True)
    frame["is_starter"] = frame["player_id"] == frame["qb_id"]
    frame["carry_share"] = frame["carries"] / frame["team_carries"].replace(0, np.nan)
    frame["opportunities"] = frame["attempts"] + frame["carries"] + frame["targets"]

    # --- producción previa del jugador --------------------------------------
    short = _ewm_prior(frame, "player_id", ROLLED, SHORT_HALFLIFE).add_suffix("_short")
    longrun = _ewm_prior(frame, "player_id", ROLLED, LONG_HALFLIFE).add_suffix("_long")
    by_season = frame.groupby(["player_id", "season"], sort=False)
    filled = frame[STATS].fillna(0)
    prior_sum = filled.groupby([frame["player_id"], frame["season"]], sort=False).cumsum() - filled
    prior_n = by_season.cumcount().replace(0, np.nan)
    season_avg = prior_sum.div(prior_n, axis=0).add_suffix("_season")
    by_player = frame.groupby("player_id", sort=False)
    extra = pd.DataFrame({
        "games_prior": by_player.cumcount(),
        "days_since_last": (frame["gameday"] - by_player["gameday"].shift(1)).dt.days.clip(upper=60),
    })
    frame = pd.concat([frame, short, longrun, season_avg, extra], axis=1)

    # --- lo que la defensa rival le permite a cada posición ------------------
    allowed = (pstats.groupby(["game_id", "opponent_team", "position_group"], as_index=False)[STATS]
               .sum().rename(columns={"opponent_team": "defense"}))
    grid = long[["game_id", "team", "gameday"]].rename(columns={"team": "defense"}).merge(
        pd.DataFrame({"position_group": list(TARGETS)}), how="cross")
    grid = grid.merge(allowed, on=["game_id", "defense", "position_group"], how="left")
    grid = grid.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    grid["key"] = grid["defense"] + "|" + grid["position_group"]
    defense = _ewm_prior(grid, "key", STATS, DEFENSE_HALFLIFE).add_prefix("def_")
    defense = pd.concat([grid[["game_id", "defense", "position_group"]], defense], axis=1)
    frame = frame.merge(defense, left_on=["game_id", "opponent_team", "position_group"],
                        right_on=["game_id", "defense", "position_group"], how="left")
    return frame.drop(columns="defense")


def eligible(frame: pd.DataFrame, position: str, min_opportunities: float = MIN_OPPORTUNITIES) -> pd.Series:
    """Renglones útiles para entrenar/evaluar: QB titulares y jugadores con rol previo."""
    mask = frame["position_group"] == position
    if position == "QB":
        return mask & frame["is_starter"]
    return mask & (frame["opportunities_long"] >= min_opportunities)


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #

def _regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=250, max_leaf_nodes=12,
        min_samples_leaf=60, l2_regularization=1.0, random_state=0)


@dataclass
class PlayerModel:
    models: dict[tuple[str, str], HistGradientBoostingRegressor] = field(default_factory=dict)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Agrega columnas `pred_<stat>` y `pred_fantasy_ppr` a los renglones dados."""
        out = frame.copy()
        for stat in STATS:
            out[f"pred_{stat}"] = np.nan
        for (position, stat), model in self.models.items():
            mask = out["position_group"] == position
            if mask.any():
                out.loc[mask, f"pred_{stat}"] = model.predict(out.loc[mask, feature_names(position)])
        pred = out[[f"pred_{s}" for s in STATS]].fillna(0)
        out["pred_fantasy_ppr"] = (
            0.04 * pred["pred_passing_yards"] + 4 * pred["pred_passing_tds"]
            - 2 * pred["pred_passing_interceptions"]
            + 0.1 * (pred["pred_rushing_yards"] + pred["pred_receiving_yards"])
            + 6 * (pred["pred_rushing_tds"] + pred["pred_receiving_tds"]) + pred["pred_receptions"])
        return out


def fit(frame: pd.DataFrame) -> PlayerModel:
    train = frame[(frame["season"] >= MIN_TRAIN_SEASON) & ~frame["upcoming"]]
    model = PlayerModel()
    for position, stats in TARGETS.items():
        rows = train[eligible(train, position)]
        features = rows[feature_names(position)]
        for stat in stats:
            model.models[(position, stat)] = _regressor().fit(features, rows[stat].clip(lower=0))
    return model


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #

def backtest(frame: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """MAE por (posición, estadística) del modelo contra dos referencias ingenuas.

    Cada temporada de prueba se predice con un modelo entrenado solo en las anteriores.
    Referencias: promedio de la temporada en curso y promedio móvil reciente.
    """
    done = frame[~frame["upcoming"]]
    scored = []
    for season in seasons:
        test = done[done["season"] == season]
        if test.empty:
            continue
        scored.append(fit(done[done["season"] < season]).predict(test))
    scored = pd.concat(scored, ignore_index=True)

    rows = []
    for position, stats in TARGETS.items():
        sample = scored[eligible(scored, position, REGULAR_OPPORTUNITIES)]
        for stat in stats:
            actual = sample[stat]
            season_avg = sample[f"{stat}_season"].fillna(sample[f"{stat}_long"])
            rows.append({
                "pos": position, "stat": stat, "n": len(sample), "promedio_real": actual.mean(),
                "mae_modelo": (sample[f"pred_{stat}"] - actual).abs().mean(),
                "mae_prom_temporada": (season_avg - actual).abs().mean(),
                "mae_prom_reciente": (sample[f"{stat}_long"] - actual).abs().mean(),
            })
    table = pd.DataFrame(rows)
    best_naive = table[["mae_prom_temporada", "mae_prom_reciente"]].min(axis=1)
    table["mejora_pct"] = 100 * (1 - table["mae_modelo"] / best_naive)
    return table
