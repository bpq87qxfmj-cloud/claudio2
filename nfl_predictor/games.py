"""Modelo de partidos: probabilidad de victoria, margen y total de puntos.

Todas las variables se calculan únicamente con información previa al partido
(ratings Elo, promedios móviles exponenciales de producción por jugada, valor
del QB titular, descanso), de modo que el backtest no tiene fuga de datos.
Las líneas de Las Vegas NO se usan como variable: sirven de punto de comparación.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .elo import add_elo

FORM_HALFLIFE = 8.0        # partidos; memoria de los promedios móviles de equipo
QB_HALFLIFE = 10.0         # partidos; memoria del valor del QB
QB_PRIOR_GAMES = 6.0       # peso (en partidos) del prior de un QB sin historial
MIN_TRAIN_SEASON = 2002    # las primeras temporadas solo "calientan" Elo y promedios
RIDGE_ALPHA = 30.0

TEAM_METRICS = ["pf", "pa", "off_epa", "def_epa", "off_pass_epa", "def_pass_epa",
                "off_rush_epa", "def_rush_epa", "off_ypp", "def_ypp", "giveaways", "takeaways"]

MARGIN_FEATURES = ["elo_diff", "qb_diff", "rest_diff", "neutral", "div_game"] + [
    f"{m}_diff" for m in TEAM_METRICS]
TOTAL_FEATURES = ["league_total", "indoor", "qb_sum", "elo_abs_diff"] + [
    f"{m}_sum" for m in ("pf", "pa", "off_epa", "def_epa", "off_pass_epa", "def_pass_epa",
                         "off_ypp", "def_ypp")]


# --------------------------------------------------------------------------- #
# Construcción de variables
# --------------------------------------------------------------------------- #

def _ewm_prior(frame: pd.DataFrame, key: str, columns: list[str], halflife: float) -> pd.DataFrame:
    """Promedio móvil exponencial por grupo usando solo renglones ANTERIORES."""
    shifted = frame.groupby(key, sort=False)[columns].shift(1)
    ewm = shifted.groupby(frame[key], sort=False).ewm(halflife=halflife, min_periods=1).mean()
    return ewm.reset_index(level=0, drop=True).reindex(frame.index)


def team_game_table(games: pd.DataFrame, pstats: pd.DataFrame) -> pd.DataFrame:
    """Tabla larga con un renglón por (partido, equipo) y su forma previa al partido."""
    box = pstats.groupby(["game_id", "team"], as_index=False).agg(
        attempts=("attempts", "sum"), sacks=("sacks_suffered", "sum"),
        carries=("carries", "sum"), pass_yards=("passing_yards", "sum"),
        rush_yards=("rushing_yards", "sum"),
        pass_epa=("passing_epa", lambda s: s.sum(min_count=1)),
        rush_epa=("rushing_epa", lambda s: s.sum(min_count=1)),
        interceptions=("passing_interceptions", "sum"),
        fumbles=("sack_fumbles_lost", "sum"), fumbles_rush=("rushing_fumbles_lost", "sum"),
        fumbles_rec=("receiving_fumbles_lost", "sum"),
    )
    dropbacks = (box["attempts"] + box["sacks"]).replace(0, np.nan)
    plays = (dropbacks.fillna(0) + box["carries"]).replace(0, np.nan)
    box["o_pass_epa"] = box["pass_epa"] / dropbacks
    box["o_rush_epa"] = box["rush_epa"] / box["carries"].replace(0, np.nan)
    box["o_epa"] = (box["pass_epa"].fillna(0) + box["rush_epa"].fillna(0)) / plays
    box["o_ypp"] = (box["pass_yards"] + box["rush_yards"]) / plays
    box["o_to"] = box["interceptions"] + box["fumbles"] + box["fumbles_rush"] + box["fumbles_rec"]
    box = box[["game_id", "team", "o_pass_epa", "o_rush_epa", "o_epa", "o_ypp", "o_to"]]

    shared = ["game_id", "season", "week", "gameday", "game_type"]
    home = games[shared].assign(team=games["home_team"], opp=games["away_team"], is_home=1,
                                pf=games["home_score"], pa=games["away_score"],
                                qb_id=games["home_qb_id"], rest=games["home_rest"])
    away = games[shared].assign(team=games["away_team"], opp=games["home_team"], is_home=0,
                                pf=games["away_score"], pa=games["home_score"],
                                qb_id=games["away_qb_id"], rest=games["away_rest"])
    long = pd.concat([home, away], ignore_index=True)
    long = long.merge(box, on=["game_id", "team"], how="left")
    opp_box = box.rename(columns={"team": "opp", "o_pass_epa": "d_pass_epa", "o_rush_epa": "d_rush_epa",
                                  "o_epa": "d_epa", "o_ypp": "d_ypp", "o_to": "d_to"})
    long = long.merge(opp_box, on=["game_id", "opp"], how="left")
    long = long.rename(columns={
        "o_epa": "off_epa", "d_epa": "def_epa", "o_pass_epa": "off_pass_epa",
        "d_pass_epa": "def_pass_epa", "o_rush_epa": "off_rush_epa", "d_rush_epa": "def_rush_epa",
        "o_ypp": "off_ypp", "d_ypp": "def_ypp", "o_to": "giveaways", "d_to": "takeaways"})
    long = long.sort_values(["gameday", "game_id", "is_home"]).reset_index(drop=True)

    form = _ewm_prior(long, "team", TEAM_METRICS, FORM_HALFLIFE)
    form.columns = [f"{c}_form" for c in TEAM_METRICS]
    long = pd.concat([long, form], axis=1)

    # Para partidos futuros sin QB anunciado se asume el último titular del equipo.
    long["qb_id"] = long.groupby("team")["qb_id"].ffill()
    long["qb_val"] = _qb_values(long, games, pstats)
    return long


def _qb_values(long: pd.DataFrame, games: pd.DataFrame, pstats: pd.DataFrame) -> pd.Series:
    """Valor del QB titular: EPA por partido (pase + carrera) suavizado y encogido a un prior."""
    qb = pstats[(pstats["position"] == "QB") & (pstats["attempts"] >= 5)].copy()
    qb["value"] = qb["passing_epa"].fillna(0) + qb["rushing_epa"].fillna(0)
    qb = qb.merge(games[["game_id", "gameday"]], on="game_id", how="inner")
    qb = qb.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    decay = 0.5 ** (1.0 / QB_HALFLIFE)
    grouped = qb.groupby("player_id", sort=False)
    mean = grouped["value"].ewm(halflife=QB_HALFLIFE).mean().reset_index(level=0, drop=True)
    count = grouped.cumcount() + 1
    weight = (1 - decay ** count) / (1 - decay)        # número efectivo de partidos
    prior = qb.loc[count <= 4, "value"].mean()         # nivel típico de un QB sin historial
    qb["qb_val"] = (mean.reindex(qb.index) * weight + QB_PRIOR_GAMES * prior) / (weight + QB_PRIOR_GAMES)

    left = long[["gameday", "qb_id"]].reset_index().dropna(subset=["qb_id"]).sort_values("gameday")
    right = qb[["gameday", "player_id", "qb_val"]].sort_values("gameday")
    merged = pd.merge_asof(left, right, on="gameday", left_by="qb_id", right_by="player_id",
                           allow_exact_matches=False)   # solo partidos estrictamente anteriores
    values = merged.set_index("index")["qb_val"].reindex(long.index)
    return values.fillna(prior)


def build_features(games: pd.DataFrame, pstats: pd.DataFrame) -> pd.DataFrame:
    """Un renglón por partido con todas las variables previas al kickoff."""
    feats = add_elo(games)
    long = team_game_table(games, pstats)
    form_cols = [f"{m}_form" for m in TEAM_METRICS] + ["qb_val"]
    for side, flag in (("home", 1), ("away", 0)):
        part = long.loc[long["is_home"] == flag, ["game_id"] + form_cols]
        part = part.rename(columns={c: f"{side}_{c}" for c in form_cols})
        feats = feats.merge(part, on="game_id", how="left")

    feats["neutral"] = (feats["location"] == "Neutral").astype(float)
    feats["indoor"] = feats["roof"].isin(["dome", "closed"]).astype(float)
    feats["elo_diff"] = feats["home_elo"] - feats["away_elo"]
    feats["elo_abs_diff"] = feats["elo_diff"].abs()
    feats["rest_diff"] = (feats["home_rest"] - feats["away_rest"]).clip(-7, 7).fillna(0)
    feats["qb_diff"] = feats["home_qb_val"] - feats["away_qb_val"]
    feats["qb_sum"] = feats["home_qb_val"] + feats["away_qb_val"]
    for metric in TEAM_METRICS:
        h, a = feats[f"home_{metric}_form"], feats[f"away_{metric}_form"]
        feats[f"{metric}_diff"] = h - a
        feats[f"{metric}_sum"] = h + a
    # Nivel de anotación de la liga (cambia con las épocas); desfasado una jornada completa.
    feats["league_total"] = feats["total"].shift(16).ewm(halflife=250, min_periods=50).mean()
    feats["home_win"] = np.where(feats["result"].isna(), np.nan, (feats["result"] > 0).astype(float))
    return feats


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #

def _ridge() -> Pipeline:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=RIDGE_ALPHA))


@dataclass
class GameModel:
    margin: Pipeline
    total: Pipeline
    margin_sd: float      # desviación de los errores del margen -> probabilidad de victoria
    total_sd: float

    def predict(self, feats: pd.DataFrame) -> pd.DataFrame:
        margin = self.margin.predict(feats[MARGIN_FEATURES])
        total = self.total.predict(feats[TOTAL_FEATURES])
        out = feats[["game_id", "season", "week", "gameday", "away_team", "home_team"]].copy()
        out["pred_margin"] = margin                       # positivo = gana el local
        out["pred_total"] = total
        out["home_win_prob"] = norm.cdf(margin / self.margin_sd)
        out["pred_home_score"] = (total + margin) / 2
        out["pred_away_score"] = (total - margin) / 2
        out["pick"] = np.where(margin >= 0, out["home_team"], out["away_team"])
        out["vegas_spread"] = feats["spread_line"].to_numpy()   # positivo = local favorito
        out["vegas_total"] = feats["total_line"].to_numpy()
        return out


def fit(train: pd.DataFrame) -> GameModel:
    train = train[(train["season"] >= MIN_TRAIN_SEASON) & train["result"].notna()]
    margin = _ridge().fit(train[MARGIN_FEATURES], train["result"])
    total = _ridge().fit(train[TOTAL_FEATURES], train["total"])
    margin_sd = float(np.std(train["result"] - margin.predict(train[MARGIN_FEATURES])))
    total_sd = float(np.std(train["total"] - total.predict(train[TOTAL_FEATURES])))
    return GameModel(margin, total, margin_sd, total_sd)


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #

def _prob_metrics(prob: np.ndarray, won: np.ndarray) -> dict[str, float]:
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return {"acierto": float(np.mean((prob > 0.5) == (won > 0.5))),
            "brier": float(np.mean((prob - won) ** 2)),
            "logloss": float(-np.mean(won * np.log(prob) + (1 - won) * np.log(1 - prob)))}


def backtest(feats: pd.DataFrame, first_season: int = 2015) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward: cada temporada se predice con un modelo entrenado solo en las anteriores.

    Devuelve (predicciones partido a partido, tabla resumen contra referencias).
    """
    played = feats[feats["result"].notna()]
    chunks = []
    for season in range(first_season, int(played["season"].max()) + 1):
        test = played[played["season"] == season]
        if test.empty:
            continue
        model = fit(played[played["season"] < season])
        pred = model.predict(test)
        pred["result"] = test["result"].to_numpy()
        pred["total"] = test["total"].to_numpy()
        pred["elo_prob"] = 1 / (1 + 10 ** (-(test["elo_diff"] + 48 * (1 - test["neutral"])) / 400)).to_numpy()
        pred["sd"] = model.margin_sd
        chunks.append(pred)
    preds = pd.concat(chunks, ignore_index=True)

    decided = preds[preds["result"] != 0]
    won = (decided["result"] > 0).to_numpy(dtype=float)
    rows = {
        "Siempre el local": _prob_metrics(np.full(len(won), won.mean()), won) | {
            "acierto": float(won.mean())},
        "Elo simple": _prob_metrics(decided["elo_prob"].to_numpy(), won),
        "Modelo": _prob_metrics(decided["home_win_prob"].to_numpy(), won),
    }
    with_line = decided[decided["vegas_spread"].notna()]
    vegas_prob = norm.cdf(with_line["vegas_spread"] / with_line["sd"])
    rows["Las Vegas (línea de cierre)"] = _prob_metrics(
        vegas_prob, (with_line["result"] > 0).to_numpy(dtype=float))
    summary = pd.DataFrame(rows).T
    summary.loc["Modelo", "mae_margen"] = (preds["pred_margin"] - preds["result"]).abs().mean()
    summary.loc["Modelo", "mae_total"] = (preds["pred_total"] - preds["total"]).abs().mean()
    lined = preds.dropna(subset=["vegas_spread", "vegas_total"])
    summary.loc["Las Vegas (línea de cierre)", "mae_margen"] = (lined["vegas_spread"] - lined["result"]).abs().mean()
    summary.loc["Las Vegas (línea de cierre)", "mae_total"] = (lined["vegas_total"] - lined["total"]).abs().mean()
    summary.loc["Siempre el local", "mae_margen"] = (preds["result"] - preds["result"].mean()).abs().mean()
    summary.loc["Siempre el local", "mae_total"] = (preds["total"] - preds["total"].mean()).abs().mean()
    summary.attrs["partidos"] = len(preds)
    return preds, summary


def against_the_spread(preds: pd.DataFrame, min_edge: float = 0.0) -> dict[str, float]:
    """Récord del modelo contra el spread de cierre cuando discrepa al menos `min_edge` puntos."""
    lined = preds.dropna(subset=["vegas_spread"])
    edge = lined["pred_margin"] - lined["vegas_spread"]
    cover = lined["result"] - lined["vegas_spread"]          # >0: el local cubrió
    bets = lined[(edge.abs() >= min_edge) & (cover != 0)]
    wins = (np.sign(edge[bets.index]) == np.sign(cover[bets.index])).sum()
    n = len(bets)
    return {"apuestas": n, "ganadas": int(wins), "pct": float(wins / n) if n else float("nan")}
