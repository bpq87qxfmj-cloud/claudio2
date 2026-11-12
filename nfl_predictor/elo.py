"""Ratings Elo con margen de victoria y regresión a la media entre temporadas."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

BASE = 1500.0
K = 20.0
HOME_ADVANTAGE = 48.0       # puntos Elo; solo se usa para actualizar los ratings
SEASON_REGRESSION = 1 / 3   # fracción que cada equipo regresa a la media en el receso


def expected_score(diff: float) -> float:
    """Probabilidad de victoria para una diferencia de Elo dada."""
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def add_elo(games: pd.DataFrame) -> pd.DataFrame:
    """Agrega `home_elo` y `away_elo`: el rating de cada equipo ANTES del partido.

    `games` debe venir en orden cronológico. Los partidos sin resultado reciben
    el rating vigente y no lo modifican.
    """
    ratings: dict[str, float] = {}
    season_seen: int | None = None
    home_elo = np.empty(len(games))
    away_elo = np.empty(len(games))

    rows = zip(games["season"].to_numpy(), games["home_team"].to_numpy(),
               games["away_team"].to_numpy(), games["result"].to_numpy(),
               (games["location"] == "Neutral").to_numpy())
    for i, (season, home, away, result, neutral) in enumerate(rows):
        if season != season_seen:
            if season_seen is not None:
                for team in ratings:
                    ratings[team] += SEASON_REGRESSION * (BASE - ratings[team])
            season_seen = season
        rh = ratings.setdefault(home, BASE)
        ra = ratings.setdefault(away, BASE)
        home_elo[i], away_elo[i] = rh, ra
        if np.isnan(result):
            continue
        diff = rh - ra + (0.0 if neutral else HOME_ADVANTAGE)
        outcome = 1.0 if result > 0 else 0.0 if result < 0 else 0.5
        # Multiplicador por margen (estilo FiveThirtyEight): premia palizas pero
        # descuenta las que consigue el favorito, para que el rating no se infle.
        winner_diff = diff if result >= 0 else -diff
        multiplier = math.log(abs(result) + 1.0) * 2.2 / (winner_diff * 0.001 + 2.2)
        delta = K * multiplier * (outcome - expected_score(diff))
        ratings[home] = rh + delta
        ratings[away] = ra - delta

    out = games.copy()
    out["home_elo"] = home_elo
    out["away_elo"] = away_elo
    return out
