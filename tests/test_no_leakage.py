"""Pruebas sin red: las variables solo pueden depender del pasado."""

import numpy as np
import pandas as pd

from nfl_predictor.elo import BASE, add_elo, expected_score
from nfl_predictor.games import _ewm_prior


def _games(results):
    n = len(results)
    return pd.DataFrame({
        "season": [2020] * n,
        "home_team": ["AAA", "BBB"] * (n // 2) + ["AAA"] * (n % 2),
        "away_team": ["BBB", "AAA"] * (n // 2) + ["BBB"] * (n % 2),
        "result": results,
        "location": ["Home"] * n,
    })


def test_elo_starts_at_base_and_rewards_winner():
    out = add_elo(_games([10.0, np.nan]))
    assert out.loc[0, "home_elo"] == BASE and out.loc[0, "away_elo"] == BASE
    # Segundo partido: BBB es local; AAA (visitante) ganó el primero.
    assert out.loc[1, "away_elo"] > BASE > out.loc[1, "home_elo"]
    assert np.isclose(out.loc[1, "away_elo"] + out.loc[1, "home_elo"], 2 * BASE)


def test_elo_ignores_the_result_of_the_same_game():
    a = add_elo(_games([3.0, 7.0, -14.0]))
    b = add_elo(_games([3.0, 7.0, 35.0]))
    assert a.loc[2, "home_elo"] == b.loc[2, "home_elo"]
    assert a.loc[2, "away_elo"] == b.loc[2, "away_elo"]


def test_unplayed_games_do_not_move_ratings():
    out = add_elo(_games([np.nan, np.nan, np.nan]))
    assert (out["home_elo"] == BASE).all() and (out["away_elo"] == BASE).all()


def test_expected_score_is_symmetric():
    assert np.isclose(expected_score(0), 0.5)
    assert np.isclose(expected_score(120) + expected_score(-120), 1.0)


def test_ewm_prior_uses_only_earlier_rows():
    frame = pd.DataFrame({"k": ["x", "x", "x", "y", "y"], "v": [1.0, 2.0, 3.0, 10.0, 20.0]})
    out = _ewm_prior(frame, "k", ["v"], halflife=1.0)["v"]
    assert np.isnan(out[0]) and np.isnan(out[3])       # sin historia previa
    assert out[1] == 1.0 and out[4] == 10.0            # solo el renglón anterior
    assert 1.0 < out[2] < 2.0                          # mezcla de 1 y 2, nunca el 3
    changed = frame.assign(v=[1.0, 2.0, 999.0, 10.0, 999.0])
    assert _ewm_prior(changed, "k", ["v"], halflife=1.0)["v"].equals(out)
