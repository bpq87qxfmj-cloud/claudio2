"""Descarga y caché de los datos públicos de nflverse.

Fuentes (https://github.com/nflverse/nflverse-data/releases):
  * schedules/games.csv            -> calendario, resultados, líneas de apuestas, QB titular
  * stats_player/stats_player_week -> estadísticas semanales por jugador (1999+)
  * injuries/injuries              -> reporte de lesiones (opcional)
"""

from __future__ import annotations

import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"
DATA_DIR = Path(os.environ.get("NFL_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
REFRESH_SECONDS = 6 * 3600  # los archivos de la temporada en curso se refrescan cada 6 h

# Franquicias que cambiaron de sede: todo se normaliza a la abreviatura actual.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

FIRST_SEASON = 1999

PLAYER_COLUMNS = [
    "player_id", "player_display_name", "position", "position_group",
    "season", "week", "season_type", "game_id", "team", "opponent_team",
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "sacks_suffered", "sack_fumbles_lost", "passing_epa",
    "carries", "rushing_yards", "rushing_tds", "rushing_fumbles_lost", "rushing_epa",
    "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_fumbles_lost",
    "target_share", "fantasy_points_ppr",
]


def _download(url: str, dest: Path, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "nfl-predictor"})
            with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as fh:
                shutil.copyfileobj(response, fh)
            tmp.replace(dest)
            return
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise FileNotFoundError(url) from exc
            last_error = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"No se pudo descargar {url}: {last_error}")


def _fetch(release: str, filename: str, *, refresh: bool) -> Path:
    """Devuelve la ruta local del archivo, descargándolo si falta o está viejo."""
    path = DATA_DIR / release / filename
    stale = path.exists() and refresh and (time.time() - path.stat().st_mtime) > REFRESH_SECONDS
    if not path.exists() or stale:
        try:
            _download(f"{BASE_URL}/{release}/{filename}", path)
        except (RuntimeError, FileNotFoundError):
            if not path.exists():
                raise
            # Sin red pero con copia local: se usa la copia.
    return path


def load_games(force: bool = False) -> pd.DataFrame:
    """Un renglón por partido (jugado o programado), ordenado cronológicamente."""
    path = DATA_DIR / "schedules" / "games.csv"
    if force and path.exists():
        path.unlink()
    games = pd.read_csv(_fetch("schedules", "games.csv", refresh=True))
    games["home_team"] = games["home_team"].replace(TEAM_ALIASES)
    games["away_team"] = games["away_team"].replace(TEAM_ALIASES)
    games["gameday"] = pd.to_datetime(games["gameday"])
    games = games[games["season"] >= FIRST_SEASON]
    return games.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def current_season(games: pd.DataFrame) -> int:
    return int(games["season"].max())


def load_player_stats(games: pd.DataFrame, first_season: int = FIRST_SEASON,
                      force: bool = False) -> pd.DataFrame:
    """Estadísticas semanales de QB/RB/WR/TE para todas las temporadas disponibles."""
    last = current_season(games)
    frames = []
    for season in range(first_season, last + 1):
        filename = f"stats_player_week_{season}.parquet"
        path = DATA_DIR / "stats_player" / filename
        if force and season == last and path.exists():
            path.unlink()
        try:
            path = _fetch("stats_player", filename, refresh=season >= last)
        except FileNotFoundError:
            continue  # temporada que todavía no tiene partidos jugados
        frame = pd.read_parquet(path, columns=PLAYER_COLUMNS)
        frames.append(frame[frame["position_group"].isin(["QB", "RB", "WR", "TE"])])
    stats = pd.concat(frames, ignore_index=True)
    stats = stats.dropna(subset=["player_id", "team"])
    stats["team"] = stats["team"].replace(TEAM_ALIASES)
    stats["opponent_team"] = stats["opponent_team"].replace(TEAM_ALIASES)
    numeric = [c for c in PLAYER_COLUMNS if c not in (
        "player_id", "player_display_name", "position", "position_group",
        "season", "week", "season_type", "game_id", "team", "opponent_team",
        "passing_epa", "rushing_epa", "target_share")]
    stats[numeric] = stats[numeric].fillna(0)
    return stats


def load_injuries(season: int) -> pd.DataFrame | None:
    """Reporte de lesiones de la temporada; None si no está disponible."""
    try:
        path = _fetch("injuries", f"injuries_{season}.csv", refresh=True)
    except (FileNotFoundError, RuntimeError):
        return None
    injuries = pd.read_csv(path)
    injuries["team"] = injuries["team"].replace(TEAM_ALIASES)
    return injuries


def next_week(games: pd.DataFrame) -> tuple[int, int]:
    """(temporada, semana) del primer partido sin resultado."""
    pending = games[games["result"].isna()]
    if pending.empty:
        raise ValueError("No hay partidos pendientes en el calendario.")
    first = pending.iloc[0]
    return int(first["season"]), int(first["week"])


