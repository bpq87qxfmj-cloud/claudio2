# claudio2 — predictor de la NFL

Predice **partidos** (ganador, probabilidad, margen, total y marcador) y
**estadísticas de jugadores** (QB, RB, WR, TE) para la siguiente jornada, usando
los datos públicos de [nflverse](https://github.com/nflverse/nflverse-data)
(calendario, resultados y estadísticas semanales desde 1999).

## Uso

```bash
pip install -r requirements.txt

python -m nfl_predictor partidos                    # próxima jornada pendiente
python -m nfl_predictor partidos --temporada 2026 --semana 12
python -m nfl_predictor jugadores                   # top 25 por posición
python -m nfl_predictor jugadores --equipo KC
python -m nfl_predictor jugadores --pos WR --top 50 --csv predictions/wr.csv
python -m nfl_predictor jugadores --jugador mahomes
python -m nfl_predictor backtest                    # precisión histórica
```

La primera corrida descarga ~15 MB a `data/`; después solo se refrescan (cada 6 h)
los archivos de la temporada en curso. Cada corrida reentrena con todo lo jugado
hasta el momento (tarda unos segundos).

## Cómo funciona

**Partidos** (`nfl_predictor/games.py`) — regresión ridge sobre variables
calculadas solo con información previa al kickoff:

- Rating Elo de cada equipo (con margen de victoria y regresión entre temporadas).
- Forma reciente (promedio móvil exponencial): puntos a favor/en contra, EPA por
  jugada ofensivo y defensivo (pase y carrera), yardas por jugada, entregas de balón.
- Valor del QB titular anunciado (EPA por partido, encogido a un prior para QBs
  con poco historial), así el modelo reacciona cuando juega un suplente.
- Descanso, campo neutral, rival divisional, estadio techado.

El margen predicho se convierte en probabilidad de victoria con la dispersión
histórica de los errores (~13 puntos). Las líneas de Las Vegas **no** son una
variable del modelo; se muestran al lado solo como comparación.

**Jugadores** (`nfl_predictor/players.py`) — un gradient boosting (pérdida de
Poisson) por cada posición y estadística, con: producción reciente y de fondo del
jugador, su participación en targets/acarreos, volumen ofensivo del equipo, lo que
la defensa rival le permite a esa posición, calidad del QB titular, localía y el
total implícito del equipo (línea de Las Vegas, o la predicción del modelo de
partidos si aún no hay línea). Se proyectan los jugadores activos en los dos
últimos partidos de su equipo, sin los reportados *Out/Doubtful*, y solo el QB
titular anunciado. Los puntos de fantasy (PPR) se derivan de esas proyecciones.

## Qué tan bueno es (backtest walk-forward)

Cada temporada se predice con un modelo entrenado únicamente en las anteriores.

Partidos, 2015–2026 (3,164 partidos):

| | Acierto | Brier | MAE margen | MAE total |
|---|---|---|---|---|
| Siempre el local | 54.7% | 0.248 | 10.97 | 11.01 |
| Elo simple | 64.2% | 0.222 | – | – |
| **Modelo** | **64.9%** | **0.218** | **10.09** | **10.63** |
| Las Vegas (cierre) | 66.2% | 0.213 | 9.79 | 10.45 |

Jugadores, 2024–2026 (QB titulares y jugadores con rol), error absoluto medio:

| Estadística | Modelo | Mejor promedio ingenuo | Mejora |
|---|---|---|---|
| QB yardas de pase | 58.5 | 63.6 | 8% |
| QB intentos | 6.7 | 7.5 | 10% |
| RB yardas por tierra | 22.3 | 23.5 | 5% |
| RB acarreos | 3.8 | 4.0 | 6% |
| WR yardas por recepción | 26.8 | 27.7 | 3% |
| WR targets | 2.3 | 2.5 | 5% |
| TE yardas por recepción | 21.7 | 22.1 | 2% |

## Limitaciones (léelas antes de apostar)

- **No le gana al mercado.** Contra el spread de cierre acierta ~50–51%
  (se necesita 52.4% para no perder con momio −110). Es una buena referencia
  independiente, no una máquina de dinero.
- Los touchdowns e intercepciones son casi puro azar semana a semana: ahí el
  modelo no mejora al promedio simple.
- No conoce lesiones de última hora, cambios de rol ni clima; las proyecciones de
  jugadores suponen que el jugador juega con su rol reciente.
- Un partido de la NFL tiene ~13 puntos de desviación: incluso una predicción
  perfecta del "valor esperado" falla seguido.

## Pruebas

```bash
python -m pytest tests
```
