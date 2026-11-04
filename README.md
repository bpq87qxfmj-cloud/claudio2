# claudio2: predictor de mercado

`predictor.py` intenta predecir si la próxima sesión del S&P 500, del NASDAQ o de otro índice cierra al alza o a la baja.

```bash
pip install -r requirements.txt
python predictor.py                 # S&P 500
python predictor.py --ticker ^IXIC  # NASDAQ Composite
python predictor.py --ticker QQQ --start 2010-01-01
```

- Calcula unas 25 señales técnicas: retornos, medias móviles, RSI, MACD, volatilidad y volumen.
- Compara tres modelos: regresión logística, Random Forest y Gradient Boosting.
- Los prueba con validación walk-forward: cada trimestre reentrena solo con datos pasados.
- Los compara con apostar siempre a que sube y con comprar y mantener.

⚠️ Es un proyecto educativo y no es asesoría financiera. Un AUC cercano a 0.50 equivale a lanzar una moneda. El backtest no cuenta comisiones ni deslizamiento.

## Scalping intradía: `scalping.py`

Hace un backtest de scalping con machine learning sobre velas de 1m, 2m o 5m: entra en largo o en corto con take-profit y stop-loss, y descuenta comisiones y deslizamiento.

```bash
python scalping.py                                   # SPY en velas de 5m (últimos 60 días)
python scalping.py --ticker QQQ --intervalo 1m --dias-entreno 3
python scalping.py --tp 0.3 --sl 0.15 --umbral 0.6 --coste-bps 0.5
```

Muestra los resultados sin costes, con costes y con entradas al azar, para ver si el modelo aporta algo de verdad. Guarda cada operación en `operaciones_scalping.csv`.

## Skills de trading + Capital.com

En `.claude/skills/` están las 11 skills de [trader-skill](https://github.com/Jakeschincariol/trader-skill) (MIT) y una nueva, `trader-capital`, que conecta todo con la API de Capital.com (demo por defecto). La guía paso a paso está en [GUIA_TRADER.md](GUIA_TRADER.md).

## Señales de mayor probabilidad: `alta_probabilidad.py`

En vez de adivinar si mañana sube (~54%), mide tres cosas con más probabilidad real, con costes de CFD y fuera de muestra:

1. **Horizonte**: probabilidad de que el índice suba según el plazo, de 1 día a 3 años.
2. **Rebote tras caída**: compra cuando el RSI(2) baja de 10 en tendencia alcista (regla de Connors). Acierta un ~70% de las operaciones.
3. **Volatilidad**: predice si mañana será un día movido o tranquilo (62% en general y ~80% en los días extremos).

```bash
python alta_probabilidad.py               # S&P 500 (US500)
python alta_probabilidad.py --ticker ^NDX # Nasdaq 100 (US100)
```

## Day trading y scalping: `daytrading.py`

```bash
python daytrading.py                                   # plan de hoy para el US500
python daytrading.py --indice us100 --cuenta 2000 --riesgo 0.5
python daytrading.py --backtest                        # qué estrategias intradía tienen ventaja
```

- **Plan del día**: rango esperado del día (acierta si será un día movido o tranquilo el 74% de las veces), hueco de apertura y probabilidad calibrada de que el precio vuelva al cierre de ayer. También calcula el stop mínimo, el tamaño en Capital.com y el límite de pérdida diaria.
- **Backtest**: gap fill (30 años), gap fill con stop (velas de 1 hora), momentum intradía y scalping RSI(2). Una estrategia solo se marca como "con ventaja" si gana en las dos mitades del periodo.
