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
