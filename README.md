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
