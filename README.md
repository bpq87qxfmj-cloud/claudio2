# claudio2 · predictor de movimientos para scalping (barras de 1 minuto)

Sistema completo para predecir el movimiento de los próximos 10 minutos y convertirlo en una
señal de scalping (COMPRAR / VENDER / ESPERAR con TP, SL y salida por tiempo), validado fuera de
muestra con costes de operación. Incluye un agente autónomo que opera solo (en simulación por
defecto), gestiona el riesgo, se reentrena y se detiene por sí mismo cuando pierde.

## Conclusión en corto

- **Mercado elegido: oro (XAUUSD).** De los tres candidatos es el único en el que el modelo
  encuentra una señal predictiva consistente fuera de muestra (acierta la dirección por encima
  del azar en el 79 % de los meses de diseño y en 6 de los 7 meses reservados).
- **La señal es real pero pequeña: no cubre los costes de una cuenta minorista.** Sin costes gana
  0,08 $ por onza y operación (t = 3,3 en 18 000 operaciones y 35 meses); el coste asumido de ida
  y vuelta es 0,30 $. En los meses reservados (may–nov 2026) el resultado neto fue **negativo**:
  −0,15 $ por operación, −213 $ por onza en total.
- **Nasdaq-100 y S&P 500: sin señal aprovechable** con datos de precio de 1 minuto.
- Por eso el modelo final publicado responde **ESPERAR** con el coste por defecto: solo emite
  órdenes si la validación encuentra ventaja neta para *tu* coste (`train --cost ...`). No hay
  evidencia de que este predictor sea rentable operando a mercado; úsalo como base de
  investigación, no como sistema para dinero real.

## Resultados

Walk-forward mensual: se entrena con los 12 meses anteriores, se opera el mes siguiente. Los
resultados están en unidades de precio del instrumento (en oro, $ por onza).

**Periodo de diseño** (ene-2024 a abr-2026, 28 meses)

| mercado | coste | IC | meses IC>0 | meses activos | ops | acierto | bruto/op | neto/op | neto total | t-stat |
|---|---|---|---|---|---|---|---|---|---|---|
| XAUUSD | 0,30 | 0,0128 | 79 % | 6/28 | 359 | 52,6 % | +0,381 | +0,081 | +29,1 | +0,39 |
| NAS100 | 1,00 | 0,0012 | 54 % | 2/28 | 122 | 44,3 % | −2,332 | −3,332 | −406,5 | −2,24 |
| SPX500 | 0,50 | 0,0015 | 64 % | 0/28 | 0 | – | – | – | 0 | – |

![diseño](reports/design_equity.png)

**Periodo reservado** (may-2026 a 6-nov-2026; no se usó para ninguna decisión y se evaluó una vez)

| mercado | coste | IC | meses IC>0 | meses activos | ops | acierto | bruto/op | neto/op | neto total | t-stat |
|---|---|---|---|---|---|---|---|---|---|---|
| XAUUSD | 0,30 | 0,0145 | 86 % | 5/7 | 1394 | 49,9 % | +0,147 | −0,153 | −213,1 | −1,43 |
| NAS100 | 1,00 | 0,0068 | 71 % | 1/7 | 660 | 49,1 % | −1,104 | −2,104 | −1388,3 | −1,94 |
| SPX500 | 0,50 | 0,0064 | 71 % | 0/7 | 0 | – | – | – | 0 | – |

![lockbox](reports/lockbox_equity.png)

**Sensibilidad al coste en oro** (ene-2024 a nov-2026; se repite todo el proceso con cada coste)

| coste ida y vuelta | meses activos | ops | acierto | bruto/op | neto/op | neto total | t-stat |
|---|---|---|---|---|---|---|---|
| 0 | 33/35 | 18 083 | 51,8 % | +0,080 | +0,080 | +1449,4 | +3,30 |
| 0,075 | 23/35 | 6 278 | 51,6 % | +0,136 | +0,061 | +382,3 | +1,03 |
| 0,15 | 20/35 | 4 700 | 50,6 % | +0,088 | −0,062 | −291,0 | −0,89 |
| 0,30 (supuesto) | 11/35 | 1 753 | 50,5 % | +0,195 | −0,105 | −184,0 | −1,10 |
| 0,60 | 1/35 | 40 | 70,0 % | +1,556 | +0,956 | +38,2 | +1,76 |

El punto de equilibrio está en torno a 0,08–0,10 $ por onza de coste total, por debajo de lo que
se paga cruzando el spread en CFD o en futuros (un tick de GC ya son 0,10 $). La fila de 0,60 son
40 operaciones en un solo mes: no es evidencia de nada.

IC = correlación de rangos entre predicción y resultado real. "Meses activos" = meses en los que
la validación encontró ventaja neta y el sistema operó; el resto se queda fuera.

## Cómo funciona

| pieza | archivo | qué hace |
|---|---|---|
| Datos | `scalper/data.py` | Histórico M1 2023→hoy de histdata.com (bid), últimos días de Yahoo (futuros) o tu CSV |
| Variables | `scalper/features.py` | 55 variables causales y adimensionales: momento/reversión a varios horizontes, régimen de volatilidad, forma de vela, posición en el rango, medias, RSI, hora y contexto de sesión |
| Objetivo | `scalper/labels.py` | Triple barrera: entrada en la apertura de la barra siguiente, TP = SL = volatilidad × √10, máximo 10 min. Si una barra toca ambas, se cuenta el stop |
| Modelo | `scalper/model.py` | LightGBM que estima el resultado esperado de un largo (negativo ⇒ corto). Umbral de entrada elegido en validación; si no hay ventaja neta, no opera |
| Backtest | `scalper/backtest.py` | Una posición a la vez, coste de ida y vuelta en cada operación, sin operar a través de cierres de sesión |
| Informe | `scalper/report.py` | Comparación de mercados, curvas y sensibilidad al coste |
| Agente | `scalper/agent.py`, `scalper/brokers.py`, `scalper/capital.py` | Bucle autónomo, gestión de riesgo, cuenta simulada y adaptadores de Capital.com y MetaTrader 5 |

Salvaguardas contra el autoengaño: purga de etiquetas solapadas entre entrenamiento, validación y
test; parámetros fijados antes de mirar el test; seis meses finales reservados; tests que
comprueban que las variables no miran al futuro y que en un paseo aleatorio no "aparece" ventaja.

También se probaron, solo sobre el periodo de diseño, horizontes de 5/20/30/60 min, un modelo más
regularizado, 24 meses de entrenamiento y variables de otros mercados (plata, EURUSD, USDJPY para
el oro; S&P para el Nasdaq). Ninguna cambió la conclusión, así que se mantuvo la configuración
inicial.

## Agente autónomo

`scalper/agent.py` convierte el predictor en un agente que opera solo, minuto a minuto:

1. Lee las barras cerradas, actualiza la posición abierta y registra las operaciones cerradas.
2. Comprueba los límites de riesgo; si alguno salta, no abre nada.
3. Pide la predicción al modelo y, si supera el umbral validado, abre una operación con TP, SL
   y caducidad a los 10 minutos, dimensionada para arriesgar un 0,25 % del capital.
4. Guarda su estado en `state/` (reanuda tras un reinicio) y un diario de operaciones en CSV.
5. Se reentrena cada 7 días con datos nuevos. Si la validación no encuentra ventaja neta, se
   queda fuera hasta el siguiente reentrenamiento.

| control | valor por defecto | efecto |
|---|---|---|
| Riesgo por operación | 0,25 % del capital | tamaño = riesgo / distancia al stop, redondeado al lote |
| Pérdida diaria máxima | 2 % | deja de abrir hasta la jornada siguiente (17:00 Nueva York) |
| Rachas | 5 pérdidas seguidas | pausa de 60 minutos |
| Caída máxima | 10 % desde el máximo | se detiene; hay que reanudarlo a mano con `--reset-halt` |
| Monitor de ventaja | t < −2 en las últimas 100 operaciones | se suspende hasta reentrenar |
| Cierre de sesión | 11 min antes del cierre diario | no abre operaciones que cruzarían el cierre |
| Datos caducados / spread alto | 3 min / spread por encima del coste indicado | no abre |
| Parada manual | crear el archivo `state/STOP` | cierra la posición y termina |

```bash
python -m scalper replay --market XAUUSD --start 2026-05        # el agente revive el histórico
python -m scalper agent  --market XAUUSD                        # en vivo, cuenta simulada, datos de Yahoo
python -m scalper agent  --market XAUUSD --feed capital         # simulado con precios en tiempo real de Capital.com
python -m scalper agent  --market XAUUSD --broker capital       # órdenes en la cuenta DEMO de Capital.com
python -m scalper agent  --market XAUUSD --broker mt5           # órdenes en MetaTrader 5 (demo)
```

**Qué habría hecho** con 10 000 $ en los meses reservados (may–nov 2026, reentrenando cada mes):
892 operaciones, 49,8 % de acierto, capital 10 000 → 9 002 $ (**−10 %**), y se detuvo solo en
septiembre al alcanzar la caída máxima. Antes de eso se suspendió dos veces por el monitor de
ventaja y pasó 59 000 minutos fuera por no encontrar ventaja validada. Es coherente con el
backtest: sin límites de riesgo el agente reproduce el walk-forward (1 390 operaciones y −211,7
frente a 1 394 y −213,1; la diferencia son las entradas junto al cierre de sesión).

### Capital.com

`scalper/capital.py` conecta con la [API oficial de Capital.com](https://open-api.capital.com):
barras de 1 minuto en tiempo real con bid y ask, y órdenes a mercado con TP y SL en el servidor.

1. En Capital.com activa la verificación en dos pasos y genera una clave en
   **Ajustes > Integraciones API** (se le pone una contraseña propia).
2. Expórtalas como variables de entorno, sin escribirlas en ningún archivo del repositorio:

   ```bash
   export CAPITAL_API_KEY=...        # la clave generada
   export CAPITAL_IDENTIFIER=...     # tu correo de Capital.com
   export CAPITAL_PASSWORD=...       # la contraseña de la clave API, no la de la cuenta
   ```
3. Comprueba la conexión y mide tu spread real, que es el dato que decide todo:

   ```bash
   python -m scalper capital --market XAUUSD
   ```
4. Entrena con ese coste y deja al agente en la cuenta demo:

   ```bash
   python -m scalper train --market XAUUSD --cost <spread medido>
   python -m scalper agent --market XAUUSD --broker capital --cost <spread medido>
   ```

Detalles del adaptador:

- Usa siempre el servidor **demo** salvo que se pase `--allow-real`.
- Capital.com no cobra comisión en CFD: el coste es el spread. El oro se anuncia con spread
  desde 0,30, justo el coste supuesto en este proyecto, con el que el modelo no encuentra ventaja
  neta. Si `capital` mide ese spread, lo esperable es que el agente se quede fuera.
- El agente no entra si el spread del momento supera en un 25 % el coste indicado.
- Las barreras del modelo están en precio bid: en los cortos se desplazan el spread porque se
  cierran en ask.
- El resultado de cada operación se toma de la variación del saldo, así que conviene una cuenta
  (o subcuenta) dedicada: si operas a mano en la misma a la vez, las cifras se mezclan.
- **Probado contra un doble de pruebas de la API documentada y contra las respuestas de error del
  servidor demo real, pero no con una sesión autenticada**: no tengo tus credenciales. La primera
  ejecución en demo es la prueba real.

Modos de ejecución:

- **Simulado (por defecto):** las órdenes se ejecutan en la apertura de la barra siguiente con
  las mismas reglas que el backtest (hay un test que exige resultados idénticos).
- **MetaTrader 5 (`--broker mt5`):** envía órdenes reales con TP/SL en el servidor. Solo funciona
  en Windows con el terminal abierto y el paquete `MetaTrader5`. Se niega a operar en una cuenta
  que no sea demo salvo que se pase `--allow-real`. **Este adaptador solo se ha probado contra un
  doble de pruebas, no contra un terminal real**: hay que validarlo en demo antes de nada.

## Uso

```bash
pip install -r requirements.txt
python -m scalper download                 # histórico M1 de los tres mercados (~1,3 M de barras cada uno)
python -m scalper compare                  # walk-forward y ranking en el periodo de diseño
python -m scalper lockbox                  # evaluación en los meses reservados
python -m scalper train --market XAUUSD --cost 0.30   # modelo final con TU coste real
python -m scalper predict --market XAUUSD             # señal con datos de Yahoo
python -m scalper predict --market XAUUSD --csv barras_mt5.csv --tz Europe/Athens --loop
pytest -q
```

Ejemplo de salida:

```
[2026-11-10 19:03:00+00:00] XAUUSD cierre=4035.30  pred=-0.0299 (umbral sin ventaja validada)  sesgo=neutral  => ESPERAR
```

## Limitaciones

- Los costes (`scalper/config.py`) son supuestos; mide el spread y la comisión reales de tu bróker.
- El histórico es cotización bid sin volumen ni libro de órdenes. El flujo de órdenes es justo la
  información que más ayuda a esta escala y aquí no está.
- Yahoo sirve los futuros con ~10 minutos de retraso: vale para probar, no para operar. Para
  tiempo real hay que alimentar `predict --csv` (o `model.live_signal`) con barras del bróker.
- El backtest asume ejecución en la apertura de la barra siguiente sin deslizamiento adicional al
  coste fijo; en noticias el deslizamiento real es mayor.
- El reloj de histdata no es el "EST fijo" que documenta: se corrigió contrastándolo con Yahoo.

Esto es software de investigación, no asesoramiento financiero.
