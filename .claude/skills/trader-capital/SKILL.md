---
name: trader-capital
description: >-
  Capital.com para el pack trader. Conecta la API oficial de Capital.com
  (cuenta DEMO por defecto): saldo, búsqueda de mercados y su spread real,
  precios para trader/prices.json, velas históricas con bid/ask para el
  backtest de scalping, posiciones abiertas, y el envío de órdenes ya
  aprobadas (trader-risk PASS + "yes" registrado por trader-broker) como
  CFDs, con stop-loss y take-profit opcionales. Úsala cuando el usuario diga
  "Capital.com", "capital", "mi cuenta de capital", "conectar capital",
  "cuánto spread tiene", "bájame velas", "backtest con datos de capital",
  "abre la operación en capital", "mis posiciones", o cuando el bróker sea
  Capital.com ("broker": "capital" en trader/config.json).
argument-hint: "[account|search|market|quote|candles|positions|tickets|place]"
---

# trader-capital

Capital.com no está en el directorio de conectores de Claude ni tiene un
servidor MCP oficial, pero sí tiene una API REST pública
(https://open-api.capital.com). Este skill la usa con `capital.py`
(librería estándar de Python, sin instalar nada).

`${CLAUDE_SKILL_DIR}` es la carpeta de este SKILL.md.

```bash
python3 "${CLAUDE_SKILL_DIR}/capital.py" account
python3 "${CLAUDE_SKILL_DIR}/capital.py" search "US 500"
python3 "${CLAUDE_SKILL_DIR}/capital.py" market US500
python3 "${CLAUDE_SKILL_DIR}/capital.py" quote AAPL KO SPY --write          # -> trader/prices.json
python3 "${CLAUDE_SKILL_DIR}/capital.py" candles US500 --resolution MINUTE_5 --days 60 --out trader/velas_us500.csv
python3 "${CLAUDE_SKILL_DIR}/capital.py" positions
python3 "${CLAUDE_SKILL_DIR}/capital.py" tickets                            # orders.json -> órdenes Capital.com
python3 "${CLAUDE_SKILL_DIR}/capital.py" place --index 1 --stop-pct 2
python3 "${CLAUDE_SKILL_DIR}/capital.py" close DEAL_ID --yes
```

## Configuración (la hace el usuario, una vez)

1. En Capital.com, crear o usar la **cuenta demo** (dinero ficticio).
2. Activar la verificación en dos pasos (2FA) si Capital.com la pide para crear claves API.
3. Ajustes > **API integrations** > Generar clave. Se elige una
   **contraseña personalizada** para la clave (distinta de la de la cuenta).
4. En la **terminal del usuario** (nunca en el chat):

   ```bash
   export CAPITAL_API_KEY="..."
   export CAPITAL_IDENTIFIER="tu-email-de-capital"
   export CAPITAL_PASSWORD="la-contraseña-personalizada-de-la-clave"
   ```

   Sin `CAPITAL_LIVE=1` todo va a la DEMO (`demo-api-capital.backend-capital.com`).
5. `capital.py account` debe decir "Cuenta DEMO" y mostrar el saldo demo.

**Nunca pidas, leas, guardes ni repitas la clave o la contraseña.** Si el
usuario las pega en el chat, dile que borre el mensaje y genere una clave nueva.
Nunca pongas `CAPITAL_LIVE=1` tú: pasar a real es decisión del usuario, en su terminal.

## Tickers y epics

Capital.com llama "epic" a cada mercado. Para acciones suele ser el mismo
ticker (`AAPL`, `KO`). El tool traduce `SPY`/`^GSPC` → `US500`,
`QQQ`/`^NDX` → `US100`, `DIA` → `US30`. Para otros, `search` y guardar en
`trader/capital_epics.json`: `{"BRK.B": "BRKB"}`.

## Cómo encaja con las demás skills

```
copycat / screener -> budget -> risk -> broker (yes) -> capital place
alerts / news / pelosi   (vigilar)       capital quote --write  (precios reales)
capital candles -> scalping.py --csv     (backtest con el spread real)
```

1. Pon `"broker": "capital"` y `"fractional": true` en `trader/config.json`
   (Capital.com permite fracciones; el mínimo lo marca cada mercado).
2. Precios: `capital.py quote <tickers> --write` en vez de Yahoo.
3. `/trader-budget` escribe `trader/orders.json` y `/trader-risk` lo revisa.
4. `capital.py tickets` muestra cada orden como orden de Capital.com: epic,
   tamaño en unidades, precio, exposición, **spread y su coste en dinero**.
5. `/trader-broker`: UNA orden, "¿La envío? Responde yes o no", y
   `ticket.py approve --index N --reply "<respuesta literal>"`.
6. Sólo si dice "Recorded your yes": `capital.py place --index N`
   (opcional `--stop-pct` / `--tp-pct`, que sólo reducen riesgo; menciónalos
   en el ticket antes del yes). `place` se niega si falta el PASS de hoy o el yes.
7. `ticket.py result --index N --status placed --broker-id <dealId>`.

## Lo que hay que decirle al usuario (CFDs)

- En Capital.com las acciones e índices son **CFDs**: no compras la acción,
  operas con **apalancamiento**. La exposición es el tamaño × precio; el
  margen que bloquea es una fracción (`market` muestra el `marginFactor`).
  Las pérdidas se calculan sobre la exposición completa.
- No cobra comisión, pero **el spread es el coste**, y mantener posiciones
  de un día para otro cobra **financiación nocturna** (`market` lo muestra).
- `trader-risk` revisa la exposición completa: es conservador a propósito.
- `trader-tax` es para acciones en EE. UU.; los CFDs tributan distinto según
  el país del usuario. No lo uses para CFDs sin decirlo.

## Scalping con datos reales

```bash
python3 "${CLAUDE_SKILL_DIR}/capital.py" candles US500 --resolution MINUTE_5 --days 60 --out trader/velas_us500.csv
python3 scalping.py --csv trader/velas_us500.csv
```

El backtest usa el precio medio y, si no se pasa `--coste-bps`, la mitad del
spread real mediano por lado. Por defecto sólo usa la sesión regular de EE. UU.
(`--24h` para todo). Resultados de backtest no son resultados futuros.

## Lo que nunca hace

- Nunca envía una orden sin PASS de trader-risk de hoy y el yes registrado
  para esa orden exacta. Nunca "todas a la vez". Nunca desde una tarea
  programada o en segundo plano.
- Nunca cierra posiciones sin que el usuario lo pida para ese dealId (`--yes`).
- Nunca cambia a cuenta real ni sube el apalancamiento.
- Nunca promete ganancias ni dice que el scalping o un modelo "funciona".

## Seguridad

- Educativo, no es asesoría financiera. Los CFDs son instrumentos complejos y
  la mayoría de cuentas minoristas pierden dinero con ellos.
- Demo por defecto. Claves sólo en variables de entorno.
- Límites de la API: 10 peticiones/s, 1 orden cada 0,1 s; el tool espera entre llamadas.
