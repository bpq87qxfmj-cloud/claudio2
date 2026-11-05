# Guía de uso: skills trader + Capital.com

Todo esto es educativo. No es asesoría financiera. Empieza **siempre en la cuenta demo**.

## 0. Preparación (una sola vez)

1. Abre Claude Code en este repo (`claudio2`). Las skills están en `.claude/skills/` y se cargan solas.
2. En Capital.com, entra en la **cuenta demo**, ve a *Ajustes > API integrations* y genera una clave API con su contraseña personalizada.
3. En **tu terminal** (no en el chat):
   ```bash
   export CAPITAL_API_KEY="..."
   export CAPITAL_IDENTIFIER="tu-email"
   export CAPITAL_PASSWORD="contraseña-de-la-clave"
   export SEC_USER_AGENT="Tu Nombre tu@email.com"   # la SEC lo exige para leer sus datos
   ```
4. Comprueba la conexión: escríbele a Claude **"/trader-capital account"**. Debe decir *Cuenta DEMO*.
5. Escríbele **"/trader-budget"**. Te pregunta tus reglas (presupuesto, % en efectivo, máximo por acción) y crea `trader/config.json`. Añade `"broker": "capital"`.

## 1. Invertir copiando a un inversor (largo plazo)

| Paso | Le dices a Claude | Qué pasa |
|---|---|---|
| 1 | "copia el top 5 de Berkshire" | `/trader-copycat` lee el informe 13F de la SEC → `trader/targets.json` |
| 2 | "precios de capital para AAPL AXP GOOGL KO BAC" | precios reales de Capital.com → `trader/prices.json` |
| 3 | "reparte 1000 dólares" | `/trader-budget` → `trader/orders.json` (sólo propuestas) |
| 4 | "revisa el riesgo" | `/trader-risk` → PASS o BLOCK por orden |
| 5 | "muéstrame los tickets de capital" | epic, unidades, exposición y **cuánto te cuesta el spread** |
| 6 | "envía la orden 1" | Claude te muestra UNA orden. Tú respondes **yes**. Se registra y `capital.py place` la envía a la demo |
| 7 | "mis posiciones" | P&L de lo abierto |

Repite el paso 6 orden por orden. "Sí a todo" no vale: lo bloquea a propósito.

## 2. Vigilar (cada día)

- "¿noticias de mis acciones?" → `/trader-news`
- "¿qué ha comprado Pelosi?" → `/trader-pelosi` (son operaciones públicas, declaradas con días o semanas de retraso)
- "avísame si AAPL baja de 320" → `/trader-alerts`
- "¿está abierta la bolsa?" → `/trader-alerts clock`

## 3. Scalping: probar antes de arriesgar

```bash
python3 .claude/skills/trader-capital/capital.py candles US500 --resolution MINUTE_5 --days 60 --out trader/velas_us500.csv
python3 scalping.py --csv trader/velas_us500.csv
python3 scalping.py --csv trader/velas_us500.csv --umbral 0.62 --tp 0.5 --sl 0.3 --horizonte 12
```

Con `--csv`, el backtest usa **el spread real de Capital.com** como coste. Antes de operar, la estrategia tendría que cumplir esto:
- El *profit factor* con costes es mayor que 1.
- Supera claramente a las entradas al azar.
- Aguanta en un periodo distinto del que usaste para ajustarla.
- Funciona en demo durante semanas.

## 4. Reglas de oro

- **CFDs = apalancamiento.** Un 5% en contra con apalancamiento 20:1 te quita el 100% del margen. Usa siempre `--stop-pct`.
- El spread y la financiación nocturna son tu coste real, aunque no paguen "comisión".
- Las claves nunca van en el chat, y la cuenta real sólo se activa con `CAPITAL_LIVE=1`, que pones tú.

## 5. Señales de mayor probabilidad

```bash
python3 alta_probabilidad.py               # US500
python3 alta_probabilidad.py --ticker ^NDX # US100
```

Al final muestra la sección **HOY**:
- Si hay **señal de rebote**, la regla compraría al cierre y vendería cuando el precio cierre por encima de la media de 5 días.
- La **probabilidad de que mañana sea un día movido** sirve para ajustar los stops y el tamaño de la posición.

## 6. Day trading / scalping

Antes de la apertura y justo después de las 9:30 (Nueva York):
```bash
python3 daytrading.py --cuenta 1000 --riesgo 1
```
1. **Rango esperado**: cuántos puntos se moverá hoy. No pongas objetivos mayores que lo que queda de rango.
2. **Stop**: como mínimo la mitad del rango esperado. Si lo pones más cerca, el ruido normal del día te saca.
3. **Tamaño**: con ese stop, el programa te dice cuántos contratos abrir para arriesgar solo el % que elegiste.
4. **Límite diario**: después de 3 operaciones perdedoras, deja de operar ese día.

`python3 daytrading.py --backtest` muestra qué reglas de entrada tienen ventaja real tras el spread. Casi ninguna la tiene.

## 7. Usarlo en VS Code (sin escribir comandos)

1. Instala Python desde python.org, marcando **"Add Python to PATH"**. Después instala VS Code.
2. Abre la carpeta `claudio2` en VS Code. Si te ofrece instalar las extensiones recomendadas de Python, acepta.
3. **Terminal → Ejecutar tarea… → "Instalar librerías"** (solo la primera vez).
4. Pulsa **Ctrl+Shift+D** (Ejecutar y depurar), elige en la lista de arriba uno de estos programas y pulsa ▶:
   - **1. Plan del día US500**: te pregunta tu saldo y el % de riesgo.
   - **3. Señal de rebote**: córrelo a las 15:45 de Nueva York (2:45 pm de CDMX en invierno, 1:45 pm en verano).
   - **4. Backtest**: muestra qué estrategias funcionan.
5. El resultado aparece abajo, en la terminal.

## 8. Agente automático (opera solo)

`agente.py` hace solo la regla del rebote RSI(2), la única que mostró ventaja estable después del spread:

- A las **15:45 de Nueva York** (2:45 pm de CDMX en invierno, 1:45 pm en verano), si el RSI(2) está por debajo de 10 y el índice está por encima de su media de 200 días, **compra** un CFD del US500 con un **stop de -5%** puesto en Capital.com.
- Los días siguientes, a la misma hora, **vende** cuando el precio supera su media de 5 días, o a los 10 días como máximo.
- No hace scalping: con 10 meses de velas de 5 minutos, todas las reglas intradía perdieron después del spread.

**Qué esperar:** desde 1996, unas 8 operaciones al año, con un acierto del 72% y un profit factor de 1.9. Con exposición 1 (sin apalancamiento) son **~3% al año** y una peor caída de -17%. Comprar y mantener el índice dio ~8% al año, pero con una caída de -57%. Pasa semanas sin operar: es normal.

### Paso 1: pruébalo sin claves
En VS Code, **Ctrl+Shift+D**:
- **7. Agente: backtest de sus reglas**: cómo le habría ido desde 1995.
- **8. Agente: ¿qué haría hoy?**: la decisión con el último precio. No envía nada.

### Paso 2: conéctalo a tu cuenta DEMO
1. En Capital.com, cambia a la **cuenta demo**. Ve a Ajustes → **API integrations** → genera una clave con una contraseña propia para la clave.
2. En **PowerShell** (nunca en el chat), una sola vez:
   ```powershell
   setx CAPITAL_API_KEY "tu-clave"
   setx CAPITAL_IDENTIFIER "tu-email-de-capital"
   setx CAPITAL_PASSWORD "la-contraseña-de-la-clave"
   ```
3. Cierra VS Code y vuelve a abrirlo, para que vea las variables nuevas.
4. **9. Agente: probar conexión** debe decir "Conectado a la cuenta DEMO", con tu saldo y el tamaño que usaría. No envía órdenes.

### Paso 3: déjalo trabajar
Tienes dos formas:
- **A. Con VS Code abierto:** **10. Agente automático** se queda encendido y espera la hora. La PC no se puede dormir.
- **B. Con el Programador de tareas (recomendado):** en la terminal de VS Code, una vez:
  ```powershell
  python agente.py --instalar-windows
  ```
  Esto crea dos tareas, de lunes a viernes a las 13:46 y a las 14:46. Una de ellas cae a las 15:46 de Nueva York; la otra cubre el cambio de horario de EE. UU. y no hace nada. La PC debe estar prendida y con tu sesión abierta a esa hora. Si acabas de usar `setx`, cierra sesión de Windows y vuelve a entrar. Para quitar las tareas: `python agente.py --desinstalar-windows`.

Lo que hace queda en `registro_agente/agente.log` (texto) y `registro_agente/log.csv` (se abre en Excel).

### Avisos al celular (opcional, Telegram)
1. En Telegram, abre **BotFather** → `/newbot` → te da un **token**.
2. Mándale cualquier mensaje a tu bot. Luego abre `https://api.telegram.org/bot<TOKEN>/getUpdates` en el navegador y copia el número de `"chat":{"id": ...}`.
3. En PowerShell: `setx TELEGRAM_BOT_TOKEN "..."` y `setx TELEGRAM_CHAT_ID "..."`.

Cada día te llega qué decidió: esperar, comprar, mantener o vender.

### Candados que trae
- **Siempre usa la DEMO.** Para la cuenta real hacen falta dos cosas que haces tú: `CAPITAL_LIVE=1` **y** `--real`. Recomendación: al menos 2 o 3 meses en la demo, comparando con el backtest, antes de pensarlo.
- Tiene una posición como máximo por índice; nunca promedia. No toca las posiciones que abras tú; si ya tienes una en el US500, no abre otra.
- No compra si el spread está anormalmente ancho, si el mercado no está operable o si la cuenta cayó más de 20% desde su máximo.
- La exposición máxima es 2 veces tu saldo. `--exposicion 1` equivale a comprar el índice con todo tu saldo, sin apalancamiento.
- Si apagas el agente, las posiciones siguen en Capital.com con su stop.

Opciones: `--indices us500,us100` (reparte la exposición entre los dos), `--exposicion 0.5`, `--stop 5`, `--max-caida 20`.

