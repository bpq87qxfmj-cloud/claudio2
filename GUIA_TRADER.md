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
   - **3. Señal de rebote**: córrelo a las 2:45 pm (hora de CDMX).
   - **4. Backtest**: muestra qué estrategias funcionan.
5. El resultado aparece abajo, en la terminal.
