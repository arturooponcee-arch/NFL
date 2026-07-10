# 🏈 Predictor NFL

Predicción de juegos NFL con datos de [nflverse](https://github.com/nflverse) y XGBoost.

## Qué predice

**Por juego:** puntos por equipo, puntos totales, spread, probabilidad de victoria calibrada.

**Por jugador (props):** yardas de pase/tierra/recepción con rango de incertidumbre propio por jugador (quantile regression), touchdowns (esperado + probabilidad de ≥1), recepciones e intercepciones (Poisson).

## Cómo funciona

- **Modelo dual:** `modelo_vegas` (usa líneas de Vegas → marcadores finos) + `modelo_puro` (sin Vegas → opinión propia). Cuando el puro discrepa >4 pts de Vegas, el juego se marca `value`.
- **Features:** forma de últimos 5 juegos (puntos, yardas, EPA ofensivo/defensivo de play-by-play), localía, descanso, juego divisional, clima (domo/temperatura/viento), líneas de Vegas.
- **Props:** rosters y depth charts oficiales de la temporada actual (titulares QB1, RB1-2, WR1-3, TE1-2), excluye lesionados Out/Doubtful.
- **Backtest walk-forward:** re-entrena semana a semana de 2025 — la métrica honesta de cómo funcionará en temporada real.

## Estructura

| Archivo | Qué es |
|---|---|
| `nfl_pred.py` | Todo el pipeline (carga, features, entrenamiento, predicción, registro) |
| `nfl_predictor.ipynb` | Interfaz interactiva (importa el módulo) |
| `actualizar.py` | Script semanal: predice la próxima semana y evalúa las pasadas |
| `.github/workflows/predicciones.yml` | GitHub Actions: corre `actualizar.py` cada martes |
| `predicciones.csv` | Registro de predicciones (lo commitea el workflow) |

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
jupyter notebook nfl_predictor.ipynb
```

```python
from nfl_pred import *
ctx = inicializar(walk_forward=True)

predecir_juego('SEA', 'NE')      # visitante NE @ local SEA (usa calendario oficial)
predecir_semana(1)               # todos los juegos de la semana, con flag value
guardar_predicciones(1)          # anexa a predicciones.csv
evaluar_predicciones()           # acierto real vs resultados y vs Vegas
```

Abreviaturas nflverse: `KC, BUF, PHI, DAL, SF, SEA, NE, DEN, LA, LAC, GB, BAL, ...`

## Automatización

GitHub Actions corre cada **martes 14:00 UTC**: descarga datos frescos, re-entrena, predice la semana próxima y commitea `predicciones.csv`. En offseason no hace nada. Disparo manual: pestaña *Actions* → *Predicciones semanales* → *Run workflow*.

Manual local: `python actualizar.py` (limpia caché y actualiza todo).

## Datos

`nflreadpy` descarga y cachea automáticamente — sin API key. Cuando la temporada actual tiene juegos jugados, el pipeline los incorpora solo al historial de forma.
