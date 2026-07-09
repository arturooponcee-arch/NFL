# 🏈 Predictor NFL

Notebook que predice juegos de NFL con datos de [nflverse](https://github.com/nflverse) y XGBoost.

## Qué predice

- Puntos por equipo y puntos totales del juego
- Probabilidad de victoria
- Yardas de pase por jugador (QB)
- Yardas por tierra por jugador (RB/QB)
- Yardas recibidas por jugador (WR/TE/RB)

## Qué usa

- **EPA por jugada** (play-by-play) — calidad ofensiva/defensiva
- **Líneas de Vegas** (spread/total del calendario oficial) como features
- **Rosters y depth charts oficiales 2026** — titulares reales (QB1, RB1-2, WR1-3, TE1-2)
- **Reportes de lesión** — excluye jugadores Out/Doubtful (durante la temporada)
- **Registro de predicciones** (`predicciones.csv`) — mide tu acierto contra resultados y contra Vegas

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
jupyter notebook nfl_predictor.ipynb
```

Ejecuta todas las celdas (entrena con 2020–2024, evalúa en 2025) y luego:

```python
predecir_juego('KC', 'BUF')      # visitante BUF @ local KC (busca líneas en calendario oficial)
predecir_semana(1)               # todos los juegos de la semana 1
guardar_predicciones(1)          # guarda la semana en predicciones.csv
evaluar_predicciones()           # acierto real vs resultados y vs Vegas
```

Abreviaturas nflverse: `KC, BUF, PHI, DAL, SF, SEA, NE, DEN, LA, LAC, GB, BAL, ...`

## Datos

`nflreadpy` descarga y cachea automáticamente — no requiere API key.
Durante la temporada, corre `nfl.clear_cache()` y re-ejecuta el notebook para datos frescos.
