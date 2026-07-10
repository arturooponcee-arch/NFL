# -*- coding: utf-8 -*-
"""Actualización semanal: predice la próxima semana y evalúa las anteriores.

Pensado para GitHub Actions (cron semanal) o ejecución manual:
    python actualizar.py
"""
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import nflreadpy as nfl
import nfl_pred


def semana_proxima(calendario, dias_max=10):
    """Primera semana con juegos pendientes en los próximos `dias_max` días."""
    pend = calendario[calendario['home_score'].isna()].copy()
    if pend.empty:
        return None
    pend['fecha'] = pd.to_datetime(pend['gameday'])
    hoy = pd.Timestamp.now().normalize()
    pend = pend[pend['fecha'] >= hoy - pd.Timedelta(days=2)]
    if pend.empty:
        return None
    prox = pend.sort_values('fecha').iloc[0]
    if (prox['fecha'] - hoy).days > dias_max:
        return None   # offseason o demasiado lejos
    return int(prox['week'])


def main():
    # en local, forzar datos frescos; en CI el runner ya arranca sin caché
    if not os.environ.get('CI'):
        nfl.clear_cache()

    nfl_pred.inicializar(walk_forward=False)

    week = semana_proxima(nfl_pred.ctx['calendario'])
    if week is None:
        print('Sin juegos en los próximos días (offseason) - nada que predecir')
    else:
        print(f'\nPrediciendo semana {week} de {nfl_pred.TEMPORADA_ACTUAL}...')
        nfl_pred.guardar_predicciones(week)

    print('\nEvaluación de predicciones guardadas:')
    nfl_pred.evaluar_predicciones()

    print('\nGenerando reporte HTML:')
    nfl_pred.generar_reporte()


if __name__ == '__main__':
    main()
