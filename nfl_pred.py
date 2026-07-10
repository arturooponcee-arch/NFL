# -*- coding: utf-8 -*-
"""Pipeline de predicción NFL: datos nflverse (nflreadpy) + ensamble XGBoost/LightGBM/Ridge.

Uso:
    import nfl_pred
    ctx = nfl_pred.inicializar()           # carga datos, features y entrena
    nfl_pred.predecir_juego('SEA', 'NE')
    nfl_pred.predecir_semana(1)
    nfl_pred.guardar_predicciones(1)
    nfl_pred.evaluar_predicciones()
    nfl_pred.generar_reporte()             # docs/index.html para GitHub Pages
"""
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import polars as pl
import nflreadpy as nfl
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error

# ---------------- Configuración ----------------
SEASONS = list(range(2020, 2026))   # historial de entrenamiento
TEMPORADA_TEST = 2025               # temporada de backtest
TEMPORADA_ACTUAL = 2026             # temporada a predecir
ROLL = 5                            # ventana de forma: últimos 5 juegos
ARCHIVO_PRED = 'predicciones.csv'
REPORTE_HTML = os.path.join('docs', 'index.html')
UMBRAL_VALUE = 4.0                  # discrepancia (pts) para marcar posible value

STAT_COLS = ['pts', 'pts_perm', 'pass_yds', 'rush_yds', 'pass_yds_perm', 'rush_yds_perm',
             'epa_of', 'epa_pase', 'epa_carr', 'epa_def']
PCOLS = ['passing_yards', 'attempts', 'rushing_yards', 'carries',
         'receiving_yards', 'targets', 'target_share',
         'passing_tds', 'rushing_tds', 'receiving_tds', 'receptions', 'passing_interceptions']
# snap counts + Next Gen Stats (pueden faltar por jugador-semana; XGBoost tolera NaN)
EXTRA_PCOLS = ['snap_pct', 'cpoe', 'time_throw', 'ryoe_att', 'rush_eff', 'separacion', 'yac_ae']

# feats = obligatorias (dropna al entrenar) | feats_opc = opcionales (NaN permitido)
PROPS = {
    'yardas_pase':      dict(target='passing_yards', pos=['QB'], tipo='yardas',
                             feats=['passing_yards_r', 'attempts_r', 'def_pase'],
                             feats_opc=['snap_pct_r', 'cpoe_r', 'time_throw_r']),
    'yardas_tierra':    dict(target='rushing_yards', pos=['RB', 'QB', 'FB'], tipo='yardas',
                             feats=['rushing_yards_r', 'carries_r', 'def_tierra'],
                             feats_opc=['snap_pct_r', 'ryoe_att_r', 'rush_eff_r']),
    'yardas_recepcion': dict(target='receiving_yards', pos=['WR', 'TE', 'RB', 'FB'], tipo='yardas',
                             feats=['receiving_yards_r', 'targets_r', 'target_share_r', 'def_pase'],
                             feats_opc=['snap_pct_r', 'separacion_r', 'yac_ae_r']),
    'td_pase':          dict(target='passing_tds', pos=['QB'], tipo='conteo',
                             feats=['passing_tds_r', 'passing_yards_r', 'attempts_r', 'def_pase'],
                             feats_opc=['snap_pct_r', 'cpoe_r']),
    'td_tierra':        dict(target='rushing_tds', pos=['RB', 'QB', 'FB'], tipo='conteo',
                             feats=['rushing_tds_r', 'rushing_yards_r', 'carries_r', 'def_tierra'],
                             feats_opc=['snap_pct_r', 'ryoe_att_r']),
    'td_recepcion':     dict(target='receiving_tds', pos=['WR', 'TE', 'RB', 'FB'], tipo='conteo',
                             feats=['receiving_tds_r', 'receiving_yards_r', 'targets_r', 'def_pase'],
                             feats_opc=['snap_pct_r', 'separacion_r', 'yac_ae_r']),
    'recepciones':      dict(target='receptions', pos=['WR', 'TE', 'RB', 'FB'], tipo='conteo',
                             feats=['receptions_r', 'targets_r', 'target_share_r', 'def_pase'],
                             feats_opc=['snap_pct_r', 'separacion_r']),
    'intercepciones':   dict(target='passing_interceptions', pos=['QB'], tipo='conteo',
                             feats=['passing_interceptions_r', 'attempts_r', 'def_pase'],
                             feats_opc=['snap_pct_r', 'cpoe_r']),
}
PROPS_POR_POS = {
    'QB': ['yardas_pase', 'td_pase', 'intercepciones', 'yardas_tierra', 'td_tierra'],
    'RB': ['yardas_tierra', 'td_tierra', 'yardas_recepcion', 'recepciones', 'td_recepcion'],
    'WR': ['yardas_recepcion', 'recepciones', 'td_recepcion'],
    'TE': ['yardas_recepcion', 'recepciones', 'td_recepcion'],
}
LIM_DEPTH = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 2}

XGB_PARAMS = dict(n_estimators=400, learning_rate=0.04, max_depth=4,
                  subsample=0.8, colsample_bytree=0.8, random_state=42)
XGB_PROP = dict(n_estimators=300, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, random_state=42)


class Ensamble:
    """Promedio de XGBoost + LightGBM + Ridge para puntos por equipo."""

    def __init__(self):
        self.modelos = [
            XGBRegressor(**XGB_PARAMS),
            LGBMRegressor(n_estimators=400, learning_rate=0.04, max_depth=4,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=42, verbose=-1),
            Ridge(alpha=1.0),
        ]

    def fit(self, X, y):
        for m in self.modelos:
            m.fit(X, y)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.modelos], axis=0)


ctx = {}   # estado compartido del pipeline


def rolling_shift(s):
    # promedio de los ROLL juegos ANTERIORES (shift evita usar el juego actual)
    return s.shift(1).rolling(ROLL, min_periods=2).mean()


# ---------------- 1. Carga de datos ----------------
def cargar_datos(seasons=None, temporada_actual=TEMPORADA_ACTUAL):
    seasons = list(seasons or SEASONS)
    calendario = nfl.load_schedules([temporada_actual]).to_pandas()

    # si la temporada actual ya tiene juegos jugados, entra al historial completo
    seasons_all = list(seasons)
    if calendario['home_score'].notna().any():
        seasons_all.append(temporada_actual)
        print(f'Incluyendo juegos ya jugados de {temporada_actual} en el historial')

    sched = nfl.load_schedules(seasons_all).to_pandas()
    stats = nfl.load_player_stats(seasons_all).to_pandas()
    pbp = pl.concat([
        nfl.load_pbp([yr]).select(['season', 'week', 'posteam', 'defteam', 'epa', 'pass', 'rush'])
        for yr in seasons_all
    ]).to_pandas()
    sched = sched[sched['home_score'].notna()].copy()
    stats = stats[stats['season_type'].isin(['REG', 'POST'])].copy()

    # snap counts (usa ids de PFR; se mapean a gsis con load_players)
    players = nfl.load_players().to_pandas()[['gsis_id', 'pfr_id']].dropna()
    snaps = nfl.load_snap_counts(seasons_all).to_pandas()
    snaps = snaps.merge(players, left_on='pfr_player_id', right_on='pfr_id')
    snaps = (snaps.groupby(['season', 'week', 'gsis_id'], as_index=False)
                  .agg(snap_pct=('offense_pct', 'max'))
                  .rename(columns={'gsis_id': 'player_id'}))

    # Next Gen Stats (semanales; solo jugadores que cumplen umbral de la NFL)
    ngs_defs = {
        'passing':   {'completion_percentage_above_expectation': 'cpoe',
                      'avg_time_to_throw': 'time_throw'},
        'rushing':   {'rush_yards_over_expected_per_att': 'ryoe_att',
                      'efficiency': 'rush_eff'},
        'receiving': {'avg_separation': 'separacion',
                      'avg_yac_above_expectation': 'yac_ae'},
    }
    ngs = []
    for tipo, cols in ngs_defs.items():
        d = nfl.load_nextgen_stats(seasons_all, stat_type=tipo).to_pandas()
        d = d[d['week'] > 0]   # week 0 = agregado de temporada
        d = d[['player_gsis_id', 'season', 'week'] + list(cols)].rename(
            columns={'player_gsis_id': 'player_id', **cols})
        ngs.append(d)

    roster = nfl.load_rosters([temporada_actual]).to_pandas()
    depth = (nfl.load_depth_charts([temporada_actual])
             .filter(pl.col('dt') == pl.col('dt').max().over('team')).to_pandas())
    try:
        lesiones = nfl.load_injuries([temporada_actual]).to_pandas()
    except Exception:
        lesiones = pd.DataFrame(columns=['team', 'week', 'gsis_id', 'full_name', 'report_status'])
        print('Aviso: reportes de lesion aun no publicados (salen con la temporada)')

    print(f'Juegos: {len(sched)} | Jugador-semanas: {len(stats)} | Jugadas pbp: {len(pbp)} | '
          f'Snaps: {len(snaps)} | NGS: {sum(len(d) for d in ngs)}')
    print(f'Calendario {temporada_actual}: {len(calendario)} juegos | Roster: {len(roster)} | '
          f'Depth chart: {len(depth)}')
    ctx.update(sched=sched, stats=stats, pbp=pbp, calendario=calendario, roster=roster,
               depth=depth, lesiones=lesiones, snaps=snaps, ngs=ngs)
    return ctx


# ---------------- 2. Features ----------------
def _clima(df, temp_def, wind_def):
    es_domo = df['roof'].isin(['dome', 'closed']).astype(int)
    temp = df['temp'].where(es_domo == 0, 70.0).fillna(temp_def)
    wind = df['wind'].where(es_domo == 0, 0.0).fillna(wind_def)
    return es_domo, temp, wind


def construir_features():
    sched, stats, pbp = ctx['sched'], ctx['stats'], ctx['pbp']

    base = ['season', 'week', 'game_id', 'gameday']
    extras = ['spread_line', 'total_line', 'roof', 'temp', 'wind', 'div_game']

    home = sched[base + ['home_team', 'away_team', 'home_score', 'away_score', 'home_rest'] + extras].copy()
    home.columns = base + ['team', 'opp', 'pts', 'pts_perm', 'rest',
                           'vegas_spread', 'vegas_total', 'roof', 'temp', 'wind', 'div_game']
    home['es_local'] = 1

    away = sched[base + ['away_team', 'home_team', 'away_score', 'home_score', 'away_rest'] + extras].copy()
    away.columns = base + ['team', 'opp', 'pts', 'pts_perm', 'rest',
                           'vegas_spread', 'vegas_total', 'roof', 'temp', 'wind', 'div_game']
    away['vegas_spread'] = -away['vegas_spread']   # spread desde la perspectiva del equipo
    away['es_local'] = 0

    tg = pd.concat([home, away], ignore_index=True)

    # clima: imputación con medianas históricas de juegos exteriores
    exterior = sched[~sched['roof'].isin(['dome', 'closed'])]
    temp_def = float(exterior['temp'].median())
    wind_def = float(exterior['wind'].median())
    tg['es_domo'], tg['temp'], tg['wind'] = _clima(tg, temp_def, wind_def)

    # yardas por equipo-juego
    yardas = stats.groupby(['season', 'week', 'team'], as_index=False).agg(
        pass_yds=('passing_yards', 'sum'), rush_yds=('rushing_yards', 'sum'))
    tg = tg.merge(yardas, on=['season', 'week', 'team'], how='left')
    tg = tg.merge(yardas.rename(columns={'team': 'opp', 'pass_yds': 'pass_yds_perm',
                                         'rush_yds': 'rush_yds_perm'}),
                  on=['season', 'week', 'opp'], how='left')

    # EPA por equipo-juego
    pj = pbp[pbp['epa'].notna() & pbp['posteam'].notna()].copy()
    epa_of = pj.groupby(['season', 'week', 'posteam'], as_index=False).agg(epa_of=('epa', 'mean'))
    epa_pase = pj[pj['pass'] == 1].groupby(['season', 'week', 'posteam'], as_index=False).agg(epa_pase=('epa', 'mean'))
    epa_carr = pj[pj['rush'] == 1].groupby(['season', 'week', 'posteam'], as_index=False).agg(epa_carr=('epa', 'mean'))
    epa_def = pj.groupby(['season', 'week', 'defteam'], as_index=False).agg(epa_def=('epa', 'mean'))

    tg = (tg.merge(epa_of, left_on=['season', 'week', 'team'], right_on=['season', 'week', 'posteam'], how='left')
            .merge(epa_pase, left_on=['season', 'week', 'team'], right_on=['season', 'week', 'posteam'], how='left', suffixes=('', '_x1'))
            .merge(epa_carr, left_on=['season', 'week', 'team'], right_on=['season', 'week', 'posteam'], how='left', suffixes=('', '_x2'))
            .merge(epa_def, left_on=['season', 'week', 'team'], right_on=['season', 'week', 'defteam'], how='left')
            .drop(columns=['posteam', 'posteam_x1', 'posteam_x2', 'defteam']))

    tg = tg.sort_values(['team', 'season', 'week']).reset_index(drop=True)
    for c in STAT_COLS:
        tg[f'{c}_r'] = tg.groupby('team')[c].transform(rolling_shift)

    feat_team = [f'{c}_r' for c in STAT_COLS]
    feats_contexto = ['es_local', 'rest', 'div_game', 'es_domo', 'temp', 'wind']
    feats_puro = feat_team + [f + '_vs' for f in feat_team] + feats_contexto
    feats_vegas = feats_puro + ['vegas_spread', 'vegas_total']

    # matriz con features del rival
    opp_feats = tg[['season', 'week', 'team'] + feat_team].rename(
        columns={'team': 'opp', **{f: f + '_vs' for f in feat_team}})
    m = tg.merge(opp_feats, on=['season', 'week', 'opp'], how='left')
    m = m.dropna(subset=feats_vegas + ['pts'])

    # tabla jugador-juego (+ snaps + Next Gen Stats)
    ps = stats[['player_id', 'player_display_name', 'position', 'season', 'week',
                'team', 'opponent_team'] + PCOLS].copy()
    ps = ps.rename(columns={'player_display_name': 'jugador', 'opponent_team': 'opp'})
    ps = ps.merge(ctx['snaps'], on=['player_id', 'season', 'week'], how='left')
    for d in ctx['ngs']:
        ps = ps.merge(d, on=['player_id', 'season', 'week'], how='left')
    ps = ps.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
    for c in PCOLS + EXTRA_PCOLS:
        ps[f'{c}_r'] = ps.groupby('player_id')[c].transform(rolling_shift)

    defensa = tg[['season', 'week', 'team', 'pass_yds_perm_r', 'rush_yds_perm_r']].rename(
        columns={'team': 'opp', 'pass_yds_perm_r': 'def_pase', 'rush_yds_perm_r': 'def_tierra'})
    ps = ps.merge(defensa, on=['season', 'week', 'opp'], how='left')

    print(f'Tabla equipo-juego: {len(tg)} filas | matriz modelo: {len(m)} filas')
    ctx.update(tg=tg, ps=ps, m=m, feats_puro=feats_puro, feats_vegas=feats_vegas,
               temp_def=temp_def, wind_def=wind_def)
    return ctx


# ---------------- 3. Entrenamiento ----------------
def _eval_juegos(test, col_pred):
    locales = test[test['es_local'] == 1][['game_id', 'season', 'pts', col_pred, 'vegas_spread']]
    visitas = test[test['es_local'] == 0][['game_id', 'pts', col_pred]].rename(
        columns={'pts': 'pts_v', col_pred: 'pred_v'})
    j = locales.merge(visitas, on='game_id')
    j['dif_pred'] = j[col_pred] - j['pred_v']
    j['dif_real'] = j['pts'] - j['pts_v']
    return j


def entrenar(temporada_test=TEMPORADA_TEST):
    m = ctx['m']
    feats_puro, feats_vegas = ctx['feats_puro'], ctx['feats_vegas']
    train = m[m['season'] < temporada_test]
    # backtest + calibración: 2025 y (si existen) juegos de la temporada actual,
    # ambos fuera del entrenamiento -> la calibración se refresca sola cada semana
    test_cal = m[m['season'] >= temporada_test].copy()

    modelo_vegas = Ensamble().fit(train[feats_vegas], train['pts'])
    modelo_puro = Ensamble().fit(train[feats_puro], train['pts'])

    test_cal['pred_vegas'] = modelo_vegas.predict(test_cal[feats_vegas])
    test_cal['pred_puro'] = modelo_puro.predict(test_cal[feats_puro])

    # métricas solo sobre la temporada de backtest
    test = test_cal[test_cal['season'] == temporada_test]
    jv = _eval_juegos(test, 'pred_vegas')
    jp = _eval_juegos(test, 'pred_puro')
    acc_vegas_model = (np.sign(jv['dif_pred']) == np.sign(jv['dif_real'])).mean()
    acc_puro = (np.sign(jp['dif_pred']) == np.sign(jp['dif_real'])).mean()
    acc_linea = (np.sign(jv['vegas_spread']) == np.sign(jv['dif_real'])).mean()
    mae_pts = mean_absolute_error(test['pts'], test['pred_vegas'])
    mae_total = mean_absolute_error(jv['pts'] + jv['pts_v'], jv['pred_vegas'] + jv['pred_v'])

    # calibrador con TODO lo fuera de muestra (2025 + temporada actual jugada)
    jp_cal = _eval_juegos(test_cal, 'pred_puro')
    Xc = np.column_stack([jp_cal['dif_pred'], jp_cal['vegas_spread']])
    yc = (jp_cal['dif_real'] > 0).astype(int)
    calibrador = LogisticRegression().fit(Xc, yc)
    acc_cal = (calibrador.predict(Xc) == yc).mean()
    n_actual = int((jp_cal['season'] == TEMPORADA_ACTUAL).sum())
    nota_cal = f' (re-calibrada con {n_actual} juegos de {TEMPORADA_ACTUAL})' if n_actual else ''

    print(f'--- Backtest {temporada_test} (split simple, ensamble XGB+LGBM+Ridge) ---')
    print(f'MAE puntos por equipo: {mae_pts:.2f} | MAE total juego: {mae_total:.2f}')
    print(f'Acierto ganador - modelo puro: {acc_puro:.1%} | modelo c/Vegas: {acc_vegas_model:.1%} | '
          f'linea Vegas: {acc_linea:.1%} | prob calibrada: {acc_cal:.1%}{nota_cal}')

    # ---- props de jugador ----
    ps = ctx['ps']
    modelos_prop = {}
    for nombre, cfg in PROPS.items():
        feats_full = cfg['feats'] + cfg['feats_opc']
        d = ps[ps['position'].isin(cfg['pos'])].dropna(subset=cfg['feats'] + [cfg['target']])
        tr, te = d[d['season'] < temporada_test], d[d['season'] == temporada_test]
        if cfg['tipo'] == 'yardas':
            mod = XGBRegressor(**XGB_PROP).fit(tr[feats_full], tr[cfg['target']])
            q16 = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.16,
                               **XGB_PROP).fit(tr[feats_full], tr[cfg['target']])
            q84 = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.84,
                               **XGB_PROP).fit(tr[feats_full], tr[cfg['target']])
            modelos_prop[nombre] = dict(mod=mod, q16=q16, q84=q84, feats=feats_full)
        else:
            mod = XGBRegressor(objective='count:poisson', **XGB_PROP).fit(
                tr[feats_full], tr[cfg['target']])
            modelos_prop[nombre] = dict(mod=mod, feats=feats_full)
        mae = mean_absolute_error(te[cfg['target']], mod.predict(te[feats_full]))
        print(f'MAE {nombre}: {mae:.2f}  ({len(te)} jugador-juegos)')

    ctx.update(modelo_vegas=modelo_vegas, modelo_puro=modelo_puro,
               calibrador=calibrador, modelos_prop=modelos_prop,
               metricas=dict(mae_pts=mae_pts, mae_total=mae_total, acc_puro=acc_puro,
                             acc_vegas_model=acc_vegas_model, acc_linea=acc_linea,
                             acc_calibrada=acc_cal, juegos_calibracion_actual=n_actual))
    return ctx


def backtest_walk_forward(temporada_test=TEMPORADA_TEST):
    """Re-entrena semana a semana: métrica honesta de temporada real."""
    m = ctx['m']
    feats_puro, feats_vegas = ctx['feats_puro'], ctx['feats_vegas']
    semanas = sorted(m[m['season'] == temporada_test]['week'].unique())
    piezas = []
    for wk in semanas:
        tr = m[(m['season'] < temporada_test) |
               ((m['season'] == temporada_test) & (m['week'] < wk))]
        te = m[(m['season'] == temporada_test) & (m['week'] == wk)].copy()
        if te.empty:
            continue
        te['pred_vegas'] = Ensamble().fit(tr[feats_vegas], tr['pts']).predict(te[feats_vegas])
        te['pred_puro'] = Ensamble().fit(tr[feats_puro], tr['pts']).predict(te[feats_puro])
        piezas.append(te)
    wf = pd.concat(piezas, ignore_index=True)
    jv, jp = _eval_juegos(wf, 'pred_vegas'), _eval_juegos(wf, 'pred_puro')
    res = dict(
        mae_pts=mean_absolute_error(wf['pts'], wf['pred_vegas']),
        acc_puro=(np.sign(jp['dif_pred']) == np.sign(jp['dif_real'])).mean(),
        acc_vegas_model=(np.sign(jv['dif_pred']) == np.sign(jv['dif_real'])).mean(),
        acc_linea=(np.sign(jv['vegas_spread']) == np.sign(jv['dif_real'])).mean(),
        n_juegos=len(jv))
    print(f'--- Backtest walk-forward {temporada_test} ({res["n_juegos"]} juegos, '
          f're-entrenando cada semana) ---')
    print(f'MAE puntos: {res["mae_pts"]:.2f} | acierto puro: {res["acc_puro"]:.1%} | '
          f'c/Vegas: {res["acc_vegas_model"]:.1%} | linea Vegas: {res["acc_linea"]:.1%}')
    ctx['walk_forward'] = res
    return res


# ---------------- 4. Predicción ----------------
def _snapshot_equipo(team):
    tg = ctx['tg']
    d = tg[tg['team'] == team].sort_values(['season', 'week']).tail(ROLL)
    if d.empty:
        raise ValueError(f'Equipo desconocido: {team}')
    return {f'{c}_r': d[c].mean() for c in STAT_COLS}


def _fila_equipo(team, opp, es_local, rest, vegas_spread, vegas_total,
                 div_game, es_domo, temp, wind, feats):
    snap, snap_vs = _snapshot_equipo(team), _snapshot_equipo(opp)
    fila = {**snap, **{k + '_vs': v for k, v in snap_vs.items()},
            'es_local': es_local, 'rest': rest, 'div_game': div_game,
            'es_domo': es_domo, 'temp': temp, 'wind': wind,
            'vegas_spread': vegas_spread, 'vegas_total': vegas_total}
    return pd.DataFrame([fila])[feats]


def _contexto_juego(g):
    """Extrae contexto (clima, división, líneas) de una fila del calendario."""
    es_domo = 1 if g['roof'] in ('dome', 'closed') else 0
    temp = 70.0 if es_domo else (g['temp'] if pd.notna(g['temp']) else ctx['temp_def'])
    wind = 0.0 if es_domo else (g['wind'] if pd.notna(g['wind']) else ctx['wind_def'])
    return dict(spread=float(g['spread_line']), total=float(g['total_line']),
                rest_l=float(g['home_rest']), rest_v=float(g['away_rest']),
                div_game=int(g['div_game']), es_domo=es_domo, temp=float(temp), wind=float(wind))


def _predecir_marcador(local, visitante, c):
    args = dict(div_game=c['div_game'], es_domo=c['es_domo'], temp=c['temp'], wind=c['wind'])
    fv_l = _fila_equipo(local, visitante, 1, c['rest_l'], c['spread'], c['total'], feats=ctx['feats_vegas'], **args)
    fv_v = _fila_equipo(visitante, local, 0, c['rest_v'], -c['spread'], c['total'], feats=ctx['feats_vegas'], **args)
    fp_l = _fila_equipo(local, visitante, 1, c['rest_l'], c['spread'], c['total'], feats=ctx['feats_puro'], **args)
    fp_v = _fila_equipo(visitante, local, 0, c['rest_v'], -c['spread'], c['total'], feats=ctx['feats_puro'], **args)
    pts_l = float(ctx['modelo_vegas'].predict(fv_l)[0])
    pts_v = float(ctx['modelo_vegas'].predict(fv_v)[0])
    dif_puro = float(ctx['modelo_puro'].predict(fp_l)[0]) - float(ctx['modelo_puro'].predict(fp_v)[0])
    p_local = float(ctx['calibrador'].predict_proba([[dif_puro, c['spread']]])[0, 1])
    return pts_l, pts_v, dif_puro, p_local


def _titulares(team, week=None):
    depth, lesiones = ctx['depth'], ctx['lesiones']
    dc = depth[(depth['team'] == team) & (depth['pos_abb'].isin(LIM_DEPTH))].copy()
    dc = dc[dc['pos_rank'] <= dc['pos_abb'].map(LIM_DEPTH)]
    if week is not None and len(lesiones):
        fuera = set(lesiones[(lesiones['team'] == team) & (lesiones['week'] == week) &
                             (lesiones['report_status'].isin(['Out', 'Doubtful']))]['gsis_id'])
        descartados = dc[dc['gsis_id'].isin(fuera)]['player_name'].tolist()
        if descartados:
            print(f'  Lesionados fuera ({team}): {", ".join(descartados)}')
        dc = dc[~dc['gsis_id'].isin(fuera)]
    return dc[['gsis_id', 'player_name', 'pos_abb']].drop_duplicates('gsis_id')


def _snapshot_jugador(pid):
    ps = ctx['ps']
    d = ps[ps['player_id'] == pid].sort_values(['season', 'week']).tail(ROLL)
    if len(d) < 2:
        return None   # sin historial suficiente (ej. novato)
    return {f'{c}_r': d[c].mean() for c in PCOLS + EXTRA_PCOLS}


def _props_equipo(team, opp_def, week=None):
    filas, sin_historial = [], []
    for _, j in _titulares(team, week).iterrows():
        snap = _snapshot_jugador(j['gsis_id'])
        if snap is None:
            sin_historial.append(f"{j['player_name']} ({j['pos_abb']})")
            continue
        for prop in PROPS_POR_POS[j['pos_abb']]:
            cfg = PROPS[prop]
            mods = ctx['modelos_prop'][prop]
            fila = pd.DataFrame([{**snap, 'def_pase': opp_def['def_pase'],
                                  'def_tierra': opp_def['def_tierra']}])[mods['feats']]
            y = float(mods['mod'].predict(fila)[0])
            r = dict(equipo=team, jugador=j['player_name'], pos=j['pos_abb'],
                     prop=prop, prediccion=round(max(0.0, y), 2))
            if cfg['tipo'] == 'yardas':
                lo = min(float(mods['q16'].predict(fila)[0]), y)
                hi = max(float(mods['q84'].predict(fila)[0]), y)
                r['rango_68pct'] = f'{max(0, lo):.0f}-{hi:.0f}'
            else:
                lam = max(0.0, y)
                r['prob_1mas'] = round(1 - np.exp(-lam), 2)
            filas.append(r)
    if sin_historial:
        print(f'  Sin historial NFL ({team}): {", ".join(sin_historial)}')
    return filas


def predecir_juego(local, visitante, week=None):
    calendario = ctx['calendario']
    g = calendario[(calendario['home_team'] == local) & (calendario['away_team'] == visitante)]
    if week is not None:
        g = g[g['week'] == week]
    if len(g):
        g = g.iloc[0]
        week = int(g['week'])
        c = _contexto_juego(g)
        print(f'Juego oficial: semana {week} ({g["gameday"]}) | '
              f'Vegas: spread {c["spread"]:+.1f}, total {c["total"]}')
    else:
        c = dict(spread=0.0, total=44.5, rest_l=7, rest_v=7, div_game=0,
                 es_domo=0, temp=ctx['temp_def'], wind=ctx['wind_def'])
        print('Aviso: juego no esta en el calendario oficial - usando lineas neutras')

    pts_l, pts_v, dif_puro, p_local = _predecir_marcador(local, visitante, c)
    discrepancia = dif_puro - c['spread']

    print(f'=== {visitante} @ {local} ===')
    print(f'Marcador predicho:  {local} {pts_l:.1f} - {visitante} {pts_v:.1f}')
    print(f'Puntos totales:     {pts_l + pts_v:.1f}')
    print(f'Modelo puro (sin Vegas): {local} por {dif_puro:+.1f} | '
          f'discrepancia vs Vegas: {discrepancia:+.1f}'
          + ('  << posible value' if abs(discrepancia) > UMBRAL_VALUE else ''))
    print(f'Prob. victoria (calibrada): {local} {p_local:.0%} | {visitante} {1 - p_local:.0%}')

    def_l, def_v = _snapshot_equipo(local), _snapshot_equipo(visitante)
    props = (_props_equipo(local, {'def_pase': def_v['pass_yds_perm_r'], 'def_tierra': def_v['rush_yds_perm_r']}, week)
             + _props_equipo(visitante, {'def_pase': def_l['pass_yds_perm_r'], 'def_tierra': def_l['rush_yds_perm_r']}, week))
    df = pd.DataFrame(props).sort_values(['equipo', 'prop', 'prediccion'],
                                         ascending=[True, True, False])
    return df.reset_index(drop=True)


def predecir_semana(week):
    filas = []
    for _, g in ctx['calendario'][ctx['calendario']['week'] == week].iterrows():
        local, visita = g['home_team'], g['away_team']
        c = _contexto_juego(g)
        pts_l, pts_v, dif_puro, p_local = _predecir_marcador(local, visita, c)
        discrepancia = dif_puro - c['spread']
        filas.append(dict(
            season=TEMPORADA_ACTUAL, week=week, fecha=g['gameday'],
            local=local, visitante=visita,
            pred_local=round(pts_l, 1), pred_visita=round(pts_v, 1),
            total_pred=round(pts_l + pts_v, 1), total_vegas=c['total'],
            # spread positivo = local favorito (misma convención que Vegas)
            spread_pred=round(pts_l - pts_v, 1), spread_puro=round(dif_puro, 1),
            spread_vegas=c['spread'], discrepancia=round(discrepancia, 1),
            value=abs(discrepancia) > UMBRAL_VALUE,
            ganador=local if p_local >= 0.5 else visita,
            prob=round(max(p_local, 1 - p_local), 2)))
    return pd.DataFrame(filas)


# ---------------- 5. Registro ----------------
def guardar_predicciones(week, archivo=ARCHIVO_PRED):
    df = predecir_semana(week)
    df['fecha_prediccion'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
    if os.path.exists(archivo):
        prev = pd.read_csv(archivo)
        prev = prev[~((prev['season'] == TEMPORADA_ACTUAL) & (prev['week'] == week))]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(archivo, index=False)
    print(f'{(df["week"] == week).sum()} juegos de semana {week} guardados en {archivo}')
    return df[df['week'] == week]


def _evaluar(log):
    """Cruza el registro con resultados reales. Devuelve (df, resumen) o (None, None)."""
    res = nfl.load_schedules([TEMPORADA_ACTUAL]).to_pandas()
    res = res[res['home_score'].notna()][['week', 'home_team', 'away_team',
                                          'home_score', 'away_score']]
    ev = log.merge(res, left_on=['week', 'local', 'visitante'],
                   right_on=['week', 'home_team', 'away_team'])
    if ev.empty:
        return None, None
    dif_real = ev['home_score'] - ev['away_score']
    ev['ganador_real'] = np.where(dif_real > 0, ev['local'], ev['visitante'])
    ev['acierto'] = ev['ganador'] == ev['ganador_real']
    resumen = dict(
        n=len(ev),
        acc_modelo=float(ev['acierto'].mean()),
        acc_vegas=float((np.sign(ev['spread_vegas']) == np.sign(dif_real)).mean()),
        mae_total=float((ev['total_pred'] - (ev['home_score'] + ev['away_score'])).abs().mean()),
        mae_spread=float((ev['spread_pred'] - dif_real).abs().mean()))
    return ev, resumen


def evaluar_predicciones(archivo=ARCHIVO_PRED):
    if not os.path.exists(archivo):
        print(f'No existe {archivo} - usa guardar_predicciones(week) primero')
        return None
    ev, resumen = _evaluar(pd.read_csv(archivo))
    if ev is None:
        print('Aun no hay resultados para las predicciones guardadas')
        return None
    print(f'Juegos evaluados: {resumen["n"]}')
    print(f'Acierto ganador - modelo: {resumen["acc_modelo"]:.1%} | Vegas: {resumen["acc_vegas"]:.1%}')
    print(f'MAE total: {resumen["mae_total"]:.2f}')
    print(f'MAE spread: {resumen["mae_spread"]:.2f}')
    return ev[['week', 'local', 'visitante', 'pred_local', 'pred_visita',
               'home_score', 'away_score', 'ganador', 'ganador_real', 'acierto', 'prob']]


# ---------------- 6. Reporte HTML (GitHub Pages) ----------------
def generar_reporte(archivo=ARCHIVO_PRED, salida=REPORTE_HTML):
    """Genera docs/index.html con la última semana guardada y el historial de aciertos."""
    if not os.path.exists(archivo):
        print(f'No existe {archivo} - nada que reportar')
        return None
    log = pd.read_csv(archivo)
    sem = int(log['week'].max())
    df = log[log['week'] == sem].sort_values('fecha')
    ev, resumen = _evaluar(log)
    ahora = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')

    if resumen:
        tarjeta_eval = f"""
    <p class="resumen">Temporada {TEMPORADA_ACTUAL}: <b>{resumen['acc_modelo']:.0%}</b> de acierto
    del modelo en {resumen['n']} juegos (Vegas: {resumen['acc_vegas']:.0%}) ·
    error medio en totales: {resumen['mae_total']:.1f} pts.</p>"""
    else:
        tarjeta_eval = '<p class="resumen">Aún sin resultados para evaluar esta temporada.</p>'

    filas_html = []
    for _, r in df.iterrows():
        badge = '<span class="value">VALUE</span>' if r['value'] else ''
        filas_html.append(f"""      <tr>
        <td>{r['fecha']}</td>
        <td class="juego">{r['visitante']} @ {r['local']}</td>
        <td>{r['local']} {r['pred_local']:.0f} - {r['pred_visita']:.0f} {r['visitante']}</td>
        <td>{r['total_pred']:.1f} <small>({r['total_vegas']:.1f})</small></td>
        <td>{r['spread_pred']:+.1f} <small>({r['spread_vegas']:+.1f})</small></td>
        <td>{r['discrepancia']:+.1f} {badge}</td>
        <td><b>{r['ganador']}</b> {r['prob']:.0%}</td>
      </tr>""")

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predicciones NFL - Semana {sem}</title>
<style>
  :root {{ color-scheme: light dark; --acento: #0b6e4f; --value: #b45309; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem auto;
         max-width: 60rem; padding: 0 1rem; line-height: 1.5; }}
  h1 {{ margin-bottom: .2rem; }}
  .sub {{ color: #888; margin-top: 0; }}
  .resumen {{ background: color-mix(in srgb, var(--acento) 12%, transparent);
             border-left: 4px solid var(--acento); padding: .7rem 1rem; border-radius: 6px; }}
  .scroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .93rem; }}
  th, td {{ text-align: left; padding: .5rem .7rem; white-space: nowrap; }}
  th {{ border-bottom: 2px solid var(--acento); }}
  tr:nth-child(even) {{ background: color-mix(in srgb, currentColor 5%, transparent); }}
  .juego {{ font-weight: 600; }}
  small {{ color: #888; }}
  .value {{ background: var(--value); color: #fff; border-radius: 4px;
           padding: .1rem .4rem; font-size: .75rem; font-weight: 700; }}
  footer {{ margin-top: 2rem; color: #888; font-size: .85rem; }}
</style>
</head>
<body>
<h1>🏈 Predicciones NFL</h1>
<p class="sub">Semana {sem} · temporada {TEMPORADA_ACTUAL} · actualizado {ahora}</p>
{tarjeta_eval}
<p><b>VALUE</b> = el modelo sin Vegas discrepa más de {UMBRAL_VALUE:.0f} pts de la línea.
Entre paréntesis, la línea de Vegas. Spread positivo = local favorito.</p>
<div class="scroll">
<table>
  <thead>
    <tr><th>Fecha</th><th>Juego</th><th>Marcador</th><th>Total (Vegas)</th>
        <th>Spread (Vegas)</th><th>Discrepancia</th><th>Ganador · prob</th></tr>
  </thead>
  <tbody>
{chr(10).join(filas_html)}
  </tbody>
</table>
</div>
<footer>Generado automáticamente con datos de nflverse ·
<a href="https://github.com/arturooponcee-arch/NFL">código</a> ·
<a href="https://raw.githubusercontent.com/arturooponcee-arch/NFL/main/predicciones.csv">predicciones.csv</a></footer>
</body>
</html>"""

    os.makedirs(os.path.dirname(salida), exist_ok=True)
    with open(salida, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Reporte de semana {sem} escrito en {salida}')
    return salida


# ---------------- Punto de entrada ----------------
def inicializar(walk_forward=False, seasons=None, temporada_actual=TEMPORADA_ACTUAL):
    cargar_datos(seasons, temporada_actual)
    construir_features()
    entrenar()
    if walk_forward:
        backtest_walk_forward()
    return ctx
