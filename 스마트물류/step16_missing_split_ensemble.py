# =============================================================================
# 스마트 창고 출고 지연 예측 AI 경진대회
# Step 16: 결측치 구조적 분리 앙상블 (Missing Value Structural Split)
# =============================================================================
# Step14 대비 변경점:
#
# [결측치 구조적 분리 전략]
# - 핵심 컬럼(congestion_score, avg_recovery_time, loading_dock_util, staging_area_util)의
#   결측은 단순 누락이 아니라 운영 상태 자체의 구조적 차이를 의미함
#   예: congestion_score=NaN → 해당 슬롯에 혼잡 이벤트 자체가 발생하지 않은 상태
#       avg_recovery_time=NaN → 로봇 장애/복구 이벤트 없음
# - "정상 운영(결측없음)"과 "이벤트 발생(결측있음)" 그룹의 지연 패턴이
#   근본적으로 다를 수 있으므로, 각 그룹에 특화된 모델을 학습
#
# [그룹 정의]
#   Group 0 (정상): nan_* 플래그 모두 0 → 결측 없는 정상 운영 슬롯
#   Group 1 (결측): nan_* 플래그 하나라도 1 → 이벤트/이상 발생 슬롯
#
# [학습 구조]
#   각 그룹별로 독립적으로 LGB+XGB+CatBoost×5seeds × GroupKFold(5) 학습
#   Test: 동일 결측 패턴으로 그룹 판별 → 해당 그룹 모델로 예측
#   OOF: 전체 train 기준 계산 (Step14와 비교 가능)
#
# 피처/모델/lr: Step14 완전 동일 (300 피처, LGB lr=0.02, XGB/Cat lr=0.05)
# 총 모델 수: 5seeds × 3모델 × 2그룹 = 30인스턴스 (Step14의 2배)
#
# 실행:
#   source ~/dacon_venv/bin/activate
#   caffeinate -i nohup python step16_missing_split_ensemble.py > step16_output.log 2>&1 &
#   tail -f step16_output.log
# =============================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

N_CPU          = os.cpu_count()
DATA_DIR       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR     = DATA_DIR
TARGET         = 'avg_delay_minutes_next_30m'
SEEDS          = [42, 123, 456, 789, 2024]
N_SPLITS       = 5
MAX_ROUNDS     = 3000       # XGB, CatBoost용
EARLY_STOP     = 150        # XGB, CatBoost용
LGB_MAX_ROUNDS = 5000       # LGB 전용
LGB_EARLY_STOP = 200        # LGB 전용

LAYOUT_TYPE_MAP = {'grid': 0, 'hybrid': 1, 'narrow': 2, 'hub_spoke': 3}

# 구조적 결측 판별 기준 컬럼 (feature_engineering 후 nan_* 플래그로 존재)
NAN_FLAG_COLS = ['nan_avg_recovery_time', 'nan_congestion_score',
                 'nan_loading_dock_util', 'nan_staging_area_util']

print(f"CPU: {N_CPU}코어 | 시드: {len(SEEDS)}개 × 3모델 × 2그룹 = {len(SEEDS)*3*2}인스턴스")

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("\n" + "="*60)
print("1. 데이터 로딩")
print("="*60)

train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

train = train.merge(layout, on='layout_id', how='left')
test  = test.merge(layout,  on='layout_id', how='left')
print(f"train: {train.shape} | test: {test.shape}")

# =============================================================================
# 2. 피처 엔지니어링 설정
# =============================================================================

AGG_COLS  = ['order_inflow_15m', 'low_battery_ratio', 'congestion_score',
             'robot_utilization', 'pack_utilization', 'robot_active',
             'charge_queue_length', 'max_zone_density']
AGG_FUNCS = ['mean', 'std', 'max', 'min']

ZSCORE_COLS = ['congestion_score', 'low_battery_ratio', 'max_zone_density',
               'pack_utilization', 'robot_utilization', 'order_inflow_15m',
               'charge_queue_length', 'robot_active', 'battery_mean',
               'aisle_traffic_score', 'blocked_path_15m']

RANK_COLS = ['congestion_score', 'low_battery_ratio', 'max_zone_density',
             'pack_utilization', 'order_inflow_15m', 'charge_queue_length',
             'battery_mean', 'robot_utilization', 'blocked_path_15m']

RATIO_COLS = ['congestion_score', 'low_battery_ratio', 'max_zone_density',
              'pack_utilization', 'order_inflow_15m', 'charge_queue_length',
              'robot_active', 'battery_mean']

SKEWED_COLS = ['task_reassign_15m', 'blocked_path_15m', 'fault_count_15m',
               'avg_charge_wait', 'near_collision_15m', 'avg_recovery_time',
               'charge_queue_length', 'robot_charging']

MEANINGFUL_NAN_COLS = ['avg_recovery_time', 'congestion_score',
                       'loading_dock_util', 'staging_area_util']

ZERO_FILL_COLS = ['charge_queue_length', 'avg_charge_wait', 'blocked_path_15m',
                  'near_collision_15m', 'fault_count_15m', 'task_reassign_15m',
                  'replenishment_overlap', 'congestion_score', 'max_zone_density',
                  'avg_recovery_time', 'loading_dock_util', 'staging_area_util']

EXPANDING_COLS = ['congestion_score', 'low_battery_ratio', 'order_inflow_15m',
                  'robot_active', 'blocked_path_15m', 'charge_queue_length',
                  'max_zone_density', 'battery_mean']

TRAJECTORY_COLS = ['congestion_score', 'low_battery_ratio', 'order_inflow_15m',
                   'robot_active', 'blocked_path_15m']


# =============================================================================
# 3. 피처 엔지니어링 함수 (Step14 완전 동일)
# =============================================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in MEANINGFUL_NAN_COLS:
        if col in df.columns:
            df[f'nan_{col}'] = df[col].isna().astype(np.int8)
    for col in ZERO_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    for col in ['congestion_score', 'blocked_path_15m', 'near_collision_15m',
                'charge_queue_length', 'avg_charge_wait', 'fault_count_15m',
                'replenishment_overlap', 'task_reassign_15m']:
        if col in df.columns:
            df[f'flag_{col}'] = (df[col] > 0).astype(np.int8)

    df['battery_stress']         = df['low_battery_ratio'] / (df['battery_mean'] + 1e-5)
    df['charge_bottleneck']      = df['charge_queue_length'] * df['avg_charge_wait']
    df['battery_volatility']     = df['battery_std'] / (df['battery_mean'] + 1e-5)
    df['battery_health']         = df['battery_mean'] - df['battery_std']
    df['order_per_robot']        = df['order_inflow_15m'] / (df['robot_active'] + 1)
    df['order_per_pack_station'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1)
    df['robot_effective_util']   = df['robot_active'] / (df['robot_total'] + 1)
    df['idle_ratio']             = df['robot_idle'] / (df['robot_active'] + df['robot_idle'] + df['robot_charging'] + 1)
    df['charging_ratio']         = df['robot_charging'] / (df['robot_total'] + 1)
    df['congestion_x_density']   = df['congestion_score'] * df['max_zone_density']
    df['traffic_severity']       = df['blocked_path_15m'] + df['near_collision_15m'] * 2
    df['aisle_load']             = df['aisle_traffic_score'] * df['congestion_score']
    df['layout_type_enc']        = df['layout_type'].map(LAYOUT_TYPE_MAP).fillna(-1).astype(np.int8)
    df['order_per_charger']      = df['order_inflow_15m'] / (df['charger_count'] + 1)
    df['robot_per_floor_area']   = df['robot_total'] / (df['floor_area_sqm'] + 1)
    df['pack_station_per_robot'] = df['pack_station_count'] / (df['robot_total'] + 1)

    for col in ['battery_mean', 'low_battery_ratio', 'congestion_score',
                'order_inflow_15m', 'robot_active', 'pack_utilization']:
        if col in df.columns:
            df[f'null_{col}'] = df[col].isna().astype(np.int8)

    for col in SKEWED_COLS:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].fillna(0))

    df['battery_crisis_index'] = (
        df['low_battery_ratio'] * df['charge_queue_length']
        + df['robot_charging'] / (df['battery_mean'] + 1e-5)
    )
    df['congestion_level'] = pd.cut(
        df['congestion_score'].fillna(0),
        bins=[-1, 0, 20, 100], labels=[0, 1, 2]
    ).astype(int)
    df['congestion_compound'] = (
        df['congestion_score'].fillna(0) * df['max_zone_density'].fillna(0)
        + df['blocked_path_15m'].fillna(0) * 2
        + df['near_collision_15m'].fillna(0) * 3
    )
    df['robot_saturation']   = 1 - (df['robot_idle'] / (df['robot_total'] + 1))
    df['operation_pressure'] = df['order_inflow_15m'] * df['low_battery_ratio'] / (df['robot_active'] + 1)
    df['triple_crisis']      = df['low_battery_ratio'] * df['congestion_score'].fillna(0) * df['order_inflow_15m']

    df['crisis_score']         = df['low_battery_ratio'] * df['congestion_score'].fillna(0)
    df['order_robot_stress']   = df['order_inflow_15m'] / (df['robot_active'] + 1) * df['low_battery_ratio']
    df['bottleneck_score']     = df['charge_queue_length'] * df['congestion_score'].fillna(0)
    df['complex_urgent_order'] = df['sku_concentration'] * df['urgent_order_ratio']
    if 'maintenance_schedule_score' in df.columns:
        df['maintenance_battery_risk'] = (1 - df['maintenance_schedule_score'].fillna(0.5)) * df['low_battery_ratio']
    df['layout_congestion'] = df['layout_type_enc'] * df['congestion_score'].fillna(0)
    df['layout_battery']    = df['layout_type_enc'] * df['low_battery_ratio']

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['slot_idx']      = df.groupby('scenario_id').cumcount()
    df['slot_progress'] = df['slot_idx'] / 24.0

    for col in EXPANDING_COLS:
        if col in df.columns:
            grp = df.groupby('scenario_id')[col]
            df[f'exp_mean_{col}'] = grp.expanding().mean().reset_index(level=0, drop=True)
            df[f'exp_max_{col}']  = grp.expanding().max().reset_index(level=0, drop=True)

    for col in ['congestion_score', 'low_battery_ratio', 'order_inflow_15m',
                'robot_active', 'blocked_path_15m', 'charge_queue_length']:
        if col in df.columns:
            df[f'lag1_{col}'] = df.groupby('scenario_id')[col].shift(1)

    return df


def add_aggregation_features(train_df, test_df):
    train_sc = train_df.groupby('scenario_id')[AGG_COLS].agg(AGG_FUNCS)
    train_sc.columns = [f'sc_{c}_{f}' for c, f in train_sc.columns]
    test_sc  = test_df.groupby('scenario_id')[AGG_COLS].agg(AGG_FUNCS)
    test_sc.columns  = [f'sc_{c}_{f}' for c, f in test_sc.columns]

    train_df = train_df.merge(train_sc.reset_index(), on='scenario_id', how='left')
    test_df  = test_df.merge(test_sc.reset_index(),   on='scenario_id', how='left')

    layout_agg = train_df.groupby('layout_id')[AGG_COLS].agg(AGG_FUNCS)
    layout_agg.columns = [f'layout_{c}_{f}' for c, f in layout_agg.columns]
    layout_agg = layout_agg.reset_index()
    lfc = [c for c in layout_agg.columns if c.startswith('layout_') and c != 'layout_id']

    train_df = train_df.merge(layout_agg[['layout_id'] + lfc], on='layout_id', how='left', suffixes=('', '_dup'))
    train_df.drop(columns=[c for c in train_df.columns if c.endswith('_dup')], inplace=True)
    test_df  = test_df.merge(layout_agg[['layout_id'] + lfc], on='layout_id', how='left')

    return train_df, test_df


def add_sc_derived_features(df):
    df = df.copy()

    for col in ZSCORE_COLS:
        m, s = f'sc_{col}_mean', f'sc_{col}_std'
        if col in df.columns and m in df.columns and s in df.columns:
            df[f'z_{col}']     = (df[col] - df[m]) / (df[s] + 1e-5)
            df[f'z_{col}_abs'] = df[f'z_{col}'].abs()

    for col in RANK_COLS:
        if col in df.columns:
            df[f'prank_{col}'] = df.groupby('scenario_id')[col].rank(pct=True, na_option='keep')

    for col in RATIO_COLS:
        if col in df.columns:
            if f'sc_{col}_max' in df.columns:
                df[f'ratio_to_max_{col}']  = df[col] / (df[f'sc_{col}_max'] + 1e-5)
                df[f'gap_to_max_{col}']    = df[f'sc_{col}_max'] - df[col]
            if f'sc_{col}_mean' in df.columns:
                df[f'ratio_to_mean_{col}'] = df[col] / (df[f'sc_{col}_mean'] + 1e-5)

    return df


def add_trajectory_features(df):
    df = df.copy()
    for col in TRAJECTORY_COLS:
        exp_col = f'exp_mean_{col}'
        sc_col  = f'sc_{col}_mean'
        if exp_col in df.columns and sc_col in df.columns:
            df[f'traj_{col}']     = df[exp_col] / (df[sc_col] + 1e-5)
            df[f'traj_dev_{col}'] = (df[exp_col] - df[sc_col]).abs()
    return df


def add_layout_target_encoding(train_df, test_df, y_log):
    gkf_te       = GroupKFold(n_splits=N_SPLITS)
    scenario_ids = train_df['scenario_id'].values

    layout_te_train = np.full(len(train_df), np.nan)

    for tr_idx, val_idx in gkf_te.split(train_df, y_log, scenario_ids):
        tr_sub = train_df.iloc[tr_idx].copy()
        tr_sub['_y'] = y_log[tr_idx]

        layout_mean      = tr_sub.groupby('layout_id')['_y'].mean()
        layout_type_mean = tr_sub.groupby('layout_type')['_y'].mean()
        global_mean      = float(y_log[tr_idx].mean())

        val_df  = train_df.iloc[val_idx]
        encoded = val_df['layout_id'].map(layout_mean).copy()

        missing_mask = encoded.isna()
        if missing_mask.any():
            encoded[missing_mask] = val_df.loc[missing_mask, 'layout_type'].map(layout_type_mean)
        encoded = encoded.fillna(global_mean)
        layout_te_train[val_idx] = encoded.values

    train_df = train_df.copy()
    train_df['layout_te'] = layout_te_train

    full_tr = train_df.copy()
    full_tr['_y'] = y_log
    layout_mean_all      = full_tr.groupby('layout_id')['_y'].mean()
    layout_type_mean_all = full_tr.groupby('layout_type')['_y'].mean()
    global_mean_all      = float(y_log.mean())

    test_df      = test_df.copy()
    test_encoded = test_df['layout_id'].map(layout_mean_all).copy()

    missing_test = test_encoded.isna()
    if missing_test.any():
        test_encoded[missing_test] = test_df.loc[missing_test, 'layout_type'].map(layout_type_mean_all)
    test_encoded   = test_encoded.fillna(global_mean_all)
    test_df['layout_te'] = test_encoded.values

    n_cold = int(missing_test.sum())
    print(f"  layout_te: train OOF 완료 | test cold-start {n_cold}행({n_cold/len(test_df)*100:.1f}%) → layout_type 평균 fallback")

    return train_df, test_df


# =============================================================================
# 4. 데이터셋 준비
# =============================================================================
print("\n" + "="*60)
print("2. 피처 엔지니어링")
print("="*60)

train = feature_engineering(train)
test  = feature_engineering(test)

print("  시간 순서 피처 추가 중...")
train = add_temporal_features(train)
test  = add_temporal_features(test)

train, test = add_aggregation_features(train, test)

train = add_sc_derived_features(train)
test  = add_sc_derived_features(test)

print("  Trajectory 피처 추가 중...")
train = add_trajectory_features(train)
test  = add_trajectory_features(test)

y_log_full = np.log1p(train[TARGET].values)

print("  Layout 타겟 인코딩 (OOF)...")
train, test = add_layout_target_encoding(train, test, y_log_full)

DROP_COLS    = {'ID', 'layout_id', 'scenario_id', 'layout_type', TARGET, 'shift_hour'}
feature_cols = [c for c in train.columns
                if c not in DROP_COLS and c in test.columns
                and train[c].dtype != object]
feature_cols = list(dict.fromkeys(feature_cols))

X_train = train[feature_cols].astype(np.float32)
y_train = y_log_full
X_test  = test[feature_cols].astype(np.float32)
groups  = train['scenario_id'].values
y_true  = train[TARGET].values

print(f"\n피처 수: {len(feature_cols)}개")
print(f"train: {X_train.shape} | test: {X_test.shape}")

# =============================================================================
# 5. 그룹 판별 (결측 구조 기반)
# =============================================================================
print("\n" + "="*60)
print("3. 결측치 구조 기반 그룹 분류")
print("="*60)

# feature_engineering 후 생성된 nan_* 플래그 사용
exist_nan_flags = [c for c in NAN_FLAG_COLS if c in X_train.columns]
print(f"  사용 nan 플래그: {exist_nan_flags}")

train_group = (X_train[exist_nan_flags].sum(axis=1) > 0).astype(int).values
test_group  = (X_test[exist_nan_flags].sum(axis=1) > 0).astype(int).values

for g in [0, 1]:
    gname = "정상(결측없음)" if g == 0 else "결측(이벤트있음)"
    t_cnt = (train_group == g).sum()
    te_cnt = (test_group == g).sum()
    t_scen = len(np.unique(groups[train_group == g]))
    print(f"  Group {g} {gname}:")
    print(f"    Train: {t_cnt:6d}행 ({t_cnt/len(train_group)*100:.1f}%) | {t_scen}개 시나리오")
    print(f"    Test : {te_cnt:6d}행 ({te_cnt/len(test_group)*100:.1f}%)")

    # 타겟 분포 확인
    g_mae_baseline = np.mean(np.abs(y_true[train_group == g] - y_true[train_group == g].mean()))
    print(f"    타겟 평균: {y_true[train_group==g].mean():.2f}분 | MAD: {g_mae_baseline:.2f}")


# =============================================================================
# 6. 모델 파라미터 (Step14 동일)
# =============================================================================
LGB_BASE = {
    'objective': 'regression_l1', 'metric': 'mae',
    'learning_rate': 0.02, 'num_leaves': 127, 'max_depth': -1,
    'min_child_samples': 50, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5,
    'lambda_l1': 0.1, 'lambda_l2': 0.1,
    'device_type': 'cpu', 'num_threads': N_CPU, 'verbose': -1,
}
XGB_BASE = {
    'objective': 'reg:absoluteerror', 'eval_metric': 'mae',
    'learning_rate': 0.05, 'max_depth': 7, 'min_child_weight': 50,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'tree_method': 'hist', 'device': 'cpu', 'nthread': N_CPU, 'verbosity': 0,
}
CAT_BASE = {
    'loss_function': 'MAE', 'eval_metric': 'MAE',
    'learning_rate': 0.05, 'depth': 8, 'l2_leaf_reg': 3.0,
    'min_data_in_leaf': 50, 'subsample': 0.8,
    'task_type': 'CPU', 'thread_count': N_CPU, 'verbose': False,
}


# =============================================================================
# 7. 그룹별 앙상블 학습 함수
# =============================================================================

def train_group_ensemble(Xg_tr, yg_tr, gg, yg_true, Xg_te, grp_name):
    """그룹별 LGB+XGB+CatBoost×5seeds 앙상블 학습
    Returns: (best_oof_pred, best_test_pred, best_mae)
    """
    gkf_g   = GroupKFold(n_splits=N_SPLITS)
    g_oof   = {'LightGBM': {}, 'XGBoost': {}, 'CatBoost': {}}
    g_test  = {'LightGBM': {}, 'XGBoost': {}, 'CatBoost': {}}
    g_score = {'LightGBM': {}, 'XGBoost': {}, 'CatBoost': {}}

    # ── LightGBM ──────────────────────────────────────────────────────────────
    print(f"\n  [{grp_name}] LightGBM × {len(SEEDS)}시드")
    for seed in SEEDS:
        t0     = time.time()
        params = {**LGB_BASE, 'seed': seed}
        oof_p  = np.zeros(len(Xg_tr))
        test_p = np.zeros(len(Xg_te))
        for tr_idx, val_idx in gkf_g.split(Xg_tr, yg_tr, gg):
            X_tr, X_val = Xg_tr.iloc[tr_idx], Xg_tr.iloc[val_idx]
            y_tr, y_val = yg_tr[tr_idx], yg_tr[val_idx]
            dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
            dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)
            model  = lgb.train(params, dtrain, num_boost_round=LGB_MAX_ROUNDS, valid_sets=[dvalid],
                               callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False),
                                          lgb.log_evaluation(9999)])
            oof_p[val_idx] = np.clip(np.expm1(model.predict(X_val,  num_iteration=model.best_iteration)), 0, None)
            test_p        += np.clip(np.expm1(model.predict(Xg_te,  num_iteration=model.best_iteration)), 0, None) / N_SPLITS
        sc = mean_absolute_error(yg_true, oof_p)
        g_oof['LightGBM'][seed] = oof_p; g_test['LightGBM'][seed] = test_p; g_score['LightGBM'][seed] = sc
        print(f"    seed={seed:5d} | OOF: {sc:.4f} | {(time.time()-t0)/60:.1f}min")
    s = list(g_score['LightGBM'].values())
    print(f"    LGB 평균: {np.mean(s):.4f} ± {np.std(s):.4f}")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    print(f"\n  [{grp_name}] XGBoost × {len(SEEDS)}시드")
    for seed in SEEDS:
        t0     = time.time()
        params = {**XGB_BASE, 'seed': seed}
        oof_p  = np.zeros(len(Xg_tr))
        test_p = np.zeros(len(Xg_te))
        for tr_idx, val_idx in gkf_g.split(Xg_tr, yg_tr, gg):
            X_tr, X_val = Xg_tr.iloc[tr_idx], Xg_tr.iloc[val_idx]
            y_tr, y_val = yg_tr[tr_idx], yg_tr[val_idx]
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dvalid = xgb.DMatrix(X_val, label=y_val)
            dte    = xgb.DMatrix(Xg_te)
            model  = xgb.train(params, dtrain, num_boost_round=MAX_ROUNDS,
                               evals=[(dvalid, 'val')], early_stopping_rounds=EARLY_STOP, verbose_eval=False)
            oof_p[val_idx] = np.clip(np.expm1(model.predict(dvalid, iteration_range=(0, model.best_iteration))), 0, None)
            test_p        += np.clip(np.expm1(model.predict(dte,    iteration_range=(0, model.best_iteration))), 0, None) / N_SPLITS
        sc = mean_absolute_error(yg_true, oof_p)
        g_oof['XGBoost'][seed] = oof_p; g_test['XGBoost'][seed] = test_p; g_score['XGBoost'][seed] = sc
        print(f"    seed={seed:5d} | OOF: {sc:.4f} | {(time.time()-t0)/60:.1f}min")
    s = list(g_score['XGBoost'].values())
    print(f"    XGB 평균: {np.mean(s):.4f} ± {np.std(s):.4f}")

    # ── CatBoost ──────────────────────────────────────────────────────────────
    print(f"\n  [{grp_name}] CatBoost × {len(SEEDS)}시드")
    for seed in SEEDS:
        t0    = time.time()
        cat_p = {**CAT_BASE, 'random_seed': seed}
        oof_p  = np.zeros(len(Xg_tr))
        test_p = np.zeros(len(Xg_te))
        for tr_idx, val_idx in gkf_g.split(Xg_tr, yg_tr, gg):
            X_tr, X_val = Xg_tr.iloc[tr_idx], Xg_tr.iloc[val_idx]
            y_tr, y_val = yg_tr[tr_idx], yg_tr[val_idx]
            model = CatBoostRegressor(iterations=MAX_ROUNDS, early_stopping_rounds=EARLY_STOP, **cat_p)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
            oof_p[val_idx] = np.clip(np.expm1(model.predict(X_val)),  0, None)
            test_p        += np.clip(np.expm1(model.predict(Xg_te)), 0, None) / N_SPLITS
        sc = mean_absolute_error(yg_true, oof_p)
        g_oof['CatBoost'][seed] = oof_p; g_test['CatBoost'][seed] = test_p; g_score['CatBoost'][seed] = sc
        print(f"    seed={seed:5d} | OOF: {sc:.4f} | {(time.time()-t0)/60:.1f}min")
    s = list(g_score['CatBoost'].values())
    print(f"    CatBoost 평균: {np.mean(s):.4f} ± {np.std(s):.4f}")

    # ── 그룹 내 앙상블 ─────────────────────────────────────────────────────────
    print(f"\n  [{grp_name}] 앙상블")
    model_oof_avg  = {}
    model_test_avg = {}
    model_oof_mae  = {}
    for m in ['LightGBM', 'XGBoost', 'CatBoost']:
        oof_mean  = np.stack(list(g_oof[m].values())).mean(axis=0)
        test_mean = np.stack(list(g_test[m].values())).mean(axis=0)
        mae       = mean_absolute_error(yg_true, oof_mean)
        model_oof_avg[m] = oof_mean; model_test_avg[m] = test_mean; model_oof_mae[m] = mae
        print(f"    {m:12s} 시드평균 OOF: {mae:.4f}")

    all_oof_list  = [g_oof[m][s]  for m in ['LightGBM', 'XGBoost', 'CatBoost'] for s in SEEDS]
    all_test_list = [g_test[m][s] for m in ['LightGBM', 'XGBoost', 'CatBoost'] for s in SEEDS]
    mae_avg  = mean_absolute_error(yg_true, np.mean(all_oof_list, axis=0))
    test_avg = np.mean(all_test_list, axis=0)

    inv = {m: 1/model_oof_mae[m] for m in model_oof_mae}
    w   = {m: inv[m]/sum(inv.values()) for m in inv}
    oof_w  = sum(w[m]*model_oof_avg[m]  for m in w)
    test_w = sum(w[m]*model_test_avg[m] for m in w)
    mae_w  = mean_absolute_error(yg_true, oof_w)

    print(f"    단순평균 OOF: {mae_avg:.4f} | 역MAE가중 OOF: {mae_w:.4f}")

    best_name, best_mae, best_oof_pred, best_test_pred = min(
        [('all_avg',   mae_avg, np.mean(all_oof_list, axis=0), test_avg),
         ('inv_mae_w', mae_w,   oof_w,                          test_w)],
        key=lambda x: x[1]
    )
    print(f"    [{grp_name}] 최적: {best_name}  OOF={best_mae:.4f}")

    return best_oof_pred, best_test_pred, best_mae


# =============================================================================
# 8. 그룹별 학습
# =============================================================================
print("\n" + "="*60)
print("4. 그룹별 학습 시작")
print("="*60)

total_start  = time.time()
group_results = {}

for grp in [0, 1]:
    grp_name  = "G0(정상)" if grp == 0 else "G1(결측)"
    g_tr_idx  = np.where(train_group == grp)[0]
    g_te_idx  = np.where(test_group  == grp)[0]

    print(f"\n{'='*60}")
    print(f"  그룹 {grp_name}: train {len(g_tr_idx)}행 | test {len(g_te_idx)}행")
    print(f"{'='*60}")

    Xg_tr   = X_train.iloc[g_tr_idx].reset_index(drop=True)
    yg_tr   = y_train[g_tr_idx]
    gg      = groups[g_tr_idx]
    yg_true = y_true[g_tr_idx]
    Xg_te   = X_test.iloc[g_te_idx].reset_index(drop=True)

    best_oof, best_test, best_mae = train_group_ensemble(
        Xg_tr, yg_tr, gg, yg_true, Xg_te, grp_name
    )

    group_results[grp] = {
        'tr_idx':  g_tr_idx,
        'te_idx':  g_te_idx,
        'oof':     best_oof,
        'test':    best_test,
        'mae':     best_mae,
        'name':    grp_name,
    }

# =============================================================================
# 9. 전체 예측 합산
# =============================================================================
print("\n" + "="*60)
print("5. 전체 예측 합산")
print("="*60)

final_oof  = np.zeros(len(X_train))
final_test = np.zeros(len(X_test))

for grp, res in group_results.items():
    final_oof[res['tr_idx']]  = res['oof']
    final_test[res['te_idx']] = res['test']
    print(f"  {res['name']}: OOF MAE={res['mae']:.4f} ({len(res['tr_idx'])}행)")

overall_mae = mean_absolute_error(y_true, final_oof)
print(f"\n  전체 OOF MAE: {overall_mae:.4f}")

# =============================================================================
# 10. 비교 요약
# =============================================================================
print("\n" + "="*60)
print("6. Step 비교")
print("="*60)
print(f"  Step7  OOF MAE: 8.6852  (197 피처, 단일 앙상블)")
print(f"  Step12 OOF MAE: 8.6050  (300 피처, lr=0.05 균일)")
print(f"  Step14 OOF MAE: 8.6037  (300 피처, LGB lr=0.02 / XGB+Cat lr=0.05, 기준)")
print(f"  Step16 OOF MAE: {overall_mae:.4f}  (결측 구조 분리, 2그룹 독립 앙상블)")
print(f"  Step14→16 변화: {overall_mae - 8.6037:+.4f}")
print(f"\n  Step14 Public: 10.0733  (현재 최고)")
print(f"  Step15 Public: 미제출 (예상 ~{overall_mae + 1.469:.4f})")
print(f"\n  총 소요 시간: {(time.time()-total_start)/60:.1f}분")

# =============================================================================
# 11. 제출 파일
# =============================================================================
submission = sample.copy()
submission[TARGET] = np.clip(final_test, 0, None)
out = os.path.join(OUTPUT_DIR, 'submission_step16_missing_split_ensemble.csv')
submission.to_csv(out, index=False)
print(f"\n제출 파일: submission_step16_missing_split_ensemble.csv")
print(f"OOF MAE: {overall_mae:.4f}")
print(f"\n예측값 분포:")
print(submission[TARGET].describe().round(3).to_string())
