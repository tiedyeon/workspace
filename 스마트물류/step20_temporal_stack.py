# =============================================================================
# Step 20: Step14 피처 + Step19 Temporal 피처 스택
# =============================================================================
# Step14 대비 추가 내용:
#   - add_lag_rolling_features(): 22개 SEQ_COLS × 6피처 = 132개 lag/rolling
#   - add_onset_features(): 충전/큐 이벤트 onset detection 10개
#   - add_threshold_features(): 배터리/충전압력 hinge 피처 6개
#   Total: Step14 ~300개 + 신규 ~148개 ≈ 448개
#
# Step14와 동일하게 유지:
#   - sc_* 집계 피처 (가장 중요)
#   - z-score, percentile rank, trajectory 피처
#   - layout_id 타겟 인코딩
#   - LGB lr=0.02 / XGB+Cat lr=0.05 (3모델 × 5시드)
#   - log1p(y) 학습
#
# 실행:
#   caffeinate -i nohup python -u -B step20_temporal_stack.py > step20_output.log 2>&1 &
#   tail -f step20_output.log
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

N_CPU      = os.cpu_count()
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = DATA_DIR
TARGET     = 'avg_delay_minutes_next_30m'
SEEDS      = [42, 123, 456, 789, 2024]
N_SPLITS   = 5
MAX_ROUNDS     = 3000
EARLY_STOP     = 150
LGB_MAX_ROUNDS = 5000
LGB_EARLY_STOP = 200

LAYOUT_TYPE_MAP = {'grid': 0, 'hybrid': 1, 'narrow': 2, 'hub_spoke': 3}

print(f"CPU: {N_CPU}코어 | 시드: {len(SEEDS)}개 × 3모델 = {len(SEEDS)*3}인스턴스")

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

# ── [NEW] Step19 temporal 피처 대상 컬럼 ──────────────────────────────────────
SEQ_COLS = [
    'order_inflow_15m', 'unique_sku_15m', 'robot_active', 'robot_idle', 'robot_charging',
    'battery_mean', 'battery_std', 'low_battery_ratio', 'charge_queue_length',
    'avg_charge_wait', 'congestion_score', 'max_zone_density', 'blocked_path_15m',
    'near_collision_15m', 'fault_count_15m', 'avg_recovery_time', 'task_reassign_15m',
    'replenishment_overlap', 'pack_utilization', 'loading_dock_util',
    'staging_area_util', 'label_print_queue',
]

# =============================================================================
# 3. 피처 엔지니어링 함수 (Step14 원본 그대로)
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

    df['battery_stress']        = df['low_battery_ratio'] / (df['battery_mean'] + 1e-5)
    df['charge_bottleneck']     = df['charge_queue_length'] * df['avg_charge_wait']
    df['battery_volatility']    = df['battery_std'] / (df['battery_mean'] + 1e-5)
    df['battery_health']        = df['battery_mean'] - df['battery_std']
    df['order_per_robot']       = df['order_inflow_15m'] / (df['robot_active'] + 1)
    df['order_per_pack_station']= df['order_inflow_15m'] / (df['pack_station_count'] + 1)
    df['robot_effective_util']  = df['robot_active'] / (df['robot_total'] + 1)
    df['idle_ratio']            = df['robot_idle'] / (df['robot_active'] + df['robot_idle'] + df['robot_charging'] + 1)
    df['charging_ratio']        = df['robot_charging'] / (df['robot_total'] + 1)
    df['congestion_x_density']  = df['congestion_score'] * df['max_zone_density']
    df['traffic_severity']      = df['blocked_path_15m'] + df['near_collision_15m'] * 2
    df['aisle_load']            = df['aisle_traffic_score'] * df['congestion_score']
    df['layout_type_enc']       = df['layout_type'].map(LAYOUT_TYPE_MAP).fillna(-1).astype(np.int8)
    df['order_per_charger']     = df['order_inflow_15m'] / (df['charger_count'] + 1)
    df['robot_per_floor_area']  = df['robot_total'] / (df['floor_area_sqm'] + 1)
    df['pack_station_per_robot']= df['pack_station_count'] / (df['robot_total'] + 1)

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

    df['crisis_score']        = df['low_battery_ratio'] * df['congestion_score'].fillna(0)
    df['order_robot_stress']  = df['order_inflow_15m'] / (df['robot_active'] + 1) * df['low_battery_ratio']
    df['bottleneck_score']    = df['charge_queue_length'] * df['congestion_score'].fillna(0)
    df['complex_urgent_order']= df['sku_concentration'] * df['urgent_order_ratio']
    if 'maintenance_schedule_score' in df.columns:
        df['maintenance_battery_risk'] = (1 - df['maintenance_schedule_score'].fillna(0.5)) * df['low_battery_ratio']
    df['layout_congestion']   = df['layout_type_enc'] * df['congestion_score'].fillna(0)
    df['layout_battery']      = df['layout_type_enc'] * df['low_battery_ratio']

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step14 원본 temporal 피처 (slot_idx, expanding, lag1×6)"""
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


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """[NEW] Step19: SEQ_COLS 전체에 lag1/lag2/diff1/rollmean3/rollmax3/dev_roll3"""
    df = df.copy()
    grp = df.groupby('scenario_id', sort=False)
    grp_key = df['scenario_id']

    for col in SEQ_COLS:
        if col not in df.columns:
            continue
        lag1 = grp[col].shift(1)
        lag2 = grp[col].shift(2)
        df[f'{col}__lag1']  = lag1
        df[f'{col}__lag2']  = lag2
        df[f'{col}__diff1'] = df[col] - lag1
        lag1_grp  = lag1.groupby(grp_key)
        roll_mean = lag1_grp.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        roll_max  = lag1_grp.rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
        df[f'{col}__rollmean3']     = roll_mean
        df[f'{col}__rollmax3']      = roll_max
        df[f'{col}__dev_rollmean3'] = df[col] - roll_mean

    return df


def add_onset_features(df: pd.DataFrame) -> pd.DataFrame:
    """[NEW] Step19: 충전/큐 이벤트 onset 감지 피처 (slot_idx 필요)"""
    df = df.copy()
    grp_key = df['scenario_id']

    def add_onset(value_col, prefix):
        if value_col not in df.columns:
            return
        positive = df[value_col].fillna(0).gt(0)
        t = df['slot_idx'].where(positive).astype(float)
        first = t.groupby(grp_key).transform(lambda s: s.ffill().cummin())
        prev  = positive.groupby(grp_key).shift(1, fill_value=False)
        df[f'{prefix}_ever_started']      = first.notna().astype(np.int8)
        df[f'{prefix}_start_idx']         = first.fillna(-1).astype(np.float32)
        df[f'{prefix}_started_now']       = (positive & ~prev).astype(np.int8)
        df[f'{prefix}_started_early']     = (first <= 5).fillna(False).astype(np.int8)
        df[f'{prefix}_steps_since_start'] = np.where(
            first.notna(), (df['slot_idx'] - first).astype(float), -1.0
        ).astype(np.float32)

    add_onset('robot_charging',      'charging')
    add_onset('charge_queue_length', 'queue')

    return df


def add_threshold_features(df: pd.DataFrame) -> pd.DataFrame:
    """[NEW] Step19: hinge/threshold 피처 + charge_pressure"""
    df = df.copy()

    # charge_pressure (충전 병목 압력 지표)
    if {'robot_charging', 'charge_queue_length', 'charger_count'}.issubset(df.columns):
        df['charge_pressure2'] = (
            (df['robot_charging'] + df['charge_queue_length'])
            / (df['charger_count'] + 1e-5)
        )

    if 'battery_mean' in df.columns:
        df['battery_mean_below_44']   = np.clip(44.0 - df['battery_mean'].fillna(44), 0, None)
    if 'charge_pressure2' in df.columns:
        df['charge_pressure_above_136'] = np.clip(df['charge_pressure2'] - 1.36, 0, None)
    if 'pack_utilization' in df.columns:
        df['pack_utilization_sq']     = df['pack_utilization'].fillna(0) ** 2
    if 'loading_dock_util' in df.columns:
        df['loading_dock_util_sq']    = df['loading_dock_util'].fillna(0) ** 2
    if 'staging_area_util' in df.columns:
        df['staging_area_util_sq']    = df['staging_area_util'].fillna(0) ** 2

    return df


def add_aggregation_features(train_df, test_df):
    """sc_* 및 layout_agg 집계 피처 추가 (Step14 원본)"""
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
    """z-score, 퍼센타일 랭크, 비율 (Step14 원본)"""
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
    """Trajectory 피처 (Step14 원본)"""
    df = df.copy()
    for col in TRAJECTORY_COLS:
        exp_col = f'exp_mean_{col}'
        sc_col  = f'sc_{col}_mean'
        if exp_col in df.columns and sc_col in df.columns:
            df[f'traj_{col}']     = df[exp_col] / (df[sc_col] + 1e-5)
            df[f'traj_dev_{col}'] = (df[exp_col] - df[sc_col]).abs()
    return df


def add_layout_target_encoding(train_df, test_df, y_log):
    """Layout_id 타겟 인코딩 (Step14 원본)"""
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
    test_encoded = test_encoded.fillna(global_mean_all)
    test_df['layout_te'] = test_encoded.values

    n_cold = int(missing_test.sum())
    print(f"  layout_te: train OOF 완료 | test cold-start {n_cold}행({n_cold/len(test_df)*100:.1f}%) → layout_type 평균 fallback")

    return train_df, test_df


# =============================================================================
# 4. 파이프라인 실행
# =============================================================================
print("\n" + "="*60)
print("2. 피처 엔지니어링")
print("="*60)

train = feature_engineering(train)
test  = feature_engineering(test)

print("  [Step14] 시간 순서 피처 (slot_idx, expanding, lag1×6)...")
train = add_temporal_features(train)
test  = add_temporal_features(test)

print("  [NEW] Lag/Rolling 피처 (22 SEQ_COLS × 6)...")
train = add_lag_rolling_features(train)
test  = add_lag_rolling_features(test)

print("  [NEW] Onset detection 피처...")
train = add_onset_features(train)
test  = add_onset_features(test)

print("  [NEW] Threshold/hinge 피처...")
train = add_threshold_features(train)
test  = add_threshold_features(test)

print("  [Step14] sc_* 집계 피처...")
train, test = add_aggregation_features(train, test)

train = add_sc_derived_features(train)
test  = add_sc_derived_features(test)

print("  [Step14] Trajectory 피처...")
train = add_trajectory_features(train)
test  = add_trajectory_features(test)

y_log_full = np.log1p(train[TARGET].values)

print("  [Step14] Layout 타겟 인코딩 (OOF)...")
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

print(f"\n  Step14 피처 수: ~300개  →  Step20 피처 수: {len(feature_cols)}개 (+{len(feature_cols)-300}개)")
print(f"  train: {X_train.shape} | test: {X_test.shape}")

# =============================================================================
# 5. 모델 파라미터 (Step14 동일)
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
# 6. 학습
# =============================================================================
print("\n" + "="*60)
print("3. 학습 시작 (LGB lr=0.02 / XGB,Cat lr=0.05)")
print("="*60)

all_oof   = {'LightGBM': {}, 'XGBoost': {}, 'CatBoost': {}}
all_test  = {'LightGBM': {}, 'XGBoost': {}, 'CatBoost': {}}
all_score = {'LightGBM': {}, 'XGBoost': {}, 'CatBoost': {}}

gkf         = GroupKFold(n_splits=N_SPLITS)
total_start = time.time()
run_count   = 0
total_runs  = len(SEEDS) * 3

# LightGBM
print("\nLightGBM × 멀티시드")
for seed in SEEDS:
    t0 = time.time()
    params = {**LGB_BASE, 'seed': seed}
    oof_p  = np.zeros(len(X_train))
    test_p = np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)
        model  = lgb.train(params, dtrain, num_boost_round=LGB_MAX_ROUNDS, valid_sets=[dvalid],
                           callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False),
                                      lgb.log_evaluation(9999)])
        oof_p[val_idx] = np.clip(np.expm1(model.predict(X_val,  num_iteration=model.best_iteration)), 0, None)
        test_p        += np.clip(np.expm1(model.predict(X_test, num_iteration=model.best_iteration)), 0, None) / N_SPLITS
    sc = mean_absolute_error(y_true, oof_p)
    all_oof['LightGBM'][seed] = oof_p
    all_test['LightGBM'][seed] = test_p
    all_score['LightGBM'][seed] = sc
    run_count += 1
    print(f"  seed={seed:5d} | OOF: {sc:.4f} | {(time.time()-t0)/60:.1f}min | [{run_count}/{total_runs}]")
s = list(all_score['LightGBM'].values())
print(f"  LGB 평균: {np.mean(s):.4f} ± {np.std(s):.4f}")

# XGBoost
print("\nXGBoost × 멀티시드")
for seed in SEEDS:
    t0 = time.time()
    params = {**XGB_BASE, 'seed': seed}
    oof_p  = np.zeros(len(X_train))
    test_p = np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        dte    = xgb.DMatrix(X_test)
        model  = xgb.train(params, dtrain, num_boost_round=MAX_ROUNDS,
                           evals=[(dvalid, 'val')], early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        oof_p[val_idx] = np.clip(np.expm1(model.predict(dvalid, iteration_range=(0, model.best_iteration))), 0, None)
        test_p        += np.clip(np.expm1(model.predict(dte,    iteration_range=(0, model.best_iteration))), 0, None) / N_SPLITS
    sc = mean_absolute_error(y_true, oof_p)
    all_oof['XGBoost'][seed] = oof_p
    all_test['XGBoost'][seed] = test_p
    all_score['XGBoost'][seed] = sc
    run_count += 1
    print(f"  seed={seed:5d} | OOF: {sc:.4f} | {(time.time()-t0)/60:.1f}min | [{run_count}/{total_runs}]")
s = list(all_score['XGBoost'].values())
print(f"  XGB 평균: {np.mean(s):.4f} ± {np.std(s):.4f}")

# CatBoost
print("\nCatBoost × 멀티시드")
for seed in SEEDS:
    t0 = time.time()
    cat_p = {**CAT_BASE, 'random_seed': seed}
    oof_p  = np.zeros(len(X_train))
    test_p = np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        model = CatBoostRegressor(iterations=MAX_ROUNDS, early_stopping_rounds=EARLY_STOP, **cat_p)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
        oof_p[val_idx] = np.clip(np.expm1(model.predict(X_val)),  0, None)
        test_p        += np.clip(np.expm1(model.predict(X_test)), 0, None) / N_SPLITS
    sc = mean_absolute_error(y_true, oof_p)
    all_oof['CatBoost'][seed] = oof_p
    all_test['CatBoost'][seed] = test_p
    all_score['CatBoost'][seed] = sc
    run_count += 1
    print(f"  seed={seed:5d} | OOF: {sc:.4f} | {(time.time()-t0)/60:.1f}min | [{run_count}/{total_runs}]")
s = list(all_score['CatBoost'].values())
print(f"  CatBoost 평균: {np.mean(s):.4f} ± {np.std(s):.4f}")

# =============================================================================
# 7. 앙상블
# =============================================================================
print("\n" + "="*60)
print("4. 앙상블")
print("="*60)

model_oof_avg  = {}
model_test_avg = {}
model_oof_mae  = {}

for m in ['LightGBM', 'XGBoost', 'CatBoost']:
    oof_mean  = np.stack(list(all_oof[m].values())).mean(axis=0)
    test_mean = np.stack(list(all_test[m].values())).mean(axis=0)
    mae       = mean_absolute_error(y_true, oof_mean)
    model_oof_avg[m] = oof_mean; model_test_avg[m] = test_mean; model_oof_mae[m] = mae
    print(f"  {m:12s} 시드평균 OOF MAE: {mae:.4f}")

all_oof_list  = [all_oof[m][s]  for m in ['LightGBM','XGBoost','CatBoost'] for s in SEEDS]
all_test_list = [all_test[m][s] for m in ['LightGBM','XGBoost','CatBoost'] for s in SEEDS]
mae_avg  = mean_absolute_error(y_true, np.mean(all_oof_list, axis=0))
test_avg = np.mean(all_test_list, axis=0)
print(f"\n  전체 단순 평균 (15개) OOF MAE: {mae_avg:.4f}")

inv = {m: 1/model_oof_mae[m] for m in model_oof_mae}
w   = {m: inv[m]/sum(inv.values()) for m in inv}
print(f"\n  역MAE 가중치: " + " | ".join(f"{m}={w[m]:.3f}" for m in w))
oof_w  = sum(w[m]*model_oof_avg[m]  for m in w)
test_w = sum(w[m]*model_test_avg[m] for m in w)
mae_w  = mean_absolute_error(y_true, oof_w)
print(f"  역MAE 가중 앙상블 OOF MAE: {mae_w:.4f}")

best_name, best_mae, best_preds = min(
    [('all_avg', mae_avg, test_avg), ('inv_mae_w', mae_w, test_w)],
    key=lambda x: x[1]
)
print(f"\n  최적: {best_name}  OOF MAE={best_mae:.4f}")

# =============================================================================
# 8. 결과 요약
# =============================================================================
print("\n" + "="*60)
print("5. Step 비교")
print("="*60)
print(f"  Step14 OOF MAE: 8.6037  (300 피처, LGB lr=0.02 / XGB+Cat lr=0.05)")
print(f"  Step20 OOF MAE: {best_mae:.4f}  ({len(feature_cols)} 피처, 동일 구조 + temporal 스택)")
print(f"  Step14→20 변화: {best_mae - 8.6037:+.4f}")
print(f"\n  Step14 Public: 10.0733")
print(f"  Step20 예상:   ~{best_mae + 1.469:.4f}  (갭 1.469 가정)")

total_elapsed = time.time() - total_start
print(f"\n  총 소요 시간: {total_elapsed/60:.1f}분")

# =============================================================================
# 9. 제출 파일
# =============================================================================
submission = sample.copy()
submission[TARGET] = np.clip(best_preds, 0, None)
out = os.path.join(OUTPUT_DIR, 'submission_step20_temporal_stack.csv')
submission.to_csv(out, index=False)
print(f"\n제출 파일: submission_step20_temporal_stack.csv")
print(f"OOF MAE: {best_mae:.4f}")
print(f"\n예측값 분포:")
print(submission[TARGET].describe().round(3).to_string())
