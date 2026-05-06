# =============================================================================
# Step 29: Extreme Value Focus - Queuing Theory + Two-Stage + Focal Weight
# =============================================================================
# Step20 대비 추가 내용:
#   1. add_queuing_theory_features():
#      - pack_queue_wait = pack_utilization / (1 - pack_utilization + eps)  [M/M/1]
#      - robot_queue_pressure = robot_utilization / (1 - robot_utilization + eps)
#      - loading_queue_wait = loading_dock_util / (1 - loading_dock_util + eps)
#      - system_saturation, total_queue_pressure
#      - little_law_pressure, arrival_service_imbalance
#      - 각 큐 피처의 lag1/diff1/worsening (악화 추이)
#      - extreme_risk_score (pack_queue_wait × order_inflow × urgent)
#      Total: ~18개 새 피처 + 시나리오 집계 ~6개
#
#   2. Focal Sample Weight:
#      0~30분: w=1.0 / 30~50분: w=2.0 / 50~100분: w=5.0 / 100분+: w=10.0
#
#   3. Two-Stage:
#      Stage A: LGB+XGB+Cat × 5 seeds (Focal Weight) - 전체 회귀
#      Stage B: LGB 이진 분류기 (≥50분 여부)
#      Stage C: LGB 극단값 전용 회귀 (≥50분 케이스만)
#      최종: pred = (1 - alpha*p)*predA + alpha*p*predC, alpha 탐색
#
# 규정: test.csv 데이터 학습 사용 전면 금지
#
# 실행:
#   source .venv/bin/activate
#   caffeinate -i nohup python -u -B step29_extreme_focus.py > step29_output.log 2>&1 &
#   tail -f step29_output.log
# =============================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score, f1_score
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
EXTREME_THRESHOLD = 50.0

LAYOUT_TYPE_MAP = {'grid': 0, 'hybrid': 1, 'narrow': 2, 'hub_spoke': 3}
print(f"CPU: {N_CPU}코어 | 시드: {len(SEEDS)}개 | 극단 임계: {EXTREME_THRESHOLD}분")

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
print(f"극단값(>={EXTREME_THRESHOLD}분): {(train[TARGET]>=EXTREME_THRESHOLD).sum()}"
      f" ({(train[TARGET]>=EXTREME_THRESHOLD).mean()*100:.1f}%)")

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
MEANINGFUL_NAN_COLS = ['avg_recovery_time', 'congestion_score', 'loading_dock_util', 'staging_area_util']
ZERO_FILL_COLS = ['charge_queue_length', 'avg_charge_wait', 'blocked_path_15m',
                  'near_collision_15m', 'fault_count_15m', 'task_reassign_15m',
                  'replenishment_overlap', 'congestion_score', 'max_zone_density',
                  'avg_recovery_time', 'loading_dock_util', 'staging_area_util']
EXPANDING_COLS = ['congestion_score', 'low_battery_ratio', 'order_inflow_15m',
                  'robot_active', 'blocked_path_15m', 'charge_queue_length',
                  'max_zone_density', 'battery_mean']
TRAJECTORY_COLS = ['congestion_score', 'low_battery_ratio', 'order_inflow_15m',
                   'robot_active', 'blocked_path_15m']
SEQ_COLS = [
    'order_inflow_15m', 'unique_sku_15m', 'robot_active', 'robot_idle', 'robot_charging',
    'battery_mean', 'battery_std', 'low_battery_ratio', 'charge_queue_length',
    'avg_charge_wait', 'congestion_score', 'max_zone_density', 'blocked_path_15m',
    'near_collision_15m', 'fault_count_15m', 'avg_recovery_time', 'task_reassign_15m',
    'replenishment_overlap', 'pack_utilization', 'loading_dock_util',
    'staging_area_util', 'label_print_queue',
]

# =============================================================================
# 3. 피처 엔지니어링 함수 (Step20 원본 + Step29 큐이론)
# =============================================================================
def feature_engineering(df):
    df = df.copy()
    for col in MEANINGFUL_NAN_COLS:
        if col in df.columns:
            df[f'nan_{col}'] = df[col].isna().astype(np.int8)
    for col in ZERO_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in ['congestion_score','blocked_path_15m','near_collision_15m',
                'charge_queue_length','avg_charge_wait','fault_count_15m',
                'replenishment_overlap','task_reassign_15m']:
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
    for col in ['battery_mean','low_battery_ratio','congestion_score',
                'order_inflow_15m','robot_active','pack_utilization']:
        if col in df.columns:
            df[f'null_{col}'] = df[col].isna().astype(np.int8)
    for col in SKEWED_COLS:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].fillna(0))
    df['battery_crisis_index'] = (df['low_battery_ratio'] * df['charge_queue_length']
                                   + df['robot_charging'] / (df['battery_mean'] + 1e-5))
    df['congestion_level'] = pd.cut(df['congestion_score'].fillna(0),
                                     bins=[-1,0,20,100], labels=[0,1,2]).astype(int)
    df['congestion_compound'] = (df['congestion_score'].fillna(0) * df['max_zone_density'].fillna(0)
                                  + df['blocked_path_15m'].fillna(0) * 2
                                  + df['near_collision_15m'].fillna(0) * 3)
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


def add_temporal_features(df):
    df = df.copy()
    df['slot_idx']      = df.groupby('scenario_id').cumcount()
    df['slot_progress'] = df['slot_idx'] / 24.0
    for col in EXPANDING_COLS:
        if col in df.columns:
            grp = df.groupby('scenario_id')[col]
            df[f'exp_mean_{col}'] = grp.expanding().mean().reset_index(level=0, drop=True)
            df[f'exp_max_{col}']  = grp.expanding().max().reset_index(level=0, drop=True)
    for col in ['congestion_score','low_battery_ratio','order_inflow_15m',
                'robot_active','blocked_path_15m','charge_queue_length']:
        if col in df.columns:
            df[f'lag1_{col}'] = df.groupby('scenario_id')[col].shift(1)
    return df


def add_lag_rolling_features(df):
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


def add_onset_features(df):
    df = df.copy()
    grp_key = df['scenario_id']
    def add_onset(value_col, prefix):
        if value_col not in df.columns: return
        positive = df[value_col].fillna(0).gt(0)
        t = df['slot_idx'].where(positive).astype(float)
        first = t.groupby(grp_key).transform(lambda s: s.ffill().cummin())
        prev  = positive.groupby(grp_key).shift(1, fill_value=False)
        df[f'{prefix}_ever_started']      = first.notna().astype(np.int8)
        df[f'{prefix}_start_idx']         = first.fillna(-1).astype(np.float32)
        df[f'{prefix}_started_now']       = (positive & ~prev).astype(np.int8)
        df[f'{prefix}_started_early']     = (first <= 5).fillna(False).astype(np.int8)
        df[f'{prefix}_steps_since_start'] = np.where(
            first.notna(), (df['slot_idx'] - first).astype(float), -1.0).astype(np.float32)
    add_onset('robot_charging',      'charging')
    add_onset('charge_queue_length', 'queue')
    return df


def add_threshold_features(df):
    df = df.copy()
    if {'robot_charging','charge_queue_length','charger_count'}.issubset(df.columns):
        df['charge_pressure2'] = ((df['robot_charging'] + df['charge_queue_length'])
                                   / (df['charger_count'] + 1e-5))
    if 'battery_mean'     in df.columns: df['battery_mean_below_44']     = np.clip(44.0 - df['battery_mean'].fillna(44), 0, None)
    if 'charge_pressure2' in df.columns: df['charge_pressure_above_136'] = np.clip(df['charge_pressure2'] - 1.36, 0, None)
    if 'pack_utilization' in df.columns: df['pack_utilization_sq']       = df['pack_utilization'].fillna(0) ** 2
    if 'loading_dock_util' in df.columns: df['loading_dock_util_sq']     = df['loading_dock_util'].fillna(0) ** 2
    if 'staging_area_util' in df.columns: df['staging_area_util_sq']     = df['staging_area_util'].fillna(0) ** 2
    return df


def add_queuing_theory_features(df):
    """
    [NEW Step29] M/M/1 대기이론 피처
    E[W] = rho/(mu*(1-rho))  =>  rho/(1-rho) 로 비선형 증폭
    pack_util 0.88 => 7.3, 0.41 => 0.69 (10배 차이)
    """
    df = df.copy()
    eps = 1e-3

    if 'pack_utilization' in df.columns:
        rho = df['pack_utilization'].fillna(0).clip(0, 1 - eps)
        df['pack_queue_wait']      = rho / (1 - rho)
        df['pack_queue_wait_sq']   = df['pack_queue_wait'] ** 2
        df['pack_near_saturation'] = (rho >= 0.8).astype(np.int8)

    if 'robot_utilization' in df.columns:
        rho = df['robot_utilization'].fillna(0).clip(0, 1 - eps)
        df['robot_queue_pressure']  = rho / (1 - rho)
        df['robot_near_saturation'] = (rho >= 0.8).astype(np.int8)

    if 'loading_dock_util' in df.columns:
        rho = df['loading_dock_util'].fillna(0).clip(0, 1 - eps)
        df['loading_queue_wait'] = rho / (1 - rho)

    util_vals = [df[c].fillna(0) for c in ['pack_utilization','robot_utilization','loading_dock_util'] if c in df.columns]
    if util_vals:
        df['system_saturation']   = np.mean(util_vals, axis=0)
        df['max_bottleneck_util'] = np.max(util_vals, axis=0)

    qw_cols = [c for c in ['pack_queue_wait','robot_queue_pressure','loading_queue_wait'] if c in df.columns]
    if qw_cols:
        df['total_queue_pressure'] = df[qw_cols].sum(axis=1)

    if all(c in df.columns for c in ['order_inflow_15m','pack_utilization','pack_station_count']):
        df['little_law_pressure'] = (df['order_inflow_15m'] * df['pack_utilization'].fillna(0)
                                      / (df['pack_station_count'] + 1))

    if all(c in df.columns for c in ['order_inflow_15m','robot_active','pack_station_count']):
        df['arrival_service_imbalance'] = (df['order_inflow_15m']
                                            / (df['robot_active'] + df['pack_station_count'] + 1))

    for q_col in ['pack_queue_wait','total_queue_pressure','system_saturation']:
        if q_col not in df.columns: continue
        lag1 = df.groupby('scenario_id', sort=False)[q_col].shift(1)
        df[f'{q_col}__lag1']      = lag1
        df[f'{q_col}__diff1']     = df[q_col] - lag1
        df[f'{q_col}__worsening'] = (df[f'{q_col}__diff1'] > 0).astype(np.int8)

    if all(c in df.columns for c in ['pack_queue_wait','order_inflow_15m','urgent_order_ratio']):
        df['extreme_risk_score'] = (df['pack_queue_wait']
                                     * df['order_inflow_15m']
                                     * (1 + df['urgent_order_ratio'].fillna(0)))
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
    gkf_te = GroupKFold(n_splits=N_SPLITS)
    scenario_ids = train_df['scenario_id'].values
    layout_te_train = np.full(len(train_df), np.nan)
    for tr_idx, val_idx in gkf_te.split(train_df, y_log, scenario_ids):
        tr_sub = train_df.iloc[tr_idx].copy()
        tr_sub['_y'] = y_log[tr_idx]
        layout_mean      = tr_sub.groupby('layout_id')['_y'].mean()
        layout_type_mean = tr_sub.groupby('layout_type')['_y'].mean()
        global_mean      = float(y_log[tr_idx].mean())
        val_df = train_df.iloc[val_idx]
        encoded = val_df['layout_id'].map(layout_mean).copy()
        missing = encoded.isna()
        if missing.any():
            encoded[missing] = val_df.loc[missing, 'layout_type'].map(layout_type_mean)
        encoded = encoded.fillna(global_mean)
        layout_te_train[val_idx] = encoded.values
    train_df = train_df.copy()
    train_df['layout_te'] = layout_te_train
    full_tr = train_df.copy(); full_tr['_y'] = y_log
    layout_mean_all      = full_tr.groupby('layout_id')['_y'].mean()
    layout_type_mean_all = full_tr.groupby('layout_type')['_y'].mean()
    global_mean_all      = float(y_log.mean())
    test_df = test_df.copy()
    test_enc = test_df['layout_id'].map(layout_mean_all).copy()
    missing_t = test_enc.isna()
    if missing_t.any():
        test_enc[missing_t] = test_df.loc[missing_t, 'layout_type'].map(layout_type_mean_all)
    test_enc = test_enc.fillna(global_mean_all)
    test_df['layout_te'] = test_enc.values
    print(f"  layout_te: cold-start {int(missing_t.sum())}행")
    return train_df, test_df


# =============================================================================
# 4. 피처 엔지니어링 실행
# =============================================================================
print("\n" + "="*60)
print("2. 피처 엔지니어링")
print("="*60)

train = feature_engineering(train);  test = feature_engineering(test)
print("  [Step20] Temporal 피처...")
train = add_temporal_features(train); test = add_temporal_features(test)
print("  [Step20] Lag/Rolling 피처 (22 SEQ_COLS x 6)...")
train = add_lag_rolling_features(train); test = add_lag_rolling_features(test)
print("  [Step20] Onset detection 피처...")
train = add_onset_features(train); test = add_onset_features(test)
print("  [Step20] Threshold/hinge 피처...")
train = add_threshold_features(train); test = add_threshold_features(test)
print("  [NEW] Queuing Theory 피처 (M/M/1)...")
train = add_queuing_theory_features(train); test = add_queuing_theory_features(test)

# 큐이론 피처 시나리오 집계
QUEUE_AGG_COLS = [c for c in ['pack_queue_wait','total_queue_pressure','system_saturation'] if c in train.columns]
if QUEUE_AGG_COLS:
    qa_tr = train.groupby('scenario_id')[QUEUE_AGG_COLS].agg(['mean','max'])
    qa_tr.columns = [f'sc_{c}_{f}' for c, f in qa_tr.columns]
    qa_te = test.groupby('scenario_id')[QUEUE_AGG_COLS].agg(['mean','max'])
    qa_te.columns = [f'sc_{c}_{f}' for c, f in qa_te.columns]
    train = train.merge(qa_tr.reset_index(), on='scenario_id', how='left')
    test  = test.merge(qa_te.reset_index(),  on='scenario_id', how='left')
    print(f"  큐이론 sc집계: {len(qa_tr.columns)}개")

print("  [Step20] sc_* 집계 피처...")
train, test = add_aggregation_features(train, test)
train = add_sc_derived_features(train); test = add_sc_derived_features(test)
print("  [Step20] Trajectory 피처...")
train = add_trajectory_features(train); test = add_trajectory_features(test)

y_log_full = np.log1p(train[TARGET].values)
y_true     = train[TARGET].values

print("  [Step20] Layout 타겟 인코딩 (OOF)...")
train, test = add_layout_target_encoding(train, test, y_log_full)

DROP_COLS = {'ID','layout_id','scenario_id','layout_type',TARGET,'shift_hour'}
feature_cols = [c for c in train.columns if c not in DROP_COLS and c in test.columns and train[c].dtype != object]
feature_cols = list(dict.fromkeys(feature_cols))

X_train = train[feature_cols].astype(np.float32)
X_test  = test[feature_cols].astype(np.float32)
groups  = train['scenario_id'].values

print(f"\n  Step20 피처: ~448개 -> Step29: {len(feature_cols)}개")
print(f"  train: {X_train.shape} | test: {X_test.shape}")

# =============================================================================
# 5. Focal Sample Weight
# =============================================================================
print("\n" + "="*60)
print("3. Focal Sample Weight")
print("="*60)

sample_weights = np.ones(len(y_true), dtype=np.float32)
sample_weights[(y_true >= 30) & (y_true < 50)]  = 2.0
sample_weights[(y_true >= 50) & (y_true < 100)] = 5.0
sample_weights[y_true >= 100]                    = 10.0

for lo, hi, ww in [(0,30,1),(30,50,2),(50,100,5),(100,9999,10)]:
    mask = (y_true >= lo) & (y_true < hi)
    print(f"  {lo:3}~{min(hi,999):3}min: {mask.sum():6d}개 ({mask.mean()*100:.1f}%) w={ww}")

# =============================================================================
# 6. 모델 파라미터
# =============================================================================
LGB_BASE = dict(objective='regression_l1', metric='mae', learning_rate=0.02,
                num_leaves=127, max_depth=-1, min_child_samples=50,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                lambda_l1=0.1, lambda_l2=0.1,
                device_type='cpu', num_threads=N_CPU, verbose=-1)
XGB_BASE = dict(objective='reg:absoluteerror', eval_metric='mae', learning_rate=0.05,
                max_depth=7, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1, tree_method='hist',
                device='cpu', nthread=N_CPU, verbosity=0)
CAT_BASE = dict(loss_function='MAE', eval_metric='MAE', learning_rate=0.05,
                depth=8, l2_leaf_reg=3.0, min_data_in_leaf=50, subsample=0.8,
                task_type='CPU', thread_count=N_CPU, verbose=False)
LGB_CLF  = dict(objective='binary', metric='auc', learning_rate=0.02,
                num_leaves=63, max_depth=-1, min_child_samples=50,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                lambda_l1=0.1, lambda_l2=0.1,
                device_type='cpu', num_threads=N_CPU, verbose=-1)

# =============================================================================
# 7. Stage A: 전체 회귀 (Focal Weight)
# =============================================================================
print("\n" + "="*60)
print("4-A. Stage A: 전체 회귀 (LGB+XGB+Cat x5seeds, Focal Weight)")
print("="*60)

gkf = GroupKFold(n_splits=N_SPLITS)
all_oof_A  = {m: {} for m in ['LightGBM','XGBoost','CatBoost']}
all_test_A = {m: {} for m in ['LightGBM','XGBoost','CatBoost']}
all_scr_A  = {m: {} for m in ['LightGBM','XGBoost','CatBoost']}
rc = 0; total_runs = len(SEEDS)*3; t_A = time.time()

# LightGBM
print("\n[A] LightGBM x5")
for seed in SEEDS:
    t0 = time.time()
    params = {**LGB_BASE, 'seed': seed}
    oof_p = np.zeros(len(X_train)); test_p = np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_log_full, groups)):
        dtrain = lgb.Dataset(X_train.iloc[tr_idx], label=y_log_full[tr_idx],
                             weight=sample_weights[tr_idx], free_raw_data=True)
        dvalid = lgb.Dataset(X_train.iloc[val_idx], label=y_log_full[val_idx],
                             reference=dtrain, free_raw_data=True)
        m = lgb.train(params, dtrain, num_boost_round=LGB_MAX_ROUNDS, valid_sets=[dvalid],
                      callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False),
                                 lgb.log_evaluation(9999)])
        oof_p[val_idx] = np.clip(np.expm1(m.predict(X_train.iloc[val_idx], num_iteration=m.best_iteration)), 0, None)
        test_p        += np.clip(np.expm1(m.predict(X_test, num_iteration=m.best_iteration)), 0, None) / N_SPLITS
    sc = mean_absolute_error(y_true, oof_p)
    mae_ext = mean_absolute_error(y_true[y_true>=50], oof_p[y_true>=50])
    all_oof_A['LightGBM'][seed]=oof_p; all_test_A['LightGBM'][seed]=test_p; all_scr_A['LightGBM'][seed]=sc
    rc+=1; print(f"  s={seed} OOF={sc:.4f} ext={mae_ext:.2f} {(time.time()-t0)/60:.1f}min [{rc}/{total_runs}]")
s=list(all_scr_A['LightGBM'].values()); print(f"  LGB avg={np.mean(s):.4f}+-{np.std(s):.4f}")

# XGBoost
print("\n[A] XGBoost x5")
for seed in SEEDS:
    t0 = time.time()
    params = {**XGB_BASE, 'seed': seed}
    oof_p = np.zeros(len(X_train)); test_p = np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_log_full, groups)):
        dtrain = xgb.DMatrix(X_train.iloc[tr_idx], label=y_log_full[tr_idx], weight=sample_weights[tr_idx])
        dvalid = xgb.DMatrix(X_train.iloc[val_idx], label=y_log_full[val_idx])
        m = xgb.train(params, dtrain, num_boost_round=MAX_ROUNDS,
                      evals=[(dvalid,'val')], early_stopping_rounds=EARLY_STOP, verbose_eval=False)
        oof_p[val_idx] = np.clip(np.expm1(m.predict(dvalid, iteration_range=(0, m.best_iteration))), 0, None)
        test_p        += np.clip(np.expm1(m.predict(xgb.DMatrix(X_test), iteration_range=(0, m.best_iteration))), 0, None) / N_SPLITS
    sc = mean_absolute_error(y_true, oof_p)
    mae_ext = mean_absolute_error(y_true[y_true>=50], oof_p[y_true>=50])
    all_oof_A['XGBoost'][seed]=oof_p; all_test_A['XGBoost'][seed]=test_p; all_scr_A['XGBoost'][seed]=sc
    rc+=1; print(f"  s={seed} OOF={sc:.4f} ext={mae_ext:.2f} {(time.time()-t0)/60:.1f}min [{rc}/{total_runs}]")
s=list(all_scr_A['XGBoost'].values()); print(f"  XGB avg={np.mean(s):.4f}+-{np.std(s):.4f}")

# CatBoost
print("\n[A] CatBoost x5")
for seed in SEEDS:
    t0 = time.time()
    oof_p = np.zeros(len(X_train)); test_p = np.zeros(len(X_test))
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_log_full, groups)):
        m = CatBoostRegressor(iterations=MAX_ROUNDS, early_stopping_rounds=EARLY_STOP,
                              random_seed=seed, **CAT_BASE)
        m.fit(X_train.iloc[tr_idx], y_log_full[tr_idx], sample_weight=sample_weights[tr_idx],
              eval_set=(X_train.iloc[val_idx], y_log_full[val_idx]), use_best_model=True, verbose=False)
        oof_p[val_idx] = np.clip(np.expm1(m.predict(X_train.iloc[val_idx])), 0, None)
        test_p        += np.clip(np.expm1(m.predict(X_test)), 0, None) / N_SPLITS
    sc = mean_absolute_error(y_true, oof_p)
    mae_ext = mean_absolute_error(y_true[y_true>=50], oof_p[y_true>=50])
    all_oof_A['CatBoost'][seed]=oof_p; all_test_A['CatBoost'][seed]=test_p; all_scr_A['CatBoost'][seed]=sc
    rc+=1; print(f"  s={seed} OOF={sc:.4f} ext={mae_ext:.2f} {(time.time()-t0)/60:.1f}min [{rc}/{total_runs}]")
s=list(all_scr_A['CatBoost'].values()); print(f"  Cat avg={np.mean(s):.4f}+-{np.std(s):.4f}")

print(f"\nStage A 완료: {(time.time()-t_A)/60:.1f}분")
oof_A  = np.mean([all_oof_A[m][s]  for m in ['LightGBM','XGBoost','CatBoost'] for s in SEEDS], axis=0)
test_A = np.mean([all_test_A[m][s] for m in ['LightGBM','XGBoost','CatBoost'] for s in SEEDS], axis=0)
mae_A  = mean_absolute_error(y_true, oof_A)
mae_A_ext = mean_absolute_error(y_true[y_true>=50], oof_A[y_true>=50])
print(f"Stage A OOF={mae_A:.4f} | ext(>=50)={mae_A_ext:.2f}")

# =============================================================================
# 8. Stage B: 이진 분류기
# =============================================================================
print("\n" + "="*60)
print("4-B. Stage B: 이진 분류기 (>=50min LGB)")
print("="*60)

y_bin = (y_true >= EXTREME_THRESHOLD).astype(np.float32)
neg_pos = (1-y_bin).sum() / y_bin.sum()
clf_params = {**LGB_CLF, 'seed': 42, 'scale_pos_weight': neg_pos * 0.5}
clf_oof  = np.zeros(len(X_train))
clf_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_bin, groups)):
    dtrain = lgb.Dataset(X_train.iloc[tr_idx], label=y_bin[tr_idx], free_raw_data=True)
    dvalid = lgb.Dataset(X_train.iloc[val_idx], label=y_bin[val_idx], reference=dtrain, free_raw_data=True)
    m = lgb.train(clf_params, dtrain, num_boost_round=LGB_MAX_ROUNDS, valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False), lgb.log_evaluation(9999)])
    clf_oof[val_idx] = m.predict(X_train.iloc[val_idx], num_iteration=m.best_iteration)
    clf_test        += m.predict(X_test, num_iteration=m.best_iteration) / N_SPLITS
    print(f"  Fold {fold+1}: AUC={roc_auc_score(y_bin[val_idx], clf_oof[val_idx]):.4f}")

auc = roc_auc_score(y_bin, clf_oof)
best_f1, best_thr = 0, 0.5
for thr in np.arange(0.1, 0.9, 0.05):
    f1 = f1_score(y_bin, (clf_oof>=thr).astype(int))
    if f1 > best_f1: best_f1, best_thr = f1, thr
print(f"  전체 AUC={auc:.4f} | 최적임계={best_thr:.2f} (F1={best_f1:.4f})")

# =============================================================================
# 9. Stage C: 극단값 전용 회귀
# =============================================================================
print("\n" + "="*60)
print("4-C. Stage C: 극단값 전용 회귀 (>=50min only, LGB)")
print("="*60)

ext_mask = y_true >= EXTREME_THRESHOLD
X_ext = X_train[ext_mask].reset_index(drop=True)
y_ext = y_true[ext_mask]; y_ext_log = np.log1p(y_ext)
g_ext = groups[ext_mask]
print(f"  극단 케이스: {ext_mask.sum()}개 ({ext_mask.mean()*100:.1f}%)")

oof_C  = np.zeros(len(X_train))
test_C = np.zeros(len(X_test))
lgb_c_params = {**LGB_BASE, 'seed': 42, 'learning_rate': 0.01, 'num_leaves': 63, 'min_child_samples': 20}

for fold, (tr_idx, val_idx) in enumerate(GroupKFold(n_splits=5).split(X_ext, y_ext_log, g_ext)):
    dtrain = lgb.Dataset(X_ext.iloc[tr_idx], label=y_ext_log[tr_idx], free_raw_data=True)
    dvalid = lgb.Dataset(X_ext.iloc[val_idx], label=y_ext_log[val_idx], reference=dtrain, free_raw_data=True)
    m = lgb.train(lgb_c_params, dtrain, num_boost_round=LGB_MAX_ROUNDS, valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(LGB_EARLY_STOP*2, verbose=False), lgb.log_evaluation(9999)])
    val_pred = np.clip(np.expm1(m.predict(X_ext.iloc[val_idx], num_iteration=m.best_iteration)), 0, None)
    print(f"  Fold {fold+1}: MAE(ext)={mean_absolute_error(y_ext[val_idx], val_pred):.2f}")
    ext_idx = np.where(ext_mask)[0]
    oof_C[ext_idx[val_idx]] = val_pred
    test_C += np.clip(np.expm1(m.predict(X_test, num_iteration=m.best_iteration)), 0, None) / 5

mae_C_ext = mean_absolute_error(y_true[ext_mask], oof_C[ext_mask])
print(f"  Stage C OOF MAE (ext): {mae_C_ext:.2f}  (Stage A: {mae_A_ext:.2f})")

# =============================================================================
# 10. 최종 앙상블
# =============================================================================
print("\n" + "="*60)
print("5. Two-Stage Blend (alpha 탐색)")
print("="*60)
print(f"  {'alpha':>5} | {'전체MAE':>10} | {'ext(>=50)':>10} | {'normal':>10}")

best_mae, best_alpha, best_oof = 999, 0.0, oof_A.copy()
normal_mask = ~ext_mask

for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    blend = np.clip((1 - alpha*clf_oof)*oof_A + alpha*clf_oof*oof_C, 0, None)
    mae_all = mean_absolute_error(y_true, blend)
    mae_ext2 = mean_absolute_error(y_true[ext_mask], blend[ext_mask])
    mae_nrm  = mean_absolute_error(y_true[normal_mask], blend[normal_mask])
    print(f"  {alpha:.1f}   | {mae_all:.4f}     | {mae_ext2:.2f}       | {mae_nrm:.4f}")
    if mae_all < best_mae:
        best_mae, best_alpha, best_oof = mae_all, alpha, blend.copy()

test_final = np.clip((1-best_alpha*clf_test)*test_A + best_alpha*clf_test*test_C, 0, None)
print(f"\n  최적 alpha={best_alpha:.1f} => OOF={best_mae:.4f}")

# =============================================================================
# 11. 저장
# =============================================================================
print("\n" + "="*60)
print("6. 결과 저장")
print("="*60)

oof_df = pd.DataFrame({'ID': train['ID'], 'scenario_id': train['scenario_id'],
                        'y_true': y_true, 'oof_A': oof_A, 'oof_C': oof_C,
                        'clf_prob': clf_oof, 'oof_blend': best_oof})
oof_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_step29_extreme.csv'), index=False)

sample[TARGET] = test_final
sample.to_csv(os.path.join(OUTPUT_DIR, 'submission_step29_extreme.csv'), index=False)

sample_A = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
sample_A[TARGET] = np.clip(test_A, 0, None)
sample_A.to_csv(os.path.join(OUTPUT_DIR, 'submission_step29_stageA.csv'), index=False)

print("  oof_step29_extreme.csv")
print("  submission_step29_extreme.csv  <- 메인 제출")
print("  submission_step29_stageA.csv   <- Stage A 단독")

# =============================================================================
# 12. 최종 요약
# =============================================================================
print("\n" + "="*60)
print("7. 구간별 MAE 비교")
print("="*60)
for lo, hi in [(0,30),(30,50),(50,100),(100,9999)]:
    mask = (y_true >= lo) & (y_true < hi)
    if mask.sum() == 0: continue
    a = mean_absolute_error(y_true[mask], oof_A[mask])
    b = mean_absolute_error(y_true[mask], best_oof[mask])
    print(f"  {lo:3}~{min(hi,999):3}min ({mask.sum():6d}개): StageA={a:.2f} Blend={b:.2f} delta={b-a:+.2f}")

print()
print(f"  Step20 OOF: ~9.02")
print(f"  Step29 StageA:  {mae_A:.4f}")
print(f"  Step29 Blend:   {best_mae:.4f}")
print(f"  분류기 AUC:     {auc:.4f}")
print(f"  예측 분포: min={test_final.min():.1f} mean={test_final.mean():.1f} max={test_final.max():.1f}")
