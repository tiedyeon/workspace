# =============================================================================
# Step 29 Fast: Queuing Theory + Two-Stage (Stage A = step20 재사용)
# =============================================================================
# Step20 test 예측을 Stage A로 재사용 → 학습 시간 2.5시간→30분
#
# 새로 학습하는 것:
#   1. OOF LGB (seed=42, 1회) - alpha 교정용 + 큐이론 피처 효과 확인
#   2. LGB 이진 분류기 (50분+ 여부)
#   3. LGB 극단값 전용 회귀 (Stage C, >=50분 케이스만)
#
# Stage A (step20 재사용):
#   - OOF: fast LGB OOF (alpha 교정용)
#   - Test: submission_step20_temporal_stack.csv 로드
#
# 규정: test.csv 학습 사용 전면 금지
#
# 실행:
#   source ~/dacon_venv/bin/activate
#   caffeinate -i nohup python -u -B step29_fast.py > step29_fast_output.log 2>&1 &
#   tail -f step29_fast_output.log
# =============================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score, f1_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

N_CPU      = os.cpu_count()
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = DATA_DIR
TARGET     = 'avg_delay_minutes_next_30m'
N_SPLITS   = 5
LGB_MAX_ROUNDS = 5000
LGB_EARLY_STOP = 200
EXTREME_THRESHOLD = 50.0
LAYOUT_TYPE_MAP = {'grid': 0, 'hybrid': 1, 'narrow': 2, 'hub_spoke': 3}

print(f"CPU: {N_CPU}코어 | 극단 임계: {EXTREME_THRESHOLD}분")
print("Step20 test 예측 재사용 → Stage A 재학습 생략")

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

# Step20 test 예측 로드 (Stage A 기반)
step20_sub = pd.read_csv(os.path.join(DATA_DIR, 'submission_step20_temporal_stack.csv'))
test_A = step20_sub[TARGET].values
print(f"train: {train.shape} | test: {test.shape}")
print(f"Step20 test 예측 로드: {len(test_A)}개 (mean={test_A.mean():.2f})")
print(f"극단값(>={EXTREME_THRESHOLD}분): {(train[TARGET]>=EXTREME_THRESHOLD).sum()} ({(train[TARGET]>=EXTREME_THRESHOLD).mean()*100:.1f}%)")

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
# 3. 피처 엔지니어링 함수
# =============================================================================
def feature_engineering(df):
    df = df.copy()
    for col in MEANINGFUL_NAN_COLS:
        if col in df.columns: df[f'nan_{col}'] = df[col].isna().astype(np.int8)
    for col in ZERO_FILL_COLS:
        if col in df.columns: df[col] = df[col].fillna(0)
    for col in ['congestion_score','blocked_path_15m','near_collision_15m',
                'charge_queue_length','avg_charge_wait','fault_count_15m',
                'replenishment_overlap','task_reassign_15m']:
        if col in df.columns: df[f'flag_{col}'] = (df[col] > 0).astype(np.int8)
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
        if col in df.columns: df[f'null_{col}'] = df[col].isna().astype(np.int8)
    for col in SKEWED_COLS:
        if col in df.columns: df[f'log_{col}'] = np.log1p(df[col].fillna(0))
    df['battery_crisis_index'] = (df['low_battery_ratio'] * df['charge_queue_length']
                                   + df['robot_charging'] / (df['battery_mean'] + 1e-5))
    df['congestion_level'] = pd.cut(df['congestion_score'].fillna(0),
                                     bins=[-1,0,20,100], labels=[0,1,2]).astype(int)
    df['congestion_compound'] = (df['congestion_score'].fillna(0) * df['max_zone_density'].fillna(0)
                                  + df['blocked_path_15m'].fillna(0)*2 + df['near_collision_15m'].fillna(0)*3)
    df['robot_saturation']   = 1 - (df['robot_idle'] / (df['robot_total'] + 1))
    df['operation_pressure'] = df['order_inflow_15m'] * df['low_battery_ratio'] / (df['robot_active'] + 1)
    df['triple_crisis']      = df['low_battery_ratio'] * df['congestion_score'].fillna(0) * df['order_inflow_15m']
    df['crisis_score']       = df['low_battery_ratio'] * df['congestion_score'].fillna(0)
    df['order_robot_stress'] = df['order_inflow_15m'] / (df['robot_active'] + 1) * df['low_battery_ratio']
    df['bottleneck_score']   = df['charge_queue_length'] * df['congestion_score'].fillna(0)
    df['complex_urgent_order']= df['sku_concentration'] * df['urgent_order_ratio']
    if 'maintenance_schedule_score' in df.columns:
        df['maintenance_battery_risk'] = (1 - df['maintenance_schedule_score'].fillna(0.5)) * df['low_battery_ratio']
    df['layout_congestion']  = df['layout_type_enc'] * df['congestion_score'].fillna(0)
    df['layout_battery']     = df['layout_type_enc'] * df['low_battery_ratio']
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
        if col not in df.columns: continue
        lag1 = grp[col].shift(1); lag2 = grp[col].shift(2)
        df[f'{col}__lag1'] = lag1; df[f'{col}__lag2'] = lag2
        df[f'{col}__diff1'] = df[col] - lag1
        lag1_grp = lag1.groupby(grp_key)
        roll_mean = lag1_grp.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        roll_max  = lag1_grp.rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
        df[f'{col}__rollmean3'] = roll_mean; df[f'{col}__rollmax3'] = roll_max
        df[f'{col}__dev_rollmean3'] = df[col] - roll_mean
    return df

def add_onset_features(df):
    df = df.copy(); grp_key = df['scenario_id']
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
    add_onset('robot_charging', 'charging'); add_onset('charge_queue_length', 'queue')
    return df

def add_threshold_features(df):
    df = df.copy()
    if {'robot_charging','charge_queue_length','charger_count'}.issubset(df.columns):
        df['charge_pressure2'] = ((df['robot_charging'] + df['charge_queue_length'])
                                   / (df['charger_count'] + 1e-5))
    if 'battery_mean'      in df.columns: df['battery_mean_below_44']     = np.clip(44.0 - df['battery_mean'].fillna(44), 0, None)
    if 'charge_pressure2'  in df.columns: df['charge_pressure_above_136'] = np.clip(df['charge_pressure2'] - 1.36, 0, None)
    if 'pack_utilization'  in df.columns: df['pack_utilization_sq']        = df['pack_utilization'].fillna(0) ** 2
    if 'loading_dock_util' in df.columns: df['loading_dock_util_sq']       = df['loading_dock_util'].fillna(0) ** 2
    if 'staging_area_util' in df.columns: df['staging_area_util_sq']       = df['staging_area_util'].fillna(0) ** 2
    return df

def add_queuing_theory_features(df):
    """[NEW] M/M/1 대기이론: rho/(1-rho) → 극단값 비선형 증폭"""
    df = df.copy(); eps = 1e-3
    if 'pack_utilization' in df.columns:
        rho = df['pack_utilization'].fillna(0).clip(0, 1-eps)
        df['pack_queue_wait']      = rho / (1 - rho)
        df['pack_queue_wait_sq']   = df['pack_queue_wait'] ** 2
        df['pack_near_saturation'] = (rho >= 0.8).astype(np.int8)
    if 'robot_utilization' in df.columns:
        rho = df['robot_utilization'].fillna(0).clip(0, 1-eps)
        df['robot_queue_pressure']  = rho / (1 - rho)
        df['robot_near_saturation'] = (rho >= 0.8).astype(np.int8)
    if 'loading_dock_util' in df.columns:
        rho = df['loading_dock_util'].fillna(0).clip(0, 1-eps)
        df['loading_queue_wait'] = rho / (1 - rho)
    util_vals = [df[c].fillna(0) for c in ['pack_utilization','robot_utilization','loading_dock_util'] if c in df.columns]
    if util_vals:
        df['system_saturation']   = np.mean(util_vals, axis=0)
        df['max_bottleneck_util'] = np.max(util_vals, axis=0)
    qw = [c for c in ['pack_queue_wait','robot_queue_pressure','loading_queue_wait'] if c in df.columns]
    if qw: df['total_queue_pressure'] = df[qw].sum(axis=1)
    if all(c in df.columns for c in ['order_inflow_15m','pack_utilization','pack_station_count']):
        df['little_law_pressure'] = (df['order_inflow_15m'] * df['pack_utilization'].fillna(0)
                                      / (df['pack_station_count'] + 1))
    if all(c in df.columns for c in ['order_inflow_15m','robot_active','pack_station_count']):
        df['arrival_service_imbalance'] = df['order_inflow_15m'] / (df['robot_active'] + df['pack_station_count'] + 1)
    for q_col in ['pack_queue_wait','total_queue_pressure','system_saturation']:
        if q_col not in df.columns: continue
        lag1 = df.groupby('scenario_id', sort=False)[q_col].shift(1)
        df[f'{q_col}__lag1']      = lag1
        df[f'{q_col}__diff1']     = df[q_col] - lag1
        df[f'{q_col}__worsening'] = (df[f'{q_col}__diff1'] > 0).astype(np.int8)
    if all(c in df.columns for c in ['pack_queue_wait','order_inflow_15m','urgent_order_ratio']):
        df['extreme_risk_score'] = (df['pack_queue_wait'] * df['order_inflow_15m']
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
    train_df = train_df.merge(layout_agg[['layout_id']+lfc], on='layout_id', how='left', suffixes=('','_dup'))
    train_df.drop(columns=[c for c in train_df.columns if c.endswith('_dup')], inplace=True)
    test_df  = test_df.merge(layout_agg[['layout_id']+lfc], on='layout_id', how='left')
    return train_df, test_df

def add_sc_derived_features(df):
    df = df.copy()
    for col in ZSCORE_COLS:
        m, s = f'sc_{col}_mean', f'sc_{col}_std'
        if col in df.columns and m in df.columns and s in df.columns:
            df[f'z_{col}'] = (df[col] - df[m]) / (df[s] + 1e-5)
            df[f'z_{col}_abs'] = df[f'z_{col}'].abs()
    for col in RANK_COLS:
        if col in df.columns:
            df[f'prank_{col}'] = df.groupby('scenario_id')[col].rank(pct=True, na_option='keep')
    for col in RATIO_COLS:
        if col in df.columns:
            if f'sc_{col}_max'  in df.columns:
                df[f'ratio_to_max_{col}']  = df[col] / (df[f'sc_{col}_max'] + 1e-5)
                df[f'gap_to_max_{col}']    = df[f'sc_{col}_max'] - df[col]
            if f'sc_{col}_mean' in df.columns:
                df[f'ratio_to_mean_{col}'] = df[col] / (df[f'sc_{col}_mean'] + 1e-5)
    return df

def add_trajectory_features(df):
    df = df.copy()
    for col in TRAJECTORY_COLS:
        ec, sc = f'exp_mean_{col}', f'sc_{col}_mean'
        if ec in df.columns and sc in df.columns:
            df[f'traj_{col}']     = df[ec] / (df[sc] + 1e-5)
            df[f'traj_dev_{col}'] = (df[ec] - df[sc]).abs()
    return df

def add_layout_target_encoding(train_df, test_df, y_log):
    gkf = GroupKFold(n_splits=N_SPLITS)
    scenario_ids = train_df['scenario_id'].values
    layout_te_train = np.full(len(train_df), np.nan)
    for tr_idx, val_idx in gkf.split(train_df, y_log, scenario_ids):
        tr_sub = train_df.iloc[tr_idx].copy(); tr_sub['_y'] = y_log[tr_idx]
        layout_mean = tr_sub.groupby('layout_id')['_y'].mean()
        lt_mean     = tr_sub.groupby('layout_type')['_y'].mean()
        g_mean      = float(y_log[tr_idx].mean())
        val_df = train_df.iloc[val_idx]
        enc = val_df['layout_id'].map(layout_mean).copy()
        miss = enc.isna()
        if miss.any(): enc[miss] = val_df.loc[miss,'layout_type'].map(lt_mean)
        layout_te_train[val_idx] = enc.fillna(g_mean).values
    train_df = train_df.copy(); train_df['layout_te'] = layout_te_train
    full = train_df.copy(); full['_y'] = y_log
    lm_all = full.groupby('layout_id')['_y'].mean()
    ltm_all = full.groupby('layout_type')['_y'].mean()
    gm_all  = float(y_log.mean())
    test_df = test_df.copy()
    te = test_df['layout_id'].map(lm_all).copy()
    mt = te.isna()
    if mt.any(): te[mt] = test_df.loc[mt,'layout_type'].map(ltm_all)
    test_df['layout_te'] = te.fillna(gm_all).values
    print(f"  layout_te: cold-start {int(mt.sum())}행")
    return train_df, test_df

# =============================================================================
# 4. 피처 엔지니어링 실행
# =============================================================================
print("\n" + "="*60)
print("2. 피처 엔지니어링")
print("="*60)

train = feature_engineering(train);  test = feature_engineering(test)
print("  Temporal 피처..."); train = add_temporal_features(train); test = add_temporal_features(test)
print("  Lag/Rolling 피처..."); train = add_lag_rolling_features(train); test = add_lag_rolling_features(test)
print("  Onset 피처..."); train = add_onset_features(train); test = add_onset_features(test)
print("  Threshold 피처..."); train = add_threshold_features(train); test = add_threshold_features(test)
print("  [NEW] Queuing Theory 피처..."); train = add_queuing_theory_features(train); test = add_queuing_theory_features(test)

QUEUE_AGG_COLS = [c for c in ['pack_queue_wait','total_queue_pressure','system_saturation'] if c in train.columns]
if QUEUE_AGG_COLS:
    qa_tr = train.groupby('scenario_id')[QUEUE_AGG_COLS].agg(['mean','max'])
    qa_tr.columns = [f'sc_{c}_{f}' for c,f in qa_tr.columns]
    qa_te = test.groupby('scenario_id')[QUEUE_AGG_COLS].agg(['mean','max'])
    qa_te.columns = [f'sc_{c}_{f}' for c,f in qa_te.columns]
    train = train.merge(qa_tr.reset_index(), on='scenario_id', how='left')
    test  = test.merge(qa_te.reset_index(),  on='scenario_id', how='left')
    print(f"  큐이론 sc집계: {len(qa_tr.columns)}개")

print("  sc_* 집계 피처..."); train, test = add_aggregation_features(train, test)
train = add_sc_derived_features(train); test = add_sc_derived_features(test)
print("  Trajectory 피처..."); train = add_trajectory_features(train); test = add_trajectory_features(test)

y_log_full = np.log1p(train[TARGET].values)
y_true     = train[TARGET].values
print("  Layout 타겟 인코딩..."); train, test = add_layout_target_encoding(train, test, y_log_full)

DROP_COLS = {'ID','layout_id','scenario_id','layout_type',TARGET,'shift_hour'}
feature_cols = [c for c in train.columns if c not in DROP_COLS and c in test.columns and train[c].dtype != object]
feature_cols = list(dict.fromkeys(feature_cols))

X_train = train[feature_cols].astype(np.float32)
X_test  = test[feature_cols].astype(np.float32)
groups  = train['scenario_id'].values
print(f"\n  피처 수: {len(feature_cols)}개 | train: {X_train.shape} | test: {X_test.shape}")

# =============================================================================
# 5. Focal Sample Weight
# =============================================================================
sample_weights = np.ones(len(y_true), dtype=np.float32)
sample_weights[(y_true>=30)&(y_true<50)]  = 2.0
sample_weights[(y_true>=50)&(y_true<100)] = 5.0
sample_weights[y_true>=100]               = 10.0
print(f"\n  Focal Weight: 0~30(×1) / 30~50(×2) / 50~100(×5) / 100+(×10)")

# =============================================================================
# 6. 모델 파라미터
# =============================================================================
LGB_BASE = dict(objective='regression_l1', metric='mae', learning_rate=0.02,
                num_leaves=127, max_depth=-1, min_child_samples=50,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                lambda_l1=0.1, lambda_l2=0.1,
                device_type='cpu', num_threads=N_CPU, verbose=-1)
LGB_CLF  = dict(objective='binary', metric='auc', learning_rate=0.02,
                num_leaves=63, max_depth=-1, min_child_samples=50,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                lambda_l1=0.1, lambda_l2=0.1,
                device_type='cpu', num_threads=N_CPU, verbose=-1)

gkf = GroupKFold(n_splits=N_SPLITS)

# =============================================================================
# 7. OOF LGB (alpha 교정용, seed=42 단일)
# =============================================================================
print("\n" + "="*60)
print("3. OOF LGB (alpha 교정 + 큐이론 효과 확인, seed=42)")
print("="*60)
t0 = time.time()
params_oof = {**LGB_BASE, 'seed': 42}
oof_A = np.zeros(len(X_train))
test_A_fast = np.zeros(len(X_test))  # 빠른 LGB test 예측 (비교용)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_log_full, groups)):
    dtrain = lgb.Dataset(X_train.iloc[tr_idx], label=y_log_full[tr_idx],
                         weight=sample_weights[tr_idx], free_raw_data=True)
    dvalid = lgb.Dataset(X_train.iloc[val_idx], label=y_log_full[val_idx],
                         reference=dtrain, free_raw_data=True)
    m = lgb.train(params_oof, dtrain, num_boost_round=LGB_MAX_ROUNDS, valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False),
                             lgb.log_evaluation(9999)])
    oof_A[val_idx] = np.clip(np.expm1(m.predict(X_train.iloc[val_idx], num_iteration=m.best_iteration)), 0, None)
    test_A_fast   += np.clip(np.expm1(m.predict(X_test, num_iteration=m.best_iteration)), 0, None) / N_SPLITS
    mae_fold = mean_absolute_error(y_true[val_idx], oof_A[val_idx])
    mae_ext  = mean_absolute_error(y_true[val_idx][y_true[val_idx]>=50], oof_A[val_idx][y_true[val_idx]>=50]) if (y_true[val_idx]>=50).sum()>0 else 0
    print(f"  Fold {fold+1}: MAE={mae_fold:.4f} | ext(>=50)={mae_ext:.2f}")

mae_oof_A = mean_absolute_error(y_true, oof_A)
mae_ext_A = mean_absolute_error(y_true[y_true>=50], oof_A[y_true>=50])
print(f"\n  OOF LGB MAE: {mae_oof_A:.4f} | ext(>=50): {mae_ext_A:.2f}")
print(f"  소요: {(time.time()-t0)/60:.1f}분")

# =============================================================================
# 8. 이진 분류기 (50분+ 여부)
# =============================================================================
print("\n" + "="*60)
print("4. 이진 분류기 (>=50분 여부)")
print("="*60)
t0 = time.time()
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
tp = ((clf_oof>=best_thr)&(y_bin==1)).sum()
fn = ((clf_oof< best_thr)&(y_bin==1)).sum()
print(f"\n  전체 AUC={auc:.4f} | 임계={best_thr:.2f} F1={best_f1:.4f}")
print(f"  극단값 탐지: {tp}/{int(y_bin.sum())} (Recall={tp/(tp+fn+1e-8):.3f})")
print(f"  소요: {(time.time()-t0)/60:.1f}분")

# =============================================================================
# 9. Stage C: 극단값 전용 회귀 (>=50분만)
# =============================================================================
print("\n" + "="*60)
print("5. Stage C: 극단값 전용 회귀 (>=50분, LGB)")
print("="*60)
t0 = time.time()
ext_mask = y_true >= EXTREME_THRESHOLD
X_ext = X_train[ext_mask].reset_index(drop=True)
y_ext = y_true[ext_mask]; y_ext_log = np.log1p(y_ext)
g_ext = groups[ext_mask]
print(f"  극단 케이스: {ext_mask.sum()}개 ({ext_mask.mean()*100:.1f}%)")

oof_C  = np.zeros(len(X_train))
test_C = np.zeros(len(X_test))
lgb_c = {**LGB_BASE, 'seed': 42, 'learning_rate': 0.01, 'num_leaves': 63, 'min_child_samples': 20}

for fold, (tr_idx, val_idx) in enumerate(GroupKFold(n_splits=5).split(X_ext, y_ext_log, g_ext)):
    dtrain = lgb.Dataset(X_ext.iloc[tr_idx], label=y_ext_log[tr_idx], free_raw_data=True)
    dvalid = lgb.Dataset(X_ext.iloc[val_idx], label=y_ext_log[val_idx], reference=dtrain, free_raw_data=True)
    m = lgb.train(lgb_c, dtrain, num_boost_round=LGB_MAX_ROUNDS, valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(LGB_EARLY_STOP*2, verbose=False), lgb.log_evaluation(9999)])
    val_pred = np.clip(np.expm1(m.predict(X_ext.iloc[val_idx], num_iteration=m.best_iteration)), 0, None)
    print(f"  Fold {fold+1}: MAE(ext)={mean_absolute_error(y_ext[val_idx], val_pred):.2f}")
    ext_idx = np.where(ext_mask)[0]
    oof_C[ext_idx[val_idx]] = val_pred
    test_C += np.clip(np.expm1(m.predict(X_test, num_iteration=m.best_iteration)), 0, None) / 5

mae_C = mean_absolute_error(y_true[ext_mask], oof_C[ext_mask])
print(f"\n  Stage C OOF MAE(ext): {mae_C:.2f} vs OOF LGB: {mae_ext_A:.2f}")
print(f"  소요: {(time.time()-t0)/60:.1f}분")

# =============================================================================
# 10. 최종 앙상블
# =============================================================================
print("\n" + "="*60)
print("6. Two-Stage Blend (alpha 탐색)")
print("="*60)
# Step20 test 예측이 Stage A 기반
# OOF는 fast LGB로 alpha 교정

ext_mask_bool = ext_mask
normal_mask   = ~ext_mask_bool
print(f"  {'alpha':>5} | {'전체MAE':>10} | {'ext>=50':>9} | {'normal':>9}")

best_mae, best_alpha, best_oof = 999, 0.0, oof_A.copy()
for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    blend = np.clip((1 - alpha*clf_oof)*oof_A + alpha*clf_oof*oof_C, 0, None)
    mae_all = mean_absolute_error(y_true, blend)
    mae_ext = mean_absolute_error(y_true[ext_mask_bool], blend[ext_mask_bool])
    mae_nrm = mean_absolute_error(y_true[normal_mask],   blend[normal_mask])
    print(f"  {alpha:.1f}   | {mae_all:.4f}     | {mae_ext:.2f}    | {mae_nrm:.4f}")
    if mae_all < best_mae:
        best_mae, best_alpha, best_oof = mae_all, alpha, blend.copy()

print(f"\n  최적 alpha={best_alpha:.1f} → OOF MAE={best_mae:.4f}")

# Test 최종 예측: Step20 기반 Stage A에 블렌드
test_final = np.clip((1-best_alpha*clf_test)*test_A + best_alpha*clf_test*test_C, 0, None)

# =============================================================================
# 11. 저장
# =============================================================================
print("\n" + "="*60)
print("7. 저장")
print("="*60)

oof_df = pd.DataFrame({'ID': train['ID'], 'scenario_id': train['scenario_id'],
                        'y_true': y_true, 'oof_A_fast': oof_A, 'oof_C': oof_C,
                        'clf_prob': clf_oof, 'oof_blend': best_oof})
oof_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_step29_fast.csv'), index=False)

sub = sample.copy(); sub[TARGET] = test_final
sub.to_csv(os.path.join(OUTPUT_DIR, 'submission_step29_fast.csv'), index=False)

# 빠른 LGB 단독도 저장 (비교용)
sub_lgb = sample.copy(); sub_lgb[TARGET] = np.clip(test_A_fast, 0, None)
sub_lgb.to_csv(os.path.join(OUTPUT_DIR, 'submission_step29_lgb_only.csv'), index=False)

print("  oof_step29_fast.csv")
print("  submission_step29_fast.csv      <- 메인 제출 (Step20 + Two-Stage)")
print("  submission_step29_lgb_only.csv  <- Fast LGB 단독 (큐이론 효과 확인)")

# =============================================================================
# 12. 최종 요약
# =============================================================================
print("\n" + "="*60)
print("8. 최종 요약")
print("="*60)
for lo, hi in [(0,30),(30,50),(50,100),(100,9999)]:
    mask = (y_true>=lo)&(y_true<hi)
    if mask.sum()==0: continue
    a = mean_absolute_error(y_true[mask], oof_A[mask])
    b = mean_absolute_error(y_true[mask], best_oof[mask])
    print(f"  {lo:3}~{min(hi,999):3}min ({mask.sum():6d}개): OOF_LGB={a:.2f} → Blend={b:.2f} ({b-a:+.2f})")

print()
print(f"  Step20 Public: 10.0606")
print(f"  OOF LGB(fast): {mae_oof_A:.4f} | ext: {mae_ext_A:.2f}")
print(f"  Blend OOF:     {best_mae:.4f}")
print(f"  분류기 AUC:    {auc:.4f}")
print(f"\n  제출: submission_step29_fast.csv")
