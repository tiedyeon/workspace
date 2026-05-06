# =============================================================================
# Step 30: Prediction-based Lag Stacking
# =============================================================================
# 핵심 아이디어:
#   타겟 자기상관 lag1 = 0.8671, sc_target_mean 상관 = 0.7964
#   test에는 타겟이 없어 직접 사용 불가
#   → Stage1 LGB OOF 예측을 "가짜 타겟 lag"으로 활용
#
# 추가 피처 (~9개):
#   sc_pred_mean  : 시나리오 내 예측 평균 (sc_target_mean ≈ 0.7964 근사)
#   sc_pred_std   : 시나리오 내 예측 편차
#   sc_pred_max   : 시나리오 내 예측 최대값
#   sc_pred_min   : 시나리오 내 예측 최소값
#   pred_lag1     : 이전 슬롯 예측값 (자기상관 0.8671 근사)
#   pred_lag2     : 2슬롯 전 예측값
#   pred_diff1    : 예측 추세 (현재 - 이전)
#   pred_rollmean3: 3슬롯 rolling 예측 평균
#   ratio_pred_sc : 현재 예측 / 시나리오 예측 평균
#
# 구조:
#   Stage1: LGB × 3seeds × 5fold → OOF + test 예측 생성 (~20분)
#   Stage2: LGB × 5seeds + XGB × 5seeds + CAT × 5seeds (~60분)
#
# 실행:
#   source .venv/bin/activate
#   caffeinate -i nohup python -u -B step30_pred_lag_stack.py > step30_output.log 2>&1 &
#   tail -f step30_output.log
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
S1_SEEDS   = [42, 123, 456]
N_SPLITS   = 5
MAX_ROUNDS     = 3000
EARLY_STOP     = 150
LGB_MAX_ROUNDS = 5000
LGB_EARLY_STOP = 200

LAYOUT_TYPE_MAP = {'grid': 0, 'hybrid': 1, 'narrow': 2, 'hub_spoke': 3}

print(f"CPU: {N_CPU}코어")
print(f"Stage1: LGB x {len(S1_SEEDS)}seeds x {N_SPLITS}fold = {len(S1_SEEDS)*N_SPLITS}runs")
print(f"Stage2: (LGB+XGB+CAT) x {len(SEEDS)}seeds x {N_SPLITS}fold = {3*len(SEEDS)*N_SPLITS}runs")

# ============================================================
# 1. 데이터 로드
# ============================================================
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

# ============================================================
# 2. 피처 엔지니어링 설정 (Step20 동일)
# ============================================================
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

SEQ_COLS = [
    'order_inflow_15m', 'unique_sku_15m', 'robot_active', 'robot_idle', 'robot_charging',
    'battery_mean', 'battery_std', 'low_battery_ratio', 'charge_queue_length',
    'avg_charge_wait', 'congestion_score', 'max_zone_density', 'blocked_path_15m',
    'near_collision_15m', 'fault_count_15m', 'avg_recovery_time', 'task_reassign_15m',
    'replenishment_overlap', 'pack_utilization', 'loading_dock_util',
    'staging_area_util', 'label_print_queue',
]


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
        exp_col, sc_col = f'exp_mean_{col}', f'sc_{col}_mean'
        if exp_col in df.columns and sc_col in df.columns:
            df[f'traj_{col}']     = df[exp_col] / (df[sc_col] + 1e-5)
            df[f'traj_dev_{col}'] = (df[exp_col] - df[sc_col]).abs()
    return df


def add_layout_target_encoding(train_df, test_df, y_log):
    gkf_te = GroupKFold(n_splits=N_SPLITS)
    sc_ids = train_df['scenario_id'].values
    layout_te_train = np.full(len(train_df), np.nan)
    for tr_idx, val_idx in gkf_te.split(train_df, y_log, sc_ids):
        tr_sub = train_df.iloc[tr_idx].copy()
        tr_sub['_y'] = y_log[tr_idx]
        layout_mean      = tr_sub.groupby('layout_id')['_y'].mean()
        layout_type_mean = tr_sub.groupby('layout_type')['_y'].mean()
        global_mean      = float(y_log[tr_idx].mean())
        val_df  = train_df.iloc[val_idx]
        encoded = val_df['layout_id'].map(layout_mean).copy()
        miss = encoded.isna()
        if miss.any():
            encoded[miss] = val_df.loc[miss,'layout_type'].map(layout_type_mean)
        encoded = encoded.fillna(global_mean)
        layout_te_train[val_idx] = encoded.values
    train_df = train_df.copy()
    train_df['layout_te'] = layout_te_train
    full_tr = train_df.copy()
    full_tr['_y'] = y_log
    layout_mean_all      = full_tr.groupby('layout_id')['_y'].mean()
    layout_type_mean_all = full_tr.groupby('layout_type')['_y'].mean()
    global_mean_all      = float(y_log.mean())
    test_df = test_df.copy()
    test_enc = test_df['layout_id'].map(layout_mean_all).copy()
    miss_t = test_enc.isna()
    if miss_t.any():
        test_enc[miss_t] = test_df.loc[miss_t,'layout_type'].map(layout_type_mean_all)
    test_enc = test_enc.fillna(global_mean_all)
    test_df['layout_te'] = test_enc.values
    print(f"  layout_te: cold-start {int(miss_t.sum())}행 ({miss_t.mean()*100:.1f}%)")
    return train_df, test_df


# ============================================================
# [핵심] OOF 예측 기반 피처 추가
# ============================================================
def add_pred_features(train_df, test_df, oof_pred, test_pred):
    """Stage1 예측값으로 sc_pred_mean, pred_lag1 등 추가"""
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df['_s1'] = oof_pred
    test_df['_s1']  = test_pred

    # 시나리오 레벨 집계
    sc_tr = train_df.groupby('scenario_id')['_s1'].agg(['mean','std','max','min'])
    sc_tr.columns = ['sc_pred_mean','sc_pred_std','sc_pred_max','sc_pred_min']
    train_df = train_df.merge(sc_tr.reset_index(), on='scenario_id', how='left')

    sc_te = test_df.groupby('scenario_id')['_s1'].agg(['mean','std','max','min'])
    sc_te.columns = ['sc_pred_mean','sc_pred_std','sc_pred_max','sc_pred_min']
    test_df = test_df.merge(sc_te.reset_index(), on='scenario_id', how='left')

    # lag 피처
    for df in [train_df, test_df]:
        grp = df.groupby('scenario_id')['_s1']
        lag1 = grp.shift(1)
        lag2 = grp.shift(2)
        df['pred_lag1']      = lag1
        df['pred_lag2']      = lag2
        df['pred_diff1']     = df['_s1'] - lag1
        df['pred_rollmean3'] = (lag1.groupby(df['scenario_id'])
                                    .rolling(3, min_periods=1).mean()
                                    .reset_index(level=0, drop=True))
        df['ratio_pred_sc']  = df['_s1'] / (df['sc_pred_mean'] + 1e-5)

    train_df.drop(columns=['_s1'], inplace=True)
    test_df.drop(columns=['_s1'],  inplace=True)
    return train_df, test_df


# ============================================================
# 3. 피처 엔지니어링 실행
# ============================================================
print("\n" + "="*60)
print("2. 피처 엔지니어링")
print("="*60)

train = feature_engineering(train)
test  = feature_engineering(test)
train = add_temporal_features(train)
test  = add_temporal_features(test)
print("  [Step20] 기본 + temporal 피처...")
train = add_lag_rolling_features(train)
test  = add_lag_rolling_features(test)
print("  [Step20] Lag/Rolling 피처 (22 SEQ_COLS x 6)...")
train = add_onset_features(train)
test  = add_onset_features(test)
train = add_threshold_features(train)
test  = add_threshold_features(test)
train, test = add_aggregation_features(train, test)
print("  [Step20] Onset + Threshold + sc_* 집계...")
train = add_sc_derived_features(train)
test  = add_sc_derived_features(test)
train = add_trajectory_features(train)
test  = add_trajectory_features(test)

y_log_full = np.log1p(train[TARGET].values)
train, test = add_layout_target_encoding(train, test, y_log_full)

DROP_COLS = {'ID','layout_id','scenario_id','layout_type',TARGET,'shift_hour'}
feature_cols_base = [c for c in train.columns
                     if c not in DROP_COLS and c in test.columns
                     and train[c].dtype != object]
feature_cols_base = list(dict.fromkeys(feature_cols_base))

X_train_base = train[feature_cols_base].astype(np.float32)
y_train      = y_log_full
X_test_base  = test[feature_cols_base].astype(np.float32)
groups       = train['scenario_id'].values
y_true       = train[TARGET].values

print(f"\n  Step20 피처 수: {len(feature_cols_base)}개")
print(f"  train: {X_train_base.shape} | test: {X_test_base.shape}")

# ============================================================
# 4. 모델 파라미터
# ============================================================
LGB_S1 = dict(objective='regression_l1', metric='mae', learning_rate=0.05,
              num_leaves=127, max_depth=-1, min_child_samples=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
              lambda_l1=0.1, lambda_l2=0.1,
              device_type='cpu', num_threads=N_CPU, verbose=-1)

LGB_S2 = dict(objective='regression_l1', metric='mae', learning_rate=0.02,
              num_leaves=127, max_depth=-1, min_child_samples=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
              lambda_l1=0.1, lambda_l2=0.1,
              device_type='cpu', num_threads=N_CPU, verbose=-1)

XGB_P  = dict(objective='reg:absoluteerror', eval_metric='mae',
              learning_rate=0.05, max_depth=7, min_child_weight=50,
              subsample=0.8, colsample_bytree=0.8,
              reg_alpha=0.1, reg_lambda=0.1,
              tree_method='hist', device='cpu', nthread=N_CPU, verbosity=0)

CAT_P  = dict(loss_function='MAE', eval_metric='MAE',
              learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
              min_data_in_leaf=50, subsample=0.8,
              task_type='CPU', thread_count=N_CPU, verbose=False)

gkf = GroupKFold(n_splits=N_SPLITS)
t_total = time.time()

# ============================================================
# 5. Stage 1: 빠른 LGB OOF 생성
# ============================================================
print("\n" + "="*60)
print("3. Stage1: LGB OOF 예측 생성")
print("="*60)

s1_oof  = np.zeros(len(X_train_base))
s1_test = np.zeros(len(X_test_base))
s1_cnt  = 0

for seed in S1_SEEDS:
    oof_s = np.zeros(len(X_train_base))
    tst_s = np.zeros(len(X_test_base))
    p = {**LGB_S1, 'seed': seed}

    for fold, (tr_idx, vl_idx) in enumerate(gkf.split(X_train_base, y_train, groups)):
        t0 = time.time()
        dtrain = lgb.Dataset(X_train_base.iloc[tr_idx], label=y_train[tr_idx])
        dval   = lgb.Dataset(X_train_base.iloc[vl_idx], label=y_train[vl_idx], reference=dtrain)
        m = lgb.train(p, dtrain, num_boost_round=LGB_MAX_ROUNDS,
                      valid_sets=[dval], valid_names=['val'],
                      callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False),
                                 lgb.log_evaluation(-1)])
        vp = np.clip(np.expm1(m.predict(X_train_base.iloc[vl_idx], num_iteration=m.best_iteration)), 0, None)
        oof_s[vl_idx] = vp
        tst_s += np.clip(np.expm1(m.predict(X_test_base, num_iteration=m.best_iteration)), 0, None) / N_SPLITS
        s1_cnt += 1
        mae = mean_absolute_error(y_true[vl_idx], vp)
        print(f"  S1 seed={seed} fold={fold+1} MAE={mae:.4f} {(time.time()-t0)/60:.1f}min [{s1_cnt}/{len(S1_SEEDS)*N_SPLITS}]")

    sc = mean_absolute_error(y_true, oof_s)
    print(f"  -> seed={seed} OOF: {sc:.4f}")
    s1_oof  += oof_s  / len(S1_SEEDS)
    s1_test += tst_s  / len(S1_SEEDS)

s1_mae = mean_absolute_error(y_true, s1_oof)
print(f"\n  Stage1 앙상블 OOF MAE: {s1_mae:.4f} ({(time.time()-t_total)/60:.1f}분)")

# ============================================================
# 6. OOF 예측 기반 피처 추가
# ============================================================
print("\n" + "="*60)
print("4. [핵심] OOF 예측 기반 피처 추가")
print("="*60)

train, test = add_pred_features(train, test, s1_oof, s1_test)

feature_cols_s2 = [c for c in train.columns
                   if c not in DROP_COLS and c in test.columns
                   and train[c].dtype != object]
feature_cols_s2 = list(dict.fromkeys(feature_cols_s2))

X_train = train[feature_cols_s2].astype(np.float32)
X_test  = test[feature_cols_s2].astype(np.float32)
print(f"  Step30 피처 수: {len(feature_cols_s2)}개 (+{len(feature_cols_s2)-len(feature_cols_base)}개)")

for col in ['sc_pred_mean','pred_lag1','pred_diff1','ratio_pred_sc']:
    if col in train.columns:
        corr = train[col].corr(train[TARGET])
        print(f"  {col} -> target 상관: {corr:.4f}")

# ============================================================
# 7. Stage 2: 풀 앙상블
# ============================================================
print("\n" + "="*60)
print("5. Stage2: LGB+XGB+CAT 풀 앙상블")
print("="*60)

all_oof   = {'LGB': {}, 'XGB': {}, 'CAT': {}}
all_test  = {'LGB': {}, 'XGB': {}, 'CAT': {}}
total_runs = len(SEEDS) * 3
run_count  = 0

# --- LightGBM ---
for seed in SEEDS:
    oof_p = np.zeros(len(X_train))
    tst_p = np.zeros(len(X_test))
    p = {**LGB_S2, 'seed': seed}
    for fold, (tr_idx, vl_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        t0 = time.time()
        dtrain = lgb.Dataset(X_train.iloc[tr_idx], label=y_train[tr_idx])
        dval   = lgb.Dataset(X_train.iloc[vl_idx], label=y_train[vl_idx], reference=dtrain)
        m = lgb.train(p, dtrain, num_boost_round=LGB_MAX_ROUNDS,
                      valid_sets=[dval], valid_names=['val'],
                      callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False),
                                 lgb.log_evaluation(-1)])
        vp = np.clip(np.expm1(m.predict(X_train.iloc[vl_idx], num_iteration=m.best_iteration)), 0, None)
        oof_p[vl_idx] = vp
        tst_p += np.clip(np.expm1(m.predict(X_test, num_iteration=m.best_iteration)), 0, None) / N_SPLITS
    sc = mean_absolute_error(y_true, oof_p)
    all_oof['LGB'][seed] = oof_p
    all_test['LGB'][seed] = tst_p
    run_count += 1
    print(f"  LGB seed={seed} OOF={sc:.4f} [{run_count}/{total_runs}]")

lgb_oof  = np.mean(list(all_oof['LGB'].values()), axis=0)
lgb_test = np.mean(list(all_test['LGB'].values()), axis=0)
print(f"  LGB 앙상블 OOF: {mean_absolute_error(y_true, lgb_oof):.4f}")

# --- XGBoost ---
for seed in SEEDS:
    oof_p = np.zeros(len(X_train))
    tst_p = np.zeros(len(X_test))
    p = {**XGB_P, 'seed': seed}
    for fold, (tr_idx, vl_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        t0 = time.time()
        dtrain = xgb.DMatrix(X_train.iloc[tr_idx], label=y_train[tr_idx])
        dval   = xgb.DMatrix(X_train.iloc[vl_idx], label=y_train[vl_idx])
        m = xgb.train(p, dtrain, num_boost_round=MAX_ROUNDS,
                      evals=[(dval,'val')], early_stopping_rounds=EARLY_STOP,
                      verbose_eval=False)
        vp = np.clip(np.expm1(m.predict(dval, iteration_range=(0, m.best_iteration))), 0, None)
        oof_p[vl_idx] = vp
        tst_p += np.clip(np.expm1(m.predict(xgb.DMatrix(X_test),
                                             iteration_range=(0, m.best_iteration))), 0, None) / N_SPLITS
    sc = mean_absolute_error(y_true, oof_p)
    all_oof['XGB'][seed] = oof_p
    all_test['XGB'][seed] = tst_p
    run_count += 1
    print(f"  XGB seed={seed} OOF={sc:.4f} [{run_count}/{total_runs}]")

xgb_oof  = np.mean(list(all_oof['XGB'].values()), axis=0)
xgb_test = np.mean(list(all_test['XGB'].values()), axis=0)
print(f"  XGB 앙상블 OOF: {mean_absolute_error(y_true, xgb_oof):.4f}")

# --- CatBoost ---
for seed in SEEDS:
    oof_p = np.zeros(len(X_train))
    tst_p = np.zeros(len(X_test))
    for fold, (tr_idx, vl_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        t0 = time.time()
        m = CatBoostRegressor(**CAT_P, iterations=MAX_ROUNDS, random_seed=seed)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx],
              eval_set=(X_train.iloc[vl_idx], y_train[vl_idx]),
              early_stopping_rounds=EARLY_STOP, use_best_model=True, verbose=False)
        vp = np.clip(np.expm1(m.predict(X_train.iloc[vl_idx])), 0, None)
        oof_p[vl_idx] = vp
        tst_p += np.clip(np.expm1(m.predict(X_test)), 0, None) / N_SPLITS
    sc = mean_absolute_error(y_true, oof_p)
    all_oof['CAT'][seed] = oof_p
    all_test['CAT'][seed] = tst_p
    run_count += 1
    print(f"  CAT seed={seed} OOF={sc:.4f} [{run_count}/{total_runs}]")

cat_oof  = np.mean(list(all_oof['CAT'].values()), axis=0)
cat_test = np.mean(list(all_test['CAT'].values()), axis=0)
print(f"  CAT 앙상블 OOF: {mean_absolute_error(y_true, cat_oof):.4f}")

# ============================================================
# 8. 앙상블 & 제출
# ============================================================
print("\n" + "="*60)
print("6. 앙상블 & 제출")
print("="*60)

final_oof  = (lgb_oof + xgb_oof + cat_oof) / 3
final_test = (lgb_test + xgb_test + cat_test) / 3
final_mae  = mean_absolute_error(y_true, final_oof)
print(f"  Stage1 OOF: {s1_mae:.4f}")
print(f"  Stage2 최종 OOF: {final_mae:.4f}")
print(f"  test: mean={final_test.mean():.2f} std={final_test.std():.2f} "
      f"max={final_test.max():.2f} 50min+={(final_test>=50).mean()*100:.1f}%")

# OOF 저장
oof_df = pd.DataFrame({
    'ID': train['ID'].values,
    'scenario_id': train['scenario_id'].values,
    'slot_idx': train['slot_idx'].values,
    'y_true': y_true,
    'oof_s1': s1_oof,
    'oof_lgb': lgb_oof,
    'oof_xgb': xgb_oof,
    'oof_cat': cat_oof,
    'oof_final': final_oof,
})
oof_path = os.path.join(OUTPUT_DIR, 'oof_step30.csv')
oof_df.to_csv(oof_path, index=False)
print(f"  OOF 저장: {oof_path}")

# 제출 파일
submission = sample.copy()
submission[TARGET] = final_test
out = os.path.join(OUTPUT_DIR, 'submission_step30_pred_lag.csv')
submission.to_csv(out, index=False)
print(f"  제출 파일: {out}")
print(f"  예측 음수: {(final_test < 0).sum()}개")

elapsed = (time.time()-t_total)/60
print("\n" + "="*60)
print(f"완료! 총 소요: {elapsed:.1f}분")
print(f"Step20 기준선: Public 10.0606")
print(f"Step30 OOF:   {final_mae:.4f}")
print("="*60)
