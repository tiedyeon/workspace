# =============================================================================
# Step 35: Lead (미래 슬롯) 피처 추가
# =============================================================================
# 핵심 아이디어:
#   대회 운영자 공식 확인: "test.csv 제공 시점에 시나리오 관측이 이미 완료"
#   → 같은 시나리오 내 미래 슬롯 피처를 현재 예측에 써도 leakage 아님
#
#   [신규 피처 그룹]
#   1) Lead 직접값: lead1_col, lead2_col, lead3_col (1~3슬롯 앞)
#   2) Lead 방향/가속: lead_diff1, lead_diff2, lead_accel (변화 방향)
#   3) Remaining 통계: remain_mean, remain_max, remain_gap (나머지 슬롯 통계)
#   4) Lead × 기존 피처 cross (dynamic risk)
#
#   타겟: avg_delay_minutes_next_30m (향후 30분 평균 지연)
#   → lead1은 바로 그 30분 구간의 창고 상태 → 매우 직접적 예측 신호
#
# 베이스: step32 (OOF 8.5729, 469 피처)
# 기대: lead 피처가 step32 sc_* 보다 훨씬 강한 상관 → OOF 8.3 이하 목표
#
# 실행:
#   caffeinate -i nohup python -u -B step35_lead_features.py > step35_output.log 2>&1 &
#   tail -f step35_output.log
# =============================================================================

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

EXTREME_THRESH = 50.0
N_CLUSTERS     = 5
LAYOUT_TYPE_MAP = {'grid': 0, 'hybrid': 1, 'narrow': 2, 'hub_spoke': 3}

print(f"CPU: {N_CPU}코어")
print(f"Step35: Lead(미래슬롯) 피처 추가 버전")
print(f"Seeds: {len(SEEDS)}개 × 3모델 = {len(SEEDS)*3}인스턴스")

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
# 2. 피처 엔지니어링 상수 (Step32 동일)
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
SEQ_COLS = [
    'order_inflow_15m', 'unique_sku_15m', 'robot_active', 'robot_idle', 'robot_charging',
    'battery_mean', 'battery_std', 'low_battery_ratio', 'charge_queue_length',
    'avg_charge_wait', 'congestion_score', 'max_zone_density', 'blocked_path_15m',
    'near_collision_15m', 'fault_count_15m', 'avg_recovery_time', 'task_reassign_15m',
    'replenishment_overlap', 'pack_utilization', 'loading_dock_util',
    'staging_area_util', 'label_print_queue',
]

# ── [NEW] Lead/Remaining 피처에 사용할 컬럼 ────────────────────────────────
LEAD_COLS = [
    'congestion_score', 'low_battery_ratio', 'order_inflow_15m',
    'robot_active', 'blocked_path_15m', 'charge_queue_length',
    'max_zone_density', 'pack_utilization', 'battery_mean',
    'aisle_traffic_score', 'robot_utilization',
]
REMAIN_COLS = [
    'congestion_score', 'low_battery_ratio', 'order_inflow_15m',
    'charge_queue_length', 'blocked_path_15m', 'max_zone_density',
    'pack_utilization', 'robot_active',
]

# =============================================================================
# 3. 피처 엔지니어링 함수 (Step32 동일)
# =============================================================================
def feature_engineering(df):
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
    for col in ['congestion_score', 'low_battery_ratio', 'order_inflow_15m',
                'robot_active', 'blocked_path_15m', 'charge_queue_length']:
        if col in df.columns:
            df[f'lag1_{col}'] = df.groupby('scenario_id')[col].shift(1)
    return df


def add_lag_rolling_features(df):
    df = df.copy()
    grp     = df.groupby('scenario_id', sort=False)
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
        if value_col not in df.columns:
            return
        positive = df[value_col].fillna(0).gt(0)
        t     = df['slot_idx'].where(positive).astype(float)
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


def add_threshold_features(df):
    df = df.copy()
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
        df['pack_utilization_sq']  = df['pack_utilization'].fillna(0) ** 2
    if 'loading_dock_util' in df.columns:
        df['loading_dock_util_sq'] = df['loading_dock_util'].fillna(0) ** 2
    if 'staging_area_util' in df.columns:
        df['staging_area_util_sq'] = df['staging_area_util'].fillna(0) ** 2
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
        val_df   = train_df.iloc[val_idx]
        encoded  = val_df['layout_id'].map(layout_mean).copy()
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
    print(f"  layout_te: OOF 완료 | test cold-start {n_cold}행")
    return train_df, test_df


# =============================================================================
# [NEW] Lead 피처 함수
# =============================================================================
def add_lead_features(df):
    """
    미래 슬롯 직접값 및 변화 방향 피처
    - lead1: t+1 슬롯값 (15분 후)
    - lead2: t+2 슬롯값 (30분 후) ← 타겟 기간과 일치!
    - lead3: t+3 슬롯값 (45분 후)
    - lead_diff1: lead1 - current (앞으로 변화 방향)
    - lead_diff2: lead2 - lead1 (2번째 변화 방향)
    - lead_accel: 가속도 (2차 미분)
    """
    df = df.copy()
    grp = df.groupby('scenario_id', sort=False)

    for col in LEAD_COLS:
        if col not in df.columns:
            continue
        lead1 = grp[col].shift(-1)
        lead2 = grp[col].shift(-2)
        lead3 = grp[col].shift(-3)

        df[f'lead1_{col}'] = lead1
        df[f'lead2_{col}'] = lead2
        df[f'lead3_{col}'] = lead3

        # 변화 방향
        df[f'lead_diff1_{col}'] = lead1 - df[col]      # 1슬롯 후 변화량
        df[f'lead_diff2_{col}'] = lead2 - lead1         # 2슬롯 구간 변화량
        # 가속도: 변화가 빨라지는지 느려지는지
        df[f'lead_accel_{col}'] = (lead2 - lead1) - (lead1 - df[col])

    # Lead 기반 복합 위험 신호
    if 'lead1_congestion_score' in df.columns and 'lead1_low_battery_ratio' in df.columns:
        df['lead1_crisis'] = df['lead1_congestion_score'] * df['lead1_low_battery_ratio']
    if 'lead2_congestion_score' in df.columns and 'lead2_order_inflow_15m' in df.columns:
        df['lead2_pressure'] = df['lead2_congestion_score'] * df['lead2_order_inflow_15m']

    return df


def add_remaining_features(df):
    """
    나머지 슬롯 통계 (현재 슬롯 이후 ~ 시나리오 끝)

    접근법:
      sc_mean * 25 = 시나리오 전체 합계
      expanding_sum = 현재까지 누적합
      remain_sum = 전체합 - 누적합
      remain_mean = remain_sum / 남은 슬롯 수

    sc_* 피처가 이미 계산되어 있어야 함.
    """
    df = df.copy()
    grp = df.groupby('scenario_id', sort=False)

    # 남은 슬롯 수 (현재 슬롯 이후)
    df['slots_remaining'] = (24 - df['slot_idx']).clip(lower=0).astype(np.float32)

    for col in REMAIN_COLS:
        if col not in df.columns:
            continue
        sc_mean_col = f'sc_{col}_mean'
        sc_max_col  = f'sc_{col}_max'

        # 현재까지 누적합
        exp_sum = grp[col].expanding().sum().reset_index(level=0, drop=True)

        if sc_mean_col in df.columns:
            total_sum  = df[sc_mean_col] * 25  # 시나리오 전체 합계
            remain_sum = (total_sum - exp_sum).clip(lower=0)
            df[f'remain_mean_{col}'] = remain_sum / (df['slots_remaining'] + 1e-5)
            # 앞으로 평균이 현재보다 높아지나 낮아지나
            df[f'remain_vs_now_{col}'] = df[f'remain_mean_{col}'] - df[col]

        # 나머지 슬롯의 최대값: 역방향 expanding max
        if sc_max_col in df.columns:
            def rev_exp_max_excl(series):
                # 역순으로 뒤집어서 expanding max → 다시 뒤집고 한칸 shift
                # = 현재 이후 슬롯들의 max
                rev = series.iloc[::-1]
                rev_max = rev.expanding().max().shift(1)  # 현재 슬롯 제외
                return rev_max.iloc[::-1]
            df[f'remain_max_{col}'] = grp[col].transform(rev_exp_max_excl)
            df[f'remain_max_gap_{col}'] = df[f'remain_max_{col}'] - df[col]

    # 미래 총 주문량 (remaining slots의 order 합)
    if 'order_inflow_15m' in df.columns and 'sc_order_inflow_15m_mean' in df.columns:
        exp_sum_order = grp['order_inflow_15m'].expanding().sum().reset_index(level=0, drop=True)
        total_order   = df['sc_order_inflow_15m_mean'] * 25
        df['remain_total_orders'] = (total_order - exp_sum_order).clip(lower=0).astype(np.float32)

    return df


# =============================================================================
# 4. 피처 엔지니어링 실행
# =============================================================================
print("\n" + "="*60)
print("2. 피처 엔지니어링")
print("="*60)

train = feature_engineering(train)
test  = feature_engineering(test)

print("  temporal...")
train = add_temporal_features(train)
test  = add_temporal_features(test)

print("  lag/rolling...")
train = add_lag_rolling_features(train)
test  = add_lag_rolling_features(test)

print("  onset...")
train = add_onset_features(train)
test  = add_onset_features(test)

print("  threshold...")
train = add_threshold_features(train)
test  = add_threshold_features(test)

print("  sc_* 집계...")
train, test = add_aggregation_features(train, test)
train = add_sc_derived_features(train)
test  = add_sc_derived_features(test)

print("  trajectory...")
train = add_trajectory_features(train)
test  = add_trajectory_features(test)

y_log_full = np.log1p(train[TARGET].values)

print("  layout TE...")
train, test = add_layout_target_encoding(train, test, y_log_full)

# [NEW] Lead 피처 (sc_* 이후에 실행)
print("  [NEW] lead 피처 (미래 슬롯 직접값)...")
train = add_lead_features(train)
test  = add_lead_features(test)

# [NEW] Remaining 피처 (sc_* 필요)
print("  [NEW] remaining 피처 (나머지 슬롯 통계)...")
train = add_remaining_features(train)
test  = add_remaining_features(test)

# Lead 피처 상관관계 확인
print("\n  [Lead 피처 상관관계 Top10]")
target_arr  = train[TARGET].values
lead_feats  = [c for c in train.columns if c.startswith('lead') or c.startswith('remain')]
lead_corrs  = {}
for c in lead_feats:
    if train[c].dtype in [np.float32, np.float64, float]:
        corr = np.corrcoef(train[c].fillna(0).values, target_arr)[0, 1]
        lead_corrs[c] = abs(corr)
top10 = sorted(lead_corrs.items(), key=lambda x: x[1], reverse=True)[:10]
for feat, corr in top10:
    raw_corr = np.corrcoef(train[feat].fillna(0).values, target_arr)[0, 1]
    print(f"    {feat:45s}: {raw_corr:+.4f}")

# =============================================================================
# 5. Cluster Extreme Risk 피처 (Step32 동일, OOF)
# =============================================================================
print("\n" + "="*60)
print("3. Cluster Extreme Risk 피처 (Step32)")
print("="*60)

sc_max_target = train.groupby('scenario_id')[TARGET].max()
is_extreme_sc = (sc_max_target >= EXTREME_THRESH).astype(int)
print(f"  극단값 시나리오: {is_extreme_sc.sum():,}/{len(is_extreme_sc):,} ({is_extreme_sc.mean()*100:.1f}%)")

sc_cols_for_cluster = [c for c in train.columns if c.startswith('sc_')]
train_sc_level = train.groupby('scenario_id')[sc_cols_for_cluster].first()
test_sc_level  = test.groupby('scenario_id')[sc_cols_for_cluster].first()
scaler  = StandardScaler()
X_sc_tr = scaler.fit_transform(train_sc_level.fillna(0))
X_sc_te = scaler.transform(test_sc_level.fillna(0))
kmeans  = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10, max_iter=300)
train_sc_level['cluster_id'] = kmeans.fit_predict(X_sc_tr)
test_sc_level['cluster_id']  = kmeans.predict(X_sc_te)
train_sc_cluster = train_sc_level['cluster_id'].to_dict()
test_sc_cluster  = test_sc_level['cluster_id'].to_dict()
train['cluster_id'] = train['scenario_id'].map(train_sc_cluster).astype(np.int8)
test['cluster_id']  = test['scenario_id'].map(test_sc_cluster).astype(np.int8)

sc_info = pd.DataFrame({
    'scenario_id': list(train_sc_cluster.keys()),
    'cluster_id':  list(train_sc_cluster.values()),
})
sc_info['is_extreme'] = sc_info['scenario_id'].map(is_extreme_sc)
cluster_er_full = sc_info.groupby('cluster_id')['is_extreme'].mean().to_dict()
layout_sc_df    = train[['scenario_id', 'layout_id']].drop_duplicates()
layout_sc_df['is_extreme'] = layout_sc_df['scenario_id'].map(is_extreme_sc)
layout_er_full  = layout_sc_df.groupby('layout_id')['is_extreme'].mean().to_dict()

oof_cluster_er = np.zeros(len(train), dtype=np.float32)
oof_layout_er  = np.zeros(len(train), dtype=np.float32)
gkf_pre      = GroupKFold(n_splits=N_SPLITS)
scenario_arr = train['scenario_id'].values
layout_arr   = train['layout_id'].values
cluster_arr  = train['cluster_id'].values

for fold_i, (tr_idx, val_idx) in enumerate(gkf_pre.split(train, y_log_full, scenario_arr)):
    tr_sc_set   = set(scenario_arr[tr_idx])
    fold_sc_info = sc_info[sc_info['scenario_id'].isin(tr_sc_set)]
    fold_cluster_er = fold_sc_info.groupby('cluster_id')['is_extreme'].mean().to_dict()
    val_clusters = cluster_arr[val_idx]
    oof_cluster_er[val_idx] = np.array([fold_cluster_er.get(c, 0.0) for c in val_clusters], dtype=np.float32)
    fold_layout_sc = layout_sc_df[layout_sc_df['scenario_id'].isin(tr_sc_set)]
    fold_layout_er = fold_layout_sc.groupby('layout_id')['is_extreme'].mean().to_dict()
    global_er      = fold_sc_info['is_extreme'].mean()
    val_layouts    = layout_arr[val_idx]
    oof_layout_er[val_idx] = np.array([fold_layout_er.get(l, global_er) for l in val_layouts], dtype=np.float32)

print(f"  cluster_extreme_ratio vs target 상관: {np.corrcoef(oof_cluster_er, target_arr)[0,1]:.4f}")
print(f"  layout_extreme_ratio  vs target 상관: {np.corrcoef(oof_layout_er,  target_arr)[0,1]:.4f}")

train['cluster_extreme_ratio'] = oof_cluster_er
train['layout_extreme_ratio']  = oof_layout_er

test_cluster_ids = test['scenario_id'].map(test_sc_cluster).values
test['cluster_extreme_ratio'] = np.array([cluster_er_full.get(c, 0.0) for c in test_cluster_ids], dtype=np.float32)
global_er_full = layout_sc_df['is_extreme'].mean()
test['layout_extreme_ratio']  = np.array([layout_er_full.get(l, global_er_full) for l in test['layout_id'].values], dtype=np.float32)

# Cross 피처
for df in [train, test]:
    df['cluster_er_x_slot']     = df['cluster_extreme_ratio'] * df['slot_progress']
    df['layout_er_x_slot']      = df['layout_extreme_ratio']  * df['slot_progress']
    df['combined_extreme_risk'] = df['cluster_extreme_ratio'] * df['layout_extreme_ratio']
    if 'sc_congestion_score_max' in df.columns and 'sc_low_battery_ratio_max' in df.columns:
        df['sc_extreme_pressure'] = (
            df['sc_congestion_score_max'].fillna(0)
            * df['sc_low_battery_ratio_max'].fillna(0)
            * df['cluster_extreme_ratio']
        )
    df['cluster_er_x_congestion'] = df['cluster_extreme_ratio'] * df['congestion_score'].fillna(0)
    df['cluster_er_x_battery']    = df['cluster_extreme_ratio'] * df['low_battery_ratio']

    # [NEW] Lead × Extreme Risk cross 피처
    if 'lead1_congestion_score' in df.columns:
        df['lead1_congestion_x_er'] = df['lead1_congestion_score'] * df['cluster_extreme_ratio']
    if 'lead1_low_battery_ratio' in df.columns:
        df['lead1_battery_x_er']    = df['lead1_low_battery_ratio'] * df['cluster_extreme_ratio']
    if 'remain_mean_congestion_score' in df.columns:
        df['remain_congestion_x_er'] = df['remain_mean_congestion_score'] * df['cluster_extreme_ratio']

# 초기 슬롯 피처 (Step32 동일)
EARLY_AGG_COLS = ['congestion_score', 'low_battery_ratio', 'order_inflow_15m',
                  'charge_queue_length', 'max_zone_density', 'blocked_path_15m']
EARLY_FUNCS = ['max', 'mean']
oof_early = {}
for col in EARLY_AGG_COLS:
    for fn in EARLY_FUNCS:
        oof_early[f'early_{col}_{fn}'] = np.full(len(train), np.nan, dtype=np.float32)

test_early = test[test['slot_idx'] <= 2].groupby('scenario_id')[EARLY_AGG_COLS].agg(EARLY_FUNCS)
test_early.columns = [f'early_{c}_{f}' for c, f in test_early.columns]

for fold_i, (tr_idx, val_idx) in enumerate(gkf_pre.split(train, y_log_full, scenario_arr)):
    val_sc_set  = set(scenario_arr[val_idx])
    val_early_fold = train[
        (train['slot_idx'] <= 2) & (train['scenario_id'].isin(val_sc_set))
    ].groupby('scenario_id')[EARLY_AGG_COLS].agg(EARLY_FUNCS)
    val_early_fold.columns = [f'early_{c}_{f}' for c, f in val_early_fold.columns]
    for feat in oof_early.keys():
        val_sc_ids = scenario_arr[val_idx]
        vals = pd.Series(val_sc_ids).map(
            val_early_fold[feat].to_dict() if feat in val_early_fold.columns else {}
        ).values.astype(np.float32)
        oof_early[feat][val_idx] = vals

for feat, arr in oof_early.items():
    train[feat] = arr

for feat in [f'early_{c}_{fn}' for c in EARLY_AGG_COLS for fn in EARLY_FUNCS]:
    if feat in test_early.columns:
        test[feat] = test['scenario_id'].map(test_early[feat].to_dict()).astype(np.float32)
    else:
        test[feat] = 0.0

# =============================================================================
# 6. 피처 선택
# =============================================================================
print("\n" + "="*60)
print("4. 피처 선택")
print("="*60)

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

lead_feat_count    = sum(1 for c in feature_cols if c.startswith('lead'))
remain_feat_count  = sum(1 for c in feature_cols if c.startswith('remain') or c == 'slots_remaining')
step32_base_count  = len(feature_cols) - lead_feat_count - remain_feat_count

print(f"  Step32 베이스: ~469피처")
print(f"  + Lead 피처:     {lead_feat_count}개")
print(f"  + Remaining 피처:{remain_feat_count}개")
print(f"  총 피처 수:      {len(feature_cols)}개")
print(f"  train: {X_train.shape} | test: {X_test.shape}")

# =============================================================================
# 7. 모델 파라미터 (Step32 동일)
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
# 8. 학습
# =============================================================================
print("\n" + "="*60)
print("5. 학습 시작")
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
    all_oof['LightGBM'][seed]   = oof_p
    all_test['LightGBM'][seed]  = test_p
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
    all_oof['XGBoost'][seed]   = oof_p
    all_test['XGBoost'][seed]  = test_p
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
    all_oof['CatBoost'][seed]   = oof_p
    all_test['CatBoost'][seed]  = test_p
    all_score['CatBoost'][seed] = sc
    run_count += 1
    print(f"  seed={seed:5d} | OOF: {sc:.4f} | {(time.time()-t0)/60:.1f}min | [{run_count}/{total_runs}]")
s = list(all_score['CatBoost'].values())
print(f"  CatBoost 평균: {np.mean(s):.4f} ± {np.std(s):.4f}")

# =============================================================================
# 9. 앙상블
# =============================================================================
print("\n" + "="*60)
print("6. 앙상블")
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

all_oof_list  = [all_oof[m][s]  for m in ['LightGBM', 'XGBoost', 'CatBoost'] for s in SEEDS]
all_test_list = [all_test[m][s] for m in ['LightGBM', 'XGBoost', 'CatBoost'] for s in SEEDS]
mae_avg  = mean_absolute_error(y_true, np.mean(all_oof_list, axis=0))
test_avg = np.mean(all_test_list, axis=0)
print(f"\n  전체 단순 평균 (15개) OOF MAE: {mae_avg:.4f}")

inv = {m: 1/model_oof_mae[m] for m in model_oof_mae}
w   = {m: inv[m]/sum(inv.values()) for m in inv}
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
# 10. 극단값 구간별 MAE 분석
# =============================================================================
print("\n" + "="*60)
print("7. 극단값 구간별 MAE 분석")
print("="*60)
oof_final = oof_w if mae_w <= mae_avg else np.mean(all_oof_list, axis=0)
bins      = [0, 10, 30, 50, 100, np.inf]
labels    = ['0-10', '10-30', '30-50', '50-100', '100+']
for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
    mask = (y_true >= lo) & (y_true < hi)
    if mask.sum() > 0:
        mae_seg  = mean_absolute_error(y_true[mask], oof_final[mask])
        bias_seg = np.mean(oof_final[mask] - y_true[mask])
        print(f"  {lab:8s}: n={mask.sum():6d} ({mask.mean()*100:4.1f}%) | "
              f"MAE={mae_seg:6.2f} | bias={bias_seg:+.2f}")

# =============================================================================
# 11. 결과 요약
# =============================================================================
print("\n" + "="*60)
print("8. Step 비교")
print("="*60)
print(f"  Step20 OOF MAE: 8.5964  (448 피처, lag/rolling/sc)")
print(f"  Step32 OOF MAE: 8.5729  (469 피처, +cluster_extreme_ratio)")
print(f"  Step35 OOF MAE: {best_mae:.4f}  ({len(feature_cols)} 피처, +lead/remaining)")
print(f"  Step32→35 변화: {best_mae - 8.5729:+.4f}")
print(f"\n  Step32 예상 Public: ~10.0329")
print(f"  Step35 예상 Public: ~{best_mae + 1.46:.4f}  (갭 1.46 가정)")
print(f"\n  총 소요 시간: {(time.time()-total_start)/60:.1f}분")

# =============================================================================
# 12. OOF 저장
# =============================================================================
oof_df = pd.DataFrame({
    'ID':           train['ID'].values if 'ID' in train.columns else np.arange(len(train)),
    'scenario_id':  train['scenario_id'].values,
    'slot_idx':     train['slot_idx'].values,
    'y_true':       y_true,
    'cluster_er':   train['cluster_extreme_ratio'].values,
    'oof_lgb':      model_oof_avg['LightGBM'],
    'oof_xgb':      model_oof_avg['XGBoost'],
    'oof_cat':      model_oof_avg['CatBoost'],
    'oof_final':    oof_final,
})
oof_path = os.path.join(OUTPUT_DIR, 'oof_step35.csv')
oof_df.to_csv(oof_path, index=False)
print(f"\nOOF 저장: {oof_path}")

# =============================================================================
# 13. 제출 파일
# =============================================================================
submission = sample.copy()
submission[TARGET] = np.clip(best_preds, 0, None)
out = os.path.join(OUTPUT_DIR, 'submission_step35_lead_features.csv')
submission.to_csv(out, index=False)
print(f"제출 파일: submission_step35_lead_features.csv")
print(f"OOF MAE: {best_mae:.4f}")
print(f"\n예측값 분포:")
print(submission[TARGET].describe().round(3).to_string())
