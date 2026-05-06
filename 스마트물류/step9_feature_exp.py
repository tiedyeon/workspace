# =============================================================================
# 스마트 창고 출고 지연 예측 AI 경진대회
# Step 9: 피처 엔지니어링 실험 A~F (OOF 비교)
# =============================================================================
# 실험 목록 (모두 Step7 베이스 197 피처 위에 추가):
#   A: 시나리오 내 z-score 피처
#   B: 타임슬롯 순서 복원 + lag/diff 피처
#   C: 타임슬롯 위치 피처 (슬롯 rank, 초반/후반 여부)
#   D: sc_* 집계 확장 (더 많은 컬럼 + median/skew/quantile)
#   E: A + B + C
#   F: A + B + C + D (전체)
# =============================================================================
# 속도: LGB 단일 시드(42) × 5폴드 — 실험당 ~1.5분, 전체 ~12분
# 실행:
#   caffeinate -i nohup python step9_feature_exp.py > step9_output.log 2>&1 &
#   tail -f step9_output.log
# =============================================================================

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)

N_CPU      = os.cpu_count()
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = DATA_DIR
TARGET     = 'avg_delay_minutes_next_30m'
SEED       = 42
N_SPLITS   = 5
MAX_ROUNDS = 2000
EARLY_STOP = 100

LAYOUT_TYPE_MAP = {'grid': 0, 'hybrid': 1, 'narrow': 2, 'hub_spoke': 3}

LGB_PARAMS = {
    'objective': 'regression_l1', 'metric': 'mae',
    'learning_rate': 0.05, 'num_leaves': 127, 'max_depth': -1,
    'min_child_samples': 50, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5,
    'lambda_l1': 0.1, 'lambda_l2': 0.1,
    'device_type': 'cpu', 'num_threads': N_CPU, 'verbose': -1, 'seed': SEED,
}

# =============================================================================
# 1. 데이터 로드
# =============================================================================
print("="*60)
print("데이터 로딩")
print("="*60)

train_raw = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_raw  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
layout    = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
sample    = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

train_raw = train_raw.merge(layout, on='layout_id', how='left')
test_raw  = test_raw.merge(layout,  on='layout_id', how='left')
print(f"train: {train_raw.shape} | test: {test_raw.shape}")

# =============================================================================
# 2. 피처 엔지니어링 함수 모음
# =============================================================================

# ── BASE (Step7과 동일) ───────────────────────────────────────────────────────
BASE_AGG_COLS  = ['order_inflow_15m', 'low_battery_ratio', 'congestion_score',
                  'robot_utilization', 'pack_utilization', 'robot_active',
                  'charge_queue_length', 'max_zone_density']
BASE_AGG_FUNCS = ['mean', 'std', 'max', 'min']


def add_base_features(df):
    df = df.copy()
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
    return df


def add_sc_agg(df, src_df, prefix='sc'):
    """src_df 기준으로 시나리오 집계 계산 후 df에 merge"""
    agg = src_df.groupby('scenario_id')[BASE_AGG_COLS].agg(BASE_AGG_FUNCS)
    agg.columns = [f'{prefix}_{c}_{f}' for c, f in agg.columns]
    agg = agg.reset_index()
    return df.merge(agg, on='scenario_id', how='left')


def add_layout_agg(df, src_df, prefix='layout'):
    agg = src_df.groupby('layout_id')[BASE_AGG_COLS].agg(BASE_AGG_FUNCS)
    agg.columns = [f'{prefix}_{c}_{f}' for c, f in agg.columns]
    agg = agg.reset_index()
    layout_feat_cols = [c for c in agg.columns if c.startswith(prefix+'_') and c != 'layout_id']
    merged = df.merge(agg[['layout_id'] + layout_feat_cols], on='layout_id', how='left', suffixes=('', '_dup'))
    merged.drop(columns=[c for c in merged.columns if c.endswith('_dup')], inplace=True)
    return merged


# ── 실험 A: 시나리오 내 z-score 피처 ─────────────────────────────────────────
ZSCORE_COLS = ['congestion_score', 'low_battery_ratio', 'max_zone_density',
               'pack_utilization', 'robot_utilization', 'order_inflow_15m',
               'charge_queue_length', 'robot_active', 'battery_mean',
               'aisle_traffic_score', 'blocked_path_15m']


def add_zscore_features(df):
    """(현재값 - sc_mean) / (sc_std + 1e-5) — 시나리오 대비 현재 이상 정도"""
    df = df.copy()
    for col in ZSCORE_COLS:
        mean_col = f'sc_{col}_mean'
        std_col  = f'sc_{col}_std'
        if col in df.columns and mean_col in df.columns and std_col in df.columns:
            df[f'z_{col}'] = (df[col] - df[mean_col]) / (df[std_col] + 1e-5)
            # 절댓값도 추가 (방향 무관 이상도)
            df[f'z_{col}_abs'] = df[f'z_{col}'].abs()
    return df


# ── 실험 B: 타임슬롯 순서 복원 + lag/diff 피처 ───────────────────────────────
LAG_COLS = ['congestion_score', 'low_battery_ratio', 'battery_mean',
            'order_inflow_15m', 'robot_active', 'pack_utilization',
            'max_zone_density', 'charge_queue_length', 'robot_utilization',
            'blocked_path_15m', 'aisle_traffic_score']


def add_lag_features(df):
    """
    shift_hour 기준으로 시나리오 내 정렬 후 lag-1, diff-1, rolling-3 피처 추가
    NaN(shift_hour 없는 행)은 정렬 시 맨 뒤로 → 첫 슬롯 lag는 NaN → 0으로 채움
    """
    df = df.copy()

    # 시나리오 내 slot 순서 부여 (shift_hour NaN은 맨 뒤)
    df['_sort_key'] = df['shift_hour'].fillna(999)
    df['slot_rank_tmp'] = df.groupby('scenario_id')['_sort_key'].rank(method='first') - 1
    df.drop(columns=['_sort_key'], inplace=True)

    # shift_hour 기준 정렬
    df = df.sort_values(['scenario_id', 'slot_rank_tmp']).reset_index(drop=True)

    for col in LAG_COLS:
        if col not in df.columns:
            continue
        grp = df.groupby('scenario_id')[col]

        # lag-1: 직전 슬롯 값
        df[f'lag1_{col}'] = grp.shift(1)

        # diff-1: 현재 - 직전 (변화량)
        df[f'diff1_{col}'] = df[col] - df[f'lag1_{col}']

        # rolling mean (window=3, 현재 포함)
        df[f'roll3_mean_{col}'] = grp.transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )

    # lag/diff NaN → 0 (첫 슬롯)
    lag_cols_new = [c for c in df.columns if c.startswith(('lag1_', 'diff1_', 'roll3_'))]
    df[lag_cols_new] = df[lag_cols_new].fillna(0)

    df.drop(columns=['slot_rank_tmp'], inplace=True)
    return df


# ── 실험 C: 타임슬롯 위치 피처 ───────────────────────────────────────────────
def add_timeslot_features(df):
    """
    시나리오 내 슬롯 순서 기반 위치 피처
    slot_rank(0~24), normalized_position(0~1), 초반/후반 플래그, 슬롯 수
    """
    df = df.copy()

    df['_sort_key'] = df['shift_hour'].fillna(999)
    df['slot_rank'] = df.groupby('scenario_id')['_sort_key'].rank(method='first') - 1
    df.drop(columns=['_sort_key'], inplace=True)

    # 시나리오 전체 슬롯 수 (결측 없는 경우 25)
    slot_count = df.groupby('scenario_id')['slot_rank'].transform('count')
    df['slot_count']           = slot_count
    df['normalized_position']  = df['slot_rank'] / (slot_count - 1).clip(lower=1)
    df['is_early_slot']        = (df['slot_rank'] <= 4).astype(np.int8)
    df['is_late_slot']         = (df['slot_rank'] >= 20).astype(np.int8)
    df['is_peak_slot']         = ((df['slot_rank'] >= 8) & (df['slot_rank'] <= 16)).astype(np.int8)
    df['slots_remaining']      = slot_count - df['slot_rank'] - 1

    # shift_hour 자체도 피처로 (NaN → -1)
    df['shift_hour_filled'] = df['shift_hour'].fillna(-1)

    return df


# ── 실험 D: sc_* 집계 확장 ───────────────────────────────────────────────────
EXT_AGG_COLS = BASE_AGG_COLS + [
    'battery_std', 'aisle_traffic_score', 'blocked_path_15m',
    'near_collision_15m', 'fault_count_15m', 'task_reassign_15m',
    'robot_idle', 'robot_charging', 'avg_charge_wait',
    'replenishment_overlap', 'wms_response_time_ms', 'pack_station_count'
]
EXT_AGG_FUNCS = ['mean', 'std', 'max', 'min', 'median',
                 lambda x: x.quantile(0.75) - x.quantile(0.25)]  # IQR


def add_ext_sc_agg(df, src_df):
    """확장된 sc_* 집계 (더 많은 컬럼 + median, IQR)"""
    valid_cols = [c for c in EXT_AGG_COLS if c in src_df.columns]

    # lambda는 이름이 <lambda>로 나와서 별도 처리
    agg_basic = src_df.groupby('scenario_id')[valid_cols].agg(
        ['mean', 'std', 'max', 'min', 'median']
    )
    agg_basic.columns = [f'ext_sc_{c}_{f}' for c, f in agg_basic.columns]

    # IQR 별도 계산
    iqr_vals = src_df.groupby('scenario_id')[valid_cols].agg(
        lambda x: x.quantile(0.75) - x.quantile(0.25)
    )
    iqr_vals.columns = [f'ext_sc_{c}_iqr' for c in iqr_vals.columns]

    agg_all = pd.concat([agg_basic, iqr_vals], axis=1).reset_index()

    # 이미 있는 컬럼과 겹치지 않는 것만 merge
    existing = set(df.columns)
    new_cols = [c for c in agg_all.columns if c not in existing or c == 'scenario_id']
    return df.merge(agg_all[new_cols], on='scenario_id', how='left')


# =============================================================================
# 3. 실험 데이터 준비 함수
# =============================================================================

def prepare_dataset(train_raw, test_raw, experiment_name, add_fns):
    """
    experiment_name: 로그용
    add_fns: 추가 피처 함수 리스트 (베이스 위에 순서대로 적용)
    """
    train = add_base_features(train_raw.copy())
    test  = add_base_features(test_raw.copy())

    # sc_* 베이스 집계 (Step7과 동일 방식)
    train = add_sc_agg(train, train, 'sc')
    test  = add_sc_agg(test,  test,  'sc')

    # layout_agg: train 기준 집계 → train/test 양쪽 merge (Step7 방식)
    layout_agg_df = train.groupby('layout_id')[BASE_AGG_COLS].agg(BASE_AGG_FUNCS)
    layout_agg_df.columns = [f'layout_{c}_{f}' for c, f in layout_agg_df.columns]
    layout_agg_df = layout_agg_df.reset_index()
    layout_feat_cols_local = [c for c in layout_agg_df.columns
                              if c.startswith('layout_') and c != 'layout_id']
    train = train.merge(layout_agg_df[['layout_id'] + layout_feat_cols_local],
                        on='layout_id', how='left', suffixes=('', '_dup'))
    train.drop(columns=[c for c in train.columns if c.endswith('_dup')], inplace=True)
    test  = test.merge(layout_agg_df[['layout_id'] + layout_feat_cols_local],
                       on='layout_id', how='left')

    # 추가 피처 함수 적용
    for fn in add_fns:
        train = fn(train)
        test  = fn(test)

    DROP_COLS    = {'ID', 'layout_id', 'scenario_id', 'layout_type', TARGET,
                    'shift_hour', 'slot_rank_tmp'}
    feature_cols = [c for c in train.columns
                    if c not in DROP_COLS and c in test.columns
                    and train[c].dtype != object]
    feature_cols = list(dict.fromkeys(feature_cols))

    X_train = train[feature_cols].astype(np.float32)
    y_train = np.log1p(train[TARGET].values)
    X_test  = test[feature_cols].astype(np.float32)
    groups  = train['scenario_id'].values

    print(f"  [{experiment_name}] 피처 수: {len(feature_cols)}개")
    return X_train, y_train, X_test, groups, feature_cols


# =============================================================================
# 4. OOF 평가 함수 (LGB 단일 시드)
# =============================================================================

def run_lgb_oof(X_train, y_train, groups, y_true, exp_name):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_preds  = np.zeros(len(X_train))
    fold_scores = []
    best_iters  = []
    t0 = time.time()

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        dtrain = lgb.Dataset(X_tr,  label=y_tr,  free_raw_data=True)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)
        model  = lgb.train(
            LGB_PARAMS, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(999)],
        )
        val_p = np.clip(np.expm1(model.predict(X_val, num_iteration=model.best_iteration)), 0, None)
        score = mean_absolute_error(y_true[val_idx], val_p)
        fold_scores.append(score)
        best_iters.append(model.best_iteration)
        oof_preds[val_idx] = val_p

    oof_score = mean_absolute_error(y_true, oof_preds)
    elapsed   = time.time() - t0
    print(f"  [{exp_name}] OOF MAE: {oof_score:.4f} | "
          f"avg_iter: {np.mean(best_iters):.0f} | {elapsed/60:.1f}min")
    return oof_score, oof_preds


# =============================================================================
# 5. 실험 실행
# =============================================================================
print("\n" + "="*60)
print("실험 시작 (LGB 단일 시드, 피처셋 비교)")
print("="*60)

train_base = add_base_features(train_raw.copy())
test_base  = add_base_features(test_raw.copy())
y_true     = train_raw[TARGET].values

results = {}   # {exp_name: oof_mae}
exp_total_start = time.time()

# ── 베이스라인 (Step7 재현) ───────────────────────────────────────────────────
print("\n[BASE] Step7 베이스라인 재현")
X_tr_b, y_tr_b, X_te_b, grp_b, fc_b = prepare_dataset(
    train_raw, test_raw, 'BASE', add_fns=[]
)
score_base, _ = run_lgb_oof(X_tr_b, y_tr_b, grp_b, y_true, 'BASE')
results['BASE (Step7)'] = score_base

# ── 실험 A: z-score ───────────────────────────────────────────────────────────
print("\n[A] 시나리오 내 z-score 피처")
X_tr_a, y_tr_a, X_te_a, grp_a, fc_a = prepare_dataset(
    train_raw, test_raw, 'A', add_fns=[add_zscore_features]
)
score_a, _ = run_lgb_oof(X_tr_a, y_tr_a, grp_a, y_true, 'A: z-score')
results['A: z-score'] = score_a

# ── 실험 B: lag/diff ──────────────────────────────────────────────────────────
print("\n[B] 타임슬롯 lag/diff 피처")
X_tr_b2, y_tr_b2, X_te_b2, grp_b2, fc_b2 = prepare_dataset(
    train_raw, test_raw, 'B', add_fns=[add_lag_features]
)
score_b, _ = run_lgb_oof(X_tr_b2, y_tr_b2, grp_b2, y_true, 'B: lag/diff')
results['B: lag/diff'] = score_b

# ── 실험 C: timeslot 위치 ─────────────────────────────────────────────────────
print("\n[C] 타임슬롯 위치 피처")
X_tr_c, y_tr_c, X_te_c, grp_c, fc_c = prepare_dataset(
    train_raw, test_raw, 'C', add_fns=[add_timeslot_features]
)
score_c, _ = run_lgb_oof(X_tr_c, y_tr_c, grp_c, y_true, 'C: timeslot pos')
results['C: timeslot pos'] = score_c

# ── 실험 D: sc_* 확장 ─────────────────────────────────────────────────────────
print("\n[D] sc_* 집계 확장 (컬럼+함수 추가)")

def add_ext_sc_agg_wrapper(df):
    # src는 동일 df 내 데이터로 (train/test 각자 자기 시나리오 집계)
    return add_ext_sc_agg(df, df)

X_tr_d, y_tr_d, X_te_d, grp_d, fc_d = prepare_dataset(
    train_raw, test_raw, 'D', add_fns=[add_ext_sc_agg_wrapper]
)
score_d, _ = run_lgb_oof(X_tr_d, y_tr_d, grp_d, y_true, 'D: ext sc_*')
results['D: ext sc_*'] = score_d

# ── 실험 E: A + B + C ─────────────────────────────────────────────────────────
print("\n[E] A + B + C 조합")
X_tr_e, y_tr_e, X_te_e, grp_e, fc_e = prepare_dataset(
    train_raw, test_raw, 'E',
    add_fns=[add_zscore_features, add_lag_features, add_timeslot_features]
)
score_e, _ = run_lgb_oof(X_tr_e, y_tr_e, grp_e, y_true, 'E: A+B+C')
results['E: A+B+C'] = score_e

# ── 실험 F: A + B + C + D (전체) ─────────────────────────────────────────────
print("\n[F] A + B + C + D 전체 조합")
X_tr_f, y_tr_f, X_te_f, grp_f, fc_f = prepare_dataset(
    train_raw, test_raw, 'F',
    add_fns=[add_zscore_features, add_lag_features, add_timeslot_features, add_ext_sc_agg_wrapper]
)
score_f, _ = run_lgb_oof(X_tr_f, y_tr_f, grp_f, y_true, 'F: A+B+C+D')
results['F: A+B+C+D'] = score_f

# =============================================================================
# 6. 결과 비교표
# =============================================================================
total_elapsed = time.time() - exp_total_start
print("\n" + "="*60)
print("피처 엔지니어링 실험 결과 비교")
print("="*60)
print(f"\n{'실험':<22} {'OOF MAE':>10} {'vs BASE':>10} {'판정':>10}")
print("-" * 55)

for name, score in results.items():
    diff  = score - score_base
    judge = ('개선' if diff < -0.002 else ('유지' if abs(diff) <= 0.002 else '악화'))
    marker = ' <-- ' if diff < -0.002 else ''
    print(f"  {name:<20} {score:>10.4f} {diff:>+10.4f} {judge:>10}{marker}")

best_exp  = min(results, key=results.get)
best_score = results[best_exp]
print(f"\n최적 실험: [{best_exp}]  OOF MAE={best_score:.4f}")
print(f"Step7 대비: {best_score - score_base:+.4f}")
print(f"\n총 소요 시간: {total_elapsed/60:.1f}분")
print("\n" + "="*60)
print("다음 단계 권장")
print("="*60)
print("OOF 개선된 실험 조합으로 step10_final.py (3모델×5시드) 작성 후 제출")
print(f"기준: Step7 Public 10.1823 — OOF {score_base:.4f} → Public 환산 약 {score_base+1.497:.4f}")
