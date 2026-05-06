# =============================================================================
# 스마트 창고 출고 지연 예측 AI 경진대회
# Step 5: 시드 앙상블 (Seed Averaging)
# =============================================================================
# 실험 목적: 시드만 다르게 하고 나머지는 Step2와 완전 동일하게 유지
# → 시드 앙상블 효과만 순수하게 측정
#
# Step2와 동일하게 유지:
#   - 피처: 197개 (Step1/2와 동일)
#   - 파라미터: Step2 LightGBM 기본 파라미터
#   - GroupKFold n_splits=5
#
# 변경점:
#   - SEEDS = [42, 123, 456, 789, 1024, 2024, 7, 777, 314, 99] (10개 시드)
#   - 각 시드별로 LGB 5-fold 학습 후 예측값 평균
# =============================================================================

import os, warnings, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

warnings.filterwarnings('ignore')

N_CPU      = os.cpu_count()
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = DATA_DIR
TARGET     = 'avg_delay_minutes_next_30m'

# ── 시드 목록 ─────────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024, 2024, 7, 777, 314, 99]
print(f"💻 CPU 코어: {N_CPU}개 | 시드 수: {len(SEEDS)}개")
print(f"총 학습 횟수: {len(SEEDS)} seeds × 5 folds = {len(SEEDS)*5}회")

# =============================================================================
# 1. 데이터 로드 & 피처 엔지니어링 (Step2와 완전 동일)
# =============================================================================
print("\n" + "="*60)
print("1. 데이터 로딩 & 피처 엔지니어링 (Step2 동일)")
print("="*60)

train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

train = train.merge(layout, on='layout_id', how='left')
test  = test.merge(layout,  on='layout_id', how='left')

LAYOUT_TYPE_MAP = {'grid': 0, 'hybrid': 1, 'narrow': 2, 'hub_spoke': 3}


def feature_engineering(df):
    df = df.copy()
    for col in ['congestion_score', 'blocked_path_15m', 'near_collision_15m',
                'charge_queue_length', 'avg_charge_wait', 'fault_count_15m',
                'replenishment_overlap', 'task_reassign_15m']:
        if col in df.columns:
            df[f'flag_{col}'] = (df[col] > 0).astype(np.int8)

    df['battery_stress']     = df['low_battery_ratio'] / (df['battery_mean'] + 1e-5)
    df['charge_bottleneck']  = df['charge_queue_length'] * df['avg_charge_wait']
    df['battery_volatility'] = df['battery_std'] / (df['battery_mean'] + 1e-5)
    df['battery_health']     = df['battery_mean'] - df['battery_std']
    df['order_per_robot']        = df['order_inflow_15m'] / (df['robot_active'] + 1)
    df['order_per_pack_station'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1)
    df['robot_effective_util']   = df['robot_active'] / (df['robot_total'] + 1)
    df['idle_ratio']  = df['robot_idle'] / (df['robot_active'] + df['robot_idle'] + df['robot_charging'] + 1)
    df['charging_ratio']       = df['robot_charging'] / (df['robot_total'] + 1)
    df['congestion_x_density'] = df['congestion_score'] * df['max_zone_density']
    df['traffic_severity']     = df['blocked_path_15m'] + df['near_collision_15m'] * 2
    df['aisle_load']           = df['aisle_traffic_score'] * df['congestion_score']
    df['layout_type_enc']        = df['layout_type'].map(LAYOUT_TYPE_MAP).fillna(-1).astype(np.int8)
    df['order_per_charger']      = df['order_inflow_15m'] / (df['charger_count'] + 1)
    df['robot_per_floor_area']   = df['robot_total'] / (df['floor_area_sqm'] + 1)
    df['pack_station_per_robot'] = df['pack_station_count'] / (df['robot_total'] + 1)
    for col in ['battery_mean', 'low_battery_ratio', 'congestion_score',
                'order_inflow_15m', 'robot_active', 'pack_utilization']:
        if col in df.columns:
            df[f'null_{col}'] = df[col].isna().astype(np.int8)
    return df


train = feature_engineering(train)
test  = feature_engineering(test)

AGG_COLS  = ['order_inflow_15m', 'low_battery_ratio', 'congestion_score',
             'robot_utilization', 'pack_utilization', 'robot_active',
             'charge_queue_length', 'max_zone_density']
AGG_FUNCS = ['mean', 'std', 'max', 'min']


def make_group_agg(df, group_col, prefix):
    agg = df.groupby(group_col)[AGG_COLS].agg(AGG_FUNCS)
    agg.columns = [f'{prefix}_{c}_{f}' for c, f in agg.columns]
    return agg.reset_index()


train_sc_agg     = make_group_agg(train, 'scenario_id', 'sc')
test_sc_agg      = make_group_agg(test,  'scenario_id', 'sc')
train_layout_agg = make_group_agg(train, 'layout_id',   'layout')
layout_feat_cols = [c for c in train_layout_agg.columns
                    if c.startswith('layout_') and c != 'layout_id']

train = train.merge(train_sc_agg, on='scenario_id', how='left')
test  = test.merge(test_sc_agg,   on='scenario_id', how='left')
train = train.merge(train_layout_agg[['layout_id'] + layout_feat_cols],
                    on='layout_id', how='left', suffixes=('', '_dup'))
test  = test.merge(train_layout_agg[['layout_id'] + layout_feat_cols],
                   on='layout_id', how='left')
train.drop(columns=[c for c in train.columns if c.endswith('_dup')], inplace=True)

DROP_COLS    = {'ID', 'layout_id', 'scenario_id', 'layout_type', TARGET}
feature_cols = [c for c in train.columns if c not in DROP_COLS and c in test.columns]

X_train = train[feature_cols].astype(np.float32)
y_train = np.log1p(train[TARGET].values)
X_test  = test[feature_cols].astype(np.float32)
groups  = train['scenario_id'].values

print(f"피처 수: {len(feature_cols)}개 (Step2와 동일)")

# =============================================================================
# 2. Step2와 동일한 LGB 파라미터
# =============================================================================
BASE_LGB_PARAMS = {
    'objective'        : 'regression_l1',
    'metric'           : 'mae',
    'learning_rate'    : 0.05,      # Step2와 동일
    'num_leaves'       : 127,       # Step2와 동일
    'max_depth'        : -1,
    'min_child_samples': 50,
    'feature_fraction' : 0.8,
    'bagging_fraction' : 0.8,
    'bagging_freq'     : 5,
    'lambda_l1'        : 0.1,
    'lambda_l2'        : 0.1,
    'device_type'      : 'cpu',
    'num_threads'      : N_CPU,
    'verbose'          : -1,
}

# =============================================================================
# 3. 시드 앙상블 학습
# =============================================================================
print("\n" + "="*60)
print(f"2. 시드 앙상블 학습 ({len(SEEDS)}개 시드 × 5 fold)")
print("="*60)

N_SPLITS   = 5
MAX_ROUNDS = 3000
EARLY_STOP = 150

gkf = GroupKFold(n_splits=N_SPLITS)

# 시드별 OOF/테스트 예측 누적
all_oof_preds  = np.zeros(len(X_train))
all_test_preds = np.zeros(len(X_test))
seed_oof_scores = []

for seed_idx, seed in enumerate(SEEDS):
    params = {**BASE_LGB_PARAMS, 'seed': seed}

    oof_preds  = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        dtrain = lgb.Dataset(X_train.iloc[tr_idx], label=y_train[tr_idx], free_raw_data=True)
        dvalid = lgb.Dataset(X_train.iloc[val_idx], label=y_train[val_idx],
                             reference=dtrain, free_raw_data=True)

        model = lgb.train(
            params, dtrain,
            num_boost_round=MAX_ROUNDS,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                       lgb.log_evaluation(9999)],
        )

        val_pred = np.clip(np.expm1(model.predict(X_train.iloc[val_idx],
                                                    num_iteration=model.best_iteration)), 0, None)
        te_pred  = np.clip(np.expm1(model.predict(X_test,
                                                    num_iteration=model.best_iteration)), 0, None)

        oof_preds[val_idx] = val_pred
        test_preds        += te_pred / N_SPLITS

    seed_score = mean_absolute_error(train[TARGET].values, oof_preds)
    seed_oof_scores.append(seed_score)
    print(f"  Seed {seed:4d} OOF MAE: {seed_score:.4f}")

    # 시드별 예측을 동등하게 평균
    all_oof_preds  += oof_preds  / len(SEEDS)
    all_test_preds += test_preds / len(SEEDS)

# =============================================================================
# 4. 최종 결과
# =============================================================================
final_oof_score = mean_absolute_error(train[TARGET].values, all_oof_preds)

print("\n" + "="*60)
print(f"✅ 시드 앙상블 OOF MAE : {final_oof_score:.4f}")
print(f"   개별 시드 OOF 평균  : {np.mean(seed_oof_scores):.4f}")
print(f"   개별 시드 OOF 범위  : {min(seed_oof_scores):.4f} ~ {max(seed_oof_scores):.4f}")
print(f"   Step2 OOF (비교)   : 8.7020")
print(f"   Step2 Public (비교): 10.1915")
print("="*60)

submission = sample.copy()
submission[TARGET] = np.clip(all_test_preds, 0, None)
out_path = os.path.join(OUTPUT_DIR, 'submission_step5_seed_ensemble.csv')
submission.to_csv(out_path, index=False)

print(f"\n저장 완료 → submission_step5_seed_ensemble.csv")
print(f"예측값 분포:")
print(submission[TARGET].describe().round(3).to_string())
