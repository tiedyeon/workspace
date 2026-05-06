# =============================================================================
# 스마트 창고 출고 지연 예측 AI 경진대회
# Step 6: 피처 선택 (Feature Selection)
# =============================================================================
# 실험 목적: 피처만 줄이고 나머지는 Step2와 완전 동일하게 유지
# → 피처 축소 효과만 순수하게 측정
#
# Step2와 동일하게 유지:
#   - 파라미터: Step2 LightGBM 기본 파라미터 (lr=0.05, num_leaves=127)
#   - GroupKFold n_splits=5, seed=42
#
# 변경점:
#   - Feature Importance 기반으로 상위 N개만 선택
#   - TOP_N = 120 (197개 → 120개, 약 39% 축소)
#   - 노이즈 피처(날씨/IT/환경) 제거 효과 기대
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
SEED       = 42   # Step2와 동일
TOP_N      = 120  # 상위 몇 개 피처를 쓸지

print(f"💻 CPU 코어: {N_CPU}개 | 선택 피처 수: TOP {TOP_N}개")

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
all_feature_cols = [c for c in train.columns if c not in DROP_COLS and c in test.columns]
print(f"전체 피처 수: {len(all_feature_cols)}개")

# =============================================================================
# 2. 피처 중요도 계산 (전체 피처로 빠르게 1회 학습)
# =============================================================================
print("\n" + "="*60)
print("2. 피처 중요도 계산 (전체 피처, 빠른 학습)")
print("="*60)

X_all   = train[all_feature_cols].astype(np.float32)
y_all   = np.log1p(train[TARGET].values)
groups  = train['scenario_id'].values
X_test_all = test[all_feature_cols].astype(np.float32)

# 빠른 중요도 계산용 파라미터 (3-fold, 적은 rounds)
imp_params = {
    'objective': 'regression_l1', 'metric': 'mae',
    'learning_rate': 0.05, 'num_leaves': 127,
    'min_child_samples': 50, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5,
    'device_type': 'cpu', 'num_threads': N_CPU,
    'verbose': -1, 'seed': SEED,
}

gkf3 = GroupKFold(n_splits=3)
importances = np.zeros(len(all_feature_cols))

for tr_idx, val_idx in gkf3.split(X_all, y_all, groups):
    dtrain = lgb.Dataset(X_all.iloc[tr_idx], label=y_all[tr_idx], free_raw_data=True)
    dvalid = lgb.Dataset(X_all.iloc[val_idx], label=y_all[val_idx],
                         reference=dtrain, free_raw_data=True)
    model = lgb.train(
        imp_params, dtrain, num_boost_round=500,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(9999)],
    )
    importances += model.feature_importance('gain') / 3

# 중요도 정렬
imp_df = pd.DataFrame({'feature': all_feature_cols, 'importance': importances})
imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)

print(f"\n상위 20개 피처:")
print(imp_df.head(20).to_string(index=False))
print(f"\n하위 20개 피처 (제거 대상):")
print(imp_df.tail(20).to_string(index=False))

# =============================================================================
# 3. 상위 TOP_N 피처 선택
# =============================================================================
selected_features = imp_df.head(TOP_N)['feature'].tolist()
print(f"\n선택된 피처: {len(selected_features)}개 / 전체 {len(all_feature_cols)}개")
print(f"제거된 피처: {len(all_feature_cols) - len(selected_features)}개")

# 제거된 피처 카테고리 확인
removed = imp_df.tail(len(all_feature_cols) - TOP_N)['feature'].tolist()
weather_removed = [f for f in removed if any(w in f for w in
    ['temp', 'humidity', 'wind', 'rain', 'precip', 'lighting', 'noise',
     'vibration', 'air_quality', 'co2', 'hvac'])]
it_removed = [f for f in removed if any(w in f for w in
    ['wifi', 'scanner', 'ups', 'network', 'wms', 'barcode'])]
print(f"  날씨/환경 피처 제거: {len(weather_removed)}개")
print(f"  IT/시스템 피처 제거: {len(it_removed)}개")

X_train = train[selected_features].astype(np.float32)
X_test  = test[selected_features].astype(np.float32)

# =============================================================================
# 4. Step2와 동일한 파라미터로 학습
# =============================================================================
print("\n" + "="*60)
print(f"3. LGB 학습 (Step2 동일 파라미터, 피처만 {TOP_N}개)")
print("="*60)

LGB_PARAMS = {
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
    'seed'             : SEED,
}

N_SPLITS   = 5
MAX_ROUNDS = 3000
EARLY_STOP = 150
gkf        = GroupKFold(n_splits=N_SPLITS)
oof_preds  = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
fold_scores= []
best_iters = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_all, groups)):
    print(f"  [Fold {fold+1}/{N_SPLITS}]", end=' ')

    dtrain = lgb.Dataset(X_train.iloc[tr_idx], label=y_all[tr_idx], free_raw_data=True)
    dvalid = lgb.Dataset(X_train.iloc[val_idx], label=y_all[val_idx],
                         reference=dtrain, free_raw_data=True)

    model = lgb.train(
        LGB_PARAMS, dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(9999)],
    )

    val_pred = np.clip(np.expm1(model.predict(X_train.iloc[val_idx],
                                               num_iteration=model.best_iteration)), 0, None)
    te_pred  = np.clip(np.expm1(model.predict(X_test,
                                               num_iteration=model.best_iteration)), 0, None)

    score = mean_absolute_error(train[TARGET].values[val_idx], val_pred)
    fold_scores.append(score)
    best_iters.append(model.best_iteration)
    print(f"MAE: {score:.4f} | best_iter: {model.best_iteration}")

    oof_preds[val_idx] = val_pred
    test_preds        += te_pred / N_SPLITS

oof_score = mean_absolute_error(train[TARGET].values, oof_preds)

print("\n" + "="*60)
print(f"✅ Step6 OOF MAE  : {oof_score:.4f}")
print(f"   Fold별         : {[round(s, 4) for s in fold_scores]}")
print(f"   평균 best_iter : {int(np.mean(best_iters))}")
print(f"   Step2 OOF (비교): 8.7020")
print(f"   Step2 Public   : 10.1915")
print("="*60)

submission = sample.copy()
submission[TARGET] = np.clip(test_preds, 0, None)
out_path = os.path.join(OUTPUT_DIR, 'submission_step6_feature_selection.csv')
submission.to_csv(out_path, index=False)

print(f"\n저장 완료 → submission_step6_feature_selection.csv")
print(f"예측값 분포:")
print(submission[TARGET].describe().round(3).to_string())
