# =============================================================================
# 스마트 창고 출고 지연 예측 AI 경진대회
# Step 7: LightGBM + XGBoost + CatBoost × 멀티시드 앙상블
# =============================================================================
# 전략: 3 모델 × 5 시드 = 15 모델 인스턴스 × 5 폴드 = 75회 학습
#       모델 다양성(Step2) + 시드 분산 감소(Step5) 동시 결합
# =============================================================================
# 실행 방법 (맥 백그라운드/슬립 방지):
#   caffeinate -i nohup python step7_multi_seed_ensemble.py > step7_output.log 2>&1 &
# 로그 확인:
#   tail -f step7_output.log
# =============================================================================

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)

N_CPU = os.cpu_count()
print(f"CPU 코어: {N_CPU}개")

DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = DATA_DIR
TARGET     = 'avg_delay_minutes_next_30m'

# 멀티시드 설정
SEEDS      = [42, 123, 456, 789, 2024]
N_SPLITS   = 5
MAX_ROUNDS = 3000
EARLY_STOP = 150

print(f"시드 수: {len(SEEDS)}개 x 3 모델 = {len(SEEDS)*3}개 인스턴스")
print(f"총 학습 횟수: {len(SEEDS)*3*N_SPLITS}회 (모델x시드x폴드)")

# =============================================================================
# 1. 데이터 로드 & 전처리
# =============================================================================
print("\n" + "="*60)
print("1. 데이터 로딩 & 전처리")
print("="*60)

train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

train = train.merge(layout, on='layout_id', how='left')
test  = test.merge(layout, on='layout_id', how='left')

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
layout_feat_cols = [c for c in train_layout_agg.columns if c.startswith('layout_') and c != 'layout_id']

train = train.merge(train_sc_agg, on='scenario_id', how='left')
test  = test.merge(test_sc_agg,   on='scenario_id', how='left')
train = train.merge(train_layout_agg[['layout_id'] + layout_feat_cols], on='layout_id', how='left', suffixes=('', '_dup'))
test  = test.merge(train_layout_agg[['layout_id'] + layout_feat_cols],  on='layout_id', how='left')
train.drop(columns=[c for c in train.columns if c.endswith('_dup')], inplace=True)

DROP_COLS    = {'ID', 'layout_id', 'scenario_id', 'layout_type', TARGET}
feature_cols = [c for c in train.columns if c not in DROP_COLS and c in test.columns]

X_train = train[feature_cols].astype(np.float32)
y_train = np.log1p(train[TARGET].values)
X_test  = test[feature_cols].astype(np.float32)
groups  = train['scenario_id'].values
y_true  = train[TARGET].values

print(f"피처 수: {len(feature_cols)}개 | train: {X_train.shape} | test: {X_test.shape}")

gkf = GroupKFold(n_splits=N_SPLITS)

# =============================================================================
# 2. 모델 베이스 파라미터
# =============================================================================

LGB_BASE = {
    'objective': 'regression_l1', 'metric': 'mae',
    'learning_rate': 0.05, 'num_leaves': 127, 'max_depth': -1,
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
# 3. 멀티시드 × 멀티모델 학습
# =============================================================================

all_model_oof   = {'LightGBM': {}, 'XGBoost': {}, 'CatBoost': {}}
all_model_test  = {'LightGBM': {}, 'XGBoost': {}, 'CatBoost': {}}
all_model_score = {'LightGBM': {}, 'XGBoost': {}, 'CatBoost': {}}

total_start = time.time()
run_count   = 0
total_runs  = len(SEEDS) * 3

# LightGBM
print("\n" + "="*60)
print("LightGBM x 멀티시드")
print("="*60)

for seed in SEEDS:
    seed_start = time.time()
    params = {**LGB_BASE, 'seed': seed}
    oof_preds  = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        dtrain = lgb.Dataset(X_tr,  label=y_tr,  free_raw_data=True)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)
        model  = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(999)],
        )
        val_p  = np.clip(np.expm1(model.predict(X_val,  num_iteration=model.best_iteration)), 0, None)
        test_p = np.clip(np.expm1(model.predict(X_test, num_iteration=model.best_iteration)), 0, None)
        oof_preds[val_idx] = val_p
        test_preds        += test_p / N_SPLITS

    oof_score = mean_absolute_error(y_true, oof_preds)
    all_model_oof['LightGBM'][seed]   = oof_preds.copy()
    all_model_test['LightGBM'][seed]  = test_preds.copy()
    all_model_score['LightGBM'][seed] = oof_score
    run_count += 1
    elapsed = time.time() - seed_start
    print(f"  seed={seed:5d} | OOF MAE: {oof_score:.4f} | {elapsed/60:.1f}min | [{run_count}/{total_runs}]")

s = list(all_model_score['LightGBM'].values())
print(f"  LGB 평균: {np.mean(s):.4f} +/- {np.std(s):.4f}")

# XGBoost
print("\n" + "="*60)
print("XGBoost x 멀티시드")
print("="*60)

for seed in SEEDS:
    seed_start = time.time()
    params = {**XGB_BASE, 'seed': seed}
    oof_preds  = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        dte    = xgb.DMatrix(X_test)
        model  = xgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            evals=[(dvalid, 'val')], early_stopping_rounds=EARLY_STOP, verbose_eval=False,
        )
        val_p  = np.clip(np.expm1(model.predict(dvalid, iteration_range=(0, model.best_iteration))), 0, None)
        test_p = np.clip(np.expm1(model.predict(dte,    iteration_range=(0, model.best_iteration))), 0, None)
        oof_preds[val_idx] = val_p
        test_preds        += test_p / N_SPLITS

    oof_score = mean_absolute_error(y_true, oof_preds)
    all_model_oof['XGBoost'][seed]   = oof_preds.copy()
    all_model_test['XGBoost'][seed]  = test_preds.copy()
    all_model_score['XGBoost'][seed] = oof_score
    run_count += 1
    elapsed = time.time() - seed_start
    print(f"  seed={seed:5d} | OOF MAE: {oof_score:.4f} | {elapsed/60:.1f}min | [{run_count}/{total_runs}]")

s = list(all_model_score['XGBoost'].values())
print(f"  XGB 평균: {np.mean(s):.4f} +/- {np.std(s):.4f}")

# CatBoost
print("\n" + "="*60)
print("CatBoost x 멀티시드")
print("="*60)

for seed in SEEDS:
    seed_start = time.time()
    cat_params = {**CAT_BASE, 'random_seed': seed}
    oof_preds  = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = CatBoostRegressor(iterations=MAX_ROUNDS, early_stopping_rounds=EARLY_STOP, **cat_params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose=False)

        val_p  = np.clip(np.expm1(model.predict(X_val)),  0, None)
        test_p = np.clip(np.expm1(model.predict(X_test)), 0, None)
        oof_preds[val_idx] = val_p
        test_preds        += test_p / N_SPLITS

    oof_score = mean_absolute_error(y_true, oof_preds)
    all_model_oof['CatBoost'][seed]   = oof_preds.copy()
    all_model_test['CatBoost'][seed]  = test_preds.copy()
    all_model_score['CatBoost'][seed] = oof_score
    run_count += 1
    elapsed = time.time() - seed_start
    print(f"  seed={seed:5d} | OOF MAE: {oof_score:.4f} | {elapsed/60:.1f}min | [{run_count}/{total_runs}]")

s = list(all_model_score['CatBoost'].values())
print(f"  CatBoost 평균: {np.mean(s):.4f} +/- {np.std(s):.4f}")

# =============================================================================
# 4. 앙상블 결합
# =============================================================================
print("\n" + "="*60)
print("앙상블 결합")
print("="*60)

# 모델별 시드 평균
model_oof_avg  = {}
model_test_avg = {}
model_oof_mae  = {}

for model_name in ['LightGBM', 'XGBoost', 'CatBoost']:
    oof_stack  = np.stack(list(all_model_oof[model_name].values()),  axis=0)
    test_stack = np.stack(list(all_model_test[model_name].values()), axis=0)
    oof_mean   = oof_stack.mean(axis=0)
    test_mean  = test_stack.mean(axis=0)
    mae        = mean_absolute_error(y_true, oof_mean)
    model_oof_avg[model_name]  = oof_mean
    model_test_avg[model_name] = test_mean
    model_oof_mae[model_name]  = mae
    print(f"  {model_name:12s} 시드 평균 OOF MAE: {mae:.4f}")

# 전체 15개 단순 평균
all_oof_list  = [all_model_oof[m][s]  for m in ['LightGBM','XGBoost','CatBoost'] for s in SEEDS]
all_test_list = [all_model_test[m][s] for m in ['LightGBM','XGBoost','CatBoost'] for s in SEEDS]
oof_all_avg   = np.mean(all_oof_list,  axis=0)
test_all_avg  = np.mean(all_test_list, axis=0)
mae_all_avg   = mean_absolute_error(y_true, oof_all_avg)
print(f"\n  전체 단순 평균 (15개) OOF MAE: {mae_all_avg:.4f}")

# 역MAE 가중 앙상블
inv_scores = {m: 1.0 / model_oof_mae[m] for m in model_oof_mae}
total_inv  = sum(inv_scores.values())
w_mae      = {m: inv_scores[m] / total_inv for m in inv_scores}
print(f"\n  모델 간 역MAE 가중치:")
for m, w in w_mae.items():
    print(f"    {m:12s}: {w:.4f}  (OOF MAE={model_oof_mae[m]:.4f})")

oof_wmae  = sum(w_mae[m] * model_oof_avg[m]  for m in w_mae)
test_wmae = sum(w_mae[m] * model_test_avg[m] for m in w_mae)
mae_wmae  = mean_absolute_error(y_true, oof_wmae)
print(f"  역MAE 가중 앙상블 OOF MAE: {mae_wmae:.4f}")

candidates = [
    ('all_avg_15', mae_all_avg, test_all_avg),
    ('inv_mae_w',  mae_wmae,    test_wmae),
]
best = min(candidates, key=lambda x: x[1])
print(f"\n최적 앙상블: {best[0]}  OOF MAE={best[1]:.4f}")
final_test_preds = best[2]

# =============================================================================
# 5. 결과 요약
# =============================================================================
print("\n" + "="*60)
print("전체 결과 요약")
print("="*60)

print("\n[개별 시드별 OOF MAE]")
for model_name in ['LightGBM', 'XGBoost', 'CatBoost']:
    scores = all_model_score[model_name]
    score_vals = list(scores.values())
    print(f"  {model_name}:")
    for seed, sc in scores.items():
        print(f"    seed={seed:5d}: {sc:.4f}")
    print(f"    => 평균: {np.mean(score_vals):.4f} | std: {np.std(score_vals):.4f}")

print("\n[모델별 시드 평균 OOF MAE]")
for m, mae in sorted(model_oof_mae.items(), key=lambda x: x[1]):
    print(f"  {m:12s}: {mae:.4f}")

print("\n[앙상블 비교]")
for name, mae, _ in candidates:
    flag = " <- 최적" if name == best[0] else ""
    print(f"  {name:20s}: {mae:.4f}{flag}")

total_elapsed = time.time() - total_start
print(f"\n총 소요 시간: {total_elapsed/60:.1f}분")

# =============================================================================
# 6. 제출 파일 저장
# =============================================================================
print("\n" + "="*60)
print("6. 제출 파일 저장")
print("="*60)

submission = sample.copy()
submission[TARGET] = np.clip(final_test_preds, 0, None)
out_path = os.path.join(OUTPUT_DIR, 'submission_step7_multi_seed_ensemble.csv')
submission.to_csv(out_path, index=False)
print(f"최적 앙상블 저장 -> {out_path}")

for name, mae, preds in candidates:
    sub = sample.copy()
    sub[TARGET] = np.clip(preds, 0, None)
    fname = f"submission_step7_{name}.csv"
    sub.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
    print(f"  백업: {fname}  (OOF MAE={mae:.4f})")

print(f"\n제출 파일: submission_step7_multi_seed_ensemble.csv")
print(f"OOF MAE: {best[1]:.4f}")
print(f"\n예측값 분포:")
print(submission[TARGET].describe().round(3).to_string())
