# =============================================================================
# 스마트 창고 출고 지연 예측 AI 경진대회
# Step 2: LightGBM + XGBoost + CatBoost 앙상블
# =============================================================================
# 환경: macOS Apple Silicon (M-series)
# - LightGBM : CPU (num_threads=전체 코어)
# - XGBoost  : CPU (tree_method='hist', nthread=전체 코어)
# - CatBoost : CPU (thread_count=전체 코어)
# - PyTorch  : MPS (Apple Silicon GPU) 사용 가능
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)

# ── 디바이스 설정 ─────────────────────────────────────────────────────────────
try:
    import torch
    if torch.backends.mps.is_available():
        TORCH_DEVICE = 'mps'
        print("✅ Apple Silicon GPU (MPS) 감지됨 — PyTorch 모델에서 사용")
    elif torch.cuda.is_available():
        TORCH_DEVICE = 'cuda'
        print("✅ NVIDIA GPU (CUDA) 감지됨")
    else:
        TORCH_DEVICE = 'cpu'
except ImportError:
    TORCH_DEVICE = 'cpu'

N_CPU = os.cpu_count()
print(f"💻 CPU 코어: {N_CPU}개")

DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = DATA_DIR
SEED       = 42
TARGET     = 'avg_delay_minutes_next_30m'

# =============================================================================
# 1. 데이터 로드 & 전처리 (Step 1과 동일)
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


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 이진 플래그
    for col in ['congestion_score', 'blocked_path_15m', 'near_collision_15m',
                'charge_queue_length', 'avg_charge_wait', 'fault_count_15m',
                'replenishment_overlap', 'task_reassign_15m']:
        if col in df.columns:
            df[f'flag_{col}'] = (df[col] > 0).astype(np.int8)

    # 배터리 복합
    df['battery_stress']     = df['low_battery_ratio'] / (df['battery_mean'] + 1e-5)
    df['charge_bottleneck']  = df['charge_queue_length'] * df['avg_charge_wait']
    df['battery_volatility'] = df['battery_std'] / (df['battery_mean'] + 1e-5)
    df['battery_health']     = df['battery_mean'] - df['battery_std']

    # 주문-로봇 균형
    df['order_per_robot']        = df['order_inflow_15m'] / (df['robot_active'] + 1)
    df['order_per_pack_station'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1)
    df['robot_effective_util']   = df['robot_active'] / (df['robot_total'] + 1)
    df['idle_ratio']             = (
        df['robot_idle'] /
        (df['robot_active'] + df['robot_idle'] + df['robot_charging'] + 1)
    )
    df['charging_ratio'] = df['robot_charging'] / (df['robot_total'] + 1)

    # 혼잡도 복합
    df['congestion_x_density'] = df['congestion_score'] * df['max_zone_density']
    df['traffic_severity']     = df['blocked_path_15m'] + df['near_collision_15m'] * 2
    df['aisle_load']           = df['aisle_traffic_score'] * df['congestion_score']

    # layout 파생
    df['layout_type_enc']        = df['layout_type'].map(LAYOUT_TYPE_MAP).fillna(-1).astype(np.int8)
    df['order_per_charger']      = df['order_inflow_15m'] / (df['charger_count'] + 1)
    df['robot_per_floor_area']   = df['robot_total'] / (df['floor_area_sqm'] + 1)
    df['pack_station_per_robot'] = df['pack_station_count'] / (df['robot_total'] + 1)

    # 결측치 플래그
    for col in ['battery_mean', 'low_battery_ratio', 'congestion_score',
                'order_inflow_15m', 'robot_active', 'pack_utilization']:
        if col in df.columns:
            df[f'null_{col}'] = df[col].isna().astype(np.int8)

    return df


train = feature_engineering(train)
test  = feature_engineering(test)

# 집계 피처
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

print(f"피처 수: {len(feature_cols)}개 | train: {X_train.shape} | test: {X_test.shape}")

# =============================================================================
# 2. 모델 파라미터 정의
# =============================================================================

# ── LightGBM ─────────────────────────────────────────────────────────────────
LGB_PARAMS = {
    'objective'        : 'regression_l1',
    'metric'           : 'mae',
    'learning_rate'    : 0.05,
    'num_leaves'       : 127,
    'max_depth'        : -1,
    'min_child_samples': 50,
    'feature_fraction' : 0.8,
    'bagging_fraction' : 0.8,
    'bagging_freq'     : 5,
    'lambda_l1'        : 0.1,
    'lambda_l2'        : 0.1,
    'device_type'      : 'cpu',    # Mac: GPU 미지원
    'num_threads'      : N_CPU,    # CPU 전체 코어
    'verbose'          : -1,
    'seed'             : SEED,
}

# ── XGBoost ──────────────────────────────────────────────────────────────────
# Mac Apple Silicon: CUDA 없음 → tree_method='hist' + nthread=N_CPU
# XGBoost 2.0+에서 device='cuda' 불가, 'cpu' 사용
XGB_PARAMS = {
    'objective'        : 'reg:absoluteerror',  # MAE 직접 최적화
    'eval_metric'      : 'mae',
    'learning_rate'    : 0.05,
    'max_depth'        : 7,
    'min_child_weight' : 50,
    'subsample'        : 0.8,
    'colsample_bytree' : 0.8,
    'reg_alpha'        : 0.1,
    'reg_lambda'       : 0.1,
    'tree_method'      : 'hist',   # 가장 빠른 CPU 알고리즘
    'device'           : 'cpu',    # Mac: CUDA 없음
    'nthread'          : N_CPU,    # CPU 전체 코어
    'seed'             : SEED,
    'verbosity'        : 0,
}

# ── CatBoost ─────────────────────────────────────────────────────────────────
# Mac Apple Silicon: CUDA 없음 → task_type='CPU', thread_count=N_CPU
CAT_PARAMS = {
    'loss_function'     : 'MAE',
    'eval_metric'       : 'MAE',
    'learning_rate'     : 0.05,
    'depth'             : 8,
    'l2_leaf_reg'       : 3.0,
    'min_data_in_leaf'  : 50,
    'subsample'         : 0.8,
    'task_type'         : 'CPU',   # Mac: GPU(CUDA) 없음
    'thread_count'      : N_CPU,   # CPU 전체 코어
    'random_seed'       : SEED,
    'verbose'           : False,
}

# =============================================================================
# 3. 앙상블 학습 함수
# =============================================================================
N_SPLITS   = 5
MAX_ROUNDS = 3000
EARLY_STOP = 150
gkf        = GroupKFold(n_splits=N_SPLITS)


def train_lgbm(X_tr, y_tr, X_val, y_val, X_te):
    dtrain = lgb.Dataset(X_tr,  label=y_tr,  free_raw_data=True)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)
    model  = lgb.train(
        LGB_PARAMS, dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(999)],
    )
    val_p  = np.clip(np.expm1(model.predict(X_val, num_iteration=model.best_iteration)), 0, None)
    test_p = np.clip(np.expm1(model.predict(X_te,  num_iteration=model.best_iteration)), 0, None)
    return val_p, test_p, model.best_iteration


def train_xgb(X_tr, y_tr, X_val, y_val, X_te):
    dtrain = xgb.DMatrix(X_tr,  label=y_tr)
    dvalid = xgb.DMatrix(X_val, label=y_val)
    dte    = xgb.DMatrix(X_te)
    model  = xgb.train(
        XGB_PARAMS, dtrain,
        num_boost_round=MAX_ROUNDS,
        evals=[(dvalid, 'val')],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=False,
    )
    val_p  = np.clip(np.expm1(model.predict(dvalid, iteration_range=(0, model.best_iteration))), 0, None)
    test_p = np.clip(np.expm1(model.predict(dte,    iteration_range=(0, model.best_iteration))), 0, None)
    return val_p, test_p, model.best_iteration


def train_cat(X_tr, y_tr, X_val, y_val, X_te):
    model = CatBoostRegressor(
        iterations=MAX_ROUNDS,
        early_stopping_rounds=EARLY_STOP,
        **CAT_PARAMS,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=False,
    )
    val_p  = np.clip(np.expm1(model.predict(X_val)), 0, None)
    test_p = np.clip(np.expm1(model.predict(X_te)),  0, None)
    return val_p, test_p, model.best_iteration_


# =============================================================================
# 4. 모델별 GroupKFold 학습
# =============================================================================
models_cfg = [
    ('LightGBM', train_lgbm),
    ('XGBoost',  train_xgb),
    ('CatBoost', train_cat),
]

# 앙상블 가중치 (추후 OOF MAE 기반 조정 가능)
WEIGHTS = {'LightGBM': 0.4, 'XGBoost': 0.3, 'CatBoost': 0.3}

all_oof   = {}   # {모델명: oof_preds 배열}
all_test  = {}   # {모델명: test_preds 배열}
all_score = {}   # {모델명: OOF MAE}

for model_name, train_fn in models_cfg:
    print("\n" + "="*60)
    print(f"📦 {model_name} GroupKFold 학습")
    print("="*60)

    oof_preds  = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr,  X_val  = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr,  y_val  = y_train[tr_idx],       y_train[val_idx]

        val_p, test_p, best_iter = train_fn(X_tr, y_tr, X_val, y_val, X_test)

        true_val = train[TARGET].values[val_idx]
        score    = mean_absolute_error(true_val, val_p)
        fold_scores.append(score)

        print(f"  Fold {fold+1} MAE: {score:.4f} | best_iter: {best_iter}")

        oof_preds[val_idx] = val_p
        test_preds        += test_p / N_SPLITS

    oof_score = mean_absolute_error(train[TARGET].values, oof_preds)
    all_oof[model_name]   = oof_preds
    all_test[model_name]  = test_preds
    all_score[model_name] = oof_score

    print(f"\n  ✅ {model_name} OOF MAE: {oof_score:.4f}")
    print(f"     Fold별: {[round(s, 4) for s in fold_scores]}")

# =============================================================================
# 5. 앙상블 결합
# =============================================================================
print("\n" + "="*60)
print("🔗 앙상블 결합")
print("="*60)

# ── 5-1. 단순 평균 앙상블 ─────────────────────────────────────────────────────
oof_avg  = np.mean([all_oof[m]  for m in all_oof],  axis=0)
test_avg = np.mean([all_test[m] for m in all_test], axis=0)
score_avg = mean_absolute_error(train[TARGET].values, oof_avg)
print(f"단순 평균 앙상블 OOF MAE : {score_avg:.4f}")

# ── 5-2. OOF MAE 기반 역가중 앙상블 ──────────────────────────────────────────
# MAE가 낮을수록 가중치 높게
inv_scores = {m: 1.0 / all_score[m] for m in all_score}
total_inv  = sum(inv_scores.values())
w_mae      = {m: inv_scores[m] / total_inv for m in inv_scores}
print(f"\nOOF 역가중치:")
for m, w in w_mae.items():
    print(f"  {m}: {w:.3f}  (OOF MAE={all_score[m]:.4f})")

oof_wmae  = sum(w_mae[m] * all_oof[m]  for m in w_mae)
test_wmae = sum(w_mae[m] * all_test[m] for m in w_mae)
score_wmae = mean_absolute_error(train[TARGET].values, oof_wmae)
print(f"\n역가중 앙상블 OOF MAE : {score_wmae:.4f}")

# ── 5-3. 고정 가중치 앙상블 ───────────────────────────────────────────────────
oof_fixed  = sum(WEIGHTS[m] * all_oof[m]  for m in WEIGHTS if m in all_oof)
test_fixed = sum(WEIGHTS[m] * all_test[m] for m in WEIGHTS if m in all_test)
score_fixed = mean_absolute_error(train[TARGET].values, oof_fixed)
print(f"고정 가중치 앙상블 OOF MAE: {score_fixed:.4f}  (weights={WEIGHTS})")

# ── 최적 앙상블 선택 ──────────────────────────────────────────────────────────
best_method = min([
    ('simple_avg',   score_avg,   test_avg),
    ('inv_mae_w',    score_wmae,  test_wmae),
    ('fixed_w',      score_fixed, test_fixed),
], key=lambda x: x[1])

print(f"\n🏆 최적 앙상블: {best_method[0]}  OOF MAE={best_method[1]:.4f}")
final_test_preds = best_method[2]

# =============================================================================
# 6. 개별 모델 성능 요약
# =============================================================================
print("\n" + "="*60)
print("📊 모델별 OOF MAE 요약")
print("="*60)
for m, s in sorted(all_score.items(), key=lambda x: x[1]):
    print(f"  {m:12s} : {s:.4f}")
print(f"  {'앙상블(최적)':12s} : {best_method[1]:.4f}")

# =============================================================================
# 7. 제출 파일 생성
# =============================================================================
print("\n" + "="*60)
print("7. 제출 파일 생성")
print("="*60)

# 최적 앙상블 제출
submission = sample.copy()
submission[TARGET] = np.clip(final_test_preds, 0, None)
out_path = os.path.join(OUTPUT_DIR, 'submission_step2_ensemble.csv')
submission.to_csv(out_path, index=False)
print(f"앙상블 제출 저장 → {out_path}")

# 개별 모델 제출도 저장
for m, preds in all_test.items():
    sub = sample.copy()
    sub[TARGET] = np.clip(preds, 0, None)
    fname = f"submission_step2_{m.lower()}.csv"
    sub.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
    print(f"{m} 제출 저장 → {fname}")

print(f"\n예측값 분포 (앙상블):")
print(submission[TARGET].describe().round(3).to_string())
print(f"\n✅ 최종 앙상블 OOF MAE: {best_method[1]:.4f}")
