# =============================================================================
# 스마트 창고 출고 지연 예측 AI 경진대회
# Step 1: LightGBM 베이스라인
# =============================================================================
# 환경: macOS Apple Silicon (M-series)
# - LightGBM/XGBoost/CatBoost : Mac GPU 미지원 → CPU 최대 스레드 활용
# - PyTorch 기반 모델          : MPS (Metal Performance Shaders) 사용
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)

# ── 디바이스 설정 ─────────────────────────────────────────────────────────────
try:
    import torch
    if torch.backends.mps.is_available():
        TORCH_DEVICE = 'mps'
        print("✅ Apple Silicon GPU (MPS) 감지됨 — PyTorch 모델에서 사용 가능")
    elif torch.cuda.is_available():
        TORCH_DEVICE = 'cuda'
        print("✅ NVIDIA GPU (CUDA) 감지됨")
    else:
        TORCH_DEVICE = 'cpu'
        print("⚠️  GPU 미감지 — CPU 모드")
except ImportError:
    TORCH_DEVICE = 'cpu'
    print("⚠️  PyTorch 미설치 — CPU 모드")

# LightGBM은 Mac GPU 미지원 → CPU 코어 전체 사용
N_CPU = os.cpu_count()
print(f"💻 CPU 코어: {N_CPU}개 | LightGBM 스레드: {N_CPU}개")

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = DATA_DIR

SEED   = 42
TARGET = 'avg_delay_minutes_next_30m'

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

print(f"train  : {train.shape}")
print(f"test   : {test.shape}")
print(f"layout : {layout.shape}")
print(f"target → mean: {train[TARGET].mean():.2f}, median: {train[TARGET].median():.2f}, max: {train[TARGET].max():.2f}")

# =============================================================================
# 2. layout_info 조인
# =============================================================================
print("\n" + "="*60)
print("2. layout_info 조인")
print("="*60)

train = train.merge(layout, on='layout_id', how='left')
test  = test.merge(layout, on='layout_id', how='left')
print(f"조인 후 → train: {train.shape}, test: {test.shape}")

# =============================================================================
# 3. 피처 엔지니어링
# =============================================================================
print("\n" + "="*60)
print("3. 피처 엔지니어링")
print("="*60)

LAYOUT_TYPE_MAP = {'grid': 0, 'hybrid': 1, 'narrow': 2, 'hub_spoke': 3}


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 3-1. 이진 플래그 (0이냐 양수냐 → 타겟 평균 3배 차이)
    binary_flag_cols = [
        'congestion_score', 'blocked_path_15m', 'near_collision_15m',
        'charge_queue_length', 'avg_charge_wait', 'fault_count_15m',
        'replenishment_overlap', 'task_reassign_15m',
    ]
    for col in binary_flag_cols:
        if col in df.columns:
            df[f'flag_{col}'] = (df[col] > 0).astype(np.int8)

    # 3-2. 배터리 복합 피처
    # battery_mean <-> low_battery_ratio 상관 0.934 → 복합 지표로 압축
    df['battery_stress']     = df['low_battery_ratio'] / (df['battery_mean'] + 1e-5)
    df['charge_bottleneck']  = df['charge_queue_length'] * df['avg_charge_wait']
    df['battery_volatility'] = df['battery_std'] / (df['battery_mean'] + 1e-5)
    df['battery_health']     = df['battery_mean'] - df['battery_std']

    # 3-3. 주문-로봇 균형 피처
    df['order_per_robot']        = df['order_inflow_15m'] / (df['robot_active'] + 1)
    df['order_per_pack_station'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1)
    df['robot_effective_util']   = df['robot_active'] / (df['robot_total'] + 1)
    df['idle_ratio']             = (
        df['robot_idle'] /
        (df['robot_active'] + df['robot_idle'] + df['robot_charging'] + 1)
    )
    df['charging_ratio'] = df['robot_charging'] / (df['robot_total'] + 1)

    # 3-4. 혼잡도 복합 피처
    df['congestion_x_density'] = df['congestion_score'] * df['max_zone_density']
    df['traffic_severity']     = df['blocked_path_15m'] + df['near_collision_15m'] * 2
    df['aisle_load']           = df['aisle_traffic_score'] * df['congestion_score']

    # 3-5. layout_info 파생 피처
    df['layout_type_enc']        = df['layout_type'].map(LAYOUT_TYPE_MAP).fillna(-1).astype(np.int8)
    df['order_per_charger']      = df['order_inflow_15m'] / (df['charger_count'] + 1)
    df['robot_per_floor_area']   = df['robot_total'] / (df['floor_area_sqm'] + 1)
    df['pack_station_per_robot'] = df['pack_station_count'] / (df['robot_total'] + 1)

    # 3-6. 결측치 플래그 (핵심 피처 — 결측 자체가 센서 고장 신호일 수 있음)
    key_null_cols = [
        'battery_mean', 'low_battery_ratio', 'congestion_score',
        'order_inflow_15m', 'robot_active', 'pack_utilization',
    ]
    for col in key_null_cols:
        if col in df.columns:
            df[f'null_{col}'] = df[col].isna().astype(np.int8)

    return df


train = feature_engineering(train)
test  = feature_engineering(test)

# 3-7. 시나리오/레이아웃 집계 피처
print("  집계 피처 생성 중...")

AGG_COLS  = [
    'order_inflow_15m', 'low_battery_ratio', 'congestion_score',
    'robot_utilization', 'pack_utilization', 'robot_active',
    'charge_queue_length', 'max_zone_density',
]
AGG_FUNCS = ['mean', 'std', 'max', 'min']


def make_group_agg(df: pd.DataFrame, group_col: str, prefix: str) -> pd.DataFrame:
    agg = df.groupby(group_col)[AGG_COLS].agg(AGG_FUNCS)
    agg.columns = [f'{prefix}_{c}_{f}' for c, f in agg.columns]
    return agg.reset_index()


# train: 시나리오 단위 집계 (타겟 누출 없음 — 피처만 집계)
train_sc_agg = make_group_agg(train, 'scenario_id', 'sc')
train = train.merge(train_sc_agg, on='scenario_id', how='left')

# test: 시나리오 단위 집계 (test 자체 피처 집계)
test_sc_agg = make_group_agg(test, 'scenario_id', 'sc')
test = test.merge(test_sc_agg, on='scenario_id', how='left')

# layout 단위 집계 (train 기준 → test에 매핑, 창고 전반적 성향 반영)
train_layout_agg = make_group_agg(train, 'layout_id', 'layout')
layout_feat_cols = [c for c in train_layout_agg.columns if c.startswith('layout_') and c != 'layout_id']

train = train.merge(train_layout_agg[['layout_id'] + layout_feat_cols],
                    on='layout_id', how='left', suffixes=('', '_dup'))
test  = test.merge(train_layout_agg[['layout_id'] + layout_feat_cols],
                   on='layout_id', how='left')

# 중복 컬럼 제거
dup_cols = [c for c in train.columns if c.endswith('_dup')]
train.drop(columns=dup_cols, inplace=True)

print(f"  완료 → train: {train.shape}, test: {test.shape}")

# =============================================================================
# 4. 피처 목록 확정
# =============================================================================
print("\n" + "="*60)
print("4. 피처 목록 확정")
print("="*60)

DROP_COLS    = {'ID', 'layout_id', 'scenario_id', 'layout_type', TARGET}
feature_cols = [c for c in train.columns
                if c not in DROP_COLS and c in test.columns]

print(f"최종 피처 수: {len(feature_cols)}개")

X_train = train[feature_cols].astype(np.float32)
y_train = np.log1p(train[TARGET].values)   # log1p 변환 (right-skewed 보정)
X_test  = test[feature_cols].astype(np.float32)
groups  = train['scenario_id'].values

# =============================================================================
# 5. LightGBM 하이퍼파라미터
# =============================================================================
# Mac Apple Silicon: device_type='cpu', num_threads=N_CPU
# LightGBM OpenCL GPU는 Mac에서 불안정 → CPU + 멀티스레드가 실질적 최선
LGB_PARAMS = {
    'objective'        : 'regression_l1',  # MAE 직접 최적화
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
    'device_type'      : 'cpu',            # Mac: GPU 미지원
    'num_threads'      : N_CPU,            # CPU 코어 전체 활용
    'verbose'          : -1,
    'seed'             : SEED,
}

# =============================================================================
# 6. GroupKFold 학습
# =============================================================================
# GroupKFold 이유: 같은 시나리오의 타임슬롯이 train/val 동시 노출 방지
# → 일반 KFold 쓰면 CV 과낙관적 → 제출 점수와 괴리 발생

print("\n" + "="*60)
print("5. LightGBM GroupKFold 학습 (n_splits=5)")
print("="*60)

N_SPLITS   = 5
MAX_ROUNDS = 3000
EARLY_STOP = 150

gkf        = GroupKFold(n_splits=N_SPLITS)
oof_preds  = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
fold_scores = []
best_iters  = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
    print(f"\n  [Fold {fold+1}/{N_SPLITS}]", end=' ')

    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[tr_idx],       y_train[val_idx]

    dtrain = lgb.Dataset(X_tr,  label=y_tr,  free_raw_data=True)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)

    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(EARLY_STOP, verbose=False),
            lgb.log_evaluation(300),
        ],
    )

    # 역변환 + 음수 클리핑
    val_pred = np.clip(np.expm1(model.predict(X_val, num_iteration=model.best_iteration)), 0, None)
    true_val = train[TARGET].values[val_idx]
    score    = mean_absolute_error(true_val, val_pred)

    fold_scores.append(score)
    best_iters.append(model.best_iteration)
    print(f"MAE: {score:.4f} | best_iter: {model.best_iteration}")

    oof_preds[val_idx] = val_pred
    test_preds += (
        np.clip(np.expm1(model.predict(X_test, num_iteration=model.best_iteration)), 0, None)
        / N_SPLITS
    )

oof_score = mean_absolute_error(train[TARGET].values, oof_preds)

print("\n" + "="*60)
print(f"✅ OOF MAE : {oof_score:.4f}")
print(f"   Fold별  : {[round(s, 4) for s in fold_scores]}")
print(f"   평균 best_iter: {int(np.mean(best_iters))}")
print("="*60)

# =============================================================================
# 7. Feature Importance (상위 30개)
# =============================================================================
print("\n" + "="*60)
print("6. Feature Importance (마지막 Fold 기준, 상위 30개)")
print("="*60)

imp_df = (
    pd.DataFrame({'feature': feature_cols,
                  'importance': model.feature_importance('gain')})
    .sort_values('importance', ascending=False)
    .head(30)
    .reset_index(drop=True)
)
print(imp_df.to_string(index=False))

# =============================================================================
# 8. 제출 파일 생성
# =============================================================================
print("\n" + "="*60)
print("7. 제출 파일 생성")
print("="*60)

submission = sample.copy()
submission[TARGET] = np.clip(test_preds, 0, None)

out_path = os.path.join(OUTPUT_DIR, 'submission_step1_lgbm.csv')
submission.to_csv(out_path, index=False)

print(f"저장 완료 → {out_path}")
print(f"\n예측값 분포:")
print(submission[TARGET].describe().round(3).to_string())
print(f"\n최종 OOF MAE: {oof_score:.4f}")
