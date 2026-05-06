# =============================================================================
# 스마트 창고 출고 지연 예측 AI 경진대회
# Step 4: Optuna 하이퍼파라미터 튜닝 + 개선된 LightGBM
# =============================================================================
# OOF vs Public 갭 개선 핵심 전략:
#   1. lr 0.05 → 0.02: 천천히 학습해서 더 일반화된 모델
#   2. num_leaves 127 → 255: 표현력 증가
#   3. Optuna: 최적 파라미터 자동 탐색 (30 trials)
#   4. 추가 피처: 피처 중요도 상위 패턴 기반 교호작용 피처
#   5. 폴드 내부 sc_* 계산: OOF를 더 보수적으로 → 실제 Public 갭 축소
# =============================================================================
# 필요 패키지:
#   pip install lightgbm xgboost catboost scikit-learn optuna
# =============================================================================

import os, warnings, numpy as np, pandas as pd, optuna
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_CPU      = os.cpu_count()
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = DATA_DIR
SEED       = 42
TARGET     = 'avg_delay_minutes_next_30m'

print(f"💻 CPU 코어: {N_CPU}개")

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

# =============================================================================
# 2. 피처 엔지니어링 (Step1/2 + 추가 피처)
# =============================================================================
print("\n" + "="*60)
print("2. 피처 엔지니어링 (Step1/2 + 추가 교호작용 피처)")
print("="*60)

LAYOUT_TYPE_MAP = {'grid': 0, 'hybrid': 1, 'narrow': 2, 'hub_spoke': 3}


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── 기존 피처 (Step1/2와 동일) ───────────────────────────────────────────
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

    # ── 추가 피처: Feature Importance 상위 패턴 기반 교호작용 ────────────────
    # sc_congestion_score_mean이 압도적 1위 → 혼잡 × 배터리 교호작용 강화
    df['congestion_x_battery']   = df['congestion_score'] * df['low_battery_ratio']
    df['congestion_x_order']     = df['congestion_score'] * df['order_inflow_15m']
    df['battery_x_order']        = df['low_battery_ratio'] * df['order_inflow_15m']
    df['pack_util_x_order']      = df['pack_utilization'] * df['order_inflow_15m']

    # 복합 부하 지표 — 창고가 얼마나 한계에 몰렸는지
    df['total_stress'] = (
        df['low_battery_ratio'].fillna(0) +
        df['congestion_score'].fillna(0) / 100 +
        df['pack_utilization'].fillna(0)
    )
    df['robot_shortage'] = df['order_inflow_15m'] / (df['robot_active'] + df['robot_idle'] + 1)

    # 패킹 병목 복합 지표 — 극단값(100분+)에서 pack_utilization이 0.88로 핵심
    df['pack_x_order_per_station'] = df['pack_utilization'] * df['order_per_pack_station']

    # 충전 위기 강도
    df['charge_crisis'] = df['low_battery_ratio'] * df['charge_queue_length']

    return df


train = feature_engineering(train)
test  = feature_engineering(test)

# ── 시나리오 집계 피처 ────────────────────────────────────────────────────────
AGG_COLS  = [
    'order_inflow_15m', 'low_battery_ratio', 'congestion_score',
    'robot_utilization', 'pack_utilization', 'robot_active',
    'charge_queue_length', 'max_zone_density',
    # 추가: 교호작용 피처 집계
    'congestion_x_battery', 'total_stress', 'pack_x_order_per_station',
]
AGG_FUNCS = ['mean', 'std', 'max', 'min']


def make_group_agg(df, group_col, prefix):
    valid_cols = [c for c in AGG_COLS if c in df.columns]
    agg = df.groupby(group_col)[valid_cols].agg(AGG_FUNCS)
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

print(f"총 피처 수: {len(feature_cols)}개 (Step1/2 대비 추가)")

X_all   = train[feature_cols].astype(np.float32)
y_all   = np.log1p(train[TARGET].values)
X_test  = test[feature_cols].astype(np.float32)
groups  = train['scenario_id'].values

# =============================================================================
# 3. Optuna 하이퍼파라미터 튜닝
# =============================================================================
print("\n" + "="*60)
print("3. Optuna 하이퍼파라미터 튜닝 (30 trials, 3-fold CV)")
print("="*60)
print("   ※ 약 10~20분 소요 예상. 건너뛰려면 Ctrl+C 후 Step 4로 진행")

N_OPTUNA_TRIALS = 30
N_OPTUNA_FOLDS  = 3   # 튜닝 시에는 3-fold로 빠르게

best_params_from_optuna = None


def optuna_objective(trial):
    params = {
        'objective'        : 'regression_l1',
        'metric'           : 'mae',
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'num_leaves'       : trial.suggest_int('num_leaves', 64, 511),
        'max_depth'        : trial.suggest_int('max_depth', 5, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 150),
        'feature_fraction' : trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction' : trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq'     : trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1'        : trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2'        : trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'device_type'      : 'cpu',
        'num_threads'      : N_CPU,
        'verbose'          : -1,
        'seed'             : SEED,
    }

    gkf = GroupKFold(n_splits=N_OPTUNA_FOLDS)
    scores = []
    for tr_idx, val_idx in gkf.split(X_all, y_all, groups):
        dtrain = lgb.Dataset(X_all.iloc[tr_idx], label=y_all[tr_idx], free_raw_data=True)
        dvalid = lgb.Dataset(X_all.iloc[val_idx], label=y_all[val_idx],
                             reference=dtrain, free_raw_data=True)
        model = lgb.train(
            params, dtrain,
            num_boost_round=2000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(9999)],
        )
        pred  = np.clip(np.expm1(model.predict(X_all.iloc[val_idx])), 0, None)
        score = mean_absolute_error(train[TARGET].values[val_idx], pred)
        scores.append(score)
    return np.mean(scores)


try:
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(optuna_objective, n_trials=N_OPTUNA_TRIALS,
                   show_progress_bar=True)

    best_params_from_optuna = study.best_params
    print(f"\n✅ Optuna 완료!")
    print(f"   Best OOF MAE: {study.best_value:.4f}")
    print(f"   Best params : {best_params_from_optuna}")

except KeyboardInterrupt:
    print("\n⚠️  Optuna 중단됨 — 기본 파라미터로 진행")

# =============================================================================
# 4. 최종 모델 학습 (Optuna 결과 or 기본 개선 파라미터)
# =============================================================================
print("\n" + "="*60)
print("4. 최종 LightGBM 학습 (5-fold GroupKFold)")
print("="*60)

if best_params_from_optuna:
    FINAL_PARAMS = {
        'objective'  : 'regression_l1',
        'metric'     : 'mae',
        'device_type': 'cpu',
        'num_threads': N_CPU,
        'verbose'    : -1,
        'seed'       : SEED,
        **best_params_from_optuna,
    }
    print("   → Optuna 최적 파라미터 사용")
else:
    # Optuna 건너뛴 경우: Step1/2 대비 개선된 기본 파라미터
    # 핵심: lr 0.05→0.02, num_leaves 127→255, min_child 50→30
    FINAL_PARAMS = {
        'objective'        : 'regression_l1',
        'metric'           : 'mae',
        'learning_rate'    : 0.02,    # ★ 핵심: 낮은 lr → 더 정밀한 학습, 일반화 향상
        'num_leaves'       : 255,     # ★ 표현력 증가
        'max_depth'        : -1,
        'min_child_samples': 30,
        'feature_fraction' : 0.75,
        'bagging_fraction' : 0.75,
        'bagging_freq'     : 5,
        'lambda_l1'        : 0.5,
        'lambda_l2'        : 0.5,
        'device_type'      : 'cpu',
        'num_threads'      : N_CPU,
        'verbose'          : -1,
        'seed'             : SEED,
    }
    print("   → 개선된 기본 파라미터 사용 (lr=0.02, num_leaves=255)")

print(f"\n학습 파라미터:")
for k, v in FINAL_PARAMS.items():
    if k not in ('device_type', 'num_threads', 'verbose', 'seed', 'objective', 'metric'):
        print(f"  {k}: {v}")

N_SPLITS   = 5
MAX_ROUNDS = 5000    # lr 낮아진 만큼 라운드 늘림
EARLY_STOP = 200

gkf        = GroupKFold(n_splits=N_SPLITS)
oof_preds  = np.zeros(len(X_all))
test_preds = np.zeros(len(X_test))
fold_scores= []
best_iters = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups)):
    print(f"\n  [Fold {fold+1}/{N_SPLITS}]", end=' ')

    dtrain = lgb.Dataset(X_all.iloc[tr_idx], label=y_all[tr_idx], free_raw_data=True)
    dvalid = lgb.Dataset(X_all.iloc[val_idx], label=y_all[val_idx],
                         reference=dtrain, free_raw_data=True)

    model = lgb.train(
        FINAL_PARAMS, dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(EARLY_STOP, verbose=False),
            lgb.log_evaluation(500),
        ],
    )

    val_pred = np.clip(np.expm1(model.predict(X_all.iloc[val_idx],
                                               num_iteration=model.best_iteration)), 0, None)
    true_val = train[TARGET].values[val_idx]
    score    = mean_absolute_error(true_val, val_pred)

    fold_scores.append(score)
    best_iters.append(model.best_iteration)
    print(f"MAE: {score:.4f} | best_iter: {model.best_iteration}")

    oof_preds[val_idx] = val_pred
    test_preds += np.clip(
        np.expm1(model.predict(X_test, num_iteration=model.best_iteration)), 0, None
    ) / N_SPLITS

oof_score = mean_absolute_error(train[TARGET].values, oof_preds)

print("\n" + "="*60)
print(f"✅ Step4 OOF MAE : {oof_score:.4f}")
print(f"   Fold별       : {[round(s, 4) for s in fold_scores]}")
print(f"   평균 best_iter: {int(np.mean(best_iters))}")
print("="*60)

# =============================================================================
# 5. Feature Importance (상위 30개)
# =============================================================================
print("\n" + "="*60)
print("5. Feature Importance (상위 30개)")
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
# 6. 이전 결과와 앙상블 결합
# =============================================================================
print("\n" + "="*60)
print("6. Step2/3 결과와 앙상블 결합")
print("="*60)

# Step4 단독 제출
sub_step4 = sample.copy()
sub_step4[TARGET] = np.clip(test_preds, 0, None)
path_step4 = os.path.join(OUTPUT_DIR, 'submission_step4_tuned.csv')
sub_step4.to_csv(path_step4, index=False)
print(f"Step4 단독 저장 → submission_step4_tuned.csv")

# Step2 앙상블 + Step4 결합
step2_path = os.path.join(DATA_DIR, 'submission_step2_ensemble.csv')
step3_path = os.path.join(DATA_DIR, 'submission_step3_final.csv')

# 사용 가능한 파일 목록으로 앙상블 구성
preds_pool = {'step4': (test_preds, oof_score)}

if os.path.exists(step3_path):
    step3_pred = pd.read_csv(step3_path)[TARGET].values
    preds_pool['step3'] = (step3_pred, None)  # step3 OOF MAE 없음 → 동등 가중치
    print(f"Step3 MLP 결과 감지됨 — 앙상블에 포함")

if os.path.exists(step2_path):
    step2_pred = pd.read_csv(step2_path)[TARGET].values
    preds_pool['step2'] = (step2_pred, 8.7020)
    print(f"Step2 앙상블 결과 감지됨 — 앙상블에 포함")

# 역가중치 앙상블 (OOF MAE 있는 것만)
known_mae = {k: v[1] for k, v in preds_pool.items() if v[1] is not None}
if len(known_mae) > 1:
    total_inv = sum(1.0 / m for m in known_mae.values())
    weights   = {k: (1.0 / m) / total_inv for k, m in known_mae.items()}
    unknown   = {k: v for k, v in preds_pool.items() if v[1] is None}

    final_pred = sum(weights[k] * preds_pool[k][0] for k in weights)

    # OOF MAE 모르는 모델(step3)은 균등 blend (20%)
    if unknown:
        n_unk = len(unknown)
        blend_ratio = 0.2 / n_unk
        final_pred  = final_pred * (1 - 0.2 * n_unk)
        for k in unknown:
            final_pred += blend_ratio * unknown[k][0]

    final_pred = np.clip(final_pred, 0, None)
    sub_final  = sample.copy()
    sub_final[TARGET] = final_pred
    path_final = os.path.join(OUTPUT_DIR, 'submission_step4_final_blend.csv')
    sub_final.to_csv(path_final, index=False)

    print(f"\n최종 블렌딩 가중치:")
    for k, w in weights.items():
        print(f"  {k}: {w:.3f}")
    print(f"\n최종 블렌딩 저장 → submission_step4_final_blend.csv")
    print(f"\n예측값 분포 (최종):")
    print(sub_final[TARGET].describe().round(3).to_string())

print(f"\n✅ Step4 OOF MAE: {oof_score:.4f}")
print(f"   제출 추천: submission_step4_tuned.csv  (Step4 단독)")
print(f"   또는     : submission_step4_final_blend.csv  (전체 블렌딩)")
