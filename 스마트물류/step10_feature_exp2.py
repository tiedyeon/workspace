# =============================================================================
# 스마트 창고 출고 지연 예측 AI 경진대회
# Step 10: 피처 엔지니어링 실험 2라운드 (shift_hour 오해 수정 후)
# =============================================================================
# Step9 교훈:
#   - shift_hour = 슬롯 순서 X, 하루 중 몇 시 O (시나리오당 평균 8.6개 유니크)
#   - lag/diff 불가 (같은 시간대 내 순서 구별 불가)
#
# 이번 실험 방향:
#   G: z-score (Step9-A 재현, 기준점)
#   H: 시나리오 내 퍼센타일 랭크 — "현재값이 이 시나리오에서 몇 번째로 나쁜가"
#   I: sc_max/mean 대비 비율 — "지금이 최악 대비 얼마나 심각한가"
#   J: shift_hour 시간대 피처 — "몇 시대 스냅샷인가 (업무초반/피크/야간)"
#   K: G + H + I (순서 불필요 피처 조합)
#   L: G + H + I + J (전체 조합)
#
# 속도: LGB 단일 시드 × 5폴드, 실험당 ~1.5분, 전체 ~12분
# 실행:
#   caffeinate -i nohup python step10_feature_exp2.py > step10_output.log 2>&1 &
#   tail -f step10_output.log
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

BASE_AGG_COLS  = ['order_inflow_15m', 'low_battery_ratio', 'congestion_score',
                  'robot_utilization', 'pack_utilization', 'robot_active',
                  'charge_queue_length', 'max_zone_density']
BASE_AGG_FUNCS = ['mean', 'std', 'max', 'min']

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
# 2. 피처 함수 정의
# =============================================================================

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


def build_base_df(raw_df, src_for_layout_agg=None):
    """베이스 피처 + sc_* + layout_agg 전부 적용"""
    df = add_base_features(raw_df.copy())

    # sc_* 집계
    sc_agg = df.groupby('scenario_id')[BASE_AGG_COLS].agg(BASE_AGG_FUNCS)
    sc_agg.columns = [f'sc_{c}_{f}' for c, f in sc_agg.columns]
    df = df.merge(sc_agg.reset_index(), on='scenario_id', how='left')

    # layout_agg (train 기준, test도 train 집계 사용)
    src = src_for_layout_agg if src_for_layout_agg is not None else df
    layout_agg = src.groupby('layout_id')[BASE_AGG_COLS].agg(BASE_AGG_FUNCS)
    layout_agg.columns = [f'layout_{c}_{f}' for c, f in layout_agg.columns]
    layout_agg = layout_agg.reset_index()
    layout_feat_cols = [c for c in layout_agg.columns if c.startswith('layout_') and c != 'layout_id']
    df = df.merge(layout_agg[['layout_id'] + layout_feat_cols],
                  on='layout_id', how='left', suffixes=('', '_dup'))
    df.drop(columns=[c for c in df.columns if c.endswith('_dup')], inplace=True)
    return df


# ── 실험 G: z-score (Step9-A 재현) ───────────────────────────────────────────
ZSCORE_COLS = ['congestion_score', 'low_battery_ratio', 'max_zone_density',
               'pack_utilization', 'robot_utilization', 'order_inflow_15m',
               'charge_queue_length', 'robot_active', 'battery_mean',
               'aisle_traffic_score', 'blocked_path_15m']

def add_zscore(df):
    df = df.copy()
    for col in ZSCORE_COLS:
        m, s = f'sc_{col}_mean', f'sc_{col}_std'
        if col in df.columns and m in df.columns and s in df.columns:
            df[f'z_{col}']     = (df[col] - df[m]) / (df[s] + 1e-5)
            df[f'z_{col}_abs'] = df[f'z_{col}'].abs()
    return df


# ── 실험 H: 시나리오 내 퍼센타일 랭크 ────────────────────────────────────────
RANK_COLS = ['congestion_score', 'low_battery_ratio', 'max_zone_density',
             'pack_utilization', 'order_inflow_15m', 'charge_queue_length',
             'battery_mean', 'robot_utilization', 'blocked_path_15m']

def add_percentile_rank(df):
    """
    시나리오 내에서 현재값의 퍼센타일 랭크 (0~1)
    순서 복원 없이도 가능 — "지금 혼잡도가 이 시나리오에서 몇 번째로 나쁜가"
    """
    df = df.copy()
    for col in RANK_COLS:
        if col not in df.columns:
            continue
        # pct=True → 0~1 비율로 반환, na_option='keep' → NaN 유지
        df[f'prank_{col}'] = df.groupby('scenario_id')[col].rank(pct=True, na_option='keep')
    return df


# ── 실험 I: sc_max/mean 대비 비율 ─────────────────────────────────────────────
RATIO_COLS = ['congestion_score', 'low_battery_ratio', 'max_zone_density',
              'pack_utilization', 'order_inflow_15m', 'charge_queue_length',
              'robot_active', 'battery_mean']

def add_ratio_features(df):
    """
    현재값 / sc_max  → "최악 대비 지금이 얼마나 심각한가" (0~1)
    현재값 / sc_mean → "평균 대비 현재 부하 (>1이면 평균 초과)"
    sc_max - 현재값  → "최악까지 얼마나 여유 있나"
    """
    df = df.copy()
    for col in RATIO_COLS:
        max_col  = f'sc_{col}_max'
        mean_col = f'sc_{col}_mean'
        if col not in df.columns:
            continue
        if max_col in df.columns:
            df[f'ratio_to_max_{col}']  = df[col] / (df[max_col] + 1e-5)
            df[f'gap_to_max_{col}']    = df[max_col] - df[col]
        if mean_col in df.columns:
            df[f'ratio_to_mean_{col}'] = df[col] / (df[mean_col] + 1e-5)
    return df


# ── 실험 J: shift_hour 시간대 피처 ───────────────────────────────────────────
def add_shift_hour_features(df):
    """
    shift_hour = 하루 중 몇 시 (0~23, NaN 있음)
    시간대별 업무 패턴 차이 포착
    """
    df = df.copy()
    sh = df['shift_hour'].fillna(-1)

    # 숫자 그대로 (NaN → -1)
    df['shift_hour_filled'] = sh

    # 시간대 구분 (업무 초반 / 피크 / 오후 / 야간 / 미상)
    df['is_early_shift']    = ((sh >= 0)  & (sh <= 5)).astype(np.int8)   # 자정~새벽
    df['is_morning_shift']  = ((sh >= 6)  & (sh <= 11)).astype(np.int8)  # 오전 피크
    df['is_afternoon_shift']= ((sh >= 12) & (sh <= 17)).astype(np.int8)  # 오후
    df['is_evening_shift']  = ((sh >= 18) & (sh <= 23)).astype(np.int8)  # 야간
    df['is_unknown_shift']  = (df['shift_hour'].isna()).astype(np.int8)

    # 사인/코사인 인코딩 (시간의 주기성 — 0시와 23시가 가까움)
    df['shift_hour_sin'] = np.sin(2 * np.pi * sh.clip(0) / 24)
    df['shift_hour_cos'] = np.cos(2 * np.pi * sh.clip(0) / 24)
    # NaN인 경우 sin/cos도 0으로
    df.loc[df['shift_hour'].isna(), ['shift_hour_sin', 'shift_hour_cos']] = 0.0

    # 시나리오 내 shift_hour 다양성 (몇 개 시간대에 걸쳐 있나)
    sc_nunique = df.groupby('scenario_id')['shift_hour'].transform('nunique')
    df['sc_shift_hour_diversity'] = sc_nunique

    # 시나리오 내 shift_hour 범위 (마지막 시간 - 처음 시간)
    sc_sh_min = df.groupby('scenario_id')['shift_hour'].transform('min')
    sc_sh_max = df.groupby('scenario_id')['shift_hour'].transform('max')
    df['sc_shift_hour_range'] = sc_sh_max - sc_sh_min

    return df


# =============================================================================
# 3. 데이터셋 준비 함수
# =============================================================================

def prepare_dataset(train_raw, test_raw, exp_name, add_fns):
    train_base = build_base_df(train_raw)
    test_base  = build_base_df(test_raw, src_for_layout_agg=train_base)

    train = train_base.copy()
    test  = test_base.copy()
    for fn in add_fns:
        train = fn(train)
        test  = fn(test)

    DROP_COLS    = {'ID', 'layout_id', 'scenario_id', 'layout_type',
                    TARGET, 'shift_hour'}
    feature_cols = [c for c in train.columns
                    if c not in DROP_COLS and c in test.columns
                    and train[c].dtype != object]
    feature_cols = list(dict.fromkeys(feature_cols))

    X_train = train[feature_cols].astype(np.float32)
    y_train = np.log1p(train[TARGET].values)
    X_test  = test[feature_cols].astype(np.float32)
    groups  = train['scenario_id'].values

    print(f"  [{exp_name}] 피처 수: {len(feature_cols)}개")
    return X_train, y_train, X_test, groups, feature_cols


# =============================================================================
# 4. OOF 평가 (LGB 단일 시드)
# =============================================================================
y_true = train_raw[TARGET].values
gkf    = GroupKFold(n_splits=N_SPLITS)


def run_lgb_oof(X_train, y_train, groups, exp_name):
    oof_preds  = np.zeros(len(X_train))
    test_preds = None
    best_iters = []
    t0 = time.time()

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        dtrain = lgb.Dataset(X_tr,  label=y_tr,  free_raw_data=True)
        dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)
        model  = lgb.train(
            LGB_PARAMS, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                       lgb.log_evaluation(999)],
        )
        val_p = np.clip(np.expm1(model.predict(X_val, num_iteration=model.best_iteration)), 0, None)
        oof_preds[val_idx] = val_p
        best_iters.append(model.best_iteration)

    oof_score = mean_absolute_error(y_true, oof_preds)
    elapsed   = time.time() - t0
    print(f"  [{exp_name}] OOF MAE: {oof_score:.4f} | "
          f"avg_iter: {np.mean(best_iters):.0f} | {elapsed/60:.1f}min")
    return oof_score


# =============================================================================
# 5. 실험 실행
# =============================================================================
print("\n" + "="*60)
print("실험 시작 (LGB 단일 시드)")
print("="*60)

results  = {}
exp_start = time.time()

# BASE
print("\n[BASE] 베이스라인")
Xb, yb, Xtb, gb, fcb = prepare_dataset(train_raw, test_raw, 'BASE', [])
results['BASE'] = run_lgb_oof(Xb, yb, gb, 'BASE')

# G: z-score
print("\n[G] z-score")
Xg, yg, Xtg, gg, fcg = prepare_dataset(train_raw, test_raw, 'G', [add_zscore])
results['G: z-score'] = run_lgb_oof(Xg, yg, gg, 'G: z-score')

# H: 퍼센타일 랭크
print("\n[H] 퍼센타일 랭크")
Xh, yh, Xth, gh, fch = prepare_dataset(train_raw, test_raw, 'H', [add_percentile_rank])
results['H: prank'] = run_lgb_oof(Xh, yh, gh, 'H: prank')

# I: 비율 피처
print("\n[I] sc_max/mean 대비 비율")
Xi, yi, Xti, gi, fci = prepare_dataset(train_raw, test_raw, 'I', [add_ratio_features])
results['I: ratio'] = run_lgb_oof(Xi, yi, gi, 'I: ratio')

# J: shift_hour 시간대
print("\n[J] shift_hour 시간대 피처")
Xj, yj, Xtj, gj, fcj = prepare_dataset(train_raw, test_raw, 'J', [add_shift_hour_features])
results['J: shift_hour'] = run_lgb_oof(Xj, yj, gj, 'J: shift_hour')

# K: G + H + I
print("\n[K] G + H + I 조합")
Xk, yk, Xtk, gk, fck = prepare_dataset(
    train_raw, test_raw, 'K', [add_zscore, add_percentile_rank, add_ratio_features])
results['K: G+H+I'] = run_lgb_oof(Xk, yk, gk, 'K: G+H+I')

# L: G + H + I + J
print("\n[L] G + H + I + J 전체")
Xl, yl, Xtl, gl, fcl = prepare_dataset(
    train_raw, test_raw, 'L',
    [add_zscore, add_percentile_rank, add_ratio_features, add_shift_hour_features])
results['L: G+H+I+J'] = run_lgb_oof(Xl, yl, gl, 'L: G+H+I+J')

# =============================================================================
# 6. 결과 비교표
# =============================================================================
total_elapsed = time.time() - exp_start
score_base    = results['BASE']

print("\n" + "="*60)
print("실험 결과 비교")
print("="*60)
print(f"\n{'실험':<22} {'OOF MAE':>10} {'vs BASE':>10} {'판정':>10}")
print("-" * 56)

for name, score in results.items():
    diff   = score - score_base
    judge  = '개선' if diff < -0.002 else ('유지' if abs(diff) <= 0.002 else '악화')
    marker = ' <--' if diff < -0.002 else ''
    print(f"  {name:<20} {score:>10.4f} {diff:>+10.4f} {judge:>10}{marker}")

best_exp   = min(results, key=results.get)
best_score = results[best_exp]
print(f"\n최적: [{best_exp}]  OOF MAE={best_score:.4f}  (vs BASE {best_score-score_base:+.4f})")
print(f"총 소요 시간: {total_elapsed/60:.1f}분")

# =============================================================================
# 7. OOF 개선 시 베스트 조합 데이터셋 저장 (다음 단계용)
# =============================================================================
THRESHOLD = -0.005  # BASE 대비 0.005 이상 개선 시 의미 있다고 판단

if best_score - score_base < THRESHOLD:
    print(f"\n개선폭 {best_score-score_base:.4f} > 임계값 {THRESHOLD}")
    print("→ 다음: step11에서 이 피처셋으로 3모델×5시드 풀 앙상블 실행")

    # 어떤 피처 조합이 best인지 기록
    fn_map = {
        'BASE':       [],
        'G: z-score': [add_zscore],
        'H: prank':   [add_percentile_rank],
        'I: ratio':   [add_ratio_features],
        'J: shift_hour': [add_shift_hour_features],
        'K: G+H+I':   [add_zscore, add_percentile_rank, add_ratio_features],
        'L: G+H+I+J': [add_zscore, add_percentile_rank, add_ratio_features, add_shift_hour_features],
    }
    print(f"베스트 실험 [{best_exp}] 피처 함수: {[f.__name__ for f in fn_map.get(best_exp, [])]}")
else:
    print(f"\n개선폭 {best_score-score_base:+.4f} — 임계값({THRESHOLD}) 미달")
    print("→ 새 피처 아이디어 추가 필요")

print("\n" + "="*60)
print("Step9 vs Step10 비교 (BASE 기준)")
print("="*60)
print(f"  Step9  BASE OOF: 8.7436  최고: A(z-score) 8.7414  개선: -0.0022")
print(f"  Step10 BASE OOF: {score_base:.4f}  최고: {best_exp} {best_score:.4f}  개선: {best_score-score_base:+.4f}")
