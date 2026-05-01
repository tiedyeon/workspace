# -*- coding: utf-8 -*-
"""
submission3_lgbm_quantile.py — LightGBM Quantile + 새 피처 디자인

구조 (옛 submission1 과 동일):
  - 모델:  LightGBM, objective='quantile', alpha=0.5
  - CV:    GroupKFold(5, by=layout_id)
  - 손실:  quantile (MAE 평가에 정렬됨)
  - 결측:  NaN 그대로 (LightGBM 자연 처리)

피처 (≈108개) — Phase 2-4 EDA 발견 반영:
  [원본]   89 numeric - 환경 노이즈 16 = 73개
  [L1]    layout_info LEFT JOIN 13 + layout_type one-hot 4 = 17개
  [L2]    robot 파생 4 (idle_ratio, charging_ratio, available_robots,
                       robot_total은 layout_info 에서)
  [L3]    capacity 정규화 6 (orders_per_robot, orders_per_pack_station 등)
  [L4]    stress 4 (flag_idle_zero, flag_charging_active, flag_active_high,
                    robot_stress_score)
  [L5]    event 1 (incident_score 만 — flag 5개는 원본과 0.99+ 중복이라 폐기)
  [L6]    interaction 5 (stress_x_inflow, load_pressure 등)

기록:
  submission1 (옛 머신, 209 피처): OOF 9.1768, Public 10.6461
  submission2 (옛 머신, 111 피처): OOF 9.1955, Public 10.6703 (폐기)
  submission3 (이 파일, ~108 피처): OOF TBD, Public TBD

산출:
  outputs/submission3_lgbm_quantile.csv   ← Dacon 제출
  outputs/submission3_oof.csv             ← OOF (inspection)
  outputs/submission3_oof.npy             ← OOF (앙상블용)
  outputs/submission3_test_pred.npy       ← test 예측 평균
  outputs/submission3_importance.csv      ← feature importance
  outputs/submission3_importance.png      ← top 30 막대
  outputs/submission3_summary.txt         ← OOF MAE, fold MAE, hyperparams

실행:
  (.smart) PS C:\...\스마트물류> python submission3_lgbm_quantile.py
  소요: 약 2~3분
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.model_selection import GroupKFold

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.default"] = "regular"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Phase 3 의 함수 재사용 (단일 출처)
from eda_phase3 import merge_layout_info, add_all_derived

# ──────────────────────────────────────────────
SUBMISSION_NUM = 3
SEED = 42
N_FOLDS = 5

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

# 환경 노이즈 컬럼 (Phase 3: mean |Spearman| < 0.03)
NOISE_COLS = [
    # environment
    "warehouse_temp_avg", "humidity_pct", "external_temp_c",
    "lighting_level_lux", "cold_storage_temp_c",
    # atmosphere
    "ambient_noise_db", "air_quality_idx", "co2_level_ppm",
    "floor_vibration_idx",
    # weather
    "wind_speed_kmh", "precipitation_mm",
    # infra_it
    "wms_response_time_ms", "wifi_signal_db", "network_latency_ms",
    "hvac_power_kw",
    # power
    "ups_battery_pct",
]

# 또한 Phase 3 에서 0.99+ 중복 확인된 event flag 5개 — incident_score 만 유지
DUPLICATE_FLAGS = [
    "flag_collision",        # near_collision_15m 와 0.993
    "flag_blocked",          # blocked_path_15m 와 0.994
    "flag_fault",            # fault_count_15m 와 0.995
    "flag_charge_queue",     # charge_queue_length 와 0.993
    "flag_congestion_hot",   # congestion_score 와 0.99+
]

# robot_active_ratio 도 robot_utilization 과 1.000 — 폐기
DUPLICATE_RATIOS = ["robot_active_ratio"]

# LightGBM 하이퍼파라미터 (옛 submission1 과 비슷한 보수 설정)
LGB_PARAMS = dict(
    objective="quantile",
    alpha=0.5,
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    n_jobs=-1,
    random_state=SEED,
    verbosity=-1,
)


# ──────────────────────────────────────────────
def section(title: str) -> None:
    print()
    print("=" * 64)
    print(f" {title}")
    print("=" * 64)


def MAE(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def prepare_features(
    train: pd.DataFrame, test: pd.DataFrame, layout_info: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """layout_info merge + layout_type one-hot + 파생 피처 (Phase 3 함수 재사용)"""
    print("layout_info LEFT JOIN...")
    train_m = merge_layout_info(train, layout_info)
    test_m = merge_layout_info(test, layout_info)

    # layout_type one-hot — train/test 공통 카테고리
    print("layout_type one-hot 인코딩...")
    all_lts = pd.concat(
        [train_m["layout_type"], test_m["layout_type"]]
    ).dropna().unique()
    for lt in all_lts:
        train_m[f"layout_type_{lt}"] = (train_m["layout_type"] == lt).astype(int)
        test_m[f"layout_type_{lt}"] = (test_m["layout_type"] == lt).astype(int)
    train_m = train_m.drop(columns=["layout_type"])
    test_m = test_m.drop(columns=["layout_type"])

    print("파생 피처 (robot 비율 + capacity + flags + interaction)...")
    train_aug = add_all_derived(train_m)
    test_aug = add_all_derived(test_m)

    return train_aug, test_aug


def get_feature_cols(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> list[str]:
    """공통 numeric 컬럼 (키·타깃·노이즈·중복 제외)"""
    exclude = (
        {"ID", "layout_id", "scenario_id", TARGET}
        | set(NOISE_COLS)
        | set(DUPLICATE_FLAGS)
        | set(DUPLICATE_RATIOS)
    )
    train_numeric = [
        c for c in df_train.columns
        if c not in exclude
        and str(df_train[c].dtype) in ("float64", "int64", "int32", "bool")
    ]
    feature_cols = [c for c in train_numeric if c in df_test.columns]
    missing_in_test = set(train_numeric) - set(df_test.columns)
    if missing_in_test:
        print(f"⚠ test 에 없어 제외: {missing_in_test}")
    return feature_cols


# ──────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    print(f">>> submission{SUBMISSION_NUM} 시작 — LightGBM Quantile + 새 디자인")

    # 1. Load
    section("1. 데이터 로드")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    layout_info = pd.read_csv(DATA_DIR / "layout_info.csv")
    print(f"train: {train.shape}, test: {test.shape}, "
          f"layout_info: {layout_info.shape}")

    # 2. Feature engineering
    section("2. 피처 엔지니어링")
    train_aug, test_aug = prepare_features(train, test, layout_info)
    print(f"train_aug: {train_aug.shape}, test_aug: {test_aug.shape}")

    feature_cols = get_feature_cols(train_aug, test_aug)
    print(f"\n사용 피처 수: {len(feature_cols)}")

    X = train_aug[feature_cols]
    y = train_aug[TARGET]
    groups = train_aug["layout_id"]
    X_test = test_aug[feature_cols]

    print(f"\nX: {X.shape}, y mean={y.mean():.2f}, median={y.median():.2f}")
    print(f"X_test: {X_test.shape}")
    print(f"groups: {groups.nunique()} unique layouts")

    # 3. Train with GroupKFold
    section(f"3. 학습 — GroupKFold({N_FOLDS}, by=layout_id)")
    gkf = GroupKFold(n_splits=N_FOLDS)

    oof = np.zeros(len(X))
    test_pred_folds = np.zeros((N_FOLDS, len(X_test)))
    importances = np.zeros(len(feature_cols))
    fold_maes = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n[fold {fold + 1}/{N_FOLDS}] "
              f"train rows={len(tr_idx)}, valid rows={len(va_idx)}")

        model = lgb.LGBMRegressor(n_estimators=3000, **LGB_PARAMS)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof[va_idx] = model.predict(X.iloc[va_idx])
        test_pred_folds[fold] = model.predict(X_test)
        importances += model.feature_importances_

        fold_mae = MAE(y.iloc[va_idx], oof[va_idx])
        fold_maes.append(fold_mae)
        print(f"  → MAE = {fold_mae:.4f}, "
              f"best_iter = {model.best_iteration_}")

    overall_mae = MAE(y, oof)
    importances /= N_FOLDS
    test_pred = test_pred_folds.mean(axis=0)
    elapsed_min = (time.time() - t0) / 60

    section("4. 결과 요약")
    print(f"OOF MAE:       {overall_mae:.4f}")
    print(f"Fold MAE:      [{', '.join(f'{m:.4f}' for m in fold_maes)}]")
    print(f"Fold std:      {np.std(fold_maes):.4f}")
    print(f"학습 시간:     {elapsed_min:.2f} 분")
    print()
    print("test 예측 통계:")
    print(f"  mean: {test_pred.mean():.2f} (train mean: {y.mean():.2f})")
    print(f"  median: {np.median(test_pred):.2f} (train median: {y.median():.2f})")
    print(f"  min: {test_pred.min():.2f}, max: {test_pred.max():.2f}")
    print(f"  음수 개수: {(test_pred < 0).sum()} (clip 처리)")

    # 5. Save submission CSV (Dacon)
    section("5. 산출물 저장")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub[TARGET] = np.clip(test_pred, 0, None)
    sub_path = OUT_DIR / f"submission{SUBMISSION_NUM}_lgbm_quantile.csv"
    sub.to_csv(sub_path, index=False)
    print(f"제출 csv → {sub_path}")

    # 6. OOF + test pred (앙상블용)
    np.save(OUT_DIR / f"submission{SUBMISSION_NUM}_oof.npy", oof)
    np.save(OUT_DIR / f"submission{SUBMISSION_NUM}_test_pred.npy", test_pred)

    oof_df = pd.DataFrame({
        "ID": train["ID"].values,
        "y_true": y.values,
        "y_pred": oof,
        "abs_err": np.abs(y.values - oof),
    })
    oof_df.to_csv(OUT_DIR / f"submission{SUBMISSION_NUM}_oof.csv",
                   index=False, encoding="utf-8-sig")
    print(f"OOF csv → {OUT_DIR / f'submission{SUBMISSION_NUM}_oof.csv'}")

    # 7. Importance
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    imp_path = OUT_DIR / f"submission{SUBMISSION_NUM}_importance.csv"
    imp_df.to_csv(imp_path, index=False, encoding="utf-8-sig")
    print(f"importance → {imp_path}")

    print("\nTop 20 features by importance:")
    print(imp_df.head(20).round(2).to_string(index=False))

    # importance figure (top 30)
    top30 = imp_df.head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.barh(range(len(top30)), top30["importance"], color="steelblue")
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30["feature"], fontsize=9)
    ax.set_xlabel("LightGBM importance (gain)")
    ax.set_title(f"submission{SUBMISSION_NUM} top 30 — "
                 f"OOF MAE = {overall_mae:.4f}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig_path = OUT_DIR / f"submission{SUBMISSION_NUM}_importance.png"
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"figure → {fig_path}")

    # 8. Summary text
    summary = f"""submission{SUBMISSION_NUM} — LightGBM Quantile + 새 디자인

OOF MAE:    {overall_mae:.4f}
Fold MAE:   [{', '.join(f'{m:.4f}' for m in fold_maes)}]
Fold std:   {np.std(fold_maes):.4f}
학습 시간:  {elapsed_min:.2f} 분

피처 수:    {len(feature_cols)}
train rows: {len(X)}, test rows: {len(X_test)}
groups:     {groups.nunique()} unique layouts

LightGBM 하이퍼파라미터:
  {LGB_PARAMS}

비교 (옛 머신):
  submission1 (LightGBM Quantile, 209 피처): OOF 9.1768, Public 10.6461
  submission2 (LightGBM log1p+MSE, 111 피처): OOF 9.1955, Public 10.6703 (폐기)
  submission3 (이 파일, {len(feature_cols)} 피처): OOF {overall_mae:.4f}, Public TBD

Top 20 features:
{imp_df.head(20).round(2).to_string(index=False)}
"""
    summary_path = OUT_DIR / f"submission{SUBMISSION_NUM}_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"summary → {summary_path}")

    print()
    print("=" * 64)
    print(f" submission{SUBMISSION_NUM} 완료")
    print("=" * 64)
    print(f"  → Dacon 제출 파일: {sub_path}")
    print(f"  → OOF MAE: {overall_mae:.4f} (옛 submission1: 9.1768)")


if __name__ == "__main__":
    main()
