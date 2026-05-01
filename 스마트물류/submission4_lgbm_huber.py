# -*- coding: utf-8 -*-
"""
submission4_lgbm_huber.py — LightGBM Huber 손실 + s3 와 동일 피처

격리 원칙 — submission3 와 변경점:
  ★ objective: 'quantile' (alpha=0.5) → 'huber' (alpha=5.0)
  나머지 (피처 115개, CV, 하이퍼파라미터, NaN 처리, clip) 모두 동일

Huber alpha=5.0 의 의미:
  - |residual| < 5 : MSE (부드러운 그래디언트로 일반 행 빠른 수렴)
  - |residual| >= 5: MAE (robust, outlier 에 안 끌려감)
  - 우리 OOF MAE 가 ~9 라 alpha=5 는 일반 행을 대체로 부드럽게 다룸

기록:
  s1 (옛, LightGBM Quantile, 209 피처): OOF 9.1768, Public 10.6461
  s2 (옛, LightGBM log1p+MSE, 111 피처): OOF 9.1955, Public 10.6703 (폐기)
  s3 (이번, LightGBM Quantile, 115 피처): OOF 9.1771, Public 10.6557
  s4 (이 파일, LightGBM Huber alpha=5.0, 115 피처): OOF TBD, Public TBD

산출:
  outputs/submission4_lgbm_huber.csv
  outputs/submission4_oof.csv / .npy
  outputs/submission4_test_pred.npy
  outputs/submission4_importance.csv / .png
  outputs/submission4_summary.txt

실행:
  (.smart) python submission4_lgbm_huber.py
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

from eda_phase3 import merge_layout_info, add_all_derived

# ──────────────────────────────────────────────
SUBMISSION_NUM = 4
SEED = 42
N_FOLDS = 5

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

# 환경 노이즈 (s3 와 동일)
NOISE_COLS = [
    "warehouse_temp_avg", "humidity_pct", "external_temp_c",
    "lighting_level_lux", "cold_storage_temp_c",
    "ambient_noise_db", "air_quality_idx", "co2_level_ppm",
    "floor_vibration_idx",
    "wind_speed_kmh", "precipitation_mm",
    "wms_response_time_ms", "wifi_signal_db", "network_latency_ms",
    "hvac_power_kw",
    "ups_battery_pct",
]

DUPLICATE_FLAGS = [
    "flag_collision", "flag_blocked", "flag_fault",
    "flag_charge_queue", "flag_congestion_hot",
]

DUPLICATE_RATIOS = ["robot_active_ratio"]

# ★ 손실만 변경 — 나머지 동일
LGB_PARAMS = dict(
    objective="huber",
    alpha=5.0,                # Huber 임계값 (residual scale)
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
    """s3 와 동일 — layout_info merge + one-hot + 파생"""
    print("layout_info LEFT JOIN...")
    train_m = merge_layout_info(train, layout_info)
    test_m = merge_layout_info(test, layout_info)

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
    """s3 와 동일 — 키·타깃·노이즈·중복 제외"""
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
    print(f">>> submission{SUBMISSION_NUM} 시작 — LightGBM Huber (alpha=5.0)")

    # 1. Load
    section("1. 데이터 로드")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    layout_info = pd.read_csv(DATA_DIR / "layout_info.csv")
    print(f"train: {train.shape}, test: {test.shape}, "
          f"layout_info: {layout_info.shape}")

    # 2. Feature engineering
    section("2. 피처 엔지니어링 (s3 와 동일)")
    train_aug, test_aug = prepare_features(train, test, layout_info)
    print(f"train_aug: {train_aug.shape}, test_aug: {test_aug.shape}")

    feature_cols = get_feature_cols(train_aug, test_aug)
    print(f"\n사용 피처 수: {len(feature_cols)} (s3 와 동일해야 함)")

    X = train_aug[feature_cols]
    y = train_aug[TARGET]
    groups = train_aug["layout_id"]
    X_test = test_aug[feature_cols]

    print(f"\nX: {X.shape}, y mean={y.mean():.2f}, median={y.median():.2f}")
    print(f"X_test: {X_test.shape}")

    # 3. Train with GroupKFold
    section(f"3. 학습 — LightGBM Huber + GroupKFold({N_FOLDS}, by=layout_id)")
    print(f"손실: huber (alpha={LGB_PARAMS['alpha']})")
    print(f"  ← s3 의 quantile (alpha=0.5) 와 차이점: 작은 residual 부드럽게")

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
    print()
    print("vs submission3 (Quantile, OOF 9.1771):")
    delta_oof = overall_mae - 9.1771
    print(f"  OOF Δ = {delta_oof:+.4f}")

    # 5. Save submission CSV
    section("5. 산출물 저장")
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub[TARGET] = np.clip(test_pred, 0, None)
    sub_path = OUT_DIR / f"submission{SUBMISSION_NUM}_lgbm_huber.csv"
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

    # importance figure
    top30 = imp_df.head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.barh(range(len(top30)), top30["importance"], color="steelblue")
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30["feature"], fontsize=9)
    ax.set_xlabel("LightGBM importance (gain)")
    ax.set_title(f"submission{SUBMISSION_NUM} (Huber α=5) top 30 — "
                 f"OOF MAE = {overall_mae:.4f}")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig_path = OUT_DIR / f"submission{SUBMISSION_NUM}_importance.png"
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"figure → {fig_path}")

    # 8. Summary text
    summary = f"""submission{SUBMISSION_NUM} — LightGBM Huber (alpha=5.0)

OOF MAE:    {overall_mae:.4f}
Fold MAE:   [{', '.join(f'{m:.4f}' for m in fold_maes)}]
Fold std:   {np.std(fold_maes):.4f}
학습 시간:  {elapsed_min:.2f} 분

피처 수:    {len(feature_cols)} (s3 와 동일)
train rows: {len(X)}, test rows: {len(X_test)}
groups:     {groups.nunique()} unique layouts

LightGBM 하이퍼파라미터 (s3 와 동일, 손실만 변경):
  {LGB_PARAMS}

격리 원칙: s3 와 단 한 가지 차이 — objective ('quantile'→'huber') + alpha (0.5→5.0)
효과 측정: OOF MAE 차이 = {overall_mae - 9.1771:+.4f}

비교:
  s1 (옛, LGBM Quantile, 209 피처): OOF 9.1768, Public 10.6461
  s2 (옛, LGBM log1p+MSE, 111 피처): OOF 9.1955, Public 10.6703 (폐기)
  s3 (이번, LGBM Quantile, 115 피처): OOF 9.1771, Public 10.6557
  s4 (이 파일, LGBM Huber α=5, 115 피처): OOF {overall_mae:.4f}, Public TBD

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
    print(f"  → OOF MAE: {overall_mae:.4f} "
          f"(s3: 9.1771, Δ {overall_mae - 9.1771:+.4f})")


if __name__ == "__main__":
    main()
