# -*- coding: utf-8 -*-
"""
eda_phase1.py — 기본 정보 + 타깃 분포

목표:
  1) 데이터 형태(shape, dtype, 메모리)
  2) 결측 현황 (전체 / 컬럼별)
  3) 타깃 분포 (raw, log1p, box, ECDF)
  4) layout overlap (seen vs unseen)
  5) 키(ID, scenario_id) 유일성 + scenario당 행수 일관성

산출:
  eda_outputs/phase1_missing_rate.csv         ← 모든 컬럼의 train/test 결측률
  eda_outputs/phase1_missing_top30.png        ← 결측률 상위 30개 컬럼 막대
  eda_outputs/phase1_target_distribution.png  ← 타깃 4분할(hist, log1p, box, ECDF)
  eda_outputs/phase1_per_layout_mean_target.png ← layout별 타깃 평균 분포

실행:
  (.smart) PS C:\...\스마트물류> python eda_phase1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 비대화형 백엔드 (서버/CI 환경 안전)
import matplotlib.pyplot as plt

# Windows 한글 폰트 (Malgun Gothic 가 기본 설치됨)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# stdout 도 utf-8 (Korean print 안 깨지게)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ──────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
OUT_DIR = Path("eda_outputs")
OUT_DIR.mkdir(exist_ok=True)

TARGET = "avg_delay_minutes_next_30m"


def section(title: str) -> None:
    print()
    print("=" * 64)
    print(f" {title}")
    print("=" * 64)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    layout = pd.read_csv(DATA_DIR / "layout_info.csv")
    samp = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return train, test, layout, samp


def step1_shapes(train, test, layout, samp) -> None:
    section("1. SHAPES & DTYPES & MEMORY")
    print(f"train:             {train.shape}")
    print(f"test:              {test.shape}")
    print(f"layout_info:       {layout.shape}")
    print(f"sample_submission: {samp.shape}")

    print("\ntrain dtypes 분포:")
    print(train.dtypes.value_counts().to_string())

    print(f"\ntrain 메모리: {train.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"test 메모리:  {test.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # train 에만 있는 컬럼 = 타깃
    only_train = sorted(set(train.columns) - set(test.columns))
    only_test = sorted(set(test.columns) - set(train.columns))
    print(f"\ntrain 에만 있는 컬럼: {only_train}")
    print(f"test 에만 있는 컬럼:  {only_test}")


def step2_missing(train, test) -> None:
    section("2. MISSING RATE")

    miss_train = train.isna().mean().sort_values(ascending=False)
    miss_test = test.reindex(columns=train.columns).isna().mean()

    miss_df = pd.DataFrame({
        "train_missing_rate": miss_train,
        "test_missing_rate": miss_test.reindex(miss_train.index),
    })
    out_csv = OUT_DIR / "phase1_missing_rate.csv"
    miss_df.to_csv(out_csv, encoding="utf-8-sig")
    print(f"전체 컬럼 결측률 → {out_csv}")

    n_cols_with_na = int((miss_train > 0).sum())
    print(f"\n결측 1개 이상 가진 컬럼: {n_cols_with_na} / {len(miss_train)}")
    print(f"평균 결측률(컬럼 전체): {miss_train.mean() * 100:.2f}%")

    rows_any_na = int(train.isna().any(axis=1).sum())
    print(f"NaN 1개 이상 행: {rows_any_na} / {len(train)} "
          f"({rows_any_na / len(train) * 100:.3f}%)")

    print("\n결측률 상위 15개 컬럼 (train):")
    print(miss_train[miss_train > 0].head(15).round(4).to_string())

    # figure: top 30 missing
    top30 = miss_train[miss_train > 0].head(30)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(range(len(top30)), top30.values[::-1], color="steelblue")
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30.index[::-1], fontsize=9)
    ax.set_xlabel("missing rate")
    ax.set_title("Top 30 columns by missing rate (train)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out_png = OUT_DIR / "phase1_missing_top30.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"figure → {out_png}")


def step3_target(train) -> None:
    section("3. TARGET DISTRIBUTION")

    y = train[TARGET]
    print(f"평균:    {y.mean():.4f}")
    print(f"중앙값:  {y.median():.4f}")
    print(f"표준편차: {y.std():.4f}")
    print(f"최소:    {y.min():.4f}")
    print(f"최대:    {y.max():.4f}")
    print(f"NaN:     {int(y.isna().sum())}")

    print("\n분위수:")
    for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        print(f"  q{q:0.2f}: {y.quantile(q):.4f}")

    # 4분할 figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) raw histogram
    axes[0, 0].hist(y, bins=80, color="steelblue", edgecolor="white")
    axes[0, 0].axvline(y.mean(), color="red", linestyle="--",
                       label=f"mean={y.mean():.2f}")
    axes[0, 0].axvline(y.median(), color="green", linestyle="--",
                       label=f"median={y.median():.2f}")
    axes[0, 0].set_xlabel(TARGET)
    axes[0, 0].set_ylabel("count")
    axes[0, 0].set_title("Target histogram (raw, right-skewed)")
    axes[0, 0].legend()

    # (0,1) log1p histogram
    axes[0, 1].hist(np.log1p(y), bins=80, color="coral", edgecolor="white")
    axes[0, 1].set_xlabel("log1p(target)")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].set_title("Target histogram (log1p) — bimodal 확인")

    # (1,0) boxplot (가로)
    axes[1, 0].boxplot(y.values, vert=False, showfliers=True,
                       flierprops={"marker": ".", "markersize": 2,
                                   "alpha": 0.3})
    axes[1, 0].set_xlabel(TARGET)
    axes[1, 0].set_yticklabels([""])
    axes[1, 0].set_title("Target boxplot (outlier 다수)")

    # (1,1) ECDF (log x)
    sorted_y = np.sort(y.values)
    ecdf = np.arange(1, len(sorted_y) + 1) / len(sorted_y)
    axes[1, 1].plot(sorted_y + 1e-3, ecdf, color="darkgreen", linewidth=1)
    axes[1, 1].set_xlabel(f"{TARGET} (+ε, log scale)")
    axes[1, 1].set_ylabel("ECDF")
    axes[1, 1].set_title("ECDF of target")
    axes[1, 1].set_xscale("log")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    out_png = OUT_DIR / "phase1_target_distribution.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"figure → {out_png}")


def step4_layout_overlap(train, test) -> None:
    section("4. LAYOUT OVERLAP (seen vs unseen)")

    train_layouts = set(train["layout_id"].unique())
    test_layouts = set(test["layout_id"].unique())
    seen = train_layouts & test_layouts
    unseen = test_layouts - train_layouts

    print(f"train layout 수:                {len(train_layouts)}")
    print(f"test layout 수:                 {len(test_layouts)}")
    print(f"seen   (test ∩ train):          {len(seen)}")
    print(f"unseen (test only):             {len(unseen)}")

    layout_mean = train.groupby("layout_id")[TARGET].mean()
    layout_med = train.groupby("layout_id")[TARGET].median()
    print(f"\nlayout별 타깃 평균 — 평균: {layout_mean.mean():.2f}, "
          f"std: {layout_mean.std():.2f}, "
          f"min: {layout_mean.min():.2f}, max: {layout_mean.max():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(layout_mean, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("per-layout mean(target)")
    axes[0].set_ylabel("# layouts")
    axes[0].set_title(f"Per-layout MEAN target (n={len(layout_mean)} layouts)")
    axes[0].grid(alpha=0.3)

    axes[1].hist(layout_med, bins=50, color="coral", edgecolor="white")
    axes[1].set_xlabel("per-layout median(target)")
    axes[1].set_ylabel("# layouts")
    axes[1].set_title(f"Per-layout MEDIAN target (n={len(layout_med)} layouts)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_png = OUT_DIR / "phase1_per_layout_mean_target.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"figure → {out_png}")


def step5_key_uniqueness(train, test) -> None:
    section("5. KEY UNIQUENESS & SCENARIO STRUCTURE")

    print(f"train.ID 유일: {train['ID'].is_unique}")
    print(f"test.ID 유일:  {test['ID'].is_unique}")
    print(f"train.ID ↔ test.ID 교집합: "
          f"{len(set(train['ID']) & set(test['ID']))} (0이어야 정상)")

    print(f"\ntrain scenario_id 수:  {train['scenario_id'].nunique()}")
    print(f"test scenario_id 수:   {test['scenario_id'].nunique()}")
    print(f"scenario_id 교집합:    "
          f"{len(set(train['scenario_id']) & set(test['scenario_id']))} (0이어야 정상)")

    # 시나리오당 행수
    rows_per_scn_tr = train.groupby("scenario_id").size()
    rows_per_scn_te = test.groupby("scenario_id").size()
    print(f"\ntrain scenario 당 행수 — "
          f"min: {rows_per_scn_tr.min()}, "
          f"max: {rows_per_scn_tr.max()}, "
          f"unique values: {sorted(rows_per_scn_tr.unique())}")
    print(f"test scenario 당 행수  — "
          f"min: {rows_per_scn_te.min()}, "
          f"max: {rows_per_scn_te.max()}, "
          f"unique values: {sorted(rows_per_scn_te.unique())}")

    # scenario_id 가 layout_id 안에 nest 되는지
    scn_per_layout_tr = train.groupby("scenario_id")["layout_id"].nunique()
    print(f"\ntrain: scenario 1개당 layout 수 max = "
          f"{scn_per_layout_tr.max()} (1이어야 정상 — scenario는 한 layout에만 속함)")


def main() -> None:
    print(">>> Phase 1 EDA 시작")
    train, test, layout, samp = load_data()
    step1_shapes(train, test, layout, samp)
    step2_missing(train, test)
    step3_target(train)
    step4_layout_overlap(train, test)
    step5_key_uniqueness(train, test)

    print()
    print("=" * 64)
    print(" Phase 1 완료")
    print("=" * 64)
    print(f"산출물: {OUT_DIR}/")
    print("  - phase1_missing_rate.csv")
    print("  - phase1_missing_top30.png")
    print("  - phase1_target_distribution.png")
    print("  - phase1_per_layout_mean_target.png")


if __name__ == "__main__":
    main()
