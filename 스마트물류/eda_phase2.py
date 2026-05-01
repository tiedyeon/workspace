# -*- coding: utf-8 -*-
"""
eda_phase2.py — 컬럼 인벤토리 + 결측 패턴 (deep dive)

Phase 1 에서 본 의문점들을 파헤친다:
  - 결측률이 모든 컬럼에서 ~12%로 거의 균일했음 → 결측이 행/시나리오 단위로 일어나나?
  - object 3개 / int 3개 컬럼의 정체는?
  - 94개 수치 컬럼의 분포 전반 (저분산·극단 outlier·이상한 형태 식별)

Part A — 94 컬럼 인벤토리 (이름 / dtype / 결측률 / 카테고리)
Part B — object·int 컬럼 정체 점검
Part C — 카테고리별 분포 히스토그램 격자
Part D — 결측 동시발생 패턴 (컬럼 간 + scenario 단위)
Part E — 저분산·상수 컬럼 탐지

산출:
  eda_outputs/phase2_column_inventory.csv          ← 94 컬럼 정리표
  eda_outputs/phase2_numeric_stats.csv             ← 수치 컬럼 통계
  eda_outputs/phase2_distributions_<카테고리>.png  ← 카테고리별 격자 (5~7개)
  eda_outputs/phase2_missing_per_row_hist.png      ← 행당 결측 컬럼 수
  eda_outputs/phase2_missing_cooccurrence.png      ← 결측 동시발생 heatmap
  eda_outputs/phase2_missing_by_scenario.png       ← scenario 단위 결측 패턴
  eda_outputs/phase2_low_variance_cols.csv         ← 저분산/상수 후보

실행:
  (.smart) PS C:\...\스마트물류> python eda_phase2.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 한글 폰트 + 마이너스 기호 fallback
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.default"] = "regular"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = Path("data")
OUT_DIR = Path("eda_outputs")
OUT_DIR.mkdir(exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"


def section(title: str) -> None:
    print()
    print("=" * 64)
    print(f" {title}")
    print("=" * 64)


# ───────────────────────────────────────────────────────────
# 컬럼 카테고리 자동 분류 (이름 prefix/keyword 기반)
# ───────────────────────────────────────────────────────────
CATEGORY_RULES = [
    # (카테고리명, 매칭할 키워드들 — OR)
    ("key",         [r"^ID$", r"^layout_id$", r"^scenario_id$"]),
    ("target",      [r"avg_delay_minutes_next_30m"]),
    ("robot_fleet", [r"robot", r"fleet", r"agv"]),
    ("battery",     [r"battery", r"charge"]),
    ("order_sku",   [r"order", r"sku", r"inventory", r"throughput"]),
    ("worker",      [r"worker", r"shift", r"operator", r"staff"]),
    ("pack_pick",   [r"pack", r"pick", r"sort", r"staging"]),
    ("congestion",  [r"congestion", r"density", r"zone", r"util"]),
    ("environment", [r"humidity", r"temp", r"weather", r"light"]),
    ("quality",     [r"barcode", r"error", r"score", r"calibration",
                     r"recovery", r"success"]),
    ("time",        [r"hour", r"day", r"time", r"15m", r"30m", r"60m"]),
]


def categorize(col: str) -> str:
    if col == TARGET:
        return "target"
    for cat, patterns in CATEGORY_RULES:
        for p in patterns:
            if re.search(p, col, flags=re.IGNORECASE):
                return cat
    return "other"


# ───────────────────────────────────────────────────────────
# Part A — 컬럼 인벤토리
# ───────────────────────────────────────────────────────────
def part_a_inventory(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    section("A. COLUMN INVENTORY (94)")

    miss_tr = train.isna().mean()
    miss_te = test.reindex(columns=train.columns).isna().mean()

    inv = pd.DataFrame({
        "column": train.columns,
        "dtype": [str(t) for t in train.dtypes],
        "n_unique": [train[c].nunique(dropna=True) for c in train.columns],
        "missing_train": miss_tr.values,
        "missing_test": miss_te.values,
        "category": [categorize(c) for c in train.columns],
    })
    inv["missing_train"] = inv["missing_train"].round(4)
    inv["missing_test"] = inv["missing_test"].round(4)

    out = OUT_DIR / "phase2_column_inventory.csv"
    inv.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"전체 컬럼 인벤토리 → {out}")

    print("\n카테고리별 컬럼 수:")
    print(inv["category"].value_counts().to_string())

    print("\nother 로 분류된 컬럼 (수동 분류 필요할 수 있음):")
    others = inv[inv["category"] == "other"]["column"].tolist()
    for c in others:
        print(f"  - {c}")

    return inv


# ───────────────────────────────────────────────────────────
# Part B — object / int 컬럼 정체
# ───────────────────────────────────────────────────────────
def part_b_obj_int(train: pd.DataFrame) -> None:
    section("B. OBJECT / INT COLUMN INSPECTION")

    obj_cols = train.select_dtypes(include="object").columns.tolist()
    int_cols = train.select_dtypes(include="int64").columns.tolist()

    print(f"object 컬럼 ({len(obj_cols)}): {obj_cols}")
    for c in obj_cols:
        n_unique = train[c].nunique()
        sample_vals = train[c].dropna().unique()[:5]
        print(f"  - {c}: nunique={n_unique}, 예시={list(sample_vals)}")

    print(f"\nint 컬럼 ({len(int_cols)}): {int_cols}")
    for c in int_cols:
        n_unique = train[c].nunique()
        vmin, vmax = train[c].min(), train[c].max()
        sample_vals = sorted(train[c].dropna().unique())[:10]
        print(f"  - {c}: nunique={n_unique}, range=[{vmin}, {vmax}], "
              f"예시={sample_vals}")


# ───────────────────────────────────────────────────────────
# Part C — 카테고리별 분포 격자
# ───────────────────────────────────────────────────────────
def part_c_distributions(train: pd.DataFrame, inv: pd.DataFrame) -> None:
    section("C. NUMERIC DISTRIBUTIONS BY CATEGORY")

    # numeric 컬럼만 (key, target 제외)
    num_inv = inv[
        (inv["dtype"].isin(["float64", "int64"])) &
        (~inv["category"].isin(["key", "target"]))
    ].copy()

    cats = num_inv["category"].unique()
    print(f"분포 격자 생성 카테고리: {list(cats)}")

    stats_rows = []

    for cat in cats:
        cols = num_inv[num_inv["category"] == cat]["column"].tolist()
        if not cols:
            continue
        n = len(cols)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3.6, nrows * 2.8))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        for i, c in enumerate(cols):
            ax = axes[i // ncols, i % ncols]
            vals = train[c].dropna()
            if len(vals) == 0:
                ax.set_visible(False)
                continue
            ax.hist(vals, bins=40, color="steelblue", edgecolor="white")
            ax.set_title(c, fontsize=9)
            ax.tick_params(labelsize=7)

            stats_rows.append({
                "column": c, "category": cat,
                "mean": vals.mean(), "std": vals.std(),
                "min": vals.min(), "q01": vals.quantile(0.01),
                "median": vals.median(),
                "q99": vals.quantile(0.99), "max": vals.max(),
                "n_unique": vals.nunique(),
            })

        # 빈 axes 숨기기
        for j in range(n, nrows * ncols):
            axes[j // ncols, j % ncols].set_visible(False)

        fig.suptitle(f"[{cat}] 컬럼 분포 (n={n})", fontsize=12)
        plt.tight_layout()
        out = OUT_DIR / f"phase2_distributions_{cat}.png"
        plt.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  figure → {out}")

    stats_df = pd.DataFrame(stats_rows)
    stats_df = stats_df.round(4)
    out_csv = OUT_DIR / "phase2_numeric_stats.csv"
    stats_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n수치 컬럼 통계 → {out_csv}")


# ───────────────────────────────────────────────────────────
# Part D — 결측 패턴
# ───────────────────────────────────────────────────────────
def part_d_missing_patterns(train: pd.DataFrame) -> None:
    section("D. MISSING CO-OCCURRENCE PATTERNS")

    feat_cols = [c for c in train.columns
                 if c not in ("ID", "layout_id", "scenario_id", TARGET)]
    M = train[feat_cols].isna()

    # D-1. 행당 결측 컬럼 수 분포
    miss_per_row = M.sum(axis=1)
    print(f"행당 결측 컬럼 수 — "
          f"mean: {miss_per_row.mean():.2f}, "
          f"median: {miss_per_row.median():.0f}, "
          f"min: {miss_per_row.min()}, max: {miss_per_row.max()}")
    print(f"결측 0개인 행: {(miss_per_row == 0).sum()} "
          f"({(miss_per_row == 0).mean() * 100:.3f}%)")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(miss_per_row, bins=range(0, miss_per_row.max() + 2),
            color="steelblue", edgecolor="white")
    ax.set_xlabel("행당 결측 컬럼 수")
    ax.set_ylabel("# rows")
    ax.set_title(f"Distribution of missing-cell count per row "
                 f"(mean={miss_per_row.mean():.1f})")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "phase2_missing_per_row_hist.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  figure → {out}")

    # D-2. 결측 동시발생 — isna flag 들의 상관관계
    # 너무 많으면 결측률 > 5% 인 컬럼만 추리기
    miss_rate = M.mean()
    cols_high_miss = miss_rate[miss_rate > 0.05].index.tolist()
    print(f"\n결측률 > 5% 컬럼 수: {len(cols_high_miss)}")

    if len(cols_high_miss) >= 2:
        corr = M[cols_high_miss].astype(float).corr()
        n = len(cols_high_miss)
        fig, ax = plt.subplots(figsize=(min(14, 0.16 * n + 5),
                                        min(14, 0.16 * n + 5)))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1,
                       aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(cols_high_miss, rotation=90, fontsize=6)
        ax.set_yticks(range(n))
        ax.set_yticklabels(cols_high_miss, fontsize=6)
        ax.set_title("Missing co-occurrence (corr of isna flags)")
        plt.colorbar(im, ax=ax, fraction=0.04)
        plt.tight_layout()
        out = OUT_DIR / "phase2_missing_cooccurrence.png"
        plt.savefig(out, dpi=120)
        plt.close(fig)
        print(f"  figure → {out}")

        # 콘솔 요약: 가장 강하게 같이 결측되는 페어 top 10
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = upper.stack().sort_values(ascending=False)
        print(f"\n결측 동시발생 상위 페어 (corr 최대값 10):")
        for (a, b), v in pairs.head(10).items():
            print(f"  {a:35s} ↔ {b:35s} : {v:.3f}")

    # D-3. scenario 단위 결측 — 같은 시나리오의 25행이 같은 컬럼에서 결측인가?
    print("\n[scenario 단위 결측 점검]")
    print("같은 scenario 의 25행 중 한 컬럼이 결측이면, 그 시나리오의 다른 24행도 같은 컬럼에서 결측일까?")

    # 결측률 top 5 컬럼에 대해 scenario 단위 결측 일관성 측정
    top5_miss_cols = miss_rate.sort_values(ascending=False).head(5).index.tolist()
    cons_rows = []
    for c in top5_miss_cols:
        # scenario별 해당 컬럼의 결측 비율
        per_scn = train.groupby("scenario_id")[c].apply(lambda s: s.isna().mean())
        # 0% 또는 100%면 시나리오 단위 결측, 중간값이 많으면 행 단위 결측
        all_or_none = ((per_scn == 0) | (per_scn == 1)).mean()
        cons_rows.append({
            "column": c,
            "overall_missing_rate": miss_rate[c],
            "scenario_all_or_none_rate": all_or_none,
            "n_scenarios_partial": int(((per_scn > 0) & (per_scn < 1)).sum()),
        })
    cons_df = pd.DataFrame(cons_rows)
    print(cons_df.round(4).to_string(index=False))

    # 시각: top 5 컬럼의 scenario별 결측 비율 히스토그램
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
    for i, c in enumerate(top5_miss_cols):
        per_scn = train.groupby("scenario_id")[c].apply(lambda s: s.isna().mean())
        axes[i].hist(per_scn, bins=[0, 0.04, 0.2, 0.4, 0.6, 0.8, 0.96, 1.0],
                     color="steelblue", edgecolor="white")
        axes[i].set_title(c, fontsize=9)
        axes[i].set_xlabel("scenario당 결측 비율")
        axes[i].set_ylabel("# scenarios")
    fig.suptitle("scenario당 결측 비율 — 0/1 양 끝에 몰리면 시나리오 단위 결측",
                 fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / "phase2_missing_by_scenario.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  figure → {out}")


# ───────────────────────────────────────────────────────────
# Part E — 저분산 / 상수 컬럼
# ───────────────────────────────────────────────────────────
def part_e_low_variance(train: pd.DataFrame, inv: pd.DataFrame) -> None:
    section("E. LOW-VARIANCE / CONSTANT COLUMNS")

    num_cols = inv[
        (inv["dtype"].isin(["float64", "int64"])) &
        (~inv["category"].isin(["key", "target"]))
    ]["column"].tolist()

    rows = []
    for c in num_cols:
        v = train[c].dropna()
        if len(v) == 0:
            continue
        rows.append({
            "column": c,
            "n_unique": v.nunique(),
            "std": v.std(),
            "mode_share": v.value_counts(normalize=True).iloc[0]
                          if v.nunique() > 0 else 1.0,
        })
    df = pd.DataFrame(rows).sort_values("n_unique")
    suspect = df[(df["n_unique"] <= 2) | (df["mode_share"] > 0.95)]
    print(f"저분산 후보 (n_unique≤2 또는 단일값 점유율>95%): {len(suspect)} 개")
    if len(suspect) > 0:
        print(suspect.head(20).round(4).to_string(index=False))
    else:
        print("  → 없음 (모든 수치 컬럼이 의미 있는 분산을 가짐)")

    out = OUT_DIR / "phase2_low_variance_cols.csv"
    df.round(4).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  저분산 검토표 → {out}")


# ───────────────────────────────────────────────────────────
def main() -> None:
    print(">>> Phase 2 EDA 시작")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    inv = part_a_inventory(train, test)
    part_b_obj_int(train)
    part_c_distributions(train, inv)
    part_d_missing_patterns(train)
    part_e_low_variance(train, inv)

    print()
    print("=" * 64)
    print(" Phase 2 완료")
    print("=" * 64)
    print(f"산출물: {OUT_DIR}/")
    print("  - phase2_column_inventory.csv (94 컬럼 정리표)")
    print("  - phase2_numeric_stats.csv (수치 컬럼 통계)")
    print("  - phase2_distributions_<카테고리>.png (카테고리별 분포 격자)")
    print("  - phase2_missing_per_row_hist.png")
    print("  - phase2_missing_cooccurrence.png")
    print("  - phase2_missing_by_scenario.png  ← 핵심")
    print("  - phase2_low_variance_cols.csv")


if __name__ == "__main__":
    main()
