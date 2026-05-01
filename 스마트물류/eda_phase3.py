# -*- coding: utf-8 -*-
"""
eda_phase3.py (v2) — 상관관계 + 시간 정렬 + layout_info + interaction

핵심 변경:
  ★ layout_info.csv LEFT JOIN (14개 layout-level 피처 추가)
  ★ robot_total 은 layout_info 의 것 사용 (우리 계산본은 검증 후 폐기)
  ★ interaction features 5개 추가 (stress × inflow 등 — 사용자 가설)

분석 대상 피처:
  - 원본 train 89개 수치
  + layout_info 13개 (layout_id 제외, layout_type 은 인코딩 후)
  + robot 파생 3개 (idle_ratio, charging_ratio, available)
    ※ robot_total 은 layout_info 에서 가져옴
  + capacity 비율 5개
  + stress flag 4개
  + event flag 5개
  + cap flag 5개
  + interaction 5개
  ≈ 130개 수치

Part A: layout_info LEFT JOIN + robot_total 일치 검증
Part B: shift_hour 정렬 가능성 (lag 운명 결판)
Part C: 피처-타깃 상관 (Pearson + Spearman, top 30)
Part D: 피처-피처 상관 heatmap (top 40 by target corr, 클러스터링)
Part E: 상위 피처 binned mean target (비선형 관계)
Part F: 카테고리별 평균 상관

산출:
  eda_outputs/phase3_layout_info_check.txt
  eda_outputs/phase3_shift_hour_check.txt
  eda_outputs/phase3_target_correlation.csv
  eda_outputs/phase3_target_correlation_top30.png
  eda_outputs/phase3_feature_correlation_heatmap.png
  eda_outputs/phase3_top_features_binned.png
  eda_outputs/phase3_category_correlation.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.default"] = "regular"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Phase 2 v2 에서 정의된 것 그대로 import
from eda_phase2_v2 import CATEGORY_MAP as CATEGORY_MAP_BASE, add_robot_derived

# Phase 3 확장 — layout_info + derived + interaction 컬럼 분류
CATEGORY_MAP_EXT: dict[str, str] = dict(CATEGORY_MAP_BASE)
CATEGORY_MAP_EXT.update({
    # layout_info (LEFT JOIN)
    "layout_type": "layout_meta",
    "aisle_width_avg": "layout_meta", "intersection_count": "layout_meta",
    "one_way_ratio": "layout_meta", "pack_station_count": "layout_meta",
    "charger_count": "layout_meta", "layout_compactness": "layout_meta",
    "zone_dispersion": "layout_meta", "robot_total": "layout_meta",
    "building_age_years": "layout_meta", "floor_area_sqm": "layout_meta",
    "ceiling_height_m": "layout_meta", "fire_sprinkler_count": "layout_meta",
    "emergency_exit_count": "layout_meta",
    # one-hot 결과
    "layout_type_grid": "layout_meta", "layout_type_hybrid": "layout_meta",
    "layout_type_narrow": "layout_meta", "layout_type_hub_spoke": "layout_meta",
    # robot 파생 (Phase 2 v2)
    "robot_active_ratio": "robot_derived", "robot_idle_ratio": "robot_derived",
    "robot_charging_ratio": "robot_derived", "available_robots": "robot_derived",
    # capacity 정규화
    "orders_per_robot": "capacity_norm", "skus_per_robot": "capacity_norm",
    "picks_per_robot": "capacity_norm", "orders_per_available": "capacity_norm",
    "orders_per_pack_station": "capacity_norm",
    "orders_per_charger": "capacity_norm",
    # stress flag
    "flag_idle_zero": "stress_flag", "flag_charging_active": "stress_flag",
    "flag_active_high": "stress_flag", "robot_stress_score": "stress_flag",
    # event flag
    "flag_collision": "event_flag", "flag_blocked": "event_flag",
    "flag_fault": "event_flag", "flag_charge_queue": "event_flag",
    "flag_congestion_hot": "event_flag", "incident_score": "event_flag",
    # cap flag
    "flag_temp_extreme": "cap_flag", "flag_humidity_extreme": "cap_flag",
    "flag_pack_saturated": "cap_flag", "flag_dock_long": "cap_flag",
    "flag_truck_wait_peak": "cap_flag",
    # interaction
    "stress_x_inflow": "interaction", "stress_x_orders_per_robot": "interaction",
    "shortage_x_inflow": "interaction", "load_pressure": "interaction",
    "incident_x_stress": "interaction",
})

CATEGORY_MAP = CATEGORY_MAP_EXT  # 다른 함수들은 이걸 사용

DATA_DIR = Path("data")
OUT_DIR = Path("eda_outputs")
OUT_DIR.mkdir(exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"


# ────────────────────────────────────────────────────────────
# layout_info LEFT JOIN
# ────────────────────────────────────────────────────────────
def merge_layout_info(df: pd.DataFrame, layout_info: pd.DataFrame) -> pd.DataFrame:
    """layout_info 의 14개 컬럼을 LEFT JOIN. layout_type 은 그대로 (object)."""
    out = df.merge(layout_info, on="layout_id", how="left",
                   suffixes=("", "_li"))
    return out


# ────────────────────────────────────────────────────────────
# 신규 파생 피처 — Phase 2 v2 발견 + 사용자 결정 반영
# ※ robot_total 은 layout_info 에 이미 있으므로 우리 계산 X
# ────────────────────────────────────────────────────────────
def add_robot_ratios_only(df: pd.DataFrame) -> pd.DataFrame:
    """비율 3개 + available (robot_total 은 layout_info 에서 옴)"""
    df = df.copy()
    df["robot_active_ratio"] = df["robot_active"] / df["robot_total"]
    df["robot_idle_ratio"] = df["robot_idle"] / df["robot_total"]
    df["robot_charging_ratio"] = df["robot_charging"] / df["robot_total"]
    df["available_robots"] = df["robot_active"] + df["robot_idle"]
    return df


def add_capacity_normalized(df: pd.DataFrame) -> pd.DataFrame:
    """창고 규모로 정규화한 부하 비율 (4개)"""
    df = df.copy()
    df["orders_per_robot"] = df["order_inflow_15m"] / df["robot_total"]
    df["skus_per_robot"] = df["unique_sku_15m"] / df["robot_total"]
    df["picks_per_robot"] = df["pick_list_length_avg"] / df["robot_total"]
    df["orders_per_available"] = df["order_inflow_15m"] / (df["available_robots"] + 1)
    # pack_station 정규화 (layout_info)
    df["orders_per_pack_station"] = df["order_inflow_15m"] / (df["pack_station_count"] + 1)
    df["orders_per_charger"] = df["order_inflow_15m"] / (df["charger_count"] + 1)
    return df


def add_stress_flags(df: pd.DataFrame) -> pd.DataFrame:
    """자원 압박 binary flag + 종합 score (4개)"""
    df = df.copy()
    df["flag_idle_zero"] = (df["robot_idle"] == 0).astype(int)
    df["flag_charging_active"] = (df["robot_charging"] > 0).astype(int)
    df["flag_active_high"] = (df["robot_active_ratio"] > 0.8).astype(int)
    df["robot_stress_score"] = (
        df["flag_idle_zero"]
        + df["flag_charging_active"]
        + df["flag_active_high"]
    )
    return df


def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """zero-inflated 컬럼들의 이벤트 발생 flag (5개) + 종합"""
    df = df.copy()
    # NaN 인 행은 0 으로 (이벤트 없음으로 가정)
    df["flag_collision"] = (df["near_collision_15m"].fillna(0) > 0).astype(int)
    df["flag_blocked"] = (df["blocked_path_15m"].fillna(0) > 0).astype(int)
    df["flag_fault"] = (df["fault_count_15m"].fillna(0) > 0).astype(int)
    df["flag_charge_queue"] = (df["charge_queue_length"].fillna(0) > 0).astype(int)
    df["flag_congestion_hot"] = (df["congestion_score"].fillna(0) > 0).astype(int)
    df["incident_score"] = (
        df["flag_collision"] + df["flag_blocked"]
        + df["flag_fault"] + df["flag_charge_queue"]
        + df["flag_congestion_hot"]
    )
    return df


def add_cap_flags(df: pd.DataFrame) -> pd.DataFrame:
    """cap 컬럼의 양 끝 spike 위치 flag (5개)"""
    df = df.copy()
    df["flag_temp_extreme"] = (
        (df["warehouse_temp_avg"] <= 15) | (df["warehouse_temp_avg"] >= 30)
    ).astype(int)
    df["flag_humidity_extreme"] = (
        (df["humidity_pct"] <= 22) | (df["humidity_pct"] >= 78)
    ).astype(int)
    df["flag_pack_saturated"] = (df["pack_utilization"] >= 0.99).astype(int)
    df["flag_dock_long"] = (df["dock_to_stock_hours"] >= 16).astype(int)
    df["flag_truck_wait_peak"] = (df["outbound_truck_wait_min"] >= 14).astype(int)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """가설: stress × inflow = delay 트리거 (5개)"""
    df = df.copy()
    # 부하 한계 + 새 주문 = 지연 폭증
    df["stress_x_inflow"] = df["robot_stress_score"] * df["order_inflow_15m"]
    df["stress_x_orders_per_robot"] = df["robot_stress_score"] * df["orders_per_robot"]
    df["shortage_x_inflow"] = df["flag_idle_zero"] * df["order_inflow_15m"]
    # 종합 부하 압력
    df["load_pressure"] = df["orders_per_available"] * (1 + df["robot_charging_ratio"])
    # 사고 + 부하 = delay 가속
    df["incident_x_stress"] = df["incident_score"] * df["robot_stress_score"]
    return df


def add_all_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 3 분석용 통합 — layout_info merge 후 호출"""
    df = add_robot_ratios_only(df)
    df = add_capacity_normalized(df)
    df = add_stress_flags(df)
    df = add_event_flags(df)
    df = add_cap_flags(df)
    df = add_interaction_features(df)
    return df


def section(title: str) -> None:
    print()
    print("=" * 64)
    print(f" {title}")
    print("=" * 64)


# ────────────────────────────────────────────────────────────
# Part A — shift_hour 검증
# ────────────────────────────────────────────────────────────
def part_a_shift_hour(train: pd.DataFrame) -> None:
    section("A. shift_hour 정렬 가능성 검증")

    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    # 1. 시나리오당 shift_hour unique 값 수
    hours_per_scn = train.groupby("scenario_id")["shift_hour"].nunique(dropna=True)
    log(f"시나리오당 shift_hour unique 값 수:")
    log(f"  count={len(hours_per_scn)}, mean={hours_per_scn.mean():.2f}, "
        f"median={hours_per_scn.median():.0f}, "
        f"min={hours_per_scn.min()}, max={hours_per_scn.max()}")
    log(f"  분포: {hours_per_scn.value_counts().sort_index().to_dict()}")
    log("")
    log("  해석:")
    if hours_per_scn.mean() >= 23:
        log("  → 평균 ≥ 23. shift_hour 가 거의 시나리오 내 unique. 15분 lag 안전!")
    elif hours_per_scn.mean() >= 6 and hours_per_scn.mean() < 10:
        log("  → 평균 ~6~10. shift_hour 는 시간 단위. 15분 lag 불가, hour 단위 lag 가능")
    else:
        log(f"  → 평균 {hours_per_scn.mean():.1f}. 중간 값. 결과 보고 결정")

    # 2. 시나리오당 (shift_hour, day_of_week) unique
    log("")
    combo = train.groupby("scenario_id").apply(
        lambda g: g[["shift_hour", "day_of_week"]].drop_duplicates().shape[0]
    )
    log(f"시나리오당 (shift_hour, day_of_week) unique 행 수:")
    log(f"  mean={combo.mean():.2f}, median={combo.median():.0f}, "
        f"min={combo.min()}, max={combo.max()}")

    # 3. 시나리오의 day_of_week 단일성
    log("")
    days_per_scn = train.groupby("scenario_id")["day_of_week"].nunique(dropna=True)
    log(f"시나리오당 day_of_week unique 수:")
    log(f"  분포: {days_per_scn.value_counts().sort_index().to_dict()}")
    log("  → 1 만 있으면 시나리오는 하루 안에 머무름")

    # 4. 샘플 시나리오 3개의 shift_hour 패턴
    log("")
    log("샘플 시나리오 3개의 shift_hour 정렬 시 패턴:")
    sample_scns = train["scenario_id"].drop_duplicates().head(3)
    for sid in sample_scns:
        sub = train[train["scenario_id"] == sid].sort_values("shift_hour")
        hours = sub["shift_hour"].dropna().astype(int).tolist()
        n_total = len(sub)
        n_hours = len(hours)
        log(f"  {sid}: 총 {n_total}행, shift_hour 비결측 {n_hours}행, "
            f"hours={hours}")

    out_txt = OUT_DIR / "phase3_shift_hour_check.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n결과 저장 → {out_txt}")


# ────────────────────────────────────────────────────────────
# Part B — 피처-타깃 상관
# ────────────────────────────────────────────────────────────
def part_b_target_corr(train_aug: pd.DataFrame) -> pd.DataFrame:
    section("B. 피처-타깃 상관 (Pearson + Spearman)")

    # 분석 대상: 수치 컬럼 (key/target/id 제외)
    exclude = {"ID", "layout_id", "scenario_id", TARGET}
    num_cols = [c for c in train_aug.columns
                if c not in exclude
                and str(train_aug[c].dtype) in ("float64", "int64", "int32")]
    print(f"분석 대상 수치 컬럼 수: {len(num_cols)}")

    # Pearson
    p_corr = train_aug[num_cols + [TARGET]].corr(method="pearson")[TARGET]
    p_corr = p_corr.drop(TARGET)

    # Spearman (랭크 기반, 비선형 관계 포착) — 큰 데이터라 시간 걸림, sample
    sample = train_aug.sample(n=min(50000, len(train_aug)), random_state=42)
    s_corr = sample[num_cols + [TARGET]].corr(method="spearman")[TARGET]
    s_corr = s_corr.drop(TARGET)

    corr_df = pd.DataFrame({
        "column": num_cols,
        "pearson": p_corr.values,
        "spearman": s_corr.reindex(num_cols).values,
        "category": [CATEGORY_MAP.get(c, "derived") for c in num_cols],
    })
    corr_df["abs_pearson"] = corr_df["pearson"].abs()
    corr_df["abs_spearman"] = corr_df["spearman"].abs()
    corr_df = corr_df.sort_values("abs_spearman", ascending=False)

    out_csv = OUT_DIR / "phase3_target_correlation.csv"
    corr_df.round(4).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"전체 상관 → {out_csv}")

    # 콘솔 — top 30
    print("\nSpearman 절대값 기준 top 30:")
    print(corr_df[["column", "category", "pearson", "spearman"]]
          .head(30).round(4).to_string(index=False))

    # figure — top 30 막대
    top30 = corr_df.head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 11))
    y = np.arange(len(top30))
    ax.barh(y - 0.2, top30["pearson"], 0.4, label="Pearson",
            color="steelblue")
    ax.barh(y + 0.2, top30["spearman"], 0.4, label="Spearman",
            color="coral")
    ax.set_yticks(y)
    ax.set_yticklabels(top30["column"], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("correlation with target")
    ax.set_title("Top 30 features by |Spearman| with target")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out_png = OUT_DIR / "phase3_target_correlation_top30.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"figure → {out_png}")

    return corr_df


# ────────────────────────────────────────────────────────────
# Part C — 피처-피처 상관 heatmap (top 40)
# ────────────────────────────────────────────────────────────
def part_c_feature_heatmap(train_aug: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    section("C. 피처-피처 상관 heatmap (top 40)")

    top40_cols = corr_df.head(40)["column"].tolist()
    sample = train_aug[top40_cols].sample(n=min(50000, len(train_aug)),
                                          random_state=42)

    M = sample.corr(method="spearman").fillna(0).values
    n = len(top40_cols)

    # 계층 클러스터링으로 컬럼 정렬
    dist = 1 - np.abs(M)
    np.fill_diagonal(dist, 0)
    cond = squareform(dist, checks=False)
    Z = linkage(cond, method="average")
    order = leaves_list(Z)
    cols_ordered = [top40_cols[i] for i in order]
    M_ord = M[order][:, order]

    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(M_ord, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(cols_ordered, rotation=90, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(cols_ordered, fontsize=8)
    ax.set_title("Top 40 features — feature-feature Spearman corr "
                 "(클러스터링 정렬)")
    plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    out_png = OUT_DIR / "phase3_feature_correlation_heatmap.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"figure → {out_png}")

    # 가장 강하게 상관된 페어 top 10
    print("\n피처 페어 상관 top 10 (자기 자신 제외):")
    upper = pd.DataFrame(M, index=top40_cols, columns=top40_cols).where(
        np.triu(np.ones((n, n)), k=1).astype(bool)
    )
    pairs = upper.abs().stack().sort_values(ascending=False).head(10)
    for (a, b), v in pairs.items():
        actual = M[top40_cols.index(a)][top40_cols.index(b)]
        print(f"  {a:30s} ↔ {b:30s}: {actual:+.3f}")


# ────────────────────────────────────────────────────────────
# Part D — 상위 피처 binned scatter (target 평균)
# ────────────────────────────────────────────────────────────
def part_d_binned_scatter(train_aug: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    section("D. 상위 피처 binned target mean (비선형 관계)")

    top9 = corr_df.head(9)["column"].tolist()
    print(f"top 9 피처: {top9}")

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for i, c in enumerate(top9):
        ax = axes[i // 3, i % 3]
        v = train_aug[[c, TARGET]].dropna()
        if v[c].nunique() < 5:
            ax.set_visible(False)
            continue
        # 동일 빈도 bin (qcut)
        try:
            v["bin"] = pd.qcut(v[c], q=20, duplicates="drop")
            agg = v.groupby("bin", observed=True).agg(
                mean_target=(TARGET, "mean"),
                median_target=(TARGET, "median"),
                bin_center=(c, "mean"),
                count=(c, "size"),
            ).reset_index(drop=True)
            ax.plot(agg["bin_center"], agg["mean_target"],
                    marker="o", color="steelblue", label="mean")
            ax.plot(agg["bin_center"], agg["median_target"],
                    marker="s", color="coral", linestyle="--", label="median")
        except Exception as e:
            ax.text(0.5, 0.5, f"err: {e}", ha="center", transform=ax.transAxes)
        ax.set_title(c, fontsize=10)
        ax.set_xlabel(c, fontsize=8)
        ax.set_ylabel("target", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    out_png = OUT_DIR / "phase3_top_features_binned.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"figure → {out_png}")


# ────────────────────────────────────────────────────────────
# Part E — 카테고리별 평균 절대상관
# ────────────────────────────────────────────────────────────
def part_e_category_corr(corr_df: pd.DataFrame) -> None:
    section("E. 카테고리별 평균 |Spearman|")

    cat_avg = corr_df.groupby("category").agg(
        n_cols=("column", "size"),
        mean_abs_pearson=("abs_pearson", "mean"),
        mean_abs_spearman=("abs_spearman", "mean"),
        max_abs_spearman=("abs_spearman", "max"),
    ).round(4).sort_values("mean_abs_spearman", ascending=False)
    print(cat_avg.to_string())

    # figure
    cat_avg2 = cat_avg.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    y = np.arange(len(cat_avg2))
    ax.barh(y, cat_avg2["mean_abs_spearman"], color="steelblue", label="mean |Spearman|")
    ax.barh(y, cat_avg2["max_abs_spearman"], color="coral", alpha=0.5, label="max |Spearman|")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} (n={n})" for c, n in
                       zip(cat_avg2.index, cat_avg2["n_cols"])], fontsize=9)
    ax.set_xlabel("|Spearman| with target")
    ax.set_title("Category-level correlation strength")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out_png = OUT_DIR / "phase3_category_correlation.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"figure → {out_png}")


# ────────────────────────────────────────────────────────────
# Part 0 — layout_info LEFT JOIN + robot_total 일치 검증
# ────────────────────────────────────────────────────────────
def part_0_layout_info(train: pd.DataFrame, layout_info: pd.DataFrame) -> pd.DataFrame:
    section("0. layout_info LEFT JOIN + robot_total 일치 검증")

    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    log(f"layout_info shape: {layout_info.shape}")
    log(f"layout_info 컬럼: {list(layout_info.columns)}")
    log("")

    # robot_total: train 행에서 active+idle+charging vs layout_info
    train_robot_total = train.groupby("layout_id").apply(
        lambda g: (g["robot_active"] + g["robot_idle"] + g["robot_charging"]).iloc[0]
    )
    li_indexed = layout_info.set_index("layout_id")["robot_total"]

    # 매칭되는 layout 만 비교
    common = train_robot_total.index.intersection(li_indexed.index)
    diff = (train_robot_total.loc[common] - li_indexed.loc[common])
    log(f"train layout 중 layout_info 와 매칭된 수: {len(common)} / {len(train_robot_total)}")
    log(f"robot_total 차이 (계산본 - layout_info):")
    log(f"  mean={diff.mean():+.4f}, std={diff.std():.4f}, "
        f"min={diff.min():+.0f}, max={diff.max():+.0f}")
    if (diff == 0).all():
        log("  → 100% 일치. 계산본은 폐기, layout_info 의 robot_total 그대로 사용")
    else:
        log(f"  → 일치 {(diff == 0).sum()} / 불일치 {(diff != 0).sum()}")

    # layout_type 분포
    log("")
    log(f"layout_type 분포: {layout_info['layout_type'].value_counts().to_dict()}")

    # LEFT JOIN
    train_merged = merge_layout_info(train, layout_info)
    log(f"\nLEFT JOIN 후 train shape: {train_merged.shape}")

    # NaN 체크 (모든 layout 이 layout_info 에 있는지)
    n_unmatched = train_merged["layout_type"].isna().sum()
    log(f"layout_info 에 없어서 NaN 인 train 행: {n_unmatched}")

    # layout_type 인코딩 (one-hot)
    train_merged = pd.get_dummies(train_merged, columns=["layout_type"],
                                   prefix="layout_type", drop_first=False, dtype=int)
    log(f"layout_type 인코딩 후 shape: {train_merged.shape}")

    out_txt = OUT_DIR / "phase3_layout_info_check.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n결과 → {out_txt}")
    return train_merged


# ────────────────────────────────────────────────────────────
def main() -> None:
    print(">>> Phase 3 v2 시작 — layout_info + 상관 + shift_hour")
    train = pd.read_csv(DATA_DIR / "train.csv")
    layout_info = pd.read_csv(DATA_DIR / "layout_info.csv")
    print(f"train: {train.shape}, layout_info: {layout_info.shape}")

    # Part 0 — layout_info merge
    train_m = part_0_layout_info(train, layout_info)

    # 신규 derived 추가 (layout_info 머지된 후 호출)
    print("\n파생 피처 추가 (robot 비율 + capacity + flags + interaction)...")
    train_aug = add_all_derived(train_m)
    n_added = train_aug.shape[1] - train_m.shape[1]
    print(f"  → {train_aug.shape[1]} 컬럼 ({n_added} 신규)")

    part_a_shift_hour(train_aug)
    corr_df = part_b_target_corr(train_aug)
    part_c_feature_heatmap(train_aug, corr_df)
    part_d_binned_scatter(train_aug, corr_df)
    part_e_category_corr(corr_df)

    print()
    print("=" * 64)
    print(" Phase 3 v2 완료")
    print("=" * 64)
    print(f"산출물: {OUT_DIR}/")
    print("  - phase3_layout_info_check.txt          ← layout_info merge 검증")
    print("  - phase3_shift_hour_check.txt           ← lag 운명 결정")
    print("  - phase3_target_correlation.csv         ← 전체 컬럼 상관")
    print("  - phase3_target_correlation_top30.png   ← top 30 막대")
    print("  - phase3_feature_correlation_heatmap.png← 클러스터 heatmap")
    print("  - phase3_top_features_binned.png        ← 비선형 관계")
    print("  - phase3_category_correlation.png       ← 카테고리 강도")


if __name__ == "__main__":
    main()
