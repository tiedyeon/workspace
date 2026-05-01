# -*- coding: utf-8 -*-
"""
eda_phase2_v2.py — Phase 2 재정리

v1 의 한계:
  - "other" 카테고리에 24개 컬럼이 빠짐 (정규식이 약했음)
  - robot 카운트 3개의 활용을 안 함

v2 의 변경:
  - CATEGORY_MAP: 94 컬럼을 명시적으로 매핑 (regex X, dict 으로)
  - 파생 피처 4개 추가: robot_total, robot_active/idle/charging_ratio
  - robot_utilization 과 robot_active_ratio 의 관계 검증
  - 카테고리별 분포 figure 깔끔히 다시 그림 (17개 카테고리)

산출:
  eda_outputs/phase2v2_column_inventory.csv      ← 94 컬럼 + category_v2
  eda_outputs/phase2v2_distributions_<cat>.png   ← 카테고리별 분포 격자
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

DATA_DIR = Path("data")
OUT_DIR = Path("eda_outputs")
OUT_DIR.mkdir(exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"


# ────────────────────────────────────────────────────────────
# CATEGORY_MAP — 94 컬럼 명시적 분류 (다음 페이즈에서도 import 해서 재사용)
# ────────────────────────────────────────────────────────────
CATEGORY_MAP: dict[str, str] = {
    # key (3)
    "ID": "key", "layout_id": "key", "scenario_id": "key",

    # target (1)
    "avg_delay_minutes_next_30m": "target",

    # robot — 로봇 자체 상태/속성 (9)
    "robot_active": "robot", "robot_idle": "robot", "robot_charging": "robot",
    "robot_utilization": "robot", "fleet_age_months_avg": "robot",
    "robot_firmware_update_days": "robot", "agv_task_success_rate": "robot",
    "robot_calibration_score": "robot", "avg_idle_duration_min": "robot",

    # battery — 배터리/충전 (7)
    "battery_mean": "battery", "battery_std": "battery",
    "low_battery_ratio": "battery", "charge_queue_length": "battery",
    "avg_charge_wait": "battery", "charge_efficiency_pct": "battery",
    "battery_cycle_count_avg": "battery",

    # order_sku — 주문 흐름 (10)
    "order_inflow_15m": "order_sku", "unique_sku_15m": "order_sku",
    "avg_items_per_order": "order_sku", "urgent_order_ratio": "order_sku",
    "sku_concentration": "order_sku", "return_order_ratio": "order_sku",
    "order_wave_count": "order_sku", "bulk_order_ratio": "order_sku",
    "inventory_turnover_rate": "order_sku", "backorder_ratio": "order_sku",

    # item_attr — 품목 특성 (3)
    "heavy_item_ratio": "item_attr", "cold_chain_ratio": "item_attr",
    "avg_package_weight_kg": "item_attr",

    # pack_pick — 포장/피킹 (6)
    "pack_utilization": "pack_pick", "pick_list_length_avg": "pack_pick",
    "sort_accuracy_pct": "pack_pick", "packaging_material_cost": "pack_pick",
    "label_print_queue": "pack_pick", "pallet_wrap_time_min": "pack_pick",

    # staging_dock — 출고/도크 (6)
    "staging_area_util": "staging_dock", "loading_dock_util": "staging_dock",
    "outbound_truck_wait_min": "staging_dock", "dock_to_stock_hours": "staging_dock",
    "cross_dock_ratio": "staging_dock", "express_lane_util": "staging_dock",

    # congestion — 혼잡/공간이용 (7)
    "congestion_score": "congestion", "max_zone_density": "congestion",
    "storage_density_pct": "congestion", "vertical_utilization": "congestion",
    "racking_height_avg_m": "congestion", "lighting_zone_variance": "congestion",
    "zone_temp_variance": "congestion",

    # traffic_path — 동선/이동 (6)
    "aisle_traffic_score": "traffic_path", "intersection_wait_time_avg": "traffic_path",
    "path_optimization_score": "traffic_path", "avg_trip_distance": "traffic_path",
    "near_collision_15m": "traffic_path", "blocked_path_15m": "traffic_path",

    # incident — 사고/오류 (3)
    "fault_count_15m": "incident", "avg_recovery_time": "incident",
    "scanner_error_rate": "incident",

    # operations — 운영 행위 (5)
    "manual_override_ratio": "operations", "replenishment_overlap": "operations",
    "quality_check_rate": "operations", "task_reassign_15m": "operations",
    "daily_forecast_accuracy": "operations",

    # environment — 실내 환경 (5)
    "warehouse_temp_avg": "environment", "humidity_pct": "environment",
    "external_temp_c": "environment", "lighting_level_lux": "environment",
    "cold_storage_temp_c": "environment",

    # atmosphere — 대기/공기 질 (4)
    "ambient_noise_db": "atmosphere", "air_quality_idx": "atmosphere",
    "co2_level_ppm": "atmosphere", "floor_vibration_idx": "atmosphere",

    # weather — 외부 기상 (2)
    "wind_speed_kmh": "weather", "precipitation_mm": "weather",

    # infra_it — IT 인프라 (4)
    "wms_response_time_ms": "infra_it", "wifi_signal_db": "infra_it",
    "network_latency_ms": "infra_it", "hvac_power_kw": "infra_it",

    # power — 전력 (1)
    "ups_battery_pct": "power",

    # safety_quality — 안전·품질 KPI (4)
    "barcode_read_success_rate": "safety_quality",
    "safety_score_monthly": "safety_quality",
    "maintenance_schedule_score": "safety_quality",
    "kpi_otd_pct": "safety_quality",

    # equipment — 비로봇 장비 (2)
    "forklift_active_count": "equipment", "conveyor_speed_mps": "equipment",

    # time — 시간 인덱스 (2)
    "shift_hour": "time", "day_of_week": "time",

    # worker — 인력 (4)
    "staff_on_floor": "worker", "prev_shift_volume": "worker",
    "worker_avg_tenure_months": "worker", "shift_handover_delay_min": "worker",
}
# 합 = 94


# ────────────────────────────────────────────────────────────
# 파생 피처 — 다음 페이즈/모델링에서 그대로 import 해서 재사용
# ────────────────────────────────────────────────────────────
def add_robot_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    robot_active/idle/charging 카운트로부터 4개 파생 피처 생성.
      - robot_total: 창고의 총 로봇 수 (= active + idle + charging)
      - robot_active_ratio:   active / total  (가동률)
      - robot_idle_ratio:     idle / total    (유휴율)
      - robot_charging_ratio: charging / total (충전 중 비율)
    """
    df = df.copy()
    df["robot_total"] = (df["robot_active"]
                         + df["robot_idle"]
                         + df["robot_charging"])
    df["robot_active_ratio"] = df["robot_active"] / df["robot_total"]
    df["robot_idle_ratio"] = df["robot_idle"] / df["robot_total"]
    df["robot_charging_ratio"] = df["robot_charging"] / df["robot_total"]
    return df


def section(title: str) -> None:
    print()
    print("=" * 64)
    print(f" {title}")
    print("=" * 64)


def plot_distributions(df: pd.DataFrame, cols: list[str], name: str) -> None:
    """카테고리별 히스토그램 격자 1장"""
    cols = [c for c in cols if c in df.columns]
    n = len(cols)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 2.8))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for i, c in enumerate(cols):
        ax = axes[i // ncols, i % ncols]
        vals = df[c].dropna()
        if len(vals) == 0:
            ax.set_visible(False)
            continue
        ax.hist(vals, bins=40, color="steelblue", edgecolor="white")
        ax.set_title(c, fontsize=9)
        ax.tick_params(labelsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle(f"[{name}] 컬럼 분포 (n={n})", fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / f"phase2v2_distributions_{name}.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  figure → {out}")


# ────────────────────────────────────────────────────────────
def main() -> None:
    print(">>> Phase 2 v2 시작 — categorize_v2 + 파생 피처")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    # ── A. 카테고리 매핑 검증 ─────────────────────────────
    section("A. CATEGORIZE_V2 검증")
    all_cols = set(train.columns)
    mapped = set(CATEGORY_MAP.keys())
    missing_cols = all_cols - mapped
    extra_cols = mapped - all_cols
    if missing_cols:
        print(f"⚠ 매핑 누락 {len(missing_cols)}: {sorted(missing_cols)}")
    if extra_cols:
        print(f"⚠ 매핑에만 있는 컬럼 {len(extra_cols)}: {sorted(extra_cols)}")
    if not missing_cols and not extra_cols:
        print(f"OK — 94 컬럼 모두 매핑됨 ({len(mapped)} entries)")

    cats_series = pd.Series(CATEGORY_MAP)
    print("\n카테고리별 컬럼 수:")
    print(cats_series.value_counts().to_string())

    # 인벤토리 v2
    miss_tr = train.isna().mean()
    miss_te = test.reindex(columns=train.columns).isna().mean()
    inv = pd.DataFrame({
        "column": train.columns,
        "category_v2": [CATEGORY_MAP.get(c, "?") for c in train.columns],
        "dtype": [str(t) for t in train.dtypes],
        "n_unique": [train[c].nunique(dropna=True) for c in train.columns],
        "missing_train": miss_tr.values.round(4),
        "missing_test": miss_te.values.round(4),
    })
    out_csv = OUT_DIR / "phase2v2_column_inventory.csv"
    inv.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n인벤토리 v2 → {out_csv}")

    # ── B. robot 파생 피처 ───────────────────────────────
    section("B. ROBOT DERIVED FEATURES (robot_total + 비율 3개)")
    train2 = add_robot_derived(train)
    test2 = add_robot_derived(test)

    # B-1. layout 안 robot_total 일정성
    rt_unique_per_layout = train2.groupby("layout_id")["robot_total"].nunique()
    print("layout 안 robot_total unique 값 수:")
    print(f"  mean={rt_unique_per_layout.mean():.2f}, "
          f"min={rt_unique_per_layout.min()}, "
          f"max={rt_unique_per_layout.max()}")
    print("  → 1이면 layout마다 시점 무관 일정 (창고 보유량)")

    # B-2. layout별 robot_total 분포
    rt_first = train2.groupby("layout_id")["robot_total"].first()
    print(f"\nlayout별 robot_total 분포 (시점0 기준):")
    print(f"  min: {int(rt_first.min())}, max: {int(rt_first.max())}, "
          f"mean: {rt_first.mean():.1f}, median: {rt_first.median():.0f}")

    # B-3. 비율 피처 통계
    print("\nrobot 비율 피처 분포 (전체 train):")
    for c in ["robot_active_ratio", "robot_idle_ratio", "robot_charging_ratio"]:
        v = train2[c].dropna()
        print(f"  {c:25s}: mean={v.mean():.3f}, std={v.std():.3f}, "
              f"min={v.min():.3f}, max={v.max():.3f}")

    # B-4. 극단 케이스 카운트
    n_all_active = int((train2["robot_active_ratio"] >= 0.95).sum())
    n_no_idle = int((train2["robot_idle"] == 0).sum())
    n_no_charge = int((train2["robot_charging"] == 0).sum())
    print(f"\nrobot_active_ratio ≥ 0.95 (거의 다 작업중) 행: "
          f"{n_all_active} ({n_all_active/len(train2)*100:.2f}%)")
    print(f"robot_idle == 0 (유휴 0개) 행:                 "
          f"{n_no_idle} ({n_no_idle/len(train2)*100:.2f}%)")
    print(f"robot_charging == 0 (충전 0개) 행:             "
          f"{n_no_charge} ({n_no_charge/len(train2)*100:.2f}%)")

    # B-5. robot_utilization vs robot_active_ratio 관계
    valid = train2[["robot_utilization", "robot_active_ratio"]].dropna()
    corr = valid.corr().iloc[0, 1]
    print(f"\nrobot_utilization ↔ robot_active_ratio 상관: {corr:.4f}")
    if abs(corr) > 0.95:
        print("  → 거의 같음 (정보 중복 — 둘 중 하나만 써도 됨)")
    elif abs(corr) > 0.5:
        print("  → 강한 상관, 그러나 다른 정의 — 둘 다 보존 가치")
    else:
        print("  → 다른 신호 — 반드시 둘 다 사용")

    # ── C. 카테고리별 분포 figure (v2) ─────────────────────
    section("C. NUMERIC DISTRIBUTIONS BY CATEGORY (v2)")

    # robot 카테고리는 파생 피처 포함해서 그림
    robot_cols = [c for c, cat in CATEGORY_MAP.items() if cat == "robot"]
    robot_cols += ["robot_total", "robot_active_ratio",
                   "robot_idle_ratio", "robot_charging_ratio"]
    plot_distributions(train2, robot_cols, "robot")

    # 나머지 카테고리들
    cat_to_cols: dict[str, list[str]] = {}
    for c, cat in CATEGORY_MAP.items():
        if cat in ("key", "target", "robot"):
            continue
        if str(train[c].dtype) not in ("float64", "int64"):
            continue
        cat_to_cols.setdefault(cat, []).append(c)

    for cat in sorted(cat_to_cols):
        plot_distributions(train2, cat_to_cols[cat], cat)

    # ── 끝 ────────────────────────────────────────────────
    print()
    print("=" * 64)
    print(" Phase 2 v2 완료")
    print("=" * 64)
    print(f"산출물: {OUT_DIR}/")
    print("  - phase2v2_column_inventory.csv (94 컬럼 + category_v2)")
    print("  - phase2v2_distributions_<category>.png (17개 카테고리)")
    print()
    print("다음(Phase 3) 코드에서:")
    print("  from eda_phase2_v2 import CATEGORY_MAP, add_robot_derived")
    print("  → 카테고리 분류 + 파생 피처 그대로 재사용")


if __name__ == "__main__":
    main()
