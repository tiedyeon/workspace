"""
Step19: Temporal Feature Engineering + Lag/Rolling + Layout Ratios
- 상위 랭커 코드 참고: lag/rolling/expanding mean 피처 + layout ratio 피처
- sc_* 전체 집계 피처 제거 → temporal 피처로 대체
- CV: GroupKFold by scenario_id (5 folds, 5 seeds) - 규정 위반 없음
- 모델: lgbm_mae_raw + lgbm_huber_log + xgb_abs_raw + catboost_mae_log
- sample_weight: 극단값(q90/q95/q99) 업웨이팅 + late_time 보너스
"""

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import time

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

# ── 경로 ──────────────────────────────────────────────
DATA_DIR  = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = Path(DATA_DIR)
TRAIN_CSV  = WORKSPACE / "train.csv"
TEST_CSV   = WORKSPACE / "test.csv"
LAYOUT_CSV = WORKSPACE / "layout_info.csv"
SAMPLE_CSV = WORKSPACE / "sample_submission.csv"
OUTPUT_CSV = WORKSPACE / "submission_step19_temporal_features.csv"

TARGET = "avg_delay_minutes_next_30m"
ID_COL = "ID"
SCENARIO_COL = "scenario_id"
LAYOUT_COL   = "layout_id"

N_CPU   = os.cpu_count()
N_FOLDS = 5
SEEDS   = [42, 123, 456, 789, 2024]

# ── sequence 피처 (lag/rolling 적용 대상) ──────────────
SEQ_COLS = [
    "order_inflow_15m", "unique_sku_15m", "robot_active", "robot_idle", "robot_charging",
    "battery_mean", "battery_std", "low_battery_ratio", "charge_queue_length",
    "avg_charge_wait", "congestion_score", "max_zone_density", "blocked_path_15m",
    "near_collision_15m", "fault_count_15m", "avg_recovery_time", "task_reassign_15m",
    "replenishment_overlap", "pack_utilization", "loading_dock_util",
    "staging_area_util", "label_print_queue",
]

# ── expanding mean 피처 (causal baseline) ──────────────
BASELINE_COLS = [
    "order_inflow_15m", "unique_sku_15m", "avg_items_per_order", "urgent_order_ratio",
    "heavy_item_ratio", "cold_chain_ratio", "sku_concentration", "bulk_order_ratio",
    "avg_trip_distance", "network_latency_ms", "air_quality_idx",
    "barcode_read_success_rate", "hvac_power_kw", "ambient_noise_db",
    "inventory_turnover_rate", "safety_score_monthly", "scanner_error_rate",
    "wms_response_time_ms", "backorder_ratio",
]

# ── 동적 피처 (결측 카운트용) ──────────────────────────
DYN_COLS = [
    "battery_mean", "low_battery_ratio", "robot_charging", "charge_queue_length",
    "avg_charge_wait", "congestion_score", "max_zone_density", "blocked_path_15m",
    "near_collision_15m", "fault_count_15m", "avg_recovery_time", "task_reassign_15m",
    "replenishment_overlap", "pack_utilization", "loading_dock_util", "staging_area_util",
]


# ══════════════════════════════════════════════════════
# Feature Engineering
# ══════════════════════════════════════════════════════

def safe_divide(num, den):
    num = pd.Series(num, dtype="float64")
    den = pd.Series(den, dtype="float64").replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan)


def build_features(df: pd.DataFrame, layout_df: pd.DataFrame,
                   layout_type_cats: list = None,
                   fit: bool = False):
    """
    전체 피처 엔지니어링:
    1. ID 순서 기반 time_idx (시나리오 내 슬롯 순서)
    2. 결측 지시자
    3. Layout 조인 + ratio 피처
    4. Robot 상태 비율 피처
    5. Interaction / composite 피처
    6. Threshold(hinge) 피처
    7. Onset detection (robot_charging, charge_queue)
    8. LAG + ROLLING 피처 (SEQ_COLS)
    9. Expanding mean 피처 (BASELINE_COLS)
    10. layout_type one-hot
    """
    df = df.copy()

    # ── ID 숫자 추출 → 시나리오 내 시간순 정렬 ──────────
    df["__id_num__"] = (
        df[ID_COL].astype(str).str.extract(r"(\d+)", expand=False)
        .fillna("0").astype(int)
    )
    df = df.sort_values([SCENARIO_COL, "__id_num__"]).reset_index(drop=True)

    # ── Layout 조인 ──────────────────────────────────────
    df = df.merge(layout_df.copy(), on=LAYOUT_COL, how="left", validate="m:1")

    grp = df.groupby(SCENARIO_COL, sort=False)
    grp_key = df[SCENARIO_COL]

    # ── 원본 수치 컬럼 목록 ──────────────────────────────
    exclude_raw = {TARGET, ID_COL, SCENARIO_COL, LAYOUT_COL, "__id_num__", "layout_type"}
    orig_num = [c for c in df.columns
                if c not in exclude_raw
                and pd.api.types.is_numeric_dtype(df[c])]

    # ── 1. Time index 피처 ───────────────────────────────
    df["time_idx"]       = grp.cumcount().astype(np.int16)
    df["time_frac"]      = (df["time_idx"] / 24.0).astype(np.float32)
    df["time_remaining"] = (24 - df["time_idx"]).astype(np.int16)
    df["time_idx_sq"]    = (df["time_frac"] ** 2).astype(np.float32)
    df["is_early_phase"] = (df["time_idx"] <= 5).astype(np.int8)
    df["is_mid_phase"]   = ((df["time_idx"] >= 6) & (df["time_idx"] <= 15)).astype(np.int8)
    df["is_late_phase"]  = (df["time_idx"] >= 16).astype(np.int8)

    # ── 2. 결측 지시자 ───────────────────────────────────
    for col in orig_num:
        if df[col].isna().any():
            df[f"{col}__is_missing"] = df[col].isna().astype(np.int8)
    df["n_missing_all_raw"]     = df[orig_num].isna().sum(axis=1).astype(np.int16)
    dyn_present = [c for c in DYN_COLS if c in df.columns]
    df["n_missing_dynamic_raw"] = df[dyn_present].isna().sum(axis=1).astype(np.int16) if dyn_present else 0
    df["missing_ratio_all_raw"] = (df["n_missing_all_raw"] / max(len(orig_num), 1)).astype(np.float32)

    # ── 3. Layout 기반 ratio/composite 피처 ─────────────
    def add_ratio(name, a, b):
        if a in df.columns and b in df.columns:
            df[name] = safe_divide(df[a], df[b]).values

    if {"floor_area_sqm", "ceiling_height_m"}.issubset(df.columns):
        df["warehouse_volume_proxy"] = (df["floor_area_sqm"] * df["ceiling_height_m"]).astype(float)
    if {"intersection_count", "floor_area_sqm"}.issubset(df.columns):
        df["intersection_density"] = safe_divide(df["intersection_count"], df["floor_area_sqm"]).values
    if {"pack_station_count", "floor_area_sqm"}.issubset(df.columns):
        df["pack_station_density"] = safe_divide(df["pack_station_count"], df["floor_area_sqm"]).values
    if {"charger_count", "floor_area_sqm"}.issubset(df.columns):
        df["charger_density"] = safe_divide(df["charger_count"], df["floor_area_sqm"]).values
    if {"robot_total", "floor_area_sqm"}.issubset(df.columns):
        df["robot_density_layout"] = safe_divide(df["robot_total"], df["floor_area_sqm"]).values
    if {"intersection_count", "aisle_width_avg"}.issubset(df.columns):
        df["movement_friction_layout"] = safe_divide(df["intersection_count"], df["aisle_width_avg"]).values
    if {"layout_compactness", "zone_dispersion"}.issubset(df.columns):
        df["layout_compactness_x_dispersion"] = (df["layout_compactness"] * df["zone_dispersion"]).astype(float)
    if {"one_way_ratio", "intersection_count", "aisle_width_avg"}.issubset(df.columns):
        df["one_way_friction"] = (df["one_way_ratio"] * safe_divide(df["intersection_count"], df["aisle_width_avg"])).astype(float)

    add_ratio("inflow_per_robot",            "order_inflow_15m",  "robot_total")
    add_ratio("inflow_per_pack_station",     "order_inflow_15m",  "pack_station_count")
    add_ratio("unique_sku_per_robot",        "unique_sku_15m",    "robot_total")
    add_ratio("charge_queue_per_charger",    "charge_queue_length", "charger_count")
    add_ratio("charging_per_charger",        "robot_charging",    "charger_count")
    add_ratio("congestion_per_width",        "congestion_score",  "aisle_width_avg")
    add_ratio("zone_density_per_width",      "max_zone_density",  "aisle_width_avg")
    add_ratio("order_per_sqm",              "order_inflow_15m",  "floor_area_sqm")
    add_ratio("dock_pressure",              "order_inflow_15m",  "staff_on_floor")
    add_ratio("label_queue_per_pack_station","label_print_queue", "pack_station_count")
    add_ratio("robot_active_per_intersection","robot_active",     "intersection_count")
    add_ratio("congestion_per_active",      "congestion_score",  "robot_active")
    add_ratio("density_per_active",         "max_zone_density",  "robot_active")
    add_ratio("fault_per_active",           "fault_count_15m",   "robot_active")
    add_ratio("collision_per_active",       "near_collision_15m","robot_active")
    add_ratio("blocked_per_active",         "blocked_path_15m",  "robot_active")
    add_ratio("inflow_per_charger",         "order_inflow_15m",  "charger_count")
    add_ratio("pack_station_per_robot",     "pack_station_count","robot_total")
    add_ratio("charger_per_robot",          "charger_count",     "robot_total")
    add_ratio("inflow_per_aisle_width",     "order_inflow_15m",  "aisle_width_avg")

    # ── 4. Robot 상태 비율 ───────────────────────────────
    if {"robot_active", "robot_idle", "robot_charging"}.issubset(df.columns):
        df["robot_total_state"] = df["robot_active"] + df["robot_idle"] + df["robot_charging"]
        if "robot_total" in df.columns:
            df["robot_total_gap"] = df["robot_total_state"] - df["robot_total"]
        df["robot_active_share"]   = safe_divide(df["robot_active"],   df["robot_total_state"]).values
        df["robot_idle_share"]     = safe_divide(df["robot_idle"],     df["robot_total_state"]).values
        df["robot_charging_share"] = safe_divide(df["robot_charging"], df["robot_total_state"]).values
        df["charging_to_active"]   = safe_divide(df["robot_charging"], df["robot_active"]).values
        df["idle_to_active"]       = safe_divide(df["robot_idle"],     df["robot_active"]).values

    # ── 5. Composite / Interaction 피처 ─────────────────
    if {"robot_charging", "charge_queue_length", "charger_count"}.issubset(df.columns):
        df["charge_pressure"] = safe_divide(df["robot_charging"] + df["charge_queue_length"], df["charger_count"]).values
    if {"order_inflow_15m", "avg_package_weight_kg"}.issubset(df.columns):
        df["demand_mass"] = (df["order_inflow_15m"] * df["avg_package_weight_kg"]).astype(float)
        if "robot_total" in df.columns:
            df["demand_mass_per_robot"] = safe_divide(df["demand_mass"], df["robot_total"]).values
    if {"order_inflow_15m", "avg_trip_distance"}.issubset(df.columns):
        df["trip_load"] = (df["order_inflow_15m"] * df["avg_trip_distance"]).astype(float)
        if "robot_total" in df.columns:
            df["trip_load_per_robot"] = safe_divide(df["trip_load"], df["robot_total"]).values
    if {"order_inflow_15m", "unique_sku_15m"}.issubset(df.columns):
        df["complexity_load"] = (df["order_inflow_15m"] * df["unique_sku_15m"]).astype(float)
        if "pack_station_count" in df.columns:
            df["complexity_load_per_pack"] = safe_divide(df["complexity_load"], df["pack_station_count"]).values
    if {"congestion_score", "low_battery_ratio"}.issubset(df.columns):
        df["congestion_x_lowbat"] = (df["congestion_score"] * df["low_battery_ratio"]).astype(float)
    if {"low_battery_ratio", "robot_active"}.issubset(df.columns):
        df["battery_pressure"] = (df["low_battery_ratio"] * df["robot_active"]).astype(float)
    if {"charge_queue_length", "avg_charge_wait"}.issubset(df.columns):
        df["queue_wait_pressure"] = (df["charge_queue_length"] * df["avg_charge_wait"]).astype(float)
    if {"loading_dock_util", "pack_utilization"}.issubset(df.columns):
        df["dock_pack_pressure"] = (df["loading_dock_util"] * df["pack_utilization"]).astype(float)
    if {"staging_area_util", "pack_utilization"}.issubset(df.columns):
        df["staging_pack_pressure"] = (df["staging_area_util"] * df["pack_utilization"]).astype(float)
    if {"avg_recovery_time", "fault_count_15m"}.issubset(df.columns):
        df["recovery_x_fault"] = (df["avg_recovery_time"] * df["fault_count_15m"]).astype(float)
    if {"near_collision_15m", "blocked_path_15m"}.issubset(df.columns):
        df["collision_x_blocked"] = (df["near_collision_15m"] * df["blocked_path_15m"]).astype(float)
    if {"charge_pressure", "congestion_score"}.issubset(df.columns):
        df["charge_pressure_x_congestion"] = (df["charge_pressure"] * df["congestion_score"]).astype(float)
    if {"inflow_per_pack_station", "charge_pressure"}.issubset(df.columns):
        df["inflow_pack_x_charge_pressure"] = (df["inflow_per_pack_station"] * df["charge_pressure"]).astype(float)

    # ── 6. Threshold (hinge) 피처 ────────────────────────
    if "battery_mean" in df.columns:
        df["battery_mean_below_44"] = np.clip(44.0 - df["battery_mean"], 0, None).astype(float)
    if "charge_pressure" in df.columns:
        df["charge_pressure_above_1_36"] = np.clip(df["charge_pressure"] - 1.36, 0, None).astype(float)
    if "pack_utilization" in df.columns:
        df["pack_utilization_sq"]     = df["pack_utilization"].astype(float) ** 2
    if "loading_dock_util" in df.columns:
        df["loading_dock_util_sq"]    = df["loading_dock_util"].astype(float) ** 2
    if "staging_area_util" in df.columns:
        df["staging_area_util_sq"]    = df["staging_area_util"].astype(float) ** 2

    # ── 7. Onset detection ───────────────────────────────
    def add_onset(value_col: str, prefix: str):
        if value_col not in df.columns:
            return
        positive = df[value_col].fillna(0).gt(0)
        t = df["time_idx"].where(positive)
        first = t.groupby(grp_key).transform(lambda s: s.ffill().cummin())
        prev  = positive.groupby(grp_key).shift(1, fill_value=False)
        df[f"{prefix}_ever_started"]      = first.notna().astype(np.int8)
        df[f"{prefix}_start_idx"]         = first.fillna(-1).astype(np.int16)
        df[f"{prefix}_started_now"]       = (positive & ~prev).astype(np.int8)
        df[f"{prefix}_started_early"]     = (first <= 5).fillna(False).astype(np.int8)
        df[f"{prefix}_steps_since_start"] = np.where(first.notna(), (df["time_idx"] - first).astype(float), -1.0).astype(np.float32)

    add_onset("robot_charging",      "charging")
    add_onset("charge_queue_length", "queue")

    # ── 8. LAG + ROLLING 피처 ────────────────────────────
    for col in SEQ_COLS:
        if col not in df.columns:
            continue
        lag1 = grp[col].shift(1)
        lag2 = grp[col].shift(2)
        df[f"{col}__lag1"]  = lag1
        df[f"{col}__lag2"]  = lag2
        df[f"{col}__diff1"] = df[col] - lag1
        # lag1 기반 rolling (과거 3슬롯 이동통계)
        lag1_grp = lag1.groupby(grp_key)
        roll_mean = lag1_grp.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        roll_max  = lag1_grp.rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
        df[f"{col}__rollmean3"]     = roll_mean
        df[f"{col}__rollmax3"]      = roll_max
        df[f"{col}__dev_rollmean3"] = df[col] - roll_mean

    # ── 9. Expanding mean 피처 (causal baseline) ─────────
    for col in BASELINE_COLS:
        if col not in df.columns:
            continue
        prev = grp[col].shift(1)
        exp_mean = (prev.groupby(grp_key)
                    .expanding(min_periods=1).mean()
                    .reset_index(level=0, drop=True))
        df[f"{col}__expmean"]       = exp_mean
        df[f"{col}__delta_expmean"] = df[col] - exp_mean

    # ── 10. layout_type one-hot ──────────────────────────
    if "layout_type" in df.columns:
        if fit or layout_type_cats is None:
            layout_type_cats = sorted(df["layout_type"].dropna().astype(str).unique().tolist())
        dummies = pd.get_dummies(
            pd.Categorical(df["layout_type"].astype(str), categories=layout_type_cats),
            prefix="layout_type", dummy_na=False
        ).astype(np.int8)
        df = pd.concat([df, dummies], axis=1)

    # ── 피처 컬럼 선택 ───────────────────────────────────
    exclude_final = {TARGET, ID_COL, SCENARIO_COL, LAYOUT_COL,
                     "layout_type", "__id_num__"}
    feat_cols = [c for c in df.columns
                 if c not in exclude_final
                 and not pd.api.types.is_object_dtype(df[c])]

    feat = df[feat_cols].astype(np.float32)
    return feat, layout_type_cats


# ══════════════════════════════════════════════════════
# Sample Weighting
# ══════════════════════════════════════════════════════

def build_sample_weight(y: np.ndarray, time_idx: np.ndarray = None) -> np.ndarray:
    """극단값 및 후반 슬롯에 높은 가중치"""
    w = np.ones(len(y), dtype=np.float32)
    q90 = np.nanquantile(y, 0.90)
    q95 = np.nanquantile(y, 0.95)
    q99 = np.nanquantile(y, 0.99)
    w += 0.15 * (y >= q90).astype(np.float32)
    w += 0.30 * (y >= q95).astype(np.float32)
    w += 0.60 * (y >= q99).astype(np.float32)
    if time_idx is not None:
        w += 0.08 * (time_idx / 24.0).astype(np.float32)
    return w


# ══════════════════════════════════════════════════════
# Model Training Helpers
# ══════════════════════════════════════════════════════

LGB_PARAMS_MAE = dict(
    objective="mae", n_estimators=1000, learning_rate=0.03,
    num_leaves=96, max_depth=-1, min_child_samples=80,
    subsample=0.9, subsample_freq=1, colsample_bytree=0.85,
    reg_alpha=0.1, reg_lambda=1.5, verbosity=-1, n_jobs=-1, num_threads=N_CPU,
    device_type="cpu",
)

LGB_PARAMS_HUBER = dict(
    objective="huber", alpha=0.9, n_estimators=1000, learning_rate=0.03,
    num_leaves=128, max_depth=-1, min_child_samples=60,
    subsample=0.9, subsample_freq=1, colsample_bytree=0.85,
    reg_alpha=0.05, reg_lambda=1.0, verbosity=-1, n_jobs=-1, num_threads=N_CPU,
    device_type="cpu",
)

XGB_PARAMS_ABS = dict(
    objective="reg:absoluteerror", n_estimators=1000, learning_rate=0.03,
    max_depth=8, min_child_weight=6.0, subsample=0.9, colsample_bytree=0.85,
    reg_lambda=1.5, reg_alpha=0.05, tree_method="hist", device="cpu", verbosity=0, nthread=N_CPU,
)

CAT_PARAMS_MAE = dict(
    loss_function="MAE", iterations=1200, learning_rate=0.03,
    depth=8, l2_leaf_reg=5.0, bootstrap_type="Bernoulli",
    subsample=0.9, task_type="CPU", thread_count=N_CPU, allow_writing_files=False, verbose=False,
)


def fit_lgb_mae(X_tr, y_tr, w_tr, X_va, y_va, seed):
    p = {**LGB_PARAMS_MAE, "random_state": seed}
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, free_raw_data=True)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain, free_raw_data=True)
    cb = lgb.train(
        p, dtrain, num_boost_round=p["n_estimators"],
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)],
    )
    pred = np.clip(cb.predict(X_va), 0, None)
    return cb, pred


def fit_lgb_huber(X_tr, y_tr, w_tr, X_va, y_va, seed):
    p = {**LGB_PARAMS_HUBER, "random_state": seed}
    y_tr_t = np.log1p(np.clip(y_tr, 0, None))
    y_va_t = np.log1p(np.clip(y_va, 0, None))
    dtrain = lgb.Dataset(X_tr, label=y_tr_t, weight=w_tr, free_raw_data=True)
    dvalid = lgb.Dataset(X_va, label=y_va_t, reference=dtrain, free_raw_data=True)
    cb = lgb.train(
        p, dtrain, num_boost_round=p["n_estimators"],
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)],
    )
    pred = np.clip(np.expm1(cb.predict(X_va)), 0, None)
    return cb, pred


def fit_xgb_abs(X_tr, y_tr, w_tr, X_va, y_va, seed):
    p = {k: v for k, v in XGB_PARAMS_ABS.items()
         if k not in ("n_estimators", "n_jobs")}
    p["seed"] = seed
    p["nthread"] = N_CPU
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)
    model = xgb.train(
        p, dtrain, num_boost_round=XGB_PARAMS_ABS["n_estimators"],
        evals=[(dvalid, "val")], early_stopping_rounds=80, verbose_eval=False,
    )
    pred = np.clip(model.predict(dvalid, iteration_range=(0, model.best_iteration)), 0, None)
    return model, pred


def fit_catboost_mae(X_tr, y_tr, w_tr, X_va, y_va, seed):
    p = {**CAT_PARAMS_MAE, "random_seed": seed}
    y_tr_t = np.log1p(np.clip(y_tr, 0, None))
    y_va_t = np.log1p(np.clip(y_va, 0, None))
    model = CatBoostRegressor(**p)
    train_pool = Pool(X_tr, label=y_tr_t, weight=w_tr)
    eval_pool  = Pool(X_va, label=y_va_t)
    model.fit(train_pool, eval_set=eval_pool,
              early_stopping_rounds=80, verbose=False)
    pred = np.clip(np.expm1(model.predict(X_va)), 0, None)
    return model, pred


MODEL_FNS = {
    "lgb_mae":   (fit_lgb_mae,   0.32),
    "lgb_huber": (fit_lgb_huber, 0.24),
    "xgb_abs":   (fit_xgb_abs,   0.22),
    "cat_mae":   (fit_catboost_mae, 0.22),
}


# ══════════════════════════════════════════════════════
# Final Prediction Helpers (full-fit)
# ══════════════════════════════════════════════════════

def fit_lgb_mae_full(X_tr, y_tr, w_tr, n_iters, seed):
    p = {**LGB_PARAMS_MAE, "random_state": seed, "n_estimators": n_iters}
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, free_raw_data=True)
    return lgb.train(p, dtrain, num_boost_round=n_iters,
                     callbacks=[lgb.log_evaluation(-1)])

def fit_lgb_huber_full(X_tr, y_tr, w_tr, n_iters, seed):
    p = {**LGB_PARAMS_HUBER, "random_state": seed, "n_estimators": n_iters}
    y_tr_t = np.log1p(np.clip(y_tr, 0, None))
    dtrain = lgb.Dataset(X_tr, label=y_tr_t, weight=w_tr, free_raw_data=True)
    return lgb.train(p, dtrain, num_boost_round=n_iters,
                     callbacks=[lgb.log_evaluation(-1)])

def fit_xgb_abs_full(X_tr, y_tr, w_tr, n_iters, seed):
    p = {k: v for k, v in XGB_PARAMS_ABS.items()
         if k not in ("n_estimators", "n_jobs")}
    p["seed"] = seed
    p["nthread"] = N_CPU
    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
    model = xgb.train(p, dtrain, num_boost_round=n_iters, verbose_eval=False)
    return model

def fit_catboost_mae_full(X_tr, y_tr, w_tr, n_iters, seed):
    p = {**CAT_PARAMS_MAE, "random_seed": seed, "iterations": n_iters}
    y_tr_t = np.log1p(np.clip(y_tr, 0, None))
    model = CatBoostRegressor(**p)
    model.fit(Pool(X_tr, label=y_tr_t, weight=w_tr), verbose=False)
    return model


# ══════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 60)
    print("Step19: Temporal Feature Engineering Ensemble")
    print("=" * 60)

    # ── 데이터 로드 ──────────────────────────────────────
    print("\n[1] 데이터 로드...")
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    layout = pd.read_csv(LAYOUT_CSV)
    sample = pd.read_csv(SAMPLE_CSV)
    print(f"  train: {train.shape}, test: {test.shape}, layout: {layout.shape}")

    y_all = train[TARGET].values

    # ── 피처 엔지니어링 ───────────────────────────────────
    print("\n[2] 피처 엔지니어링...")
    X_train, layout_type_cats = build_features(train, layout, fit=True)
    X_test,  _                = build_features(test,  layout, layout_type_cats=layout_type_cats, fit=False)
    print(f"  train features: {X_train.shape[1]}개")
    print(f"  test  features: {X_test.shape[1]}개")

    # time_idx (sample weight용)
    train_sorted = train.copy()
    train_sorted["__id_num__"] = train_sorted[ID_COL].astype(str).str.extract(r"(\d+)", expand=False).fillna("0").astype(int)
    train_sorted = train_sorted.sort_values([SCENARIO_COL, "__id_num__"]).reset_index(drop=True)
    time_idx_all = train_sorted.groupby(SCENARIO_COL).cumcount().values

    # ── GroupKFold CV ─────────────────────────────────────
    print("\n[3] GroupKFold CV (scenario_id, 5folds × 5seeds)...")
    groups = train_sorted[SCENARIO_COL].values

    # OOF 저장
    all_oof = {name: [] for name in MODEL_FNS}
    all_test_preds = {name: [] for name in MODEL_FNS}
    best_iters = {name: [] for name in MODEL_FNS}

    feat_cols = X_train.columns.tolist()
    X_train_np = X_train.values
    X_test_np  = X_test.values

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  -- Seed {seed} ({seed_idx+1}/{len(SEEDS)}) --")
        gkf = GroupKFold(n_splits=N_FOLDS)
        oof_seed = {name: np.zeros(len(y_all)) for name in MODEL_FNS}

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train_np, y_all, groups)):
            X_tr, X_va = X_train_np[tr_idx], X_train_np[va_idx]
            y_tr, y_va = y_all[tr_idx], y_all[va_idx]
            ti_tr = time_idx_all[tr_idx]
            w_tr  = build_sample_weight(y_tr, ti_tr)

            fold_maes = {}
            for name, (fn, _) in MODEL_FNS.items():
                t0 = time.time()
                model, pred_va = fn(X_tr, y_tr, w_tr, X_va, y_va, seed)
                oof_seed[name][va_idx] = pred_va
                fold_mae = mean_absolute_error(y_va, pred_va)
                fold_maes[name] = fold_mae
                elapsed = time.time() - t0
                print(f"    fold{fold+1} {name}: MAE={fold_mae:.4f} ({elapsed:.1f}s)")

                # best iteration 저장
                if hasattr(model, 'best_iteration'):
                    best_iters[name].append(model.best_iteration)
                elif hasattr(model, 'best_iteration_'):
                    best_iters[name].append(model.best_iteration_)

        for name in MODEL_FNS:
            oof_mae = mean_absolute_error(y_all, oof_seed[name])
            print(f"  Seed {seed} | {name} OOF MAE: {oof_mae:.4f}")
            all_oof[name].append(oof_seed[name])

    # ── OOF 앙상블 평가 ───────────────────────────────────
    print("\n[4] OOF 앙상블 평가...")
    weights = {name: w for name, (_, w) in MODEL_FNS.items()}
    total_w = sum(weights.values())

    oof_by_model = {}
    for name in MODEL_FNS:
        oof_by_model[name] = np.mean(all_oof[name], axis=0)
        mae_val = mean_absolute_error(y_all, oof_by_model[name])
        print(f"  {name}: OOF MAE = {mae_val:.4f}")

    # 가중 앙상블 OOF
    oof_ensemble = sum(
        oof_by_model[name] * (weights[name] / total_w)
        for name in MODEL_FNS
    )
    oof_mae_final = mean_absolute_error(y_all, oof_ensemble)
    print(f"\n  [앙상블 OOF MAE]: {oof_mae_final:.4f}")

    # ── Full-fit 최종 예측 ───────────────────────────────
    print("\n[5] Full-fit 최종 예측...")
    w_all = build_sample_weight(y_all, time_idx_all)

    test_preds = {name: [] for name in MODEL_FNS}
    full_fn_map = {
        "lgb_mae":   fit_lgb_mae_full,
        "lgb_huber": fit_lgb_huber_full,
        "xgb_abs":   fit_xgb_abs_full,
        "cat_mae":   fit_catboost_mae_full,
    }

    for seed in SEEDS:
        print(f"  seed {seed} full-fit...")
        for name, fn in full_fn_map.items():
            # best iter 결정
            bi_list = best_iters[name]
            n_iters = int(np.mean(bi_list)) if bi_list else 700
            n_iters = max(50, n_iters)
            print(f"    {name}: n_iters={n_iters}")

            model = fn(X_train_np, y_all, w_all, n_iters, seed)

            if name in ("lgb_mae",):
                pred = np.clip(model.predict(X_test_np), 0, None)
            elif name in ("lgb_huber", "cat_mae"):
                pred = np.clip(np.expm1(model.predict(X_test_np)), 0, None)
            elif name == "xgb_abs":
                dtest = xgb.DMatrix(X_test_np)
                pred = np.clip(model.predict(dtest), 0, None)
            else:
                pred = np.clip(model.predict(X_test_np), 0, None)

            test_preds[name].append(pred)

    # ── 최종 제출 파일 ────────────────────────────────────
    print("\n[6] 제출 파일 생성...")
    final_test = np.zeros(len(test))
    for name, (_, w) in MODEL_FNS.items():
        model_avg = np.mean(test_preds[name], axis=0)
        final_test += model_avg * (w / total_w)

    sample[TARGET] = final_test
    sample.to_csv(OUTPUT_CSV, index=False)

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"완료! 총 소요시간: {elapsed_total/60:.1f}분")
    print(f"OOF MAE: {oof_mae_final:.4f}")
    print(f"제출 파일: {OUTPUT_CSV.name}")
    print(f"예측값 범위: [{final_test.min():.2f}, {final_test.max():.2f}]")
    print(f"예측값 평균: {final_test.mean():.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
