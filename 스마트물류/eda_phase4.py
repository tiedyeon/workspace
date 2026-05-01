# -*- coding: utf-8 -*-
"""
eda_phase4.py — seen vs unseen layout 분포 비교 + adversarial validation

목표: OOF→Public 갭 1.47 의 원인 진단
  → test 의 unseen 50 layout 이 train 250 layout 과 얼마나 다른가?
  → 어떤 피처가 train↔test 를 구분하는가?
  → 갭의 원인이 layout 자체인가, 피처 분포인가, 시점인가?

Part A: seen/unseen layout 식별 + layout_info 매칭 검증
Part B: layout_info 메타 분포 비교 (train+seen vs unseen)
Part C: 강한 피처 KS test (train vs test 전체)
Part D: adversarial validation — LightGBM 으로 train↔test 분류, OOF AUC + importance
Part E: 종합 진단

산출:
  eda_outputs/phase4_layout_overlap.txt
  eda_outputs/phase4_layout_meta_ks.csv
  eda_outputs/phase4_layout_meta_comparison.png
  eda_outputs/phase4_feature_ks_train_vs_test.csv
  eda_outputs/phase4_feature_distribution_comparison.png
  eda_outputs/phase4_adversarial_oof_auc.txt
  eda_outputs/phase4_adversarial_importance.csv
  eda_outputs/phase4_adversarial_importance.png

실행:
  python eda_phase4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.default"] = "regular"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Phase 3 에서 정의된 함수 + 카테고리
from eda_phase3 import merge_layout_info, add_all_derived, CATEGORY_MAP

DATA_DIR = Path("data")
OUT_DIR = Path("eda_outputs")
OUT_DIR.mkdir(exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"


def section(title: str) -> None:
    print()
    print("=" * 64)
    print(f" {title}")
    print("=" * 64)


# ────────────────────────────────────────────────────────────
# Part A — layout overlap 분석
# ────────────────────────────────────────────────────────────
def part_a_layout_overlap(
    train: pd.DataFrame, test: pd.DataFrame, layout_info: pd.DataFrame
) -> tuple[set, set]:
    section("A. seen / unseen layout 식별 + layout_info 매칭")

    lines = []

    def log(msg):
        print(msg)
        lines.append(msg)

    train_layouts = set(train["layout_id"].unique())
    test_layouts = set(test["layout_id"].unique())
    li_layouts = set(layout_info["layout_id"].unique())

    seen = train_layouts & test_layouts
    unseen = test_layouts - train_layouts
    train_only = train_layouts - test_layouts

    log(f"train layouts: {len(train_layouts)}")
    log(f"test layouts:  {len(test_layouts)}")
    log(f"  seen   (train ∩ test): {len(seen)}")
    log(f"  unseen (test only):    {len(unseen)}")
    log("")
    log(f"layout_info layouts: {len(li_layouts)}")
    log(f"  train layout 중 layout_info 에 있음: "
        f"{len(train_layouts & li_layouts)} / {len(train_layouts)}")
    log(f"  unseen layout 중 layout_info 에 있음: "
        f"{len(unseen & li_layouts)} / {len(unseen)}")
    log(f"  seen layout 중 layout_info 에 있음:   "
        f"{len(seen & li_layouts)} / {len(seen)}")
    log(f"  layout_info 에만 있는 layout (data 에 안 나옴): "
        f"{len(li_layouts - train_layouts - test_layouts)}")
    log("")
    log(f"검산 — train_only + seen + unseen = {len(train_only)} + {len(seen)} + {len(unseen)} "
        f"= {len(train_only) + len(seen) + len(unseen)}")
    log(f"      train ∪ test = {len(train_layouts | test_layouts)}")

    out_txt = OUT_DIR / "phase4_layout_overlap.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n결과 → {out_txt}")

    return seen, unseen


# ────────────────────────────────────────────────────────────
# Part B — layout_info 메타 분포 비교 (train+seen vs unseen)
# ────────────────────────────────────────────────────────────
def part_b_layout_meta(
    layout_info: pd.DataFrame, seen: set, unseen: set
) -> None:
    section("B. layout_info 메타 분포: (train+seen) vs unseen 50")

    train_seen_mask = ~layout_info["layout_id"].isin(unseen)
    train_seen_layouts = layout_info[train_seen_mask]
    unseen_layouts = layout_info[layout_info["layout_id"].isin(unseen)]

    print(f"비교 대상: train+seen={len(train_seen_layouts)}, "
          f"unseen={len(unseen_layouts)}")

    numeric_cols = [c for c in layout_info.columns
                    if c not in ("layout_id", "layout_type")
                    and str(layout_info[c].dtype) in ("float64", "int64")]

    # KS test
    rows = []
    for c in numeric_cols:
        ks, p = ks_2samp(train_seen_layouts[c].dropna(),
                          unseen_layouts[c].dropna())
        rows.append({
            "column": c,
            "train_seen_mean": train_seen_layouts[c].mean(),
            "unseen_mean": unseen_layouts[c].mean(),
            "diff": unseen_layouts[c].mean() - train_seen_layouts[c].mean(),
            "ks_stat": ks,
            "p_value": p,
        })
    df_ks = pd.DataFrame(rows).sort_values("ks_stat", ascending=False).round(4)
    print("\nlayout_info 컬럼별 KS test (값 클수록 분포 다름):")
    print(df_ks.to_string(index=False))
    out_csv = OUT_DIR / "phase4_layout_meta_ks.csv"
    df_ks.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"  → {out_csv}")

    # 히스토그램 격자
    n = len(numeric_cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 2.8))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    for i, c in enumerate(numeric_cols):
        ax = axes[i // ncols, i % ncols]
        ax.hist(train_seen_layouts[c].dropna(), bins=20, alpha=0.55,
                label=f"train+seen (n={len(train_seen_layouts)})",
                color="steelblue", density=True)
        ax.hist(unseen_layouts[c].dropna(), bins=20, alpha=0.55,
                label=f"unseen (n={len(unseen_layouts)})",
                color="coral", density=True)
        ks_val = df_ks.set_index("column").loc[c, "ks_stat"]
        ax.set_title(f"{c}\nKS={ks_val:.3f}", fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    fig.suptitle("layout_info: train+seen vs unseen 50 — 분포 비교",
                 fontsize=12)
    plt.tight_layout()
    out_png = OUT_DIR / "phase4_layout_meta_comparison.png"
    plt.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  figure → {out_png}")

    # layout_type 분포
    print("\nlayout_type 분포 비교:")
    print(f"  train+seen: {train_seen_layouts['layout_type'].value_counts().to_dict()}")
    print(f"  unseen:     {unseen_layouts['layout_type'].value_counts().to_dict()}")


# ────────────────────────────────────────────────────────────
# Part C — 강한 피처 KS test (train vs test 행 단위)
# ────────────────────────────────────────────────────────────
def part_c_feature_ks(train_aug: pd.DataFrame, test_aug: pd.DataFrame) -> None:
    section("C. 강한 피처 KS test (train vs test 전체 행)")

    # phase3 의 top 30 가져옴
    corr_csv = OUT_DIR / "phase3_target_correlation.csv"
    if not corr_csv.exists():
        print(f"⚠ {corr_csv} 없음. Phase 3 먼저 실행해야 함")
        return
    corr_df = pd.read_csv(corr_csv)
    top_cols = corr_df.head(30)["column"].tolist()

    rows = []
    for c in top_cols:
        if c not in train_aug.columns or c not in test_aug.columns:
            continue
        t = train_aug[c].dropna()
        e = test_aug[c].dropna()
        if len(t) == 0 or len(e) == 0:
            continue
        ts = t.sample(min(50000, len(t)), random_state=42)
        es = e.sample(min(50000, len(e)), random_state=42)
        ks, p = ks_2samp(ts, es)
        rows.append({
            "column": c,
            "train_mean": t.mean(),
            "test_mean": e.mean(),
            "train_std": t.std(),
            "test_std": e.std(),
            "ks_stat": ks,
            "p_value": p,
        })
    df_ks = pd.DataFrame(rows).sort_values("ks_stat", ascending=False).round(4)
    print("\nTop 30 강한 피처의 train vs test KS test:")
    print(df_ks.to_string(index=False))
    out_csv = OUT_DIR / "phase4_feature_ks_train_vs_test.csv"
    df_ks.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"  → {out_csv}")

    # 상위 12개 분포 비교 figure
    top12 = df_ks.head(12)["column"].tolist()
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    for i, c in enumerate(top12):
        ax = axes[i // 4, i % 4]
        t_sample = train_aug[c].dropna().sample(
            min(50000, train_aug[c].notna().sum()), random_state=42)
        e_sample = test_aug[c].dropna().sample(
            min(50000, test_aug[c].notna().sum()), random_state=42)
        ax.hist(t_sample, bins=40, alpha=0.5, label="train",
                color="steelblue", density=True)
        ax.hist(e_sample, bins=40, alpha=0.5, label="test",
                color="coral", density=True)
        ks_val = df_ks.set_index("column").loc[c, "ks_stat"]
        ax.set_title(f"{c}\nKS={ks_val:.3f}", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    fig.suptitle("KS top 12 — train (steelblue) vs test (coral) 분포", fontsize=11)
    plt.tight_layout()
    out_png = OUT_DIR / "phase4_feature_distribution_comparison.png"
    plt.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  figure → {out_png}")


# ────────────────────────────────────────────────────────────
# Part D — Adversarial Validation
# ────────────────────────────────────────────────────────────
def part_d_adversarial(
    train_aug: pd.DataFrame, test_aug: pd.DataFrame
) -> float:
    section("D. Adversarial validation — LightGBM 으로 train↔test 분류")

    # 공통 수치 컬럼만
    common_cols = [c for c in train_aug.columns
                   if c in test_aug.columns
                   and c not in ("ID", "layout_id", "scenario_id", TARGET)
                   and str(train_aug[c].dtype) in ("float64", "int64",
                                                     "int32", "bool")]
    print(f"공통 수치 피처: {len(common_cols)}")

    # combined
    train_x = train_aug[common_cols].copy()
    train_x["_is_test"] = 0
    train_x["_layout_id"] = train_aug["layout_id"].values

    test_x = test_aug[common_cols].copy()
    test_x["_is_test"] = 1
    test_x["_layout_id"] = test_aug["layout_id"].values

    combined = pd.concat([train_x, test_x], ignore_index=True)
    print(f"combined: {combined.shape}")

    X = combined[common_cols]
    y = combined["_is_test"]
    groups = combined["_layout_id"]

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(X))
    importances = np.zeros(len(common_cols))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=200,
            n_jobs=-1,
            random_state=42,
            verbosity=-1,
        )
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        importances += model.feature_importances_
        fold_auc = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        print(f"  fold {fold + 1}: AUC = {fold_auc:.4f}")

    overall_auc = roc_auc_score(y, oof)
    importances /= 5

    print(f"\nOverall OOF AUC: {overall_auc:.4f}")
    if overall_auc < 0.55:
        verdict = "분포 거의 같음 — 갭 1.47 의 원인은 분포 외 (예: layout 차)"
    elif overall_auc < 0.70:
        verdict = "분포 약간 다름 — 일부 피처가 차이. 보정 가능성"
    elif overall_auc < 0.85:
        verdict = "분포 명확히 다름 — unseen layout 영향 큼"
    else:
        verdict = "매우 다름 — 모델이 train/test 거의 완벽 구분. 큰 보정 필요"
    print(f"  → {verdict}")

    # importance df
    imp_df = pd.DataFrame({
        "column": common_cols,
        "importance": importances,
        "category": [CATEGORY_MAP.get(c, "?") for c in common_cols],
    }).sort_values("importance", ascending=False)

    print("\n분포 차이 만드는 컬럼 top 20 (adversarial importance):")
    print(imp_df.head(20).round(2).to_string(index=False))

    out_csv = OUT_DIR / "phase4_adversarial_importance.csv"
    imp_df.round(4).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"  → {out_csv}")

    out_txt = OUT_DIR / "phase4_adversarial_oof_auc.txt"
    out_txt.write_text(f"OOF AUC: {overall_auc:.4f}\n{verdict}\n",
                       encoding="utf-8")

    # figure
    top20 = imp_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top20)), top20["importance"], color="steelblue")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["column"], fontsize=9)
    ax.set_xlabel("LightGBM importance (gain)")
    ax.set_title(f"Adversarial validation top 20 — train↔test 구분 컬럼\n"
                 f"OOF AUC = {overall_auc:.4f} ({verdict})")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out_png = OUT_DIR / "phase4_adversarial_importance.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"  figure → {out_png}")

    return overall_auc


# ────────────────────────────────────────────────────────────
def main() -> None:
    print(">>> Phase 4 시작 — seen vs unseen + adversarial validation")

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    layout_info = pd.read_csv(DATA_DIR / "layout_info.csv")
    print(f"train: {train.shape}, test: {test.shape}, "
          f"layout_info: {layout_info.shape}")

    # layout_info merge + layout_type one-hot (train/test 동일하게)
    print("\nlayout_info merge + layout_type one-hot...")
    train_m = merge_layout_info(train, layout_info)
    test_m = merge_layout_info(test, layout_info)

    all_lts = pd.concat([train_m["layout_type"], test_m["layout_type"]]).unique()
    for lt in all_lts:
        if pd.isna(lt):
            continue
        train_m[f"layout_type_{lt}"] = (train_m["layout_type"] == lt).astype(int)
        test_m[f"layout_type_{lt}"] = (test_m["layout_type"] == lt).astype(int)
    train_m = train_m.drop(columns=["layout_type"])
    test_m = test_m.drop(columns=["layout_type"])

    # 파생 피처
    train_aug = add_all_derived(train_m)
    test_aug = add_all_derived(test_m)
    print(f"  train_aug: {train_aug.shape}, test_aug: {test_aug.shape}")

    # ── Part A
    seen, unseen = part_a_layout_overlap(train, test, layout_info)

    # ── Part B
    part_b_layout_meta(layout_info, seen, unseen)

    # ── Part C
    part_c_feature_ks(train_aug, test_aug)

    # ── Part D
    auc = part_d_adversarial(train_aug, test_aug)

    # ── Part E
    section("E. 종합 진단 — OOF→Public 갭 1.47 의 원인")
    print(f"adversarial OOF AUC: {auc:.4f}")
    print()
    if auc < 0.6:
        print("진단: 분포 차이 작음")
        print("  → 갭 1.47 은 layout 자체의 random effect 가 큼")
        print("  → submission 에서 sample weighting 등은 효과 작을 듯")
        print("  → 다양한 모델 앙상블이 가장 효과적인 갭 축소 방법")
    elif auc < 0.75:
        print("진단: 분포 차이 중간")
        print("  → 일부 피처가 train/test 를 가르고 있음")
        print("  → adversarial importance top 컬럼들을 weight or 제거 검토")
        print("  → unseen layout 행에 더 큰 가중치 (또는 작은 가중치) 시도")
    else:
        print("진단: 분포 차이 큼")
        print("  → 갭 1.47 의 주 원인은 분포 자체")
        print("  → adversarial importance 상위 컬럼은 OOF→Public 갭 만드는 주범")
        print("  → 분포 보정 (target encoding, 분포 정규화) 또는 unseen 가중치 필수")

    print()
    print("=" * 64)
    print(" Phase 4 완료")
    print("=" * 64)
    print(f"산출물: {OUT_DIR}/")
    print("  - phase4_layout_overlap.txt")
    print("  - phase4_layout_meta_ks.csv")
    print("  - phase4_layout_meta_comparison.png        ← 메타 분포 비교")
    print("  - phase4_feature_ks_train_vs_test.csv")
    print("  - phase4_feature_distribution_comparison.png")
    print("  - phase4_adversarial_oof_auc.txt           ← 핵심 진단")
    print("  - phase4_adversarial_importance.csv")
    print("  - phase4_adversarial_importance.png        ← 갭 만드는 컬럼")


if __name__ == "__main__":
    main()
