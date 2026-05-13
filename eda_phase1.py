# -*- coding: utf-8 -*-
"""
eda_phase1.py — 기본 정보 + 좌표 분포 + 라벨 분포

목표:
  1) 데이터 형태 확인 (train/test 파일 수, 행 수, 컬럼)
  2) 시점(timestep_ms) 일관성 — 정말 -400 ~ 0, 11개 시점인가
  3) 좌표 (x, y, z) 분포 — 범위, 분포, 이상치
  4) 라벨 (80ms 후 좌표) 분포
  5) 결측 / NaN 체크
  6) 샘플 궤적 시각화 (5~10개 모기 비행 경로)
  7) Constant Velocity 베이스라인 sanity check
     — 마지막 두 시점 외삽 벡터가 (target - last) 와 얼마나 비슷한지

산출:
  cache/train_combined.parquet     ← 10000개 csv → 1개 parquet (재사용)
  cache/test_combined.parquet
  eda_outputs/phase1_coord_dist.png
  eda_outputs/phase1_label_dist.png
  eda_outputs/phase1_sample_trajectories.png
  eda_outputs/phase1_time_evolution.png
  eda_outputs/phase1_cv_sanity.png
  eda_outputs/phase1_per_sample_stats.csv

실행:
  (.smart) PS C:\...\모기궤적> python eda_phase1.py
  소요: ~1~2분 (첫 실행 시 csv 로딩 ~30초, 캐시 이후 빠름)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.default"] = "regular"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
LABELS_PATH = DATA_DIR / "train_labels.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

OUT_DIR = Path("eda_outputs")
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)


def section(title: str) -> None:
    print()
    print("=" * 64)
    print(f" {title}")
    print("=" * 64)


# ──────────────────────────────────────────────────────────────
# 1. 로딩 (캐시 사용)
# ──────────────────────────────────────────────────────────────
def load_all_csvs(folder: Path, prefix: str, cache_path: Path) -> pd.DataFrame:
    """폴더 안 모든 csv 를 하나의 long-format DataFrame 으로 통합. parquet 캐시."""
    if cache_path.exists():
        print(f"  cache 로드: {cache_path}")
        return pd.read_parquet(cache_path)

    files = sorted(folder.glob(f"{prefix}_*.csv"))
    print(f"  {len(files)}개 csv 로딩 중...")

    dfs = []
    for f in tqdm(files, ncols=80):
        df = pd.read_csv(f)
        df["id"] = f.stem  # TRAIN_00001 등
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(cache_path)
    print(f"  cache 저장: {cache_path}")
    return combined


def main() -> None:
    t0 = time.time()
    print(">>> Phase 1 EDA 시작 — 모기 비행 궤적 예측")

    section("1. 데이터 로딩")
    train_df = load_all_csvs(TRAIN_DIR, "TRAIN", CACHE_DIR / "train_combined.parquet")
    test_df = load_all_csvs(TEST_DIR, "TEST", CACHE_DIR / "test_combined.parquet")
    labels = pd.read_csv(LABELS_PATH)

    print(f"\ntrain (long format): {train_df.shape}")
    print(f"test  (long format): {test_df.shape}")
    print(f"labels:              {labels.shape}")

    section("2. 기본 정보")
    print(f"train 컬럼: {list(train_df.columns)}")
    print(f"test 컬럼:  {list(test_df.columns)}")
    print(f"labels 컬럼: {list(labels.columns)}")
    print(f"\ntrain dtypes:")
    print(train_df.dtypes.to_string())

    # 샘플 당 행 수 검증
    train_per_sample = train_df.groupby("id").size()
    test_per_sample = test_df.groupby("id").size()
    print(f"\n샘플당 행 수:")
    print(f"  train: min={train_per_sample.min()}, max={train_per_sample.max()}, "
          f"unique={train_per_sample.unique()}")
    print(f"  test:  min={test_per_sample.min()}, max={test_per_sample.max()}, "
          f"unique={test_per_sample.unique()}")

    # 시점 일관성
    print(f"\ntimestep_ms unique values:")
    print(f"  train: {sorted(train_df['timestep_ms'].unique())}")
    print(f"  test:  {sorted(test_df['timestep_ms'].unique())}")

    section("3. 결측 / NaN")
    print(f"train NaN 개수:")
    print(train_df.isna().sum().to_string())
    print(f"\ntest NaN 개수:")
    print(test_df.isna().sum().to_string())
    print(f"\nlabels NaN 개수:")
    print(labels.isna().sum().to_string())

    section("4. 좌표 (x, y, z) 분포")
    coord_stats = train_df[["x", "y", "z"]].describe()
    print("train 좌표 통계:")
    print(coord_stats.round(4).to_string())
    print()
    print("test 좌표 통계:")
    print(test_df[["x", "y", "z"]].describe().round(4).to_string())
    print()
    print("label 좌표 통계 (80ms 후):")
    print(labels[["x", "y", "z"]].describe().round(4).to_string())

    # figure: 좌표 분포 — train vs test vs label
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, axis in enumerate(["x", "y", "z"]):
        # train 관측
        axes[0, i].hist(train_df[axis], bins=80, color="steelblue",
                        edgecolor="white", alpha=0.85)
        axes[0, i].set_title(f"train 관측 {axis} (모든 시점)", fontsize=10)
        axes[0, i].grid(alpha=0.3)

        # test 관측
        axes[1, i].hist(test_df[axis], bins=80, color="coral",
                        edgecolor="white", alpha=0.85)
        axes[1, i].set_title(f"test 관측 {axis} (모든 시점)", fontsize=10)
        axes[1, i].grid(alpha=0.3)

        # label
        axes[2, i].hist(labels[axis], bins=80, color="seagreen",
                        edgecolor="white", alpha=0.85)
        axes[2, i].set_title(f"label {axis} (80ms 후)", fontsize=10)
        axes[2, i].grid(alpha=0.3)

    plt.tight_layout()
    out_png = OUT_DIR / "phase1_coord_dist.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"\n좌표 분포 figure → {out_png}")

    section("5. 시점별 좌표 평균 (시간 진화)")
    time_evol = train_df.groupby("timestep_ms")[["x", "y", "z"]].agg(["mean", "std"])
    print(time_evol.round(4).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, axis in enumerate(["x", "y", "z"]):
        means = train_df.groupby("timestep_ms")[axis].mean()
        stds = train_df.groupby("timestep_ms")[axis].std()
        axes[i].plot(means.index, means.values, "o-", color="steelblue",
                     label="mean")
        axes[i].fill_between(means.index, means - stds, means + stds,
                             alpha=0.3, color="steelblue", label="±1σ")
        axes[i].set_xlabel("timestep_ms")
        axes[i].set_ylabel(axis)
        axes[i].set_title(f"{axis} 시간 진화 (train 10000 모기 평균)")
        axes[i].legend()
        axes[i].grid(alpha=0.3)
    plt.tight_layout()
    out_png = OUT_DIR / "phase1_time_evolution.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"figure → {out_png}")

    section("6. 샘플 궤적 시각화 (12개 무작위 모기)")
    np.random.seed(42)
    sample_ids = np.random.choice(train_df["id"].unique(), 12, replace=False)
    print(f"선택된 샘플: {list(sample_ids)}")

    fig = plt.figure(figsize=(18, 12))
    for i, sid in enumerate(sample_ids):
        ax = fig.add_subplot(3, 4, i + 1, projection="3d")
        sample = train_df[train_df["id"] == sid].sort_values("timestep_ms")
        label_row = labels[labels["id"] == sid].iloc[0]

        # 관측 궤적 (파란 점·선)
        ax.plot(sample["x"], sample["y"], sample["z"],
                "o-", color="steelblue", markersize=4, label="관측 (11)")
        # 시작점 (초록 큰 점)
        ax.scatter([sample["x"].iloc[0]], [sample["y"].iloc[0]],
                   [sample["z"].iloc[0]], color="green", s=80, label="-400ms")
        # 현재 (빨강 큰 점)
        ax.scatter([sample["x"].iloc[-1]], [sample["y"].iloc[-1]],
                   [sample["z"].iloc[-1]], color="red", s=80, label="0ms")
        # 80ms 후 정답 (보라 별)
        ax.scatter([label_row["x"]], [label_row["y"]], [label_row["z"]],
                   color="purple", s=120, marker="*", label="+80ms (정답)")
        # 현재 → 정답 직선
        ax.plot([sample["x"].iloc[-1], label_row["x"]],
                [sample["y"].iloc[-1], label_row["y"]],
                [sample["z"].iloc[-1], label_row["z"]],
                "--", color="purple", alpha=0.5)
        ax.set_title(sid, fontsize=9)
        ax.set_xlabel("x", fontsize=7)
        ax.set_ylabel("y", fontsize=7)
        ax.set_zlabel("z", fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6)

    plt.suptitle("모기 비행 궤적 샘플 12개 — 초록(시작) → 파랑(관측) → 빨강(현재) → 보라★(정답)",
                 fontsize=11)
    plt.tight_layout()
    out_png = OUT_DIR / "phase1_sample_trajectories.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"figure → {out_png}")

    section("7. 샘플별 통계 — 이동 거리·속도·가속도 간단 분석")

    # 각 샘플의 wide 화 (id × 11시점 × xyz)
    print("샘플별 통계 계산 중 (10000 샘플)...")

    # wide pivot
    def get_per_sample_stats(df: pd.DataFrame) -> pd.DataFrame:
        """샘플별 path length, total displacement, mean velocity 등"""
        sorted_df = df.sort_values(["id", "timestep_ms"])
        # 좌표 array (id 별로 11×3)
        rows = []
        for sid, group in tqdm(sorted_df.groupby("id"), ncols=80,
                                total=df["id"].nunique()):
            coords = group[["x", "y", "z"]].to_numpy()  # (11, 3)
            # 점 사이 거리 (10개)
            diffs = np.diff(coords, axis=0)  # (10, 3)
            step_dists = np.linalg.norm(diffs, axis=1)  # (10,)
            total_path = step_dists.sum()
            displacement = np.linalg.norm(coords[-1] - coords[0])
            mean_vel = step_dists.mean()  # 40ms 당 평균 이동
            # 마지막 40ms 이동
            last_step = step_dists[-1]
            # 가속도 (속도 변화)
            vel_changes = np.diff(step_dists)
            mean_acc = np.abs(vel_changes).mean()
            rows.append({
                "id": sid,
                "total_path_m": total_path,
                "displacement_m": displacement,
                "mean_vel_per_40ms": mean_vel,
                "last_step_m": last_step,
                "mean_abs_acc": mean_acc,
                "x_start": coords[0, 0], "y_start": coords[0, 1],
                "z_start": coords[0, 2],
                "x_end": coords[-1, 0], "y_end": coords[-1, 1],
                "z_end": coords[-1, 2],
            })
        return pd.DataFrame(rows)

    stats = get_per_sample_stats(train_df)
    print("\n샘플별 통계 (train 10000):")
    print(stats[["total_path_m", "displacement_m", "mean_vel_per_40ms",
                 "last_step_m", "mean_abs_acc"]].describe().round(4).to_string())

    stats.to_csv(OUT_DIR / "phase1_per_sample_stats.csv", index=False)
    print(f"  → {OUT_DIR / 'phase1_per_sample_stats.csv'}")

    # 분포 figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    cols = ["total_path_m", "displacement_m", "mean_vel_per_40ms",
            "last_step_m", "mean_abs_acc"]
    titles = ["총 이동 거리 (10 step)", "변위 (시작→끝 직선)",
              "40ms 당 평균 이동", "마지막 40ms 이동",
              "|가속도| 평균"]
    for i, (col, title) in enumerate(zip(cols, titles)):
        ax = axes[i // 3, i % 3]
        ax.hist(stats[col], bins=80, color="steelblue", edgecolor="white")
        ax.set_xlabel(col)
        ax.set_ylabel("# samples")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[1, 2].set_visible(False)
    plt.suptitle("샘플별 운동 통계 분포 (train 10000)", fontsize=12)
    plt.tight_layout()
    out_png = OUT_DIR / "phase1_motion_stats.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"figure → {out_png}")

    section("8. Constant Velocity sanity check")
    # 마지막 두 시점의 이동 벡터 (Δ) × 2 가 80ms 후 변화와 얼마나 가까운가
    train_df_sorted = train_df.sort_values(["id", "timestep_ms"])

    # 각 샘플의 마지막 두 시점 (-40ms, 0ms)
    last_two = train_df_sorted.groupby("id").tail(2)
    last_two_pivot = last_two.pivot(index="id", columns="timestep_ms",
                                     values=["x", "y", "z"])
    # 마지막 step delta (40ms 동안)
    delta_last = last_two_pivot.iloc[:, [3, 4, 5]].values - \
                 last_two_pivot.iloc[:, [0, 1, 2]].values
    # 현재 위치 (0ms)
    cur_pos = last_two_pivot.iloc[:, [3, 4, 5]].values
    # constant velocity 예측 (80ms 후)
    cv_pred = cur_pos + 2.0 * delta_last
    # 정답
    label_arr = labels.set_index("id").loc[
        last_two_pivot.index, ["x", "y", "z"]
    ].values

    # 진짜 변화 (80ms 동안)
    delta_true = label_arr - cur_pos

    # constant velocity 의 예측 오차 (3D 거리)
    cv_error = np.linalg.norm(cv_pred - label_arr, axis=1)

    print(f"Constant Velocity 예측 오차 (3D 거리) 통계:")
    print(f"  mean:    {cv_error.mean():.6f} m")
    print(f"  median:  {np.median(cv_error):.6f} m")
    print(f"  std:     {cv_error.std():.6f} m")
    print(f"  max:     {cv_error.max():.6f} m")
    print(f"  q99:     {np.quantile(cv_error, 0.99):.6f} m")

    # R-Hit @ 1cm
    R_HIT = 0.01
    hit_rate = float(np.mean(cv_error <= R_HIT))
    print(f"\nR-Hit@1cm (1cm 이내 적중률): {hit_rate:.4f} = {hit_rate * 100:.2f}%")
    print(f"  → 우리 베이스라인. 이걸 넘어야 모델이 의미 있음.")

    # figure: CV 예측 오차 분포 + 1cm 표시
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(cv_error, bins=100, color="steelblue", edgecolor="white")
    axes[0].axvline(R_HIT, color="red", linestyle="--",
                    label=f"R_HIT = {R_HIT}m (1cm)")
    axes[0].set_xlabel("CV 예측 오차 (m)")
    axes[0].set_ylabel("# samples")
    axes[0].set_title(f"Constant Velocity 오차 분포\nR-Hit@1cm = {hit_rate:.4f}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # log scale 로 한 번 더
    axes[1].hist(cv_error, bins=100, color="coral", edgecolor="white")
    axes[1].axvline(R_HIT, color="red", linestyle="--",
                    label=f"R_HIT = {R_HIT}m")
    axes[1].set_xlabel("CV 예측 오차 (m)")
    axes[1].set_ylabel("# samples (log)")
    axes[1].set_yscale("log")
    axes[1].set_title("CV 오차 분포 (log y-scale)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out_png = OUT_DIR / "phase1_cv_sanity.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"figure → {out_png}")

    # 마지막 step 벡터 vs 진짜 변화 벡터 — 방향 일치도
    last_norm = np.linalg.norm(delta_last, axis=1)
    true_norm = np.linalg.norm(delta_true, axis=1)
    # cosine similarity
    valid = (last_norm > 1e-8) & (true_norm > 1e-8)
    cosine = np.sum(delta_last[valid] * delta_true[valid], axis=1) / \
             (last_norm[valid] * true_norm[valid])
    print(f"\n마지막 40ms 벡터 vs 진짜 80ms 벡터 방향 일치도 (cosine):")
    print(f"  mean cosine:  {cosine.mean():.4f}")
    print(f"  median:       {np.median(cosine):.4f}")
    print(f"  > 0.9 비율:    {(cosine > 0.9).mean() * 100:.2f}% (거의 같은 방향)")
    print(f"  > 0.99 비율:   {(cosine > 0.99).mean() * 100:.2f}% (매우 일치)")
    print(f"  < 0   비율:    {(cosine < 0).mean() * 100:.2f}% (반대 방향)")

    # 라벨 분포 figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, axis in enumerate(["x", "y", "z"]):
        axes[i].hist(labels[axis], bins=80, color="seagreen",
                     edgecolor="white")
        axes[i].set_title(f"label {axis} (80ms 후)", fontsize=10)
        axes[i].set_xlabel(axis)
        axes[i].grid(alpha=0.3)
    plt.tight_layout()
    out_png = OUT_DIR / "phase1_label_dist.png"
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"\nlabel 분포 figure → {out_png}")

    section("9. 핵심 요약")
    print(f"""
    데이터 구조:
      train 샘플 수:     {train_df['id'].nunique()}
      test 샘플 수:      {test_df['id'].nunique()}
      샘플 당 시점:      11 ({sorted(train_df['timestep_ms'].unique())[0]} ~ {sorted(train_df['timestep_ms'].unique())[-1]} ms)
      좌표 단위:         meter (sensor-local)
      결측:              {train_df.isna().sum().sum()} (train)

    Constant Velocity 베이스라인:
      평균 오차:         {cv_error.mean():.4f} m
      중앙값:            {np.median(cv_error):.4f} m
      R-Hit@1cm:         {hit_rate:.4f} ({hit_rate * 100:.2f}%)
      → 이 점수가 우리의 최소 기준선

    운동 패턴:
      40ms 당 평균 이동: {stats['mean_vel_per_40ms'].mean():.4f} m
      마지막 40ms 평균: {stats['last_step_m'].mean():.4f} m
      방향 일치도:       {cosine.mean():.4f} (cosine, 마지막 vs 80ms 후)
      거의 직선 비율:    {(cosine > 0.9).mean() * 100:.1f}%
    """)

    print()
    print("=" * 64)
    print(f" Phase 1 완료 — {(time.time() - t0) / 60:.1f}분 소요")
    print("=" * 64)
    print(f"산출물: {OUT_DIR}/")
    print("  - phase1_coord_dist.png         좌표 분포 (train/test/label)")
    print("  - phase1_label_dist.png         라벨 분포")
    print("  - phase1_time_evolution.png     시점별 좌표 평균")
    print("  - phase1_sample_trajectories.png 12개 모기 3D 궤적")
    print("  - phase1_motion_stats.png       샘플별 운동 통계 분포")
    print("  - phase1_cv_sanity.png          Constant Velocity 오차 분포")
    print("  - phase1_per_sample_stats.csv   샘플별 path, displacement 등")
    print()
    print("cache:")
    print(f"  - {CACHE_DIR / 'train_combined.parquet'}")
    print(f"  - {CACHE_DIR / 'test_combined.parquet'}")
    print()
    print("다음: Phase 2 — 운동 패턴 군집화, 속도/가속도 깊이 분석")


if __name__ == "__main__":
    main()
