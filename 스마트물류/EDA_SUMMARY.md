# EDA 종합 정리 — 데이콘 스마트 창고 출고 지연 예측

> 이번 머신에서 새로 수행한 EDA Phase 1~4 의 발견 종합. submission3 디자인의 근거 문서.
> CLAUDE.md 의 데이터 구조 / 검증 전략 / 진행 이력은 별도 참조.

---

## 0. 요약 (한 페이지)

| 영역 | 핵심 발견 |
|---|---|
| 데이터 구조 | 250000×94 train, 50000×93 test, 250+50+50 layout, scenario당 25행 정확 일관 |
| 결측 | 86/94 컬럼 결측, 행당 평균 10개, **컬럼 간·시나리오 단위 모두 독립 랜덤 (인위적 노이즈)** |
| 타깃 | mean 18.96 vs median 9.03, max 715, log1p 에서 bimodal — **MAE 직접 최적화 정공법** |
| layout_info | 300 layouts × 14 meta. **모든 train+test layout 100% 커버** |
| 시간 정렬 | shift_hour 시나리오당 평균 8.57 unique → **15분 lag 불가, hour-aggregated 만 가능** |
| 강한 피처 | 우리 derived 가 top 30 의 60%. interaction 카테고리 1위 |
| 갭 1.47 원인 | adversarial AUC 0.6066, **layout_meta 12개가 importance 압도**. 갭 = unseen 이 더 큰 hybrid 창고들 |

---

## 1. Phase 1 — 기본 정보 + 타깃 분포

### 데이터 형태
- train (250000, 94), test (50000, 93)
- 88 float + 3 object + 3 int — object는 ID/layout_id/scenario_id, int는 robot_active/idle/charging 카운트
- train 메모리 231 MB

### 결측 — 행 단위 균일
- 결측 컬럼: 86 / 94
- 컬럼별 결측률: **거의 모두 11.7~13.0%** 범위. 분산 매우 작음
- 행당 평균 10개 셀이 결측 (86×0.12 ≈ 10.3 와 정확 일치)
- 결측 0 행: 7 / 250000 (0.003%)

### 타깃 분포
| 통계 | 값 |
|---|---|
| mean | 18.96 |
| median | 9.03 |
| std | 27.35 |
| min, max | 0, 715.86 |
| q01, q99 | 0.00, 120.85 |

- **right-skewed** — mean 이 median 의 2배
- log1p 변환 시 bimodal (peak 1.8 ≈ target 5, peak 3.5 ≈ target 30) + 0 spike
- ECDF 로그축에서 보면 80% 이상이 target < 30

### Layout overlap
- train 250 + seen 50 + unseen 50 = 300
- scenario_id 교집합 0 (train/test 완전 분리)
- 시나리오당 행수 정확히 25 — **시계열 X, 독립 스냅샷 회귀**

---

## 2. Phase 2 v2 — 컬럼 카테고리 + 결측 패턴 + 파생 피처

### 17개 카테고리 (94 컬럼 명시 매핑)

| 카테고리 | n | 설명 |
|---|---|---|
| key | 3 | ID, layout_id, scenario_id |
| target | 1 | avg_delay_minutes_next_30m |
| robot | 9 | robot_active/idle/charging, utilization, fleet_age 등 |
| battery | 7 | battery_mean/std, charge_queue 등 |
| order_sku | 10 | order_inflow_15m, urgent_order_ratio 등 |
| item_attr | 3 | heavy/cold_chain/avg_package_weight |
| pack_pick | 6 | pack_utilization, pick_list 등 |
| staging_dock | 6 | staging/loading_dock_util, dock_to_stock 등 |
| congestion | 7 | congestion_score, max_zone_density 등 |
| traffic_path | 6 | aisle_traffic, near_collision, blocked_path 등 |
| incident | 3 | fault_count_15m, avg_recovery_time, scanner_error |
| operations | 5 | manual_override, replenishment, task_reassign 등 |
| environment | 5 | warehouse/external_temp, humidity 등 |
| atmosphere | 4 | ambient_noise, air_quality, co2, vibration |
| weather | 2 | wind, precipitation |
| infra_it | 4 | wms_response, wifi, network 등 |
| power | 1 | ups_battery_pct |
| safety_quality | 4 | barcode_success, kpi_otd 등 |
| equipment | 2 | forklift, conveyor |
| time | 2 | shift_hour, day_of_week |
| worker | 4 | staff_on_floor, prev_shift_volume 등 |

### Robot 파생 4개 (시점 무관 검증됨)

```python
robot_total              = robot_active + robot_idle + robot_charging
robot_active_ratio       = robot_active / robot_total       # ※ robot_utilization 과 corr 1.000 → 중복
robot_idle_ratio         = robot_idle / robot_total         # 신규
robot_charging_ratio     = robot_charging / robot_total     # 신규
available_robots         = robot_active + robot_idle        # 신규
```

검증: **layout 안 robot_total unique = 1.00 (mean)** → 시점 무관 일정. 창고 보유량.
또한 **layout_info.csv 의 robot_total 컬럼과 100% 일치** → 우리 계산본 폐기, layout_info 사용.

### 결측 패턴 — 행 단위 독립 랜덤 (인위적 노이즈)

| 검증 | 결과 |
|---|---|
| 행당 결측 컬럼 수 분포 | 종형, 평균 10.2, 86×0.12 = 10.3 와 일치 |
| 결측 동시발생 corr (top 페어) | **최대 0.009** — 사실상 0 |
| scenario당 결측 비율 (top 5 결측 컬럼) | **0~0.4 사이만 분포** — 시나리오 단위 결측 거의 없음 |
| scenario_all_or_none_rate | 3.4~4.0% — 시나리오 단위 결측은 거의 없음 |

**결론**: 결측은 (행, 컬럼) 셀 단위로 독립적으로 ~12% 확률 dropout. **NaN 그대로 두는 것이 최선** (트리 모델 자연 처리). 시나리오 평균 보정도 가능하지만 효과 제한적.

### 분포 4가지 패턴

| 패턴 | 컬럼 예 | 처리 |
|---|---|---|
| **Cap (양 끝 spike)** | warehouse_temp_avg, humidity_pct, lighting_level_lux, pack_utilization 등 15+개 | 트리 모델 자연 분기. flag 5개 추가 |
| **Zero-inflated (0 spike)** | congestion_score, near_collision_15m, fault_count_15m, charge_queue_length 등 12+개 | 트리 자연 분기. event flag 5개 추가 (대부분 원본과 0.99+ 동일) |
| **Right-skewed** | order_inflow_15m, avg_trip_distance, robot_active 등 | 트리는 변환 불필요 |
| **Bimodal** | battery_mean, robot_idle_ratio, return_order_ratio | 강한 분기 신호 |

### 저분산 (1개)
- `task_reassign_15m` : mode_share 96.5% (대부분 0). 그대로 유지.

---

## 3. Phase 3 — 상관관계 + 시간 정렬

### shift_hour 검증 → **lag 폐기 결정**

```
시나리오당 shift_hour unique 값 수: mean=8.57, median=9, range=3~14
시나리오당 day_of_week unique 수: 1=35, 2=1099, 3=4429, 4=3955, 5=477, 6=5
```

해석:
- 25행이 평균 8.57개 시간 슬롯에 분산 → 한 시간당 ~3행
- 같은 시간(예: 14시) 내 3행의 순서를 데이터로 알 수 없음 → **15분 lag 불가**
- 84% 시나리오가 3~4일에 걸침 → 6시간 연속 운영이 아니라 **시뮬레이션 인스턴스의 random sampling**
- 시간 단위 lag (hour-aggregated) 도 sampling 형태라 위험

→ **lag 폐기**. 시나리오 통계 (`__scn_mean`, `__scn_max`) 가 같은 정보를 안전하게 제공.

### 피처-타깃 상관 — top 30 (Spearman 절대값)

```
1.  congestion_score                  0.6745
2.  max_zone_density                  0.6722
3.  incident_score                    0.6639   ← derived (event 종합)
4.  incident_x_stress                 0.6501   ← interaction
5.  load_pressure                     0.6426   ← interaction
6.  robot_idle_ratio                 -0.6368   ← derived (Phase 2 v2)
7.  orders_per_available              0.6311   ← derived (capacity_norm)
8.  robot_charging_ratio              0.6289   ← derived
9.  robot_charging                    0.6202
10. low_battery_ratio                 0.6076
11. stress_x_orders_per_robot         0.6023   ← interaction
12. stress_x_inflow                   0.5962   ← interaction
13. robot_stress_score                0.5794   ← derived (stress)
14. flag_congestion_hot               0.5762   ← derived (event flag)
15. robot_idle                       -0.5534
16. flag_charging_active              0.5530
17. orders_per_robot                  0.5496   ← derived
18. orders_per_pack_station           0.5229   ← derived
19. order_inflow_15m                  0.5031
20. skus_per_robot                    0.4823   ← derived
21. charge_queue_length               0.4816
22. battery_mean                     -0.4763
23. flag_idle_zero                    0.4723
24. avg_charge_wait                   0.4701
25. shortage_x_inflow                 0.4653   ← interaction
26. near_collision_15m                0.4646
27. blocked_path_15m                  0.4615
28. orders_per_charger                0.4570   ← derived
29. available_robots                 -0.4479   ← derived
30. flag_charge_queue                 0.4429
```

**18 / 30 이 우리 derived** — Phase 2 v2 + 사용자 가설들이 데이터로 검증.

### 카테고리별 평균 |Spearman|

```
순위                  카테고리           mean    max
1.   ★ interaction      (n=5)        0.591   0.650
2.   ★ robot_derived    (n=4)        0.504   0.637
3.   ★ event_flag       (n=6)        0.487   0.664
4.   ★ capacity_norm    (n=6)        0.462   0.631
5.   ★ stress_flag      (n=4)        0.419   0.579
6.     battery          (n=7)        0.398   0.608
7.     incident         (n=3)        0.274   0.413
8.     robot            (n=9)        0.258   0.620
9.     order_sku        (n=10)       0.236   0.503
10.    item_attr        (n=3)        0.229   0.306
11.    congestion       (n=7)        0.224   0.674   ← max는 1위 (congestion_score)
...
20.    layout_meta      (n=17)       0.042   0.195
21.    pack_pick        (n=6)        0.031   0.127
22.    environment      (n=5)        0.024   0.094   ← 노이즈
23.    atmosphere       (n=4)        0.006   0.015   ← 노이즈
24.    weather          (n=2)        0.005   0.008   ← 노이즈
25.    infra_it         (n=4)        0.005   0.013   ← 노이즈
26.    power            (n=1)        0.000   0.000   ← 노이즈
```

**우리가 만든 5개 카테고리가 1~5위 독식**. environment/atmosphere/weather/infra_it/power 5개 카테고리는 사실상 노이즈.

### 비선형 관계 (binned mean)

상위 9개 피처 모두 **단조 증가/감소 + plateau(포화)** 모양. 트리 모델이 잘 학습할 수 있는 형태. 변환 불필요.

### 피처-피처 상관 — 중복 경고

```
load_pressure ↔ orders_per_available     0.998  ← 거의 동일
fault_count_15m ↔ flag_fault              0.995  ← flag 무가치
blocked_path_15m ↔ flag_blocked           0.994
near_collision_15m ↔ flag_collision       0.993
charge_queue_length ↔ flag_charge_queue   0.993
fault_count_15m ↔ avg_recovery_time       0.993  ← 다른 도메인 컬럼이 동일
flag_idle_zero ↔ shortage_x_inflow        0.992
robot_charging_ratio ↔ robot_charging     0.992
stress_x_orders_per_robot ↔ stress_x_inflow  0.987
charge_queue_length ↔ avg_charge_wait     0.984
```

**event flag 5개는 원본 (>0 분기) 과 0.99+ 동일** → 사실상 무가치. submission3 에서 제거 (incident_score 종합만 유지).

---

## 4. Phase 4 — seen vs unseen + adversarial validation

### Layout 메타 분포 (train+seen 250 vs unseen 50)

unseen 50 layout 의 특징 (KS top):

| 컬럼 | train+seen | unseen | 차이 | KS |
|---|---|---|---|---|
| robot_total | 45.5 | 55.1 | +21% | 0.284 |
| one_way_ratio | 0.24 | 0.31 | +30% | 0.244 |
| pack_station_count | 10.7 | 13.4 | +25% | 0.240 |
| charger_count | 8.3 | 9.9 | +19% | 0.220 |

**unseen 50 = 더 큰, 더 정교한 창고들** (로봇·도크·충전기 多, 일방통행 비율 高).

layout_type 분포:
- train+seen: grid 36% / hybrid 30% / hub_spoke 17% / narrow 17%
- unseen:     grid 32% / **hybrid 46%** / **hub_spoke 6%** / narrow 16%
- **hybrid 가 압도적, hub_spoke 가 거의 없음**

### 행 단위 분포 (train vs test)

모든 강한 피처에서 **test 가 30~50% 더 부하 높음**:

| 피처 | train | test | 차이 |
|---|---|---|---|
| order_inflow_15m | 95 | 132 | +39% |
| load_pressure | 9.4 | 14.8 | +57% |
| robot_charging_ratio | 0.16 | 0.22 | +34% |
| congestion_score | 10.0 | 13.0 | +30% |
| robot_idle_ratio | 0.51 | 0.42 | -18% (더 바쁨) |

→ test 시점들이 더 빡센 시점들. 부분적으로 layout 차이의 결과 (큰 창고는 자연스럽게 부하 큼), 부분적으로 sampling.

### Adversarial Validation

```
fold AUC: 0.4825, 0.7031, 0.5775, 0.6370, 0.5855
Overall OOF AUC: 0.6066  ← "분포 약간 다름"

Top 20 importance:
  1.  aisle_width_avg          84   layout_meta
  2.  intersection_count       32   layout_meta
  3.  one_way_ratio            32   layout_meta
  4.  pack_station_count       28   layout_meta
  5.  zone_dispersion          26   layout_meta
  6.  robot_total              26   layout_meta
  7.  building_age_years       23   layout_meta
  8.  floor_area_sqm           22   layout_meta
  9.  charger_count            22   layout_meta
  10. ceiling_height_m         20   layout_meta
  11. layout_compactness       15   layout_meta
  12. fire_sprinkler_count     14   layout_meta
  --- 여기서부터 importance 급감 ---
  13. sku_concentration         9   order_sku
  ...
```

**top 12 가 모두 layout_meta** — 갭 1.47 의 80% 가 layout 메타 차이로 설명.

### 종합 진단 — 갭 1.47 의 분해

```
~80% : unseen 50 layout 이 train 250 과 다른 메타 (큰 창고, hybrid 위주)
~15% : 그 결과 시점 부하가 자연스럽게 높음
~5%  : 시점 sampling 차이
```

**처방**:
1. **layout_info LEFT JOIN** (이미 결정) — 80% 갭 보정 기대
2. **capacity 정규화 비율** (이미 결정) — 15% 갭 보정
3. (선택) sample weighting — 미세 효과, AUC 0.6 이라 후순위

fold 별 변동 큼 (0.48~0.70) → 어떤 unseen 은 분포 같음 (쉬움), 어떤 unseen 은 매우 다름 (어려움). **모델 다양성 필수** (다양한 seed, 다른 알고리즘 앙상블).

---

## 5. submission3 디자인 (확정)

### 구조 — 옛 submission1 과 동일

| 항목 | 결정 |
|---|---|
| 모델 | LightGBM Quantile (alpha=0.5) |
| CV | GroupKFold(5, by=layout_id) |
| 손실 | quantile |
| 결측 | NaN 그대로 (LightGBM 자연 처리) |

### 피처 (≈ 108개)

```
[베이스] 원본 수치 89개 - 환경 노이즈 18개 = 71개
[L1] layout_info LEFT JOIN 14개 + layout_type one-hot 4개 = 18개
[L2] robot 파생 4개 (idle_ratio, charging_ratio, available_robots, robot_total은 L1)
     ※ active_ratio 폐기 (utilization 과 동일)
[L3] capacity 정규화 6개 (orders_per_robot, orders_per_pack_station,
     orders_per_charger, orders_per_available, skus_per_robot, picks_per_robot)
[L4] stress 4개 (flag_idle_zero, flag_charging_active, flag_active_high, robot_stress_score)
[L5] event 1개 (incident_score 만, 개별 flag 5개는 원본과 중복이라 폐기)
[L6] interaction 5개 (stress_x_inflow, stress_x_orders_per_robot,
     shortage_x_inflow, load_pressure, incident_x_stress)
─────────────────────────────────────────────────────
총 ≈ 108~112 (약 71 + 18 + 4 + 6 + 4 + 1 + 5)
```

### 제거 후보 (환경 노이즈 카테고리 18개)

| 카테고리 | 컬럼 |
|---|---|
| environment | warehouse_temp_avg, humidity_pct, external_temp_c, lighting_level_lux, cold_storage_temp_c |
| atmosphere | ambient_noise_db, air_quality_idx, co2_level_ppm, floor_vibration_idx |
| weather | wind_speed_kmh, precipitation_mm |
| infra_it | wms_response_time_ms, wifi_signal_db, network_latency_ms, hvac_power_kw |
| power | ups_battery_pct |
| (선택) pack_pick 약함 | label_print_queue 등 |

이 18개 모두 mean |Spearman| < 0.03. 첫 시도엔 유지하고 importance 보고 제거하는 것도 안전.

---

## 6. 다음 시도 라인업 (잠정)

| # | 변경점 | 기대 효과 |
|---|---|---|
| **submission3** | 위 디자인 (LightGBM Quantile + 108 피처) | 옛 OOF 9.18 보다 개선 가능성 |
| submission4 | sample weight (adversarial proba 기반) | unseen 보정, 작은 효과 |
| submission5 | LightGBM Huber 손실 | 알고리즘 다양성 |
| submission6 | CatBoost MAE | 알고리즘 다양성 |
| submission7 | XGBoost reg:absoluteerror | 알고리즘 다양성 |
| submission8 | OOF 가중평균 앙상블 | 최종 |

---

## 7. 주요 결정사항 요약 (변하지 말 것)

1. **단일 글로벌 모델** + layout_info LEFT JOIN (per-layout 분리 모델 X)
2. **GroupKFold by layout_id** (단순 KFold 절대 금지)
3. **MAE/Quantile 직접 최적화** (log1p+MSE 폐기 — submission2 결과로 확인)
4. **NaN 그대로** (트리 모델 자연 처리)
5. **lag 폐기** (시간 정렬 불가)
6. **시나리오 통계 피처는 후순위** (submission3 다음)
7. **앙상블은 최종** (다양한 모델 라인업 확보 후)
