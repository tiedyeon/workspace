# =============================================================================
# Step 25: TabM – Parameter-Efficient MLP Ensembling (ICLR 2025, Yandex)
# =============================================================================
# 논문: "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"
#       Gorishniy et al., ICLR 2025  |  https://arxiv.org/abs/2401.03893
# 라이선스: Apache 2.0 ✅ (당사 자체 구현, 외부 패키지 불필요)
#
# 핵심 아이디어:
#   K개의 MLP 앙상블 멤버가 가중치를 완전히 공유하고,
#   입력에 곱해지는 소형 adapter 벡터(α_k ∈ R^d)만 각 멤버별로 독립 학습.
#   최종 예측 = (1/K) Σ_k  MLP(x ⊙ α_k)
#   → 파라미터 효율적으로 강력한 앙상블 효과 달성
#
# 특징:
#   - TabReD 벤치마크(산업용, 수백 피처, 분포 이동) 에서 GBDT 수준 성능
#   - step20 동일 피처 448개 사용
#   - 3 seeds × 5-fold GroupKFold  (GBM과 동일 CV 구조)
#   - OOF 저장 → step23 TabPFN 스태킹에 활용
#
# 실행:
#   caffeinate -i nohup python -u -B step25_tabm.py > step25_output.log 2>&1 &
#   tail -f step25_output.log
# =============================================================================

import os, time, warnings, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── 디바이스 선택 ──────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

print(f"Device: {DEVICE}")

DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = DATA_DIR
TARGET     = 'avg_delay_minutes_next_30m'
SEEDS      = [42, 123, 456]
N_SPLITS   = 5

# TabM 하이퍼파라미터
K           = 32       # 앙상블 멤버 수 (논문 기본값)
HIDDEN_DIMS = [512, 256, 128]
DROPOUT     = 0.15
LR          = 1e-3
WEIGHT_DECAY= 1e-5
BATCH_SIZE  = 4096
MAX_EPOCHS  = 200
PATIENCE    = 20       # early stopping

LAYOUT_TYPE_MAP = {'grid': 0, 'hybrid': 1, 'narrow': 2, 'hub_spoke': 3}

# =============================================================================
# 1. 피처 엔지니어링 (Step20 동일)
# =============================================================================
AGG_COLS  = ['order_inflow_15m', 'low_battery_ratio', 'congestion_score',
             'robot_utilization', 'pack_utilization', 'robot_active',
             'charge_queue_length', 'max_zone_density']
AGG_FUNCS = ['mean', 'std', 'max', 'min']

ZSCORE_COLS = ['congestion_score', 'low_battery_ratio', 'max_zone_density',
               'pack_utilization', 'robot_utilization', 'order_inflow_15m',
               'charge_queue_length', 'robot_active', 'battery_mean',
               'aisle_traffic_score', 'blocked_path_15m']
RANK_COLS   = ['congestion_score', 'low_battery_ratio', 'max_zone_density',
               'pack_utilization', 'order_inflow_15m', 'charge_queue_length',
               'battery_mean', 'robot_utilization', 'blocked_path_15m']
RATIO_COLS  = ['congestion_score', 'low_battery_ratio', 'max_zone_density',
               'pack_utilization', 'order_inflow_15m', 'charge_queue_length',
               'robot_active', 'battery_mean']
SKEWED_COLS = ['task_reassign_15m', 'blocked_path_15m', 'fault_count_15m',
               'avg_charge_wait', 'near_collision_15m', 'avg_recovery_time',
               'charge_queue_length', 'robot_charging']
MEANINGFUL_NAN_COLS = ['avg_recovery_time', 'congestion_score',
                       'loading_dock_util', 'staging_area_util']
ZERO_FILL_COLS = ['charge_queue_length', 'avg_charge_wait', 'blocked_path_15m',
                  'near_collision_15m', 'fault_count_15m', 'task_reassign_15m',
                  'replenishment_overlap', 'congestion_score', 'max_zone_density',
                  'avg_recovery_time', 'loading_dock_util', 'staging_area_util']
EXPANDING_COLS = ['congestion_score', 'low_battery_ratio', 'order_inflow_15m',
                  'robot_active', 'blocked_path_15m', 'charge_queue_length',
                  'max_zone_density', 'battery_mean']
TRAJECTORY_COLS = ['congestion_score', 'low_battery_ratio', 'order_inflow_15m',
                   'robot_active', 'blocked_path_15m']
SEQ_COLS = [
    'order_inflow_15m', 'unique_sku_15m', 'robot_active', 'robot_idle', 'robot_charging',
    'battery_mean', 'battery_std', 'low_battery_ratio', 'charge_queue_length',
    'avg_charge_wait', 'congestion_score', 'max_zone_density', 'blocked_path_15m',
    'near_collision_15m', 'fault_count_15m', 'avg_recovery_time', 'task_reassign_15m',
    'replenishment_overlap', 'pack_utilization', 'loading_dock_util',
    'staging_area_util', 'label_print_queue',
]


def feature_engineering(df):
    df = df.copy()
    for col in MEANINGFUL_NAN_COLS:
        if col in df.columns:
            df[f'nan_{col}'] = df[col].isna().astype(np.int8)
    for col in ZERO_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    for col in ['congestion_score', 'blocked_path_15m', 'near_collision_15m',
                'charge_queue_length', 'avg_charge_wait', 'fault_count_15m',
                'replenishment_overlap', 'task_reassign_15m']:
        if col in df.columns:
            df[f'flag_{col}'] = (df[col] > 0).astype(np.int8)
    df['battery_stress']         = df['low_battery_ratio'] / (df['battery_mean'] + 1e-5)
    df['charge_bottleneck']      = df['charge_queue_length'] * df['avg_charge_wait']
    df['battery_volatility']     = df['battery_std'] / (df['battery_mean'] + 1e-5)
    df['battery_health']         = df['battery_mean'] - df['battery_std']
    df['order_per_robot']        = df['order_inflow_15m'] / (df['robot_active'] + 1)
    df['order_per_pack_station'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1)
    df['robot_effective_util']   = df['robot_active'] / (df['robot_total'] + 1)
    df['idle_ratio']             = df['robot_idle'] / (df['robot_active'] + df['robot_idle'] + df['robot_charging'] + 1)
    df['charging_ratio']         = df['robot_charging'] / (df['robot_total'] + 1)
    df['congestion_x_density']   = df['congestion_score'] * df['max_zone_density']
    df['traffic_severity']       = df['blocked_path_15m'] + df['near_collision_15m'] * 2
    df['aisle_load']             = df['aisle_traffic_score'] * df['congestion_score']
    df['layout_type_enc']        = df['layout_type'].map(LAYOUT_TYPE_MAP).fillna(-1).astype(np.int8)
    df['order_per_charger']      = df['order_inflow_15m'] / (df['charger_count'] + 1)
    df['robot_per_floor_area']   = df['robot_total'] / (df['floor_area_sqm'] + 1)
    df['pack_station_per_robot'] = df['pack_station_count'] / (df['robot_total'] + 1)
    for col in ['battery_mean', 'low_battery_ratio', 'congestion_score',
                'order_inflow_15m', 'robot_active', 'pack_utilization']:
        if col in df.columns:
            df[f'null_{col}'] = df[col].isna().astype(np.int8)
    for col in SKEWED_COLS:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].fillna(0))
    df['battery_crisis_index'] = (
        df['low_battery_ratio'] * df['charge_queue_length']
        + df['robot_charging'] / (df['battery_mean'] + 1e-5))
    df['congestion_level'] = pd.cut(
        df['congestion_score'].fillna(0), bins=[-1, 0, 20, 100], labels=[0, 1, 2]).astype(int)
    df['congestion_compound'] = (
        df['congestion_score'].fillna(0) * df['max_zone_density'].fillna(0)
        + df['blocked_path_15m'].fillna(0) * 2
        + df['near_collision_15m'].fillna(0) * 3)
    df['robot_saturation']   = 1 - (df['robot_idle'] / (df['robot_total'] + 1))
    df['operation_pressure'] = df['order_inflow_15m'] * df['low_battery_ratio'] / (df['robot_active'] + 1)
    df['triple_crisis']      = df['low_battery_ratio'] * df['congestion_score'].fillna(0) * df['order_inflow_15m']
    df['crisis_score']       = df['low_battery_ratio'] * df['congestion_score'].fillna(0)
    df['order_robot_stress'] = df['order_inflow_15m'] / (df['robot_active'] + 1) * df['low_battery_ratio']
    df['bottleneck_score']   = df['charge_queue_length'] * df['congestion_score'].fillna(0)
    df['complex_urgent_order'] = df['sku_concentration'] * df['urgent_order_ratio']
    if 'maintenance_schedule_score' in df.columns:
        df['maintenance_battery_risk'] = (1 - df['maintenance_schedule_score'].fillna(0.5)) * df['low_battery_ratio']
    df['layout_congestion']  = df['layout_type_enc'] * df['congestion_score'].fillna(0)
    df['layout_battery']     = df['layout_type_enc'] * df['low_battery_ratio']
    return df


def add_temporal_features(df):
    df = df.copy()
    df['slot_idx']      = df.groupby('scenario_id').cumcount()
    df['slot_progress'] = df['slot_idx'] / 24.0
    for col in EXPANDING_COLS:
        if col in df.columns:
            grp = df.groupby('scenario_id')[col]
            df[f'exp_mean_{col}'] = grp.expanding().mean().reset_index(level=0, drop=True)
            df[f'exp_max_{col}']  = grp.expanding().max().reset_index(level=0, drop=True)
    for col in ['congestion_score', 'low_battery_ratio', 'order_inflow_15m',
                'robot_active', 'blocked_path_15m', 'charge_queue_length']:
        if col in df.columns:
            df[f'lag1_{col}'] = df.groupby('scenario_id')[col].shift(1)
    return df


def add_lag_rolling_features(df):
    df = df.copy()
    grp = df.groupby('scenario_id', sort=False)
    grp_key = df['scenario_id']
    for col in SEQ_COLS:
        if col not in df.columns:
            continue
        lag1 = grp[col].shift(1)
        lag2 = grp[col].shift(2)
        df[f'{col}__lag1']  = lag1
        df[f'{col}__lag2']  = lag2
        df[f'{col}__diff1'] = df[col] - lag1
        lag1_grp  = lag1.groupby(grp_key)
        roll_mean = lag1_grp.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        roll_max  = lag1_grp.rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
        df[f'{col}__rollmean3']     = roll_mean
        df[f'{col}__rollmax3']      = roll_max
        df[f'{col}__dev_rollmean3'] = df[col] - roll_mean
    return df


def add_onset_features(df):
    df = df.copy()
    grp_key = df['scenario_id']
    def add_onset(value_col, prefix):
        if value_col not in df.columns:
            return
        positive = df[value_col].fillna(0).gt(0)
        t = df['slot_idx'].where(positive).astype(float)
        first = t.groupby(grp_key).transform(lambda s: s.ffill().cummin())
        prev  = positive.groupby(grp_key).shift(1, fill_value=False)
        df[f'{prefix}_ever_started']      = first.notna().astype(np.int8)
        df[f'{prefix}_start_idx']         = first.fillna(-1).astype(np.float32)
        df[f'{prefix}_started_now']       = (positive & ~prev).astype(np.int8)
        df[f'{prefix}_started_early']     = (first <= 5).fillna(False).astype(np.int8)
        df[f'{prefix}_steps_since_start'] = np.where(
            first.notna(), (df['slot_idx'] - first).astype(float), -1.0).astype(np.float32)
    add_onset('robot_charging',      'charging')
    add_onset('charge_queue_length', 'queue')
    return df


def add_threshold_features(df):
    df = df.copy()
    if {'robot_charging', 'charge_queue_length', 'charger_count'}.issubset(df.columns):
        df['charge_pressure2'] = (
            (df['robot_charging'] + df['charge_queue_length']) / (df['charger_count'] + 1e-5))
    if 'battery_mean' in df.columns:
        df['battery_mean_below_44']     = np.clip(44.0 - df['battery_mean'].fillna(44), 0, None)
    if 'charge_pressure2' in df.columns:
        df['charge_pressure_above_136'] = np.clip(df['charge_pressure2'] - 1.36, 0, None)
    if 'pack_utilization' in df.columns:
        df['pack_utilization_sq']       = df['pack_utilization'].fillna(0) ** 2
    if 'loading_dock_util' in df.columns:
        df['loading_dock_util_sq']      = df['loading_dock_util'].fillna(0) ** 2
    if 'staging_area_util' in df.columns:
        df['staging_area_util_sq']      = df['staging_area_util'].fillna(0) ** 2
    return df


def add_aggregation_features(train_df, test_df):
    train_sc = train_df.groupby('scenario_id')[AGG_COLS].agg(AGG_FUNCS)
    train_sc.columns = [f'sc_{c}_{f}' for c, f in train_sc.columns]
    test_sc  = test_df.groupby('scenario_id')[AGG_COLS].agg(AGG_FUNCS)
    test_sc.columns  = [f'sc_{c}_{f}' for c, f in test_sc.columns]
    train_df = train_df.merge(train_sc.reset_index(), on='scenario_id', how='left')
    test_df  = test_df.merge(test_sc.reset_index(),   on='scenario_id', how='left')
    layout_agg = train_df.groupby('layout_id')[AGG_COLS].agg(AGG_FUNCS)
    layout_agg.columns = [f'layout_{c}_{f}' for c, f in layout_agg.columns]
    layout_agg = layout_agg.reset_index()
    lfc = [c for c in layout_agg.columns if c.startswith('layout_') and c != 'layout_id']
    train_df = train_df.merge(layout_agg[['layout_id'] + lfc], on='layout_id', how='left', suffixes=('', '_dup'))
    train_df.drop(columns=[c for c in train_df.columns if c.endswith('_dup')], inplace=True)
    test_df  = test_df.merge(layout_agg[['layout_id'] + lfc], on='layout_id', how='left')
    return train_df, test_df


def add_sc_derived_features(df):
    df = df.copy()
    for col in ZSCORE_COLS:
        m, s = f'sc_{col}_mean', f'sc_{col}_std'
        if col in df.columns and m in df.columns and s in df.columns:
            df[f'z_{col}']     = (df[col] - df[m]) / (df[s] + 1e-5)
            df[f'z_{col}_abs'] = df[f'z_{col}'].abs()
    for col in RANK_COLS:
        if col in df.columns:
            df[f'prank_{col}'] = df.groupby('scenario_id')[col].rank(pct=True, na_option='keep')
    for col in RATIO_COLS:
        if col in df.columns:
            if f'sc_{col}_max' in df.columns:
                df[f'ratio_to_max_{col}']  = df[col] / (df[f'sc_{col}_max'] + 1e-5)
                df[f'gap_to_max_{col}']    = df[f'sc_{col}_max'] - df[col]
            if f'sc_{col}_mean' in df.columns:
                df[f'ratio_to_mean_{col}'] = df[col] / (df[f'sc_{col}_mean'] + 1e-5)
    return df


def add_trajectory_features(df):
    df = df.copy()
    for col in TRAJECTORY_COLS:
        exp_col = f'exp_mean_{col}'
        sc_col  = f'sc_{col}_mean'
        if exp_col in df.columns and sc_col in df.columns:
            df[f'traj_{col}']     = df[exp_col] / (df[sc_col] + 1e-5)
            df[f'traj_dev_{col}'] = (df[exp_col] - df[sc_col]).abs()
    return df


def add_layout_target_encoding(train_df, test_df, y_log):
    gkf_te = GroupKFold(n_splits=N_SPLITS)
    scenario_ids = train_df['scenario_id'].values
    layout_te_train = np.full(len(train_df), np.nan)
    for _, (tr_idx, val_idx) in enumerate(gkf_te.split(train_df, y_log, scenario_ids)):
        tr_sub = train_df.iloc[tr_idx].copy()
        tr_sub['_y'] = y_log[tr_idx]
        layout_mean      = tr_sub.groupby('layout_id')['_y'].mean()
        layout_type_mean = tr_sub.groupby('layout_type')['_y'].mean()
        global_mean      = float(y_log[tr_idx].mean())
        val_df = train_df.iloc[val_idx]
        encoded = val_df['layout_id'].map(layout_mean).copy()
        missing_mask = encoded.isna()
        if missing_mask.any():
            encoded[missing_mask] = val_df.loc[missing_mask, 'layout_type'].map(layout_type_mean)
        encoded = encoded.fillna(global_mean)
        layout_te_train[val_idx] = encoded.values
    train_df = train_df.copy()
    train_df['layout_te'] = layout_te_train
    full_tr = train_df.copy(); full_tr['_y'] = y_log
    layout_mean_all      = full_tr.groupby('layout_id')['_y'].mean()
    layout_type_mean_all = full_tr.groupby('layout_type')['_y'].mean()
    global_mean_all      = float(y_log.mean())
    test_df = test_df.copy()
    test_encoded = test_df['layout_id'].map(layout_mean_all).copy()
    missing_test = test_encoded.isna()
    if missing_test.any():
        test_encoded[missing_test] = test_df.loc[missing_test, 'layout_type'].map(layout_type_mean_all)
    test_encoded = test_encoded.fillna(global_mean_all)
    test_df['layout_te'] = test_encoded.values
    print(f"  layout_te cold-start: {int(missing_test.sum())}행")
    return train_df, test_df


# =============================================================================
# 2. TabM 모델 (논문 구조 충실 구현)
# =============================================================================
class TabM(nn.Module):
    """
    TabM: Multiplying Adapter 방식 파라미터 효율적 앙상블
    - K개의 adapter 벡터(α_k)가 입력에 원소별 곱셈
    - MLP 가중치는 K개 멤버 완전 공유
    - 최종 예측 = (1/K) Σ_k MLP(x ⊙ α_k)
    """
    def __init__(self, input_dim, hidden_dims=HIDDEN_DIMS, k=K, dropout=DROPOUT):
        super().__init__()
        self.K = k

        # 입력 정규화 (BN을 adapter 적용 전에 수행)
        self.input_bn = nn.BatchNorm1d(input_dim)

        # K개 adapter: 초기값 1.0, 작은 노이즈 → 초기엔 모두 동일 출력
        self.adapters = nn.Parameter(
            torch.ones(k, input_dim) + 0.02 * torch.randn(k, input_dim))

        # 공유 MLP (LayerNorm 사용 → reshape 후에도 per-sample 독립 정규화)
        dims = [input_dim] + hidden_dims
        self.blocks = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.blocks.append(nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

        # 출력 헤드
        self.head = nn.Linear(hidden_dims[-1], 1)

        # 초기화
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, D)
        B, D = x.shape
        x_norm = self.input_bn(x)                           # (B, D)
        x_k = x_norm.unsqueeze(1) * self.adapters.unsqueeze(0)  # (B, K, D)
        h = x_k.reshape(B * self.K, D)                     # (B*K, D)
        for block in self.blocks:
            h = block(h)                                    # (B*K, H)
        y = self.head(h).reshape(B, self.K).mean(dim=1)    # (B,)
        return y


# =============================================================================
# 3. 학습 유틸리티
# =============================================================================
def mae_loss(pred, target):
    return torch.mean(torch.abs(pred - target))


def train_one_fold(X_tr, y_tr, X_val, y_val, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TabM(input_dim=X_tr.shape[1]).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=1, eta_min=1e-5)

    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_np = y_val  # numpy for MAE

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    loader   = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(DEVICE.type == 'cuda'))

    best_val_mae = float('inf')
    best_weights = None
    no_improve   = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = mae_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # 검증
        model.eval()
        with torch.no_grad():
            val_pred_log = model(X_val_t).cpu().numpy()

        val_pred = np.clip(np.expm1(val_pred_log), 0, None)
        val_mae  = mean_absolute_error(np.expm1(y_val_np), val_pred)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    # 최적 가중치 복원
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})
    return model, best_val_mae, epoch - PATIENCE


def predict_tabm(model, X_np, batch_size=8192):
    model.eval()
    X_t = torch.tensor(X_np, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb = X_t[i:i+batch_size].to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


# =============================================================================
# 4. 데이터 로드 & 피처 엔지니어링
# =============================================================================
print("\n" + "="*60)
print("1. 데이터 로딩")
print("="*60)

train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

train = train.merge(layout, on='layout_id', how='left')
test  = test.merge(layout,  on='layout_id', how='left')
print(f"train: {train.shape} | test: {test.shape}")

print("\n" + "="*60)
print("2. 피처 엔지니어링 (Step20 동일)")
print("="*60)

train = feature_engineering(train)
test  = feature_engineering(test)
train = add_temporal_features(train)
test  = add_temporal_features(test)
train = add_lag_rolling_features(train)
test  = add_lag_rolling_features(test)
train = add_onset_features(train)
test  = add_onset_features(test)
train = add_threshold_features(train)
test  = add_threshold_features(test)
train, test = add_aggregation_features(train, test)
train = add_sc_derived_features(train)
test  = add_sc_derived_features(test)
train = add_trajectory_features(train)
test  = add_trajectory_features(test)

y_log_full = np.log1p(train[TARGET].values)
train, test = add_layout_target_encoding(train, test, y_log_full)

DROP_COLS    = {'ID', 'layout_id', 'scenario_id', 'layout_type', TARGET, 'shift_hour'}
feature_cols = [c for c in train.columns
                if c not in DROP_COLS and c in test.columns
                and train[c].dtype != object]
feature_cols = list(dict.fromkeys(feature_cols))

X_all  = train[feature_cols].astype(np.float32).values
y_all  = y_log_full.astype(np.float32)
X_test = test[feature_cols].astype(np.float32).values
y_true = train[TARGET].values
groups = train['scenario_id'].values

print(f"\n  피처 수: {len(feature_cols)}개")
print(f"  train: {X_all.shape} | test: {X_test.shape}")

# StandardScaler 적합 (전체 train으로)
scaler  = StandardScaler()
X_all   = scaler.fit_transform(X_all).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

# NaN → 0 처리
X_all[~np.isfinite(X_all)]   = 0.0
X_test[~np.isfinite(X_test)] = 0.0

# =============================================================================
# 5. 학습 (3 seeds × 5 folds)
# =============================================================================
print("\n" + "="*60)
print(f"3. TabM 학습 ({len(SEEDS)}seeds × {N_SPLITS}folds, K={K}, hidden={HIDDEN_DIMS})")
print("="*60)

gkf = GroupKFold(n_splits=N_SPLITS)
all_oof  = {}
all_test = {}

total_start = time.time()

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)

    oof_p  = np.zeros(len(X_all))
    test_p = np.zeros(len(X_test))
    fold_maes = []
    t_seed = time.time()

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups)):
        t_fold = time.time()
        X_tr, X_val = X_all[tr_idx], X_all[val_idx]
        y_tr, y_val = y_all[tr_idx], y_all[val_idx]

        model, val_mae, stopped_ep = train_one_fold(X_tr, y_tr, X_val, y_val, seed + fold)

        # OOF 예측
        oof_log = predict_tabm(model, X_val)
        oof_p[val_idx] = np.clip(np.expm1(oof_log), 0, None)

        # Test 예측 (fold 평균)
        test_log = predict_tabm(model, X_test)
        test_p  += np.clip(np.expm1(test_log), 0, None) / N_SPLITS

        fold_maes.append(val_mae)
        elapsed = time.time() - t_fold
        print(f"  seed={seed} fold={fold+1} | val MAE={val_mae:.4f} | "
              f"stopped@ep{stopped_ep} | {elapsed:.1f}s")

        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    seed_mae = mean_absolute_error(y_true, oof_p)
    print(f"  seed={seed} OOF MAE: {seed_mae:.4f} ± fold_std={np.std(fold_maes):.4f} | "
          f"{(time.time()-t_seed)/60:.1f}min\n")

    all_oof[seed]  = oof_p
    all_test[seed] = test_p

# =============================================================================
# 6. 시드 앙상블 & 결과
# =============================================================================
print("="*60)
print("4. 앙상블")
print("="*60)

oof_mean  = np.stack(list(all_oof.values())).mean(axis=0)
test_mean = np.stack(list(all_test.values())).mean(axis=0)
final_mae = mean_absolute_error(y_true, oof_mean)

for seed in SEEDS:
    m = mean_absolute_error(y_true, all_oof[seed])
    print(f"  seed={seed:5d} OOF MAE: {m:.4f}")
print(f"\n  ▶ 앙상블 OOF MAE : {final_mae:.4f}")
print(f"  ▶ 총 소요 시간    : {(time.time()-total_start)/60:.1f}분")
print(f"\n  예측값 분포: min={oof_mean.min():.2f} | "
      f"mean={oof_mean.mean():.2f} | max={oof_mean.max():.2f} | "
      f"p95={np.percentile(oof_mean,95):.2f}")

print("\n  Step 비교:")
print(f"    Step20 (LGB+XGB+Cat)  OOF: 8.5964  Public: 10.0606")
print(f"    Step25 TabM           OOF: {final_mae:.4f}")

# =============================================================================
# 7. 저장
# =============================================================================
# OOF 저장 (step23/step27 TabPFN 스태킹 용)
oof_df = pd.DataFrame({
    'scenario_id': train['scenario_id'].values,
    'slot_idx':    train.groupby('scenario_id').cumcount().values,
    'true':        y_true,
    'oof_pred':    oof_mean,
})
oof_path = os.path.join(OUTPUT_DIR, 'oof_step25_tabm.csv')
oof_df.to_csv(oof_path, index=False)
print(f"\n  OOF 저장: oof_step25_tabm.csv")

# 단독 제출
sub_solo = sample.copy()
sub_solo[TARGET] = np.clip(test_mean, 0, None)
sub_solo.to_csv(os.path.join(OUTPUT_DIR, 'submission_step25_tabm_only.csv'), index=False)

# Step20 (OOF 기준 가중) 블렌드
# Step20 OOF MAE=8.5964, TabM OOF=final_mae
w20 = 1 / 8.5964
w25 = 1 / max(final_mae, 0.01)
wsum = w20 + w25

# Step20 test predictions가 있다면 블렌드, 없으면 TabM만
step20_sub_path = os.path.join(OUTPUT_DIR, 'submission_step20_temporal_stack.csv')
if os.path.exists(step20_sub_path):
    pred20 = pd.read_csv(step20_sub_path)[TARGET].values
    blend  = (w20 * pred20 + w25 * test_mean) / wsum
    sub_blend = sample.copy()
    sub_blend[TARGET] = np.clip(blend, 0, None)
    sub_blend.to_csv(os.path.join(OUTPUT_DIR, 'submission_step25_blend.csv'), index=False)
    print(f"  블렌드 저장: submission_step25_blend.csv")
    print(f"  블렌드 가중치: step20={w20/wsum:.3f} | tabm={w25/wsum:.3f}")
else:
    print("  (step20 제출 파일 없음 → 블렌드 생략)")

print(f"\n제출 파일: submission_step25_tabm_only.csv")
print(f"OOF MAE: {final_mae:.4f}")
