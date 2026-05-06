# =============================================================================
# 스마트 창고 출고 지연 예측 AI 경진대회
# Step 3: PyTorch MLP — Apple Silicon GPU (MPS) 활용
# =============================================================================
# Mac Apple Silicon에서 실제로 GPU를 쓰는 유일한 구간.
# 트리 모델(LGB/XGB/CAT)이 잡지 못하는 비선형 패턴을 MLP가 보완.
# 마지막에 Step2 앙상블 예측과 결합해 최종 제출 파일까지 생성.
# =============================================================================
# 필요 패키지:
#   pip install torch scikit-learn numpy pandas
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)

# =============================================================================
# 디바이스 설정 — Mac Apple Silicon GPU (MPS) 우선
# =============================================================================
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("✅ Apple Silicon GPU (MPS) 사용")
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("✅ NVIDIA GPU (CUDA) 사용")
else:
    DEVICE = torch.device('cpu')
    print("⚠️  CPU 모드")

N_CPU      = os.cpu_count()
DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = DATA_DIR
SEED       = 42
TARGET     = 'avg_delay_minutes_next_30m'

torch.manual_seed(SEED)
np.random.seed(SEED)

# =============================================================================
# 1. 데이터 로드 & 피처 엔지니어링 (Step1/2와 동일)
# =============================================================================
print("\n" + "="*60)
print("1. 데이터 로딩 & 피처 엔지니어링")
print("="*60)

train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

train = train.merge(layout, on='layout_id', how='left')
test  = test.merge(layout,  on='layout_id', how='left')

LAYOUT_TYPE_MAP = {'grid': 0, 'hybrid': 1, 'narrow': 2, 'hub_spoke': 3}


def feature_engineering(df):
    df = df.copy()
    for col in ['congestion_score', 'blocked_path_15m', 'near_collision_15m',
                'charge_queue_length', 'avg_charge_wait', 'fault_count_15m',
                'replenishment_overlap', 'task_reassign_15m']:
        if col in df.columns:
            df[f'flag_{col}'] = (df[col] > 0).astype(np.float32)

    df['battery_stress']     = df['low_battery_ratio'] / (df['battery_mean'] + 1e-5)
    df['charge_bottleneck']  = df['charge_queue_length'] * df['avg_charge_wait']
    df['battery_volatility'] = df['battery_std'] / (df['battery_mean'] + 1e-5)
    df['battery_health']     = df['battery_mean'] - df['battery_std']
    df['order_per_robot']        = df['order_inflow_15m'] / (df['robot_active'] + 1)
    df['order_per_pack_station'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1)
    df['robot_effective_util']   = df['robot_active'] / (df['robot_total'] + 1)
    df['idle_ratio']  = df['robot_idle'] / (df['robot_active'] + df['robot_idle'] + df['robot_charging'] + 1)
    df['charging_ratio']       = df['robot_charging'] / (df['robot_total'] + 1)
    df['congestion_x_density'] = df['congestion_score'] * df['max_zone_density']
    df['traffic_severity']     = df['blocked_path_15m'] + df['near_collision_15m'] * 2
    df['aisle_load']           = df['aisle_traffic_score'] * df['congestion_score']
    df['layout_type_enc']        = df['layout_type'].map(LAYOUT_TYPE_MAP).fillna(-1).astype(np.float32)
    df['order_per_charger']      = df['order_inflow_15m'] / (df['charger_count'] + 1)
    df['robot_per_floor_area']   = df['robot_total'] / (df['floor_area_sqm'] + 1)
    df['pack_station_per_robot'] = df['pack_station_count'] / (df['robot_total'] + 1)
    for col in ['battery_mean', 'low_battery_ratio', 'congestion_score',
                'order_inflow_15m', 'robot_active', 'pack_utilization']:
        if col in df.columns:
            df[f'null_{col}'] = df[col].isna().astype(np.float32)
    return df


train = feature_engineering(train)
test  = feature_engineering(test)

AGG_COLS  = ['order_inflow_15m', 'low_battery_ratio', 'congestion_score',
             'robot_utilization', 'pack_utilization', 'robot_active',
             'charge_queue_length', 'max_zone_density']
AGG_FUNCS = ['mean', 'std', 'max', 'min']


def make_group_agg(df, group_col, prefix):
    agg = df.groupby(group_col)[AGG_COLS].agg(AGG_FUNCS)
    agg.columns = [f'{prefix}_{c}_{f}' for c, f in agg.columns]
    return agg.reset_index()


train_sc_agg     = make_group_agg(train, 'scenario_id', 'sc')
test_sc_agg      = make_group_agg(test,  'scenario_id', 'sc')
train_layout_agg = make_group_agg(train, 'layout_id',   'layout')
layout_feat_cols = [c for c in train_layout_agg.columns
                    if c.startswith('layout_') and c != 'layout_id']

train = train.merge(train_sc_agg, on='scenario_id', how='left')
test  = test.merge(test_sc_agg,   on='scenario_id', how='left')
train = train.merge(train_layout_agg[['layout_id'] + layout_feat_cols],
                    on='layout_id', how='left', suffixes=('', '_dup'))
test  = test.merge(train_layout_agg[['layout_id'] + layout_feat_cols],
                   on='layout_id', how='left')
train.drop(columns=[c for c in train.columns if c.endswith('_dup')], inplace=True)

DROP_COLS    = {'ID', 'layout_id', 'scenario_id', 'layout_type', TARGET}
feature_cols = [c for c in train.columns if c not in DROP_COLS and c in test.columns]

print(f"피처 수: {len(feature_cols)}개")

# =============================================================================
# 2. 전처리 — MLP는 결측치 대체 + 정규화 필수
# =============================================================================
print("\n" + "="*60)
print("2. 전처리 (결측치 대체 + 정규화)")
print("="*60)

X_raw   = train[feature_cols].values.astype(np.float32)
X_te_raw= test[feature_cols].values.astype(np.float32)
y_raw   = np.log1p(train[TARGET].values).astype(np.float32)
groups  = train['scenario_id'].values

# 결측치: 중앙값 대체 (트리와 달리 MLP는 NaN 처리 필수)
imputer = SimpleImputer(strategy='median')
X_raw   = imputer.fit_transform(X_raw)
X_te_raw= imputer.transform(X_te_raw)

# 정규화: StandardScaler (트리는 불필요하지만 MLP는 필수)
scaler  = StandardScaler()
X_raw   = scaler.fit_transform(X_raw).astype(np.float32)
X_te_raw= scaler.transform(X_te_raw).astype(np.float32)

print(f"X_train: {X_raw.shape} | X_test: {X_te_raw.shape}")
print(f"결측치 처리 완료 (중앙값 대체) | 정규화 완료 (StandardScaler)")

# =============================================================================
# 3. MLP 모델 아키텍처
# =============================================================================

class ResidualBlock(nn.Module):
    """잔차 연결(Residual Connection) 블록 — 깊은 MLP에서 학습 안정성 향상"""
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))   # 잔차 연결


class WarehouseMLP(nn.Module):
    """
    창고 지연 예측용 MLP
    구조: Input → Embedding → Residual Blocks × N → Output
    """
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()

        # 입력 임베딩 레이어
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 잔차 블록들
        res_blocks = []
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] != hidden_dims[i + 1]:
                # 차원이 바뀌는 경우 프로젝션 레이어
                res_blocks.append(nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ))
            else:
                res_blocks.append(ResidualBlock(hidden_dims[i], dropout))
        self.res_blocks = nn.ModuleList(res_blocks)

        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.GELU(),
            nn.Linear(hidden_dims[-1] // 2, 1),
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x).squeeze(-1)


# =============================================================================
# 4. 학습 설정
# =============================================================================
INPUT_DIM   = X_raw.shape[1]
HIDDEN_DIMS = [512, 512, 256, 256, 128]   # 잔차 블록 차원 배열
DROPOUT     = 0.3
BATCH_SIZE  = 2048
MAX_EPOCHS  = 200
PATIENCE    = 20                            # Early stopping 기준 epoch
LR          = 1e-3
WEIGHT_DECAY= 1e-4
N_SPLITS    = 5

# =============================================================================
# 5. GroupKFold 학습
# =============================================================================
print("\n" + "="*60)
print(f"3. PyTorch MLP GroupKFold 학습 (device={DEVICE})")
print("="*60)

gkf        = GroupKFold(n_splits=N_SPLITS)
oof_preds  = np.zeros(len(X_raw))
test_preds = np.zeros(len(X_te_raw))
fold_scores= []

X_te_tensor = torch.tensor(X_te_raw, dtype=torch.float32).to(DEVICE)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_raw, y_raw, groups)):
    print(f"\n  [Fold {fold+1}/{N_SPLITS}]")

    X_tr  = torch.tensor(X_raw[tr_idx],  dtype=torch.float32)
    y_tr  = torch.tensor(y_raw[tr_idx],  dtype=torch.float32)
    X_val = torch.tensor(X_raw[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y_raw[val_idx], dtype=torch.float32)

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds   = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 4, shuffle=False,
                              num_workers=0, pin_memory=False)

    model     = WarehouseMLP(INPUT_DIM, HIDDEN_DIMS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=LR * 0.01
    )
    criterion = nn.L1Loss()   # MAE loss (log1p 스케일)

    best_val_loss = float('inf')
    best_state    = None
    no_improve    = 0

    for epoch in range(MAX_EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_preds_log = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(DEVICE)
                val_preds_log.append(model(X_batch).cpu().numpy())
        val_preds_log = np.concatenate(val_preds_log)
        val_loss      = np.mean(np.abs(val_preds_log - y_raw[val_idx]))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"    Epoch {epoch+1:3d} | val_loss(log): {val_loss:.4f} "
                  f"| best: {best_val_loss:.4f} | lr: {lr_now:.2e} | patience: {no_improve}/{PATIENCE}")

        if no_improve >= PATIENCE:
            print(f"    → Early stopping at epoch {epoch+1}")
            break

    # ── Best 모델로 예측 ───────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        # Validation OOF
        chunks = []
        for X_batch, _ in val_loader:
            chunks.append(model(X_batch.to(DEVICE)).cpu().numpy())
        val_pred_log = np.concatenate(chunks)
        val_pred     = np.clip(np.expm1(val_pred_log), 0, None)

        # Test
        te_chunks = []
        for i in range(0, len(X_te_tensor), BATCH_SIZE * 4):
            te_chunks.append(model(X_te_tensor[i:i + BATCH_SIZE * 4]).cpu().numpy())
        te_pred_log = np.concatenate(te_chunks)
        te_pred     = np.clip(np.expm1(te_pred_log), 0, None)

    true_val = train[TARGET].values[val_idx]
    score    = mean_absolute_error(true_val, val_pred)
    fold_scores.append(score)
    print(f"  → Fold {fold+1} MAE: {score:.4f}")

    oof_preds[val_idx] = val_pred
    test_preds        += te_pred / N_SPLITS

    # 메모리 해제
    del model, optimizer, scheduler
    del X_tr, y_tr, X_val, y_val, train_ds, val_ds, train_loader, val_loader
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()

oof_score = mean_absolute_error(train[TARGET].values, oof_preds)

print("\n" + "="*60)
print(f"✅ MLP OOF MAE : {oof_score:.4f}")
print(f"   Fold별      : {[round(s, 4) for s in fold_scores]}")
print("="*60)

# =============================================================================
# 6. Step2 앙상블과 결합 — 최종 앙상블
# =============================================================================
print("\n" + "="*60)
print("4. Step2 앙상블 + MLP 최종 결합")
print("="*60)

step2_path = os.path.join(DATA_DIR, 'submission_step2_ensemble.csv')

if os.path.exists(step2_path):
    step2_sub  = pd.read_csv(step2_path)
    step2_pred = step2_sub[TARGET].values

    # OOF 기반 역가중 결합
    # Step2 OOF MAE는 8.7020 (하드코딩 — 필요 시 step2 결과로 업데이트)
    step2_oof_mae = 8.7020
    mlp_oof_mae   = oof_score

    w_step2 = (1.0 / step2_oof_mae)
    w_mlp   = (1.0 / mlp_oof_mae)
    total_w = w_step2 + w_mlp

    w_step2 /= total_w
    w_mlp   /= total_w

    print(f"Step2 앙상블 가중치: {w_step2:.3f}  (OOF MAE={step2_oof_mae:.4f})")
    print(f"MLP         가중치: {w_mlp:.3f}  (OOF MAE={mlp_oof_mae:.4f})")

    final_pred = w_step2 * step2_pred + w_mlp * test_preds
    final_pred = np.clip(final_pred, 0, None)

    submission = sample.copy()
    submission[TARGET] = final_pred
    out_path = os.path.join(OUTPUT_DIR, 'submission_step3_final.csv')
    submission.to_csv(out_path, index=False)
    print(f"\n최종 앙상블 저장 → {out_path}")
    print(f"\n예측값 분포:")
    print(submission[TARGET].describe().round(3).to_string())
else:
    print("⚠️  step2 파일 없음 — MLP 단독 제출 파일 생성")

# MLP 단독 제출 파일도 저장
sub_mlp = sample.copy()
sub_mlp[TARGET] = np.clip(test_preds, 0, None)
mlp_path = os.path.join(OUTPUT_DIR, 'submission_step3_mlp_only.csv')
sub_mlp.to_csv(mlp_path, index=False)
print(f"\nMLP 단독 저장 → {mlp_path}")

print(f"\n✅ MLP OOF MAE      : {oof_score:.4f}")
if os.path.exists(step2_path):
    print(f"✅ Step2 OOF MAE    : {step2_oof_mae:.4f}")
    print(f"✅ 최종 앙상블 파일 : submission_step3_final.csv")
