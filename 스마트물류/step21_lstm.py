"""
Step21: LSTM Sequence Model + Step20 GBM 블렌딩
- 각 시나리오를 25 time step 시퀀스로 처리 (LSTM)
- sc_* 제외한 인과적 피처만 사용 → GBM과 상호보완
- Step20 제출 파일(GBM)과 블렌딩하여 최종 예측

실행:
    caffeinate -i nohup python -u -B step21_lstm.py > step21_output.log 2>&1 &
    tail -f step21_output.log
"""

import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings('ignore')

# ── 설정 ─────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
TARGET      = 'avg_delay_minutes_next_30m'
N_SPLITS    = 5
SEEDS       = [42, 123, 456]
EPOCHS      = 100
BATCH_SIZE  = 64
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.3
LR          = 1e-3
PATIENCE    = 15
SEQ_LEN     = 25
BLEND_ALPHA = 0.30  # LSTM 비중 (GBM 0.70 + LSTM 0.30)

LAYOUT_TYPE_MAP = {'grid': 0, 'hybrid': 1, 'narrow': 2, 'hub_spoke': 3}

ZERO_FILL_COLS = [
    'charge_queue_length', 'avg_charge_wait', 'blocked_path_15m',
    'near_collision_15m', 'fault_count_15m', 'task_reassign_15m',
    'replenishment_overlap', 'congestion_score', 'max_zone_density',
    'avg_recovery_time', 'loading_dock_util', 'staging_area_util',
    'label_print_queue',
]

# LSTM 입력 피처 (인과적, sc_* 제외)
LSTM_FEATS_CAND = [
    'order_inflow_15m', 'unique_sku_15m',
    'robot_active', 'robot_idle', 'robot_charging',
    'battery_mean', 'battery_std', 'low_battery_ratio',
    'charge_queue_length', 'avg_charge_wait',
    'congestion_score', 'max_zone_density', 'blocked_path_15m',
    'near_collision_15m', 'fault_count_15m', 'avg_recovery_time',
    'task_reassign_15m', 'replenishment_overlap', 'pack_utilization',
    'loading_dock_util', 'staging_area_util', 'label_print_queue',
    'slot_idx', 'slot_progress',
    'floor_area_sqm', 'aisle_width_avg', 'intersection_count',
    'pack_station_count', 'charger_count', 'robot_total',
    'layout_compactness', 'zone_dispersion',
    'layout_type_enc',
]

# ── Device 설정 ───────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

print("=" * 60)
print("Step21: LSTM Sequence Model")
print("=" * 60)
print(f"Device: {DEVICE} | Seeds: {SEEDS} | Epochs: {EPOCHS} | Blend: GBM {1-BLEND_ALPHA:.0%} + LSTM {BLEND_ALPHA:.0%}")

# ── 1. 데이터 로드 ─────────────────────────────────────────────────────────────
print("\n[1] 데이터 로드...")
train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
sample = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
step20_sub = pd.read_csv(os.path.join(DATA_DIR, 'submission_step20_temporal_stack.csv'))

train = train.merge(layout, on='layout_id', how='left')
test  = test.merge(layout,  on='layout_id', how='left')
print(f"  train: {train.shape} | test: {test.shape}")

# ── 2. 전처리 ─────────────────────────────────────────────────────────────────
print("\n[2] 전처리...")
for col in ZERO_FILL_COLS:
    for df in [train, test]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

for df in [train, test]:
    df['layout_type_enc'] = df['layout_type'].map(LAYOUT_TYPE_MAP).fillna(-1).astype(np.float32)
    df['slot_idx']        = df.groupby('scenario_id').cumcount().astype(np.float32)
    df['slot_progress']   = (df['slot_idx'] / 24.0).astype(np.float32)

# 실제 존재하는 피처만 선택
lstm_feats = [f for f in LSTM_FEATS_CAND if f in train.columns]
print(f"  LSTM 입력 피처: {len(lstm_feats)}개")

# 나머지 NaN → 0
for col in lstm_feats:
    train[col] = train[col].fillna(0).astype(np.float32)
    test[col]  = test[col].fillna(0).astype(np.float32)

# 전역 정규화 (train 통계로)
feat_mean = train[lstm_feats].mean().values.astype(np.float32)
feat_std  = train[lstm_feats].std().values.astype(np.float32)
feat_std[feat_std < 1e-8] = 1.0

# ── 3. 시퀀스 변환 ─────────────────────────────────────────────────────────────
print("\n[3] 시퀀스 변환 (n_scenarios, 25, n_feats)...")

def make_sequences(df, feats, mean_v, std_v, target=None):
    """DataFrame → (n_scenarios, SEQ_LEN, n_feats), optional (n_scenarios, SEQ_LEN)"""
    scenarios = sorted(df['scenario_id'].unique())
    n = len(scenarios)
    X = np.zeros((n, SEQ_LEN, len(feats)), dtype=np.float32)
    y = np.zeros((n, SEQ_LEN), dtype=np.float32) if target else None
    scen2idx = {}

    for i, scen_id in enumerate(scenarios):
        grp = df[df['scenario_id'] == scen_id].sort_values('slot_idx')
        vals = grp[feats].values[:SEQ_LEN]  # (25, n_feats)
        X[i, :len(vals)] = (vals - mean_v) / std_v
        if target:
            t = grp[target].values[:SEQ_LEN]
            y[i, :len(t)] = np.log1p(np.clip(t, 0, None))
        scen2idx[scen_id] = i

    return X, y, scenarios, scen2idx

X_train_seq, y_train_seq, train_scenarios, train_scen2idx = \
    make_sequences(train, lstm_feats, feat_mean, feat_std, target=TARGET)
X_test_seq,  _,          test_scenarios,  test_scen2idx  = \
    make_sequences(test,  lstm_feats, feat_mean, feat_std)

print(f"  train: {X_train_seq.shape} | test: {X_test_seq.shape}")

y_true_row = train[TARGET].values  # 행 단위 실제값 (MAE 계산용)

# 행 → 시나리오 인덱스 매핑
train['__scen_idx__'] = train['scenario_id'].map(train_scen2idx)
train['__slot_idx__'] = train.groupby('scenario_id').cumcount()

# ── 4. LSTM 모델 정의 ─────────────────────────────────────────────────────────
class WarehouseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)       # (batch, seq, hidden)
        pred   = self.head(out)     # (batch, seq, 1)
        return pred.squeeze(-1)     # (batch, seq)


# ── 5. CV 학습 ────────────────────────────────────────────────────────────────
print("\n[4] GroupKFold CV 학습...")

n_train_row = len(train)
oof_pred_sum  = np.zeros(n_train_row, dtype=np.float64)
oof_pred_cnt  = np.zeros(n_train_row, dtype=np.float64)
test_pred_all = []

total_start = time.time()

for seed_i, seed in enumerate(SEEDS):
    torch.manual_seed(seed)
    np.random.seed(seed)

    gkf = GroupKFold(n_splits=N_SPLITS)
    groups_row = train['scenario_id'].values

    for fold, (tr_row, val_row) in enumerate(gkf.split(train, y_true_row, groups_row)):
        t_fold = time.time()

        # 해당 fold의 시나리오 인덱스
        tr_scen = np.unique(train.iloc[tr_row]['scenario_id'].values)
        val_scen = np.unique(train.iloc[val_row]['scenario_id'].values)
        tr_idx   = np.array([train_scen2idx[s] for s in tr_scen])
        val_idx  = np.array([train_scen2idx[s] for s in val_scen])

        X_tr = torch.FloatTensor(X_train_seq[tr_idx])
        y_tr = torch.FloatTensor(y_train_seq[tr_idx])
        X_val_t = torch.FloatTensor(X_train_seq[val_idx]).to(DEVICE)
        y_val_np = y_train_seq[val_idx]  # numpy (n_val_scen, 25)

        loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        )

        model = WarehouseLSTM(len(lstm_feats), HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
        opt   = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sch   = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.05)
        crit  = nn.L1Loss()

        best_val_mae = float('inf')
        best_state   = None
        patience_cnt = 0

        for epoch in range(EPOCHS):
            model.train()
            for Xb, yb in loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()

            model.eval()
            with torch.no_grad():
                val_log = model(X_val_t).cpu().numpy()
            val_orig = np.clip(np.expm1(val_log), 0, None)
            y_val_orig = np.clip(np.expm1(y_val_np), 0, None)
            val_mae = mean_absolute_error(y_val_orig.ravel(), val_orig.ravel())

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    break

        # best model로 OOF + test 예측
        model.load_state_dict(best_state)
        model.to(DEVICE)
        model.eval()

        with torch.no_grad():
            val_pred_log = model(X_val_t).cpu().numpy()
            X_test_t = torch.FloatTensor(X_test_seq).to(DEVICE)
            test_pred_log = model(X_test_t).cpu().numpy()

        val_pred_orig  = np.clip(np.expm1(val_pred_log),  0, None)  # (n_val_scen, 25)
        test_pred_orig = np.clip(np.expm1(test_pred_log), 0, None)  # (n_test_scen, 25)
        test_pred_all.append(test_pred_orig)

        # OOF 행 단위 매핑
        for j, scen_id in enumerate(val_scen):
            grp_mask = train['scenario_id'] == scen_id
            slot_sorted_idx = train[grp_mask].sort_values('__slot_idx__').index
            pred_slots = val_pred_orig[j][:len(slot_sorted_idx)]
            oof_pred_sum[slot_sorted_idx] += pred_slots
            oof_pred_cnt[slot_sorted_idx] += 1

        elapsed = time.time() - t_fold
        print(f"  seed={seed} fold={fold+1} | best_val_mae={best_val_mae:.4f} | stop@ep{EPOCHS - patience_cnt + PATIENCE if patience_cnt >= PATIENCE else epoch+1} | {elapsed/60:.1f}min")

    # Seed 단위 OOF MAE
    valid_mask = oof_pred_cnt > 0
    oof_so_far = np.where(valid_mask, oof_pred_sum / np.maximum(oof_pred_cnt, 1), 0)
    print(f"  Seed {seed} 누적 OOF MAE (유효행): {mean_absolute_error(y_true_row[valid_mask], oof_so_far[valid_mask]):.4f}")

# ── 6. 최종 OOF MAE ──────────────────────────────────────────────────────────
oof_final = oof_pred_sum / np.maximum(oof_pred_cnt, 1)
lstm_oof_mae = mean_absolute_error(y_true_row, oof_final)
print(f"\n[5] LSTM OOF MAE: {lstm_oof_mae:.4f}")
print(f"    (Step20 GBM OOF: 8.5964 | Step14 GBM OOF: 8.6037)")

# ── 7. Test 예측 ─────────────────────────────────────────────────────────────
print("\n[6] Test 예측 생성...")
test_pred_mean = np.mean(test_pred_all, axis=0)  # (n_test_scen, 25)

lstm_test_flat = np.zeros(len(test), dtype=np.float32)
for i, scen_id in enumerate(test_scenarios):
    grp_mask = test['scenario_id'] == scen_id
    slot_sorted_idx = test[grp_mask].sort_values('slot_idx').index
    preds = test_pred_mean[i][:len(slot_sorted_idx)]
    lstm_test_flat[slot_sorted_idx] = preds

# ── 8. 제출 파일 생성 ─────────────────────────────────────────────────────────
print("\n[7] 제출 파일 생성...")

# LSTM 단독
sub_lstm = sample.copy()
sub_lstm[TARGET] = np.clip(lstm_test_flat, 0, None)
sub_lstm.to_csv(os.path.join(DATA_DIR, 'submission_step21_lstm_only.csv'), index=False)

# GBM + LSTM 블렌딩
gbm_pred = step20_sub[TARGET].values
blended  = (1 - BLEND_ALPHA) * gbm_pred + BLEND_ALPHA * lstm_test_flat
sub_blend = sample.copy()
sub_blend[TARGET] = np.clip(blended, 0, None)
sub_blend.to_csv(os.path.join(DATA_DIR, 'submission_step21_blend.csv'), index=False)

total_elapsed = time.time() - total_start
print(f"\n{'='*60}")
print(f"완료! 총 소요시간: {total_elapsed/60:.1f}분")
print(f"LSTM OOF MAE:    {lstm_oof_mae:.4f}")
print(f"Step20 GBM OOF:  8.5964")
print(f"예상 Public (LSTM 단독): ~{lstm_oof_mae + 1.469:.4f}")
print(f"예상 Public (Blend):     ~{(lstm_oof_mae*BLEND_ALPHA + 8.5964*(1-BLEND_ALPHA)) + 1.469:.4f}")
print(f"\n제출 파일:")
print(f"  LSTM 단독: submission_step21_lstm_only.csv")
print(f"  GBM+LSTM:  submission_step21_blend.csv")
print(f"\n예측값 분포 (Blend):")
print(sub_blend[TARGET].describe().round(3).to_string())
print(f"{'='*60}")
