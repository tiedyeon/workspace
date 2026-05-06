#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
step22_lstm_v2.py
=================
LSTM v2 : Step20 전체 피처(448개) 사용 — sc_* 포함
- Step21과 달리 sc_*/layout 집계 피처 모두 LSTM 입력에 포함
- InputLayerNorm → LSTM(2layer, hidden=256) → Head
- 200 epoch, patience=25, ReduceLROnPlateau 스케줄러
- 3 seeds × 5 folds GroupKFold CV
- OOF CSV 저장(step23 TabPFN 메타러너용)

실행:
  cd ~/Desktop/데이콘/스마트\ 창고\ 출고\ 지연\ 예측\ AI\ 경진대회
  source ~/dacon_venv/bin/activate
  caffeinate -i nohup python -u -B step22_lstm_v2.py > step22_output.log 2>&1 &
"""
import os, warnings, time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

# ============================================================
# 설정
# ============================================================
DATA_DIR    = '.'
OUTPUT_DIR  = '.'
TARGET      = 'avg_delay_minutes_next_30m'
N_SPLITS    = 5
SEEDS       = [42, 123, 456]
EPOCHS      = 200
BATCH_SIZE  = 64
HIDDEN_DIM  = 256
NUM_LAYERS  = 2
DROPOUT     = 0.3
LR          = 1e-3
PATIENCE    = 25
LR_PATIENCE = 10
LR_FACTOR   = 0.5
BLEND_GBM   = 0.70
N_CPU       = os.cpu_count()
SLOTS       = 25
LAYOUT_TYPE_MAP = {'grid': 0, 'hybrid': 1, 'narrow': 2, 'hub_spoke': 3}

device = (torch.device('mps')  if torch.backends.mps.is_available() else
          torch.device('cuda') if torch.cuda.is_available() else
          torch.device('cpu'))

t0 = time.time()
print("="*62)
print(f"Step22 LSTM v2 | sc_* 포함 전체 피처 | device={device}")
print(f"Seeds={SEEDS} | EPOCHS={EPOCHS} | PATIENCE={PATIENCE}")
print("="*62)

# ============================================================
# 1. 데이터 로드
# ============================================================
print("\n[1] 데이터 로드...")
train  = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test   = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
layout = pd.read_csv(os.path.join(DATA_DIR, 'layout_info.csv'))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

train = train.merge(layout, on='layout_id', how='left')
test  = test.merge(layout, on='layout_id', how='left')
print(f"  train: {train.shape} | test: {test.shape}")

# ============================================================
# 2. 피처 엔지니어링 (step20 완전 동일)
# ============================================================
AGG_COLS  = ['order_inflow_15m', 'low_battery_ratio', 'congestion_score',
             'robot_utilization', 'pack_utilization', 'robot_active',
             'charge_queue_length', 'max_zone_density']
AGG_FUNCS = ['mean', 'std', 'max', 'min']
ZSCORE_COLS = ['congestion_score', 'low_battery_ratio', 'max_zone_density',
               'pack_utilization', 'robot_utilization', 'order_inflow_15m',
               'charge_queue_length', 'robot_active', 'battery_mean',
               'aisle_traffic_score', 'blocked_path_15m']
RANK_COLS  = ['congestion_score', 'low_battery_ratio', 'max_zone_density',
              'pack_utilization', 'order_inflow_15m', 'charge_queue_length',
              'battery_mean', 'robot_utilization', 'blocked_path_15m']
RATIO_COLS = ['congestion_score', 'low_battery_ratio', 'max_zone_density',
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
        if col in df.columns: df[f'nan_{col}'] = df[col].isna().astype(np.int8)
    for col in ZERO_FILL_COLS:
        if col in df.columns: df[col] = df[col].fillna(0)
    for col in ['congestion_score','blocked_path_15m','near_collision_15m',
                'charge_queue_length','avg_charge_wait','fault_count_15m',
                'replenishment_overlap','task_reassign_15m']:
        if col in df.columns: df[f'flag_{col}'] = (df[col] > 0).astype(np.int8)
    df['battery_stress']         = df['low_battery_ratio'] / (df['battery_mean'] + 1e-5)
    df['charge_bottleneck']      = df['charge_queue_length'] * df['avg_charge_wait']
    df['battery_volatility']     = df['battery_std'] / (df['battery_mean'] + 1e-5)
    df['battery_health']         = df['battery_mean'] - df['battery_std']
    df['order_per_robot']        = df['order_inflow_15m'] / (df['robot_active'] + 1)
    df['order_per_pack_station'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1)
    df['robot_effective_util']   = df['robot_active'] / (df['robot_total'] + 1)
    df['idle_ratio']             = df['robot_idle'] / (df['robot_active']+df['robot_idle']+df['robot_charging']+1)
    df['charging_ratio']         = df['robot_charging'] / (df['robot_total'] + 1)
    df['congestion_x_density']   = df['congestion_score'] * df['max_zone_density']
    df['traffic_severity']       = df['blocked_path_15m'] + df['near_collision_15m'] * 2
    df['aisle_load']             = df['aisle_traffic_score'] * df['congestion_score']
    df['layout_type_enc']        = df['layout_type'].map(LAYOUT_TYPE_MAP).fillna(-1).astype(np.int8)
    df['order_per_charger']      = df['order_inflow_15m'] / (df['charger_count'] + 1)
    df['robot_per_floor_area']   = df['robot_total'] / (df['floor_area_sqm'] + 1)
    df['pack_station_per_robot'] = df['pack_station_count'] / (df['robot_total'] + 1)
    for col in ['battery_mean','low_battery_ratio','congestion_score',
                'order_inflow_15m','robot_active','pack_utilization']:
        if col in df.columns: df[f'null_{col}'] = df[col].isna().astype(np.int8)
    for col in SKEWED_COLS:
        if col in df.columns: df[f'log_{col}'] = np.log1p(df[col].fillna(0))
    df['battery_crisis_index'] = (df['low_battery_ratio']*df['charge_queue_length']
                                   + df['robot_charging']/(df['battery_mean']+1e-5))
    df['congestion_level']    = pd.cut(df['congestion_score'].fillna(0),
                                        bins=[-1,0,20,100], labels=[0,1,2]).astype(int)
    df['congestion_compound'] = (df['congestion_score'].fillna(0)*df['max_zone_density'].fillna(0)
                                  + df['blocked_path_15m'].fillna(0)*2
                                  + df['near_collision_15m'].fillna(0)*3)
    df['robot_saturation']    = 1 - (df['robot_idle']/(df['robot_total']+1))
    df['operation_pressure']  = df['order_inflow_15m']*df['low_battery_ratio']/(df['robot_active']+1)
    df['triple_crisis']       = df['low_battery_ratio']*df['congestion_score'].fillna(0)*df['order_inflow_15m']
    df['crisis_score']        = df['low_battery_ratio']*df['congestion_score'].fillna(0)
    df['order_robot_stress']  = df['order_inflow_15m']/(df['robot_active']+1)*df['low_battery_ratio']
    df['bottleneck_score']    = df['charge_queue_length']*df['congestion_score'].fillna(0)
    df['complex_urgent_order']= df['sku_concentration']*df['urgent_order_ratio']
    if 'maintenance_schedule_score' in df.columns:
        df['maintenance_battery_risk'] = (1-df['maintenance_schedule_score'].fillna(0.5))*df['low_battery_ratio']
    df['layout_congestion'] = df['layout_type_enc']*df['congestion_score'].fillna(0)
    df['layout_battery']    = df['layout_type_enc']*df['low_battery_ratio']
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
    for col in ['congestion_score','low_battery_ratio','order_inflow_15m',
                'robot_active','blocked_path_15m','charge_queue_length']:
        if col in df.columns:
            df[f'lag1_{col}'] = df.groupby('scenario_id')[col].shift(1)
    return df


def add_lag_rolling_features(df):
    df = df.copy()
    grp     = df.groupby('scenario_id', sort=False)
    grp_key = df['scenario_id']
    for col in SEQ_COLS:
        if col not in df.columns: continue
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
        if value_col not in df.columns: return
        positive = df[value_col].fillna(0).gt(0)
        t     = df['slot_idx'].where(positive).astype(float)
        first = t.groupby(grp_key).transform(lambda s: s.ffill().cummin())
        prev  = positive.groupby(grp_key).shift(1, fill_value=False)
        df[f'{prefix}_ever_started']      = first.notna().astype(np.int8)
        df[f'{prefix}_start_idx']         = first.fillna(-1).astype(np.float32)
        df[f'{prefix}_started_now']       = (positive & ~prev).astype(np.int8)
        df[f'{prefix}_started_early']     = (first <= 5).fillna(False).astype(np.int8)
        df[f'{prefix}_steps_since_start'] = np.where(
            first.notna(), (df['slot_idx']-first).astype(float), -1.0).astype(np.float32)
    add_onset('robot_charging',      'charging')
    add_onset('charge_queue_length', 'queue')
    return df


def add_threshold_features(df):
    df = df.copy()
    if {'robot_charging','charge_queue_length','charger_count'}.issubset(df.columns):
        df['charge_pressure2'] = ((df['robot_charging']+df['charge_queue_length'])
                                   / (df['charger_count']+1e-5))
    if 'battery_mean'     in df.columns: df['battery_mean_below_44']    = np.clip(44.0-df['battery_mean'].fillna(44),0,None)
    if 'charge_pressure2' in df.columns: df['charge_pressure_above_136']= np.clip(df['charge_pressure2']-1.36,0,None)
    if 'pack_utilization' in df.columns: df['pack_utilization_sq']      = df['pack_utilization'].fillna(0)**2
    if 'loading_dock_util'in df.columns: df['loading_dock_util_sq']     = df['loading_dock_util'].fillna(0)**2
    if 'staging_area_util'in df.columns: df['staging_area_util_sq']     = df['staging_area_util'].fillna(0)**2
    return df


def add_aggregation_features(train_df, test_df):
    train_sc = train_df.groupby('scenario_id')[AGG_COLS].agg(AGG_FUNCS)
    train_sc.columns = [f'sc_{c}_{f}' for c,f in train_sc.columns]
    test_sc  = test_df.groupby('scenario_id')[AGG_COLS].agg(AGG_FUNCS)
    test_sc.columns  = [f'sc_{c}_{f}' for c,f in test_sc.columns]
    train_df = train_df.merge(train_sc.reset_index(), on='scenario_id', how='left')
    test_df  = test_df.merge(test_sc.reset_index(),   on='scenario_id', how='left')
    layout_agg = train_df.groupby('layout_id')[AGG_COLS].agg(AGG_FUNCS)
    layout_agg.columns = [f'layout_{c}_{f}' for c,f in layout_agg.columns]
    layout_agg = layout_agg.reset_index()
    lfc = [c for c in layout_agg.columns if c.startswith('layout_') and c != 'layout_id']
    train_df = train_df.merge(layout_agg[['layout_id']+lfc], on='layout_id', how='left', suffixes=('','_dup'))
    train_df.drop(columns=[c for c in train_df.columns if c.endswith('_dup')], inplace=True)
    test_df  = test_df.merge(layout_agg[['layout_id']+lfc], on='layout_id', how='left')
    return train_df, test_df


def add_sc_derived_features(df):
    df = df.copy()
    for col in ZSCORE_COLS:
        m, s = f'sc_{col}_mean', f'sc_{col}_std'
        if col in df.columns and m in df.columns and s in df.columns:
            df[f'z_{col}']     = (df[col]-df[m])/(df[s]+1e-5)
            df[f'z_{col}_abs'] = df[f'z_{col}'].abs()
    for col in RANK_COLS:
        if col in df.columns:
            df[f'prank_{col}'] = df.groupby('scenario_id')[col].rank(pct=True, na_option='keep')
    for col in RATIO_COLS:
        if col in df.columns:
            if f'sc_{col}_max'  in df.columns:
                df[f'ratio_to_max_{col}']  = df[col]/(df[f'sc_{col}_max']+1e-5)
                df[f'gap_to_max_{col}']    = df[f'sc_{col}_max']-df[col]
            if f'sc_{col}_mean' in df.columns:
                df[f'ratio_to_mean_{col}'] = df[col]/(df[f'sc_{col}_mean']+1e-5)
    return df


def add_trajectory_features(df):
    df = df.copy()
    for col in TRAJECTORY_COLS:
        exp_col = f'exp_mean_{col}'
        sc_col  = f'sc_{col}_mean'
        if exp_col in df.columns and sc_col in df.columns:
            df[f'traj_{col}']     = df[exp_col]/(df[sc_col]+1e-5)
            df[f'traj_dev_{col}'] = (df[exp_col]-df[sc_col]).abs()
    return df


def add_layout_target_encoding(train_df, test_df, y_log):
    gkf_te       = GroupKFold(n_splits=N_SPLITS)
    scenario_ids = train_df['scenario_id'].values
    layout_te_train = np.full(len(train_df), np.nan)
    for tr_idx, val_idx in gkf_te.split(train_df, y_log, scenario_ids):
        tr_sub = train_df.iloc[tr_idx].copy(); tr_sub['_y'] = y_log[tr_idx]
        layout_mean      = tr_sub.groupby('layout_id')['_y'].mean()
        layout_type_mean = tr_sub.groupby('layout_type')['_y'].mean()
        global_mean      = float(y_log[tr_idx].mean())
        val_df   = train_df.iloc[val_idx]
        encoded  = val_df['layout_id'].map(layout_mean).copy()
        missing  = encoded.isna()
        if missing.any(): encoded[missing] = val_df.loc[missing,'layout_type'].map(layout_type_mean)
        encoded  = encoded.fillna(global_mean)
        layout_te_train[val_idx] = encoded.values
    train_df = train_df.copy(); train_df['layout_te'] = layout_te_train
    full_tr  = train_df.copy(); full_tr['_y'] = y_log
    layout_mean_all      = full_tr.groupby('layout_id')['_y'].mean()
    layout_type_mean_all = full_tr.groupby('layout_type')['_y'].mean()
    global_mean_all      = float(y_log.mean())
    test_df      = test_df.copy()
    test_encoded = test_df['layout_id'].map(layout_mean_all).copy()
    missing_test = test_encoded.isna()
    if missing_test.any(): test_encoded[missing_test] = test_df.loc[missing_test,'layout_type'].map(layout_type_mean_all)
    test_encoded = test_encoded.fillna(global_mean_all)
    test_df['layout_te'] = test_encoded.values
    return train_df, test_df


# ── 피처 엔지니어링 실행 ──
print("\n[2] 피처 엔지니어링...")
train = feature_engineering(train); test = feature_engineering(test)
train = add_temporal_features(train); test = add_temporal_features(test)
train = add_lag_rolling_features(train); test = add_lag_rolling_features(test)
train = add_onset_features(train); test = add_onset_features(test)
train = add_threshold_features(train); test = add_threshold_features(test)
train, test = add_aggregation_features(train, test)
train = add_sc_derived_features(train); test = add_sc_derived_features(test)
train = add_trajectory_features(train); test = add_trajectory_features(test)
y_log_full = np.log1p(train[TARGET].values)
train, test = add_layout_target_encoding(train, test, y_log_full)

DROP_COLS    = {'ID','layout_id','scenario_id','layout_type',TARGET,'shift_hour'}
feature_cols = [c for c in train.columns
                if c not in DROP_COLS and c in test.columns and train[c].dtype != object]
feature_cols = list(dict.fromkeys(feature_cols))

X_train = train[feature_cols].values.astype(np.float32)
y_train = y_log_full.astype(np.float32)
X_test  = test[feature_cols].values.astype(np.float32)
groups  = train['scenario_id'].values
y_true  = train[TARGET].values
IDs_tr  = train['ID'].values
n_feats = len(feature_cols)
print(f"  피처 수: {n_feats}개 | train: {X_train.shape} | test: {X_test.shape}")

# ============================================================
# 3. 시퀀스 변환 (n_scenarios, 25, n_feats)
# ============================================================
print("\n[3] 시퀀스 변환...")

def make_sequences(X, sc_ids, y=None):
    unique_sc = pd.unique(sc_ids)
    n_sc      = len(unique_sc)
    sc_to_i   = {sc: i for i, sc in enumerate(unique_sc)}
    seq_X = np.zeros((n_sc, SLOTS, X.shape[1]), dtype=np.float32)
    seq_y = np.zeros((n_sc, SLOTS), dtype=np.float32) if y is not None else None
    # slot_idx 위치 찾기
    if 'slot_idx' in feature_cols:
        si_pos = feature_cols.index('slot_idx')
        for r in range(len(X)):
            i  = sc_to_i[sc_ids[r]]
            sl = int(round(X[r, si_pos]))
            if 0 <= sl < SLOTS:
                seq_X[i, sl] = X[r]
                if seq_y is not None: seq_y[i, sl] = y[r]
    else:
        ctr = {}
        for r in range(len(X)):
            i  = sc_to_i[sc_ids[r]]
            sl = ctr.get(i, 0)
            if sl < SLOTS:
                seq_X[i, sl] = X[r]
                if seq_y is not None: seq_y[i, sl] = y[r]
            ctr[i] = sl + 1
    seq_X = np.nan_to_num(seq_X, nan=0.0)
    return seq_X, seq_y, unique_sc

tr_sc_ids = groups
te_sc_ids = test['scenario_id'].values
X_tr_seq, y_tr_seq, sc_train = make_sequences(X_train, tr_sc_ids, y_train)
X_te_seq, _, sc_test          = make_sequences(X_test,  te_sc_ids)
print(f"  train seq: {X_tr_seq.shape} | test seq: {X_te_seq.shape}")

# ============================================================
# 4. Dataset / Model
# ============================================================
class SeqDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return (self.X[i], self.y[i]) if self.y is not None else self.X[i]


class WarehouseLSTMv2(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        x   = self.input_norm(x)
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1)   # (batch, 25)

# ============================================================
# 5. GroupKFold CV
# ============================================================
print("\n[4] GroupKFold CV 학습...")
gkf        = GroupKFold(n_splits=N_SPLITS)
sc_groups  = np.arange(len(sc_train))

# OOF: per-scenario 예측 합산용
oof_sum    = np.zeros(len(sc_train))
test_sum   = np.zeros((len(sc_test), SLOTS))
seed_count = 0

for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    seed_oof  = np.zeros((len(sc_train), SLOTS))
    seed_test = np.zeros((len(sc_test),  SLOTS))

    for fi, (tr_idx, val_idx) in enumerate(gkf.split(X_tr_seq, y_tr_seq, sc_groups)):
        t_fold = time.time()
        tr_loader  = DataLoader(SeqDataset(X_tr_seq[tr_idx], y_tr_seq[tr_idx]),
                                batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(SeqDataset(X_tr_seq[val_idx], y_tr_seq[val_idx]),
                                batch_size=BATCH_SIZE, shuffle=False)
        te_loader  = DataLoader(SeqDataset(X_te_seq),
                                batch_size=BATCH_SIZE, shuffle=False)

        model     = WarehouseLSTMv2(n_feats, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE)
        criterion = nn.L1Loss()

        best_mae  = np.inf
        best_state= None
        no_imp    = 0

        for ep in range(EPOCHS):
            model.train()
            for Xb, yb in tr_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            vp = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    vp.append(model(Xb.to(device)).cpu().numpy())
            vp    = np.concatenate(vp)           # (n_val, 25)
            v_mae = np.mean(np.abs(np.expm1(vp) - np.expm1(y_tr_seq[val_idx])))
            scheduler.step(v_mae)

            if v_mae < best_mae:
                best_mae   = v_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_imp     = 0
            else:
                no_imp += 1
                if no_imp >= PATIENCE:
                    stop_ep = f"stop@ep{ep+1}"; break
        else:
            stop_ep = f"ep{EPOCHS}(full)"

        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            oof_p = np.concatenate([model(Xb.to(device)).cpu().numpy()
                                    for Xb, _ in val_loader])
            te_p  = np.concatenate([model(Xb.to(device)).cpu().numpy()
                                    for Xb in te_loader])
        seed_oof[val_idx] = oof_p
        seed_test        += te_p

        elapsed = (time.time()-t_fold)/60
        print(f"  seed={seed} fold={fi+1} | best_val_mae={best_mae:.4f} | {stop_ep} | {elapsed:.1f}min")

    seed_test /= N_SPLITS
    # per-slot → per-row OOF MAE
    oof_row  = np.expm1(seed_oof).ravel()   # 250K
    true_row = y_true                        # 250K
    seed_mae = np.mean(np.abs(oof_row - true_row))
    print(f"  Seed {seed} OOF MAE (per-slot): {seed_mae:.4f}")

    oof_sum   += np.expm1(seed_oof).mean(axis=1)  # scenario-level avg
    test_sum  += np.expm1(seed_test)
    seed_count += 1

oof_sc_final  = oof_sum / seed_count     # (n_sc,)
test_sc_final = test_sum / seed_count    # (n_te_sc, 25)

# ============================================================
# 6. OOF MAE (per-slot 기준 — 마지막 seed 기준)
# ============================================================
print(f"\n[5] 최종 결과:")
# 간편 계산: seed_oof는 마지막 seed의 값. 대략적 비교용
print(f"  LSTM v2 OOF (시나리오 평균, {seed_count} seeds):")
y_true_sc = np.array([y_true[tr_sc_ids == sc].mean() for sc in sc_train])
oof_mae   = np.mean(np.abs(oof_sc_final - y_true_sc))
print(f"    시나리오 MAE = {oof_mae:.4f}")
print(f"  비교: Step21 LSTM(sc_* 없음) OOF=9.1052 | Step20 GBM OOF=8.5964")

# ============================================================
# 7. 제출 파일 & OOF 저장
# ============================================================
print("\n[6] 제출 파일 생성...")
# test_sc_final: (n_te_sc, 25) → per-row 예측
# 각 시나리오의 각 슬롯에 예측값 매핑
te_sc_to_pred = {}
for i, sc in enumerate(sc_test):
    te_sc_to_pred[sc] = test_sc_final[i]   # (25,)

# per-row (slot별) 예측
slot_idx_col = test['slot_idx'].values if 'slot_idx' in test.columns else None
te_pred_row  = np.zeros(len(test))
for r in range(len(test)):
    sc = te_sc_ids[r]
    if slot_idx_col is not None:
        sl = int(round(slot_idx_col[r])); sl = max(0, min(sl, SLOTS-1))
    else:
        sl = 0
    te_pred_row[r] = te_sc_to_pred[sc][sl]
te_pred_row = np.clip(te_pred_row, 0, None)

sub_lstm = sample_sub.copy()
sub_lstm['avg_delay_minutes_next_30m'] = te_pred_row
sub_lstm.to_csv(os.path.join(OUTPUT_DIR, 'submission_step22_lstm_v2_only.csv'), index=False)
print("  저장: submission_step22_lstm_v2_only.csv")

step20_csv = os.path.join(DATA_DIR, 'submission_step20_temporal_stack.csv')
if os.path.exists(step20_csv):
    gbm_sub = pd.read_csv(step20_csv)
    blend   = BLEND_GBM*gbm_sub['avg_delay_minutes_next_30m'].values + (1-BLEND_GBM)*te_pred_row
    sub_blend = sample_sub.copy()
    sub_blend['avg_delay_minutes_next_30m'] = np.clip(blend, 0, None)
    sub_blend.to_csv(os.path.join(OUTPUT_DIR, 'submission_step22_blend.csv'), index=False)
    print("  저장: submission_step22_blend.csv (GBM70+LSTMv2 30%)")

# OOF 저장 (step23용)
tr_sc_to_oof = dict(zip(sc_train, oof_sc_final))
oof_row_arr  = np.array([tr_sc_to_oof[sc] for sc in tr_sc_ids])
oof_df = pd.DataFrame({'ID': IDs_tr, 'lstm_v2_oof': oof_row_arr})
oof_df.to_csv(os.path.join(OUTPUT_DIR, 'oof_step22_lstm_v2.csv'), index=False)
print("  저장: oof_step22_lstm_v2.csv  (step23 TabPFN 메타러너용)")

# ============================================================
# 요약
# ============================================================
total_min = (time.time()-t0)/60
print("\n" + "="*62)
print(f"완료! 총 소요: {total_min:.1f}분")
print(f"LSTM v2 시나리오 OOF MAE : {oof_mae:.4f}")
print(f"비교 — Step21 LSTM       : 9.1052  (sc_* 없음)")
print(f"비교 — Step20 GBM        : 8.5964")
print(f"예상 Public (LSTM 단독)  : ~{oof_mae+1.464:.4f}")
print(f"예측 분포: mean={te_pred_row.mean():.3f} std={te_pred_row.std():.3f} "
      f"min={te_pred_row.min():.3f} max={te_pred_row.max():.3f}")
print("="*62)
