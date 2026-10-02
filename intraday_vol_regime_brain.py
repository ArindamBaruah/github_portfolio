import os
import sys
import re
import math
import time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# 1. CONFIGURATION & DIRECTORY PATHS
# =====================================================================
SPOT_BASE_DIR = r"C:\Users\RISHI\nifty_data\nifty_spot"
OPTIONS_BASE_DIR = r"C:\Users\RISHI\nifty_data\nifty_options"
CACHE_DIR = r"C:\Users\RISHI\nifty_data\model_cache"

# Set to True on first run to train models with the new threshold and cache to disk
FORCE_RETRAIN = False 

LOT_SIZE = 50
RISK_FREE_RATE = 0.07
TARGET_SPIKE_WINDOW = 15      # 15-minute forward predictive window
TARGET_VOL_THRESHOLD = 0.004 # 0.22% expansion (~55 pts on Nifty 25k)
TARGET_OOS_DATE = "09_09_2024" # OOS Date (DD_MM_YYYY)
SEQ_LEN = 15                  # Sequence history for Attention network

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

def log(msg):
    """Guarantees unbuffered immediate console output in Spyder/IPython."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# =====================================================================
# 2. FAST SINGLE-PASS DIRECTORY INDEXER
# =====================================================================
def index_dataset_directory(base_dir):
    """Crawls folder tree in a single pass O(N), indexing files by date 'DD_MM_YYYY'."""
    date_map = {}
    if not os.path.exists(base_dir):
        log(f"WARNING: Directory not found: {base_dir}")
        return date_map
        
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                m = re.search(r"(\d{2}_\d{2}_\d{4})", f)
                if m:
                    date_str = m.group(1)
                    date_map[date_str] = os.path.join(root, f)
    return date_map

def match_spot_and_options(spot_base, opt_base):
    log("Scanning directory trees for Spot and Options pairings...")
    t0 = time.time()
    spot_index = index_dataset_directory(spot_base)
    opt_index = index_dataset_directory(opt_base)
    
    common_dates = sorted(
        list(set(spot_index.keys()) & set(opt_index.keys())),
        key=lambda d: datetime.strptime(d, "%d_%m_%Y")
    )
    
    matched = [{'date_str': d, 'spot_path': spot_index[d], 'opt_path': opt_index[d]} for d in common_dates]
    log(f"Found {len(matched)} matched sessions across folders in {time.time()-t0:.2f}s.")
    return matched

# =====================================================================
# 3. VECTORIZED MICROSTRUCTURE & GREEKS ENGINE
# =====================================================================
def parse_option_symbol(symbol):
    """Parses strike and option type reliably across NSE option symbols."""
    m = re.match(r"^NIFTY(.*?)([0-9]{4,5})(CE|PE)$", str(symbol).strip())
    if m:
        return m.group(1), float(m.group(2)), m.group(3)
    return None, None, None

def extract_session_features(spot_df, opt_df):
    """
    Vectorized computation of GEX, VEX, Garman-Klass Vol, and PCR metrics.
    Strictly causal (09:15 -> 15:30 IST) with zero forward lookahead.
    """
    spot_df.columns = [c.strip().lower() for c in spot_df.columns]
    opt_df.columns = [c.strip().lower() for c in opt_df.columns]
    
    spot_df['time'] = spot_df['time'].astype(str).str.strip()
    spot_df = spot_df[(spot_df['time'] >= '09:15:00') & (spot_df['time'] <= '15:30:00')].sort_values('time').reset_index(drop=True)
    if spot_df.empty:
        return None
        
    opt_df['time'] = opt_df['time'].astype(str).str.strip()
    opt_df = opt_df[(opt_df['time'] >= '09:15:00') & (opt_df['time'] <= '15:30:00')].copy()
    
    parsed = [parse_option_symbol(s) for s in opt_df['symbol'].values]
    opt_df['strike'] = [p[1] for p in parsed]
    opt_df['opt_type'] = [p[2] for p in parsed]
    opt_df = opt_df.dropna(subset=['strike', 'opt_type']).copy()
    
    opt_grouped = dict(tuple(opt_df.groupby('time')))
    
    results = []
    prev_metrics = None
    gk_vars = []
    
    for _, s_row in spot_df.iterrows():
        t = s_row['time']
        S = float(s_row['close'])
        H = float(s_row['high'])
        L = float(s_row['low'])
        O = float(s_row['open'])
        
        # Garman-Klass Realized Volatility
        log_hl = np.log(max(H / max(L, 1e-4), 1.0))
        log_co = np.log(max(S / max(O, 1e-4), 1.0))
        gk_var = 0.5 * (log_hl**2) - (2.0 * np.log(2.0) - 1.0) * (log_co**2)
        gk_vars.append(max(0.0, gk_var))
        rv_gk = np.sqrt(np.mean(gk_vars[-5:])) * np.sqrt(252 * 375) if len(gk_vars) >= 5 else 0.15
        
        if t in opt_grouped:
            opts = opt_grouped[t]
            strikes = opts['strike'].values
            closes = opts['close'].values
            types = opts['opt_type'].values
            ois = opts['oi'].values
            vols = opts['volume'].values
            
            # Find ATM Strike
            atm_strike = strikes[np.argmin(np.abs(strikes - S))]
            ce_mask = (strikes == atm_strike) & (types == 'CE')
            pe_mask = (strikes == atm_strike) & (types == 'PE')
            
            ce_p = closes[ce_mask][0] if np.any(ce_mask) else 0.0
            pe_p = closes[pe_mask][0] if np.any(pe_mask) else 0.0
            atm_straddle = ce_p + pe_p
            
            # ATM Implied Volatility
            T_exp = 3.0 / 365.0
            atm_iv = max(0.05, (atm_straddle / (S * np.sqrt(T_exp) + 1e-4)) * 0.8) if atm_straddle > 0 else 0.15
            
            # Vectorized Greeks (Gamma & Vanna)
            sigma = max(atm_iv, 0.05)
            d1 = (np.log(S / strikes) + (RISK_FREE_RATE + 0.5 * sigma**2) * T_exp) / (sigma * np.sqrt(T_exp))
            d2 = d1 - sigma * np.sqrt(T_exp)
            pdf_d1 = norm.pdf(d1)
            
            gamma_arr = pdf_d1 / (S * sigma * np.sqrt(T_exp))
            vanna_arr = -np.exp(-RISK_FREE_RATE * T_exp) * pdf_d1 * (d2 / sigma)
            
            mult = np.where(types == 'CE', 1.0, -1.0)
            total_gex = float(np.sum(mult * gamma_arr * ois * S * LOT_SIZE))
            total_vex = float(np.sum(mult * vanna_arr * ois * S * LOT_SIZE))
            
            call_oi = np.sum(ois[types == 'CE'])
            put_oi = np.sum(ois[types == 'PE'])
            call_vol = np.sum(vols[types == 'CE'])
            put_vol = np.sum(vols[types == 'PE'])
            
            pcr_oi = float(put_oi / max(call_oi, 1.0))
            pcr_vol = float(put_vol / max(call_vol, 1.0))
            
            current_metrics = {
                'atm_straddle': atm_straddle,
                'atm_iv': atm_iv,
                'total_gex': total_gex,
                'total_vex': total_vex,
                'pcr_oi': pcr_oi,
                'pcr_vol': pcr_vol
            }
            prev_metrics = current_metrics
        else:
            current_metrics = prev_metrics if prev_metrics else {
                'atm_straddle': 0.0, 'atm_iv': 0.15, 'total_gex': 0.0,
                'total_vex': 0.0, 'pcr_oi': 1.0, 'pcr_vol': 1.0
            }
            
        row_data = {
            'date': s_row['date'],
            'time': t,
            'spot': S,
            'rv_gk': rv_gk,
            'vol_premium_gap': rv_gk - current_metrics['atm_iv'],
            **current_metrics
        }
        results.append(row_data)
        
    df = pd.DataFrame(results)
    if len(df) < 20:
        return None
        
    # Dynamic 5-minute Momentum Rates
    df['straddle_roc_5m'] = df['atm_straddle'].pct_change(5).fillna(0)
    df['pcr_oi_delta_5m'] = df['pcr_oi'].diff(5).fillna(0)
    df['pcr_vol_delta_5m'] = df['pcr_vol'].diff(5).fillna(0)
    df['gex_delta_5m'] = df['total_gex'].pct_change(5).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Ground Truth Label for Training (Forward 15-minute range expansion)
    fwd_max = df['spot'].iloc[::-1].rolling(TARGET_SPIKE_WINDOW).max().iloc[::-1]
    fwd_min = df['spot'].iloc[::-1].rolling(TARGET_SPIKE_WINDOW).min().iloc[::-1]
    df['label_spike'] = (((fwd_max - fwd_min) / df['spot']) > TARGET_VOL_THRESHOLD).astype(int)
    df.loc[df.index[-TARGET_SPIKE_WINDOW:], 'label_spike'] = np.nan
    return df

# =====================================================================
# 4. NEURAL TEMPORAL ATTENTION NETWORK (RAW LOGITS)
# =====================================================================
class TemporalAttentionBlock(nn.Module):
    def __init__(self, d_model=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ffn(x))

class ExtendedVolatilityPredictor(nn.Module):
    def __init__(self, in_features, d_model=32):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.temporal_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.attn_block = TemporalAttentionBlock(d_model=d_model, num_heads=4)
        # Outputs raw logits for BCEWithLogitsLoss
        self.head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        h = self.input_proj(x)
        h = h.transpose(1, 2)
        h = torch.relu(self.temporal_conv(h)).transpose(1, 2)
        h = self.attn_block(h)
        return self.head(h[:, -1, :]).squeeze(-1)

class TimeSeriesWindowDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# =====================================================================
# 5. MODEL CHECKPOINTING UTILITIES
# =====================================================================
FEATURE_COLS = [
    'atm_straddle', 'atm_iv', 'rv_gk', 'vol_premium_gap',
    'total_gex', 'total_vex', 'pcr_oi', 'pcr_vol',
    'straddle_roc_5m', 'pcr_oi_delta_5m', 'pcr_vol_delta_5m', 'gex_delta_5m'
]

def get_cache_paths():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return {
        'xgb': os.path.join(CACHE_DIR, "xgb_vol_model.json"),
        'attn': os.path.join(CACHE_DIR, "attn_vol_model.pt"),
        'scaler': os.path.join(CACHE_DIR, "scaler_params.npz")
    }

def has_cached_models():
    paths = get_cache_paths()
    return os.path.exists(paths['xgb']) and os.path.exists(paths['attn']) and os.path.exists(paths['scaler'])

# =====================================================================
# 6. MAIN CONTROLLER & INTRADAY SIMULATION
# =====================================================================
def main():
    log("=" * 65)
    log("NIFTY REAL-TIME VOLATILITY REGIME DETECTOR")
    log("=" * 65)
    
    paths = get_cache_paths()
    sessions = match_spot_and_options(SPOT_BASE_DIR, OPTIONS_BASE_DIR)
    
    if not sessions:
        log("No matched spot & option datasets found. Check directory paths.")
        return

    # Match OOS Test Session
    oos_session = next((s for s in sessions if s['date_str'] == TARGET_OOS_DATE), None)
    if oos_session is None:
        oos_session = sessions[-1]
        train_sessions = sessions[:-1]
        log(f"Date {TARGET_OOS_DATE} not found. Defaulting to latest: {oos_session['date_str']}")
    else:
        train_sessions = [s for s in sessions if s['date_str'] != TARGET_OOS_DATE]

    xgb_clf = xgb.XGBClassifier()
    attn_model = ExtendedVolatilityPredictor(in_features=len(FEATURE_COLS)).to(DEVICE)
    
    # -------------------------------------------------------------
    # MODEL TRAINING OR CACHE RETRIEVAL
    # -------------------------------------------------------------
    if has_cached_models() and not FORCE_RETRAIN:
        log("Found pre-trained checkpoints in cache. Loading models...")
        xgb_clf.load_model(paths['xgb'])
        attn_model.load_state_dict(torch.load(paths['attn'], map_location=DEVICE))
        
        scaler_data = np.load(paths['scaler'])
        means = pd.Series(scaler_data['means'], index=FEATURE_COLS)
        stds = pd.Series(scaler_data['stds'], index=FEATURE_COLS)
        log("Cached weights and scalers restored successfully.")
    else:
        log(f"Extracting features across {len(train_sessions)} training session(s)...")
        train_dfs = []
        for i, s in enumerate(train_sessions):
            s_df = pd.read_csv(s['spot_path'])
            o_df = pd.read_csv(s['opt_path'])
            feat_df = extract_session_features(s_df, o_df)
            if feat_df is not None:
                train_dfs.append(feat_df)
            if (i + 1) % 25 == 0 or (i + 1) == len(train_sessions):
                log(f"Ingested {i + 1}/{len(train_sessions)} sessions...")
                
        if not train_dfs:
            log("No training data extracted. Terminating.")
            return

        full_train = pd.concat(train_dfs, ignore_index=True).dropna(subset=['label_spike'])
        pos_cnt = full_train['label_spike'].sum()
        neg_cnt = len(full_train) - pos_cnt
        scale_weight = float(neg_cnt / max(pos_cnt, 1.0))
        
        log(f"Training bars: {len(full_train)} | Spike bars: {int(pos_cnt)} | Scale Weight: {scale_weight:.2f}")

        means = full_train[FEATURE_COLS].mean()
        stds = full_train[FEATURE_COLS].std().replace(0, 1.0)
        full_train[FEATURE_COLS] = (full_train[FEATURE_COLS] - means) / stds

        # Train XGBoost with Imbalance Weighting
        log("Training XGBoost Classifier...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.03,
            scale_pos_weight=scale_weight,
            random_state=42
        )
        xgb_clf.fit(full_train[FEATURE_COLS], full_train['label_spike'])

        # Train Attention Network with Balanced BCEWithLogitsLoss
        log("Training Calibrated Temporal Multi-Head Attention Network...")
        X_list, y_list = [], []
        feats_arr = full_train[FEATURE_COLS].to_numpy(dtype=np.float32)
        labels_arr = full_train['label_spike'].to_numpy(dtype=np.float32)
        
        for i in range(SEQ_LEN, len(full_train)):
            X_list.append(feats_arr[i-SEQ_LEN:i])
            y_list.append(labels_arr[i])
            
        dataset = TimeSeriesWindowDataset(np.array(X_list), np.array(y_list))
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = torch.optim.AdamW(attn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Penalize missed volatility spikes equally in PyTorch
        pos_weight_t = torch.tensor([scale_weight], dtype=torch.float32).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
        
        attn_model.train()
        for epoch in range(3):
            loss_total = 0.0
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                pred = attn_model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                loss_total += loss.item()
            log(f"Epoch {epoch+1}/3 - Loss: {loss_total/len(loader):.4f}")

        # Save to Cache
        log("Caching model weights and scaler to disk...")
        xgb_clf.save_model(paths['xgb'])
        torch.save(attn_model.state_dict(), paths['attn'])
        np.savez(paths['scaler'], means=means.values, stds=stds.values)
        log("Checkpoints successfully saved.")

    # =================================================================
    # 7. OUT-OF-SAMPLE STREAMING SIMULATION (NO LOOKAHEAD)
    # =================================================================
    log("=" * 65)
    log(f"STREAMING REAL-TIME SIMULATION: {oos_session['date_str']}")
    log("=" * 65)

    oos_spot_df = pd.read_csv(oos_session['spot_path'])
    oos_opt_df = pd.read_csv(oos_session['opt_path'])
    oos_df = extract_session_features(oos_spot_df, oos_opt_df)
    
    if oos_df is None:
        log("Failed to process test session.")
        return

    oos_scaled = oos_df.copy()
    oos_scaled[FEATURE_COLS] = (oos_scaled[FEATURE_COLS] - means) / stds

    # Pre-extract float32 NumPy arrays to prevent pandas object downcasting
    feature_matrix = oos_scaled[FEATURE_COLS].to_numpy(dtype=np.float32)
    time_series = oos_scaled['time'].values
    spot_series = oos_df['spot'].values
    iv_series = oos_df['atm_iv'].values
    gex_series = oos_df['total_gex'].values

    print(f"\n{'TIME':<10} | {'SPOT':<9} | {'ATM IV':<7} | {'GEX':<11} | {'XGB':<6} | {'ATTN':<6} | {'ALERT STATUS'}", flush=True)
    print("-" * 85, flush=True)

    stream_window = []
    alert_count = 0
    attn_model.eval()

    for idx in range(len(oos_scaled)):
        t = time_series[idx]
        S = spot_series[idx]
        iv = iv_series[idx]
        gex = gex_series[idx]
        
        feat_vec = feature_matrix[idx].reshape(1, -1)
        stream_window.append(feature_matrix[idx])
        
        # XGBoost inference
        p_xgb = float(xgb_clf.predict_proba(feat_vec)[0, 1])
        
        # Neural Network inference (activates once history >= SEQ_LEN)
        p_attn = 0.0
        if len(stream_window) >= SEQ_LEN:
            seq_np = np.array(stream_window[-SEQ_LEN:], dtype=np.float32)
            seq_t = torch.from_numpy(seq_np).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                p_attn = float(torch.sigmoid(attn_model(seq_t)).cpu().item())

        # Decoupled trigger: fires if either model detects expansion or during negative gamma squeeze
        is_spike = p_xgb >= 0.65 and p_attn >= 0.65 or (p_xgb >= 0.65 and gex < 0) or (p_attn >= 0.65 and gex < 0)
        alert = ">> VOL SPIKE IMMINENT <<" if is_spike else "NORMAL"
        
        if "IMMINENT" in alert:
            alert_count += 1

        # Emit row every 15 minutes or immediately upon trigger
        if idx % 15 == 0 or "IMMINENT" in alert:
            print(f"{t:<10} | {S:<9.2f} | {iv*100:<6.2f}% | {gex:<11.0f} | {p_xgb:<6.3f} | {p_attn:<6.3f} | {alert}", flush=True)

    log("=" * 65)
    log(f"SIMULATION COMPLETE. Spike Alerts Triggered: {alert_count}")
    log("=" * 65)

if __name__ == "__main__":
    main()