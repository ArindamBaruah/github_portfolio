import argparse
import warnings
 
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
import sys
 
warnings.filterwarnings('ignore', category=FutureWarning)
 
# =============================================================================
# 1. garch_module.py
# =============================================================================
def fit_garch11(returns_pct):
    """Fit GARCH(1,1) via MLE on a 1D array of percent returns. Returns (omega, alpha, beta)."""
    r = np.asarray(returns_pct)
 
    def negloglik(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        n = len(r)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(r)
        for t in range(1, n):
            sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + r ** 2 / sigma2)
        return -ll
 
    res = minimize(negloglik, [0.05, 0.08, 0.90], method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    return res.x  # omega, alpha, beta
 
 
def garch_sigma2_series(returns_pct, omega, alpha, beta):
    """Recursive in-sample conditional variance sigma2_t, using only returns up to t-1 (no leakage)."""
    r = np.asarray(returns_pct)
    n = len(r)
    sigma2 = np.zeros(n)
    sigma2[0] = np.var(r)
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2
 
 
def garch_one_step_forecast(last_return_pct, last_sigma2, omega, alpha, beta):
    return omega + alpha * last_return_pct ** 2 + beta * last_sigma2
 
 
# =============================================================================
# 2. gbm_factory.py
# =============================================================================
try:
    import xgboost as xgb
    BACKEND = 'xgboost'
except ImportError:
    try:
        import lightgbm as lgb
        BACKEND = 'lightgbm'
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        BACKEND = 'sklearn_hgb'
 
_WARNED = False
 
 
def backend_name():
    global _WARNED
    if BACKEND == 'sklearn_hgb' and not _WARNED:
        warnings.warn(
            "xgboost/lightgbm not installed (no internet access in this sandbox) -- "
            "falling back to sklearn HistGradientBoosting, LightGBM's closest algorithmic "
            "cousin in scikit-learn. Install xgboost/lightgbm and this module will use them "
            "automatically, no code changes needed.", RuntimeWarning)
        _WARNED = True
    return BACKEND
 
 
PARAM_GRID = [
    {'max_depth': 3, 'lr': 0.05, 'n_estimators': 150},
    {'max_depth': 4, 'lr': 0.05, 'n_estimators': 150},
    {'max_depth': 4, 'lr': 0.03, 'n_estimators': 250},
    {'max_depth': 6, 'lr': 0.05, 'n_estimators': 100},
]
 
 
def _make_model(task, params, backend):
    if backend == 'xgboost':
        common = dict(max_depth=params['max_depth'], learning_rate=params['lr'],
                      n_estimators=params['n_estimators'], subsample=0.8, colsample_bytree=0.8,
                      random_state=42, n_jobs=-1)
        return xgb.XGBClassifier(**common, eval_metric='logloss') if task == 'clf' else xgb.XGBRegressor(**common)
    elif backend == 'lightgbm':
        common = dict(max_depth=params['max_depth'], learning_rate=params['lr'],
                      n_estimators=params['n_estimators'], subsample=0.8, colsample_bytree=0.8,
                      random_state=42, n_jobs=-1, verbose=-1)
        return lgb.LGBMClassifier(**common) if task == 'clf' else lgb.LGBMRegressor(**common)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        common = dict(max_depth=params['max_depth'], learning_rate=params['lr'],
                      max_iter=params['n_estimators'], random_state=42)
        return HistGradientBoostingClassifier(**common) if task == 'clf' else HistGradientBoostingRegressor(**common)
 
 
def fit_best_gbm(X_train, y_train, task='clf'):
    """Small time-respecting hyperparameter search + final fit on the full training fold."""
    backend = backend_name()
    n = len(X_train)
    cut = int(n * 0.8)
    Xtr, Xval = X_train[:cut], X_train[cut:]
    ytr, yval = y_train[:cut], y_train[cut:]
 
    best_score, best_params = -np.inf if task == 'clf' else np.inf, PARAM_GRID[0]
    for params in PARAM_GRID:
        m = _make_model(task, params, backend)
        m.fit(Xtr, ytr)
        if task == 'clf':
            proba = m.predict_proba(Xval)[:, 1]
            score = roc_auc_score(yval, proba) if len(np.unique(yval)) > 1 else 0.5
            if score > best_score:
                best_score, best_params = score, params
        else:
            pred = m.predict(Xval)
            score = mean_squared_error(yval, pred)
            if score < best_score:
                best_score, best_params = score, params
 
    final_model = _make_model(task, best_params, backend)
    final_model.fit(X_train, y_train)
    return final_model, best_params, backend
 
 
# =============================================================================
# 3. numpy_lstm.py  (kept for parity with the full pipeline; not used by
#    the report generator itself, which only needs the GBM + GARCH pieces)
# =============================================================================
class NumpyLSTM:
    """
    Minimal single-layer LSTM + linear head, trained with plain-numpy BPTT + Adam.
    Used because torch/tensorflow are unavailable in this offline sandbox.
    Supports 'regression' (MSE) and 'classification' (BCE) via `task`.
    Input:  X of shape (n_samples, seq_len, n_features)
    Output: y of shape (n_samples,)
    """
    def __init__(self, n_features, hidden=16, task='regression', lr=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.h = hidden
        self.n_features = n_features
        self.task = task
        z = hidden + n_features
        scale = 1.0 / np.sqrt(z)
        self.Wf = rng.normal(0, scale, (z, hidden)); self.bf = np.zeros(hidden)
        self.Wi = rng.normal(0, scale, (z, hidden)); self.bi = np.zeros(hidden)
        self.Wc = rng.normal(0, scale, (z, hidden)); self.bc = np.zeros(hidden)
        self.Wo = rng.normal(0, scale, (z, hidden)); self.bo = np.zeros(hidden)
        self.Wy = rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, 1)); self.by = np.zeros(1)
        self.lr = lr
        self.params = ['Wf', 'bf', 'Wi', 'bi', 'Wc', 'bc', 'Wo', 'bo', 'Wy', 'by']
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0
 
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))
 
    def forward(self, X):
        n, T, f = X.shape
        h = self.h
        Hs = np.zeros((n, T + 1, h)); Cs = np.zeros((n, T + 1, h))
        cache = {'f': np.zeros((n, T, h)), 'i': np.zeros((n, T, h)), 'cbar': np.zeros((n, T, h)),
                 'o': np.zeros((n, T, h)), 'z': np.zeros((n, T, h + f))}
        for t in range(T):
            zt = np.concatenate([Hs[:, t, :], X[:, t, :]], axis=1)
            ft = self.sigmoid(zt @ self.Wf + self.bf)
            it = self.sigmoid(zt @ self.Wi + self.bi)
            cbar = np.tanh(zt @ self.Wc + self.bc)
            ot = self.sigmoid(zt @ self.Wo + self.bo)
            Cs[:, t + 1, :] = ft * Cs[:, t, :] + it * cbar
            Hs[:, t + 1, :] = ot * np.tanh(Cs[:, t + 1, :])
            cache['f'][:, t, :] = ft; cache['i'][:, t, :] = it; cache['cbar'][:, t, :] = cbar
            cache['o'][:, t, :] = ot; cache['z'][:, t, :] = zt
        h_last = Hs[:, T, :]
        logits = h_last @ self.Wy + self.by
        pred = self.sigmoid(logits) if self.task == 'classification' else logits
        return pred.ravel(), (Hs, Cs, cache, h_last)
 
    def backward(self, X, y, pred, state):
        n, T, f = X.shape
        h = self.h
        Hs, Cs, cache, h_last = state
        y = y.reshape(-1, 1); pred_col = pred.reshape(-1, 1)
        dlogits = (pred_col - y) / n if self.task == 'classification' else 2 * (pred_col - y) / n
 
        grads = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        grads['Wy'] = h_last.T @ dlogits
        grads['by'] = dlogits.sum(axis=0)
        dh_next = dlogits @ self.Wy.T
        dc_next = np.zeros((n, h))
 
        for t in reversed(range(T)):
            ft = cache['f'][:, t, :]; it = cache['i'][:, t, :]; cbar = cache['cbar'][:, t, :]
            ot = cache['o'][:, t, :]; zt = cache['z'][:, t, :]
            c_t = Cs[:, t + 1, :]; c_prev = Cs[:, t, :]
            tanh_c = np.tanh(c_t)
 
            dh = dh_next
            do = dh * tanh_c
            dc = dc_next + dh * ot * (1 - tanh_c ** 2)
            df = dc * c_prev
            di = dc * cbar
            dcbar = dc * it
            dc_prev = dc * ft
 
            do_raw = do * ot * (1 - ot)
            df_raw = df * ft * (1 - ft)
            di_raw = di * it * (1 - it)
            dcbar_raw = dcbar * (1 - cbar ** 2)
 
            grads['Wf'] += zt.T @ df_raw; grads['bf'] += df_raw.sum(axis=0)
            grads['Wi'] += zt.T @ di_raw; grads['bi'] += di_raw.sum(axis=0)
            grads['Wc'] += zt.T @ dcbar_raw; grads['bc'] += dcbar_raw.sum(axis=0)
            grads['Wo'] += zt.T @ do_raw; grads['bo'] += do_raw.sum(axis=0)
 
            dz = df_raw @ self.Wf.T + di_raw @ self.Wi.T + dcbar_raw @ self.Wc.T + do_raw @ self.Wo.T
            dh_next = dz[:, :h]
            dc_next = dc_prev
        return grads
 
    def adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for p in self.params:
            g = grads[p]
            self.m[p] = beta1 * self.m[p] + (1 - beta1) * g
            self.v[p] = beta2 * self.v[p] + (1 - beta2) * (g ** 2)
            mhat = self.m[p] / (1 - beta1 ** self.t)
            vhat = self.v[p] / (1 - beta2 ** self.t)
            setattr(self, p, getattr(self, p) - self.lr * mhat / (np.sqrt(vhat) + eps))
 
    def fit(self, X, y, epochs=30, batch_size=64, verbose=False):
        n = X.shape[0]
        rng = np.random.default_rng(0)
        for ep in range(epochs):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                b = idx[start:start + batch_size]
                pred, state = self.forward(X[b])
                grads = self.backward(X[b], y[b], pred, state)
                self.adam_step(grads)
            if verbose and ep % 10 == 0:
                pred_all, _ = self.forward(X)
                if self.task == 'classification':
                    eps = 1e-9
                    loss = -np.mean(y * np.log(pred_all + eps) + (1 - y) * np.log(1 - pred_all + eps))
                else:
                    loss = np.mean((pred_all - y) ** 2)
                print(f"  epoch {ep} loss {loss:.5f}")
 
    def predict(self, X):
        pred, _ = self.forward(X)
        return pred
 
 
# =============================================================================
# 4. nifty_pipeline_v2.py  -- data loading, feature engineering, CV, forecast
# =============================================================================
DEFAULT_CONFIG = dict(
    nifty_xlsx='Nifty_OHLC_Direction_Indicators.xlsx',
    vix_xlsx='INDIAVIX_OHLC_Direction_Indicators.xlsx',
    us_csv='us_overnight_market_features.csv',
    gift_csv='Gift_Nifty_50_Futures_Historical_Data.csv',
    end_date=None,       # <- set from --cutoff-date on the command line
    train_window=750,    # ~2 trading years
    test_window=63,       # ~3 months (used only by run_walkforward, not the report)
    purge=1,
    seq_len=30,           # LSTM window
    lstm_hidden=16,
    lstm_epochs=25,
)
 
 
def load_and_merge(cfg):
    nifty = pd.read_excel(cfg['nifty_xlsx'], sheet_name='OHLC Data')
    nifty['Date'] = pd.to_datetime(nifty['Date'])
    nifty = nifty.sort_values('Date').reset_index(drop=True)
    nifty = nifty.rename(columns={'Open': 'O', 'High': 'H', 'Low': 'L', 'Close': 'C', 'Volume': 'V'})
 
    vix = pd.read_excel(cfg['vix_xlsx'], sheet_name='OHLC Data')
    vix['Date'] = pd.to_datetime(vix['Date'])
    vix = vix[['Date', 'Close']].rename(columns={'Close': 'VIX_Close'})
 
    us = pd.read_csv(cfg['us_csv'])
    us['Date'] = pd.to_datetime(us['Date'])
 
    gift = pd.read_csv(cfg['gift_csv'])
    gift['Date'] = pd.to_datetime(gift['Date'], format='%m/%d/%Y')
    for c in ['Price', 'Open', 'High', 'Low']:
        gift[c] = gift[c].astype(str).str.replace(',', '').astype(float)
    gift = gift.sort_values('Date').reset_index(drop=True)
    gift = gift.rename(columns={'Price': 'GIFT_Close', 'Open': 'GIFT_Open',
                                 'High': 'GIFT_High', 'Low': 'GIFT_Low'})
    gift = gift[['Date', 'GIFT_Open', 'GIFT_High', 'GIFT_Low', 'GIFT_Close']]
 
    common_start = max(nifty['Date'].min(), vix['Date'].min(), us['Date'].min(), gift['Date'].min())
    print(f"[load] Restricting ALL data to >= {common_start.date()} "
          f"(first date common to all 4 sources, per instruction)")
 
    df = nifty.merge(vix, on='Date', how='left').merge(us, on='Date', how='left').merge(gift, on='Date', how='left')
    df = df.sort_values('Date').reset_index(drop=True)
    df['VIX_Close'] = df['VIX_Close'].ffill()
    for c in ['SP500_Return', 'NASDAQ_Return', 'Crude_Change', 'DXY_Change', 'US_VIX_Change',
              'GIFT_Open', 'GIFT_High', 'GIFT_Low', 'GIFT_Close']:
        df[c] = df[c].ffill()
 
    df = df[df['Date'] >= common_start].reset_index(drop=True)
 
    if cfg.get('end_date'):
        end_date = pd.Timestamp(cfg['end_date'])
        df = df[df['Date'] <= end_date].reset_index(drop=True)
        print(f"[load] Also capping data at <= {end_date.date()} (cutoff date)")
 
    return df
 
 
def engineer_features(df):
    rng = (df['H'] - df['L']).replace(0, np.nan)
    df['body_ratio'] = (df['C'] - df['O']) / rng
    df['upper_wick_ratio'] = (df['H'] - df[['O', 'C']].max(axis=1)) / rng
    df['lower_wick_ratio'] = (df[['O', 'C']].min(axis=1) - df['L']) / rng
 
    df['prev_C'] = df['C'].shift(1)
    df['gap_return'] = np.log(df['O'] / df['prev_C'])
    df['intraday_return'] = np.log(df['C'] / df['O'])
    df['daily_return'] = np.log(df['C'] / df['prev_C'])
 
    df['parkinson_vol'] = np.sqrt((1.0 / (4 * np.log(2))) * (np.log(df['H'] / df['L'])) ** 2)
    log_hl, log_co = np.log(df['H'] / df['L']), np.log(df['C'] / df['O'])
    df['gk_vol'] = np.sqrt(0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2)
 
    tr = pd.concat([df['H'] - df['L'], (df['H'] - df['prev_C']).abs(),
                     (df['L'] - df['prev_C']).abs()], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()
    df['ATR14_norm'] = df['ATR14'] / df['C']
    df['realized_vol_20'] = df['daily_return'].rolling(20).std()
 
    delta = df['C'].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan)
    df['RSI14_norm'] = (100 - 100 / (1 + rs) - 50) / 50
 
    low14, high14 = df['L'].rolling(14).min(), df['H'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['C'] - low14) / (high14 - low14)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
 
    ema20, ema50 = df['C'].ewm(span=20, adjust=False).mean(), df['C'].ewm(span=50, adjust=False).mean()
    df['dist_EMA20'] = (df['C'] - ema20) / ema20
    df['dist_EMA50'] = (df['C'] - ema50) / ema50
 
    df['VIX_change'] = df['VIX_Close'].pct_change()
    df['VIX_level_norm'] = (df['VIX_Close'] - df['VIX_Close'].rolling(60).mean()) / df['VIX_Close'].rolling(60).std()
 
    # ---- GIFT Nifty overnight indicator ----
    df['gift_gap_indicator'] = np.log(df['GIFT_Open'] / df['prev_C'])
    df['gift_range_pct'] = (df['GIFT_High'] - df['GIFT_Low']) / df['GIFT_Open']
    df['gift_own_change_lag1'] = (df['GIFT_Close'] / df['GIFT_Close'].shift(1) - 1).shift(1)
 
    # ---- US/SGX overnight moves ----
    us_cols = ['SP500_Return', 'NASDAQ_Return', 'Crude_Change', 'DXY_Change', 'US_VIX_Change']
 
    df['target_dir'] = (df['C'] > df['O']).astype(int)
    df['target_dH'] = np.log(df['H'] / df['O'])
    df['target_dL'] = np.log(df['L'] / df['O'])
    df['target_gap'] = df['gap_return']  # Open target: overnight gap
 
    lag_feats = ['body_ratio', 'upper_wick_ratio', 'lower_wick_ratio', 'gap_return', 'intraday_return',
                 'daily_return', 'parkinson_vol', 'gk_vol', 'ATR14_norm', 'realized_vol_20',
                 'RSI14_norm', 'Stoch_K', 'Stoch_D', 'dist_EMA20', 'dist_EMA50',
                 'VIX_Close', 'VIX_change', 'VIX_level_norm']
    contemp_feats = ['gift_gap_indicator', 'gift_range_pct', 'gift_own_change_lag1'] + us_cols
 
    Xlag = df[lag_feats].shift(1)
    Xlag.columns = [c + '_lag1' for c in lag_feats]
    Xcontemp = df[contemp_feats].copy()  # pre-open info, used as-is
 
    data = pd.concat([df[['Date', 'target_dir', 'target_dH', 'target_dL', 'target_gap', 'daily_return']],
                       Xlag, Xcontemp], axis=1)
    data = data.dropna().reset_index(drop=True)
    feat_cols = list(Xlag.columns) + contemp_feats
    return data, feat_cols
 
 
def build_sequences(data, feat_cols, seq_len):
    """Row i's sequence = feat_cols values for rows [i-seq_len+1 .. i] -> predicts target at row i."""
    X = data[feat_cols].values.astype(float)
    n = len(data)
    seqs = np.full((n, seq_len, len(feat_cols)), np.nan)
    for i in range(seq_len - 1, n):
        seqs[i] = X[i - seq_len + 1:i + 1]
    valid_from = seq_len - 1
    return seqs, valid_from
 
 
def run_walkforward(data, feat_cols, cfg):
    """Full purged walk-forward CV over the entire history (GBM + LSTM benchmark). Optional utility,
    not required for the N-day report but kept here for parity with the original pipeline."""
    seqs, valid_from = build_sequences(data, feat_cols, cfg['seq_len'])
    n = len(data)
    train_win, test_win, purge, seq_len = cfg['train_window'], cfg['test_window'], cfg['purge'], cfg['seq_len']
 
    results = []
    start = valid_from
    fold = 0
    while start + train_win + purge + test_win <= n:
        tr_idx = np.arange(start, start + train_win)
        te_idx = np.arange(start + train_win + purge, start + train_win + purge + test_win)
 
        Xtr, Xte = data.iloc[tr_idx][feat_cols].values, data.iloc[te_idx][feat_cols].values
        y_dir_tr, y_dir_te = data.iloc[tr_idx]['target_dir'].values, data.iloc[te_idx]['target_dir'].values
        y_dH_tr, y_dH_te = data.iloc[tr_idx]['target_dH'].values, data.iloc[te_idx]['target_dH'].values
        y_dL_tr, y_dL_te = data.iloc[tr_idx]['target_dL'].values, data.iloc[te_idx]['target_dL'].values
        y_gap_tr, y_gap_te = data.iloc[tr_idx]['target_gap'].values, data.iloc[te_idx]['target_gap'].values
 
        r_pct_tr = data.iloc[tr_idx]['daily_return'].values * 100
        omega, alpha, beta = fit_garch11(r_pct_tr)
        sigma2_tr = garch_sigma2_series(r_pct_tr, omega, alpha, beta)
        r_pct_te = data.iloc[te_idx]['daily_return'].values * 100
        sigma2_te = np.zeros(len(te_idx))
        last_sigma2, last_r = sigma2_tr[-1], r_pct_tr[-1]
        for i in range(len(te_idx)):
            s2 = garch_one_step_forecast(last_r, last_sigma2, omega, alpha, beta)
            sigma2_te[i] = s2
            last_sigma2, last_r = s2, r_pct_te[i]
 
        Xtr_garch = np.column_stack([Xtr, sigma2_tr])
        Xte_garch = np.column_stack([Xte, sigma2_te])
 
        clf, clf_params, backend = fit_best_gbm(Xtr, y_dir_tr, task='clf')
        proba = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y_dir_te, proba) if len(np.unique(y_dir_te)) > 1 else np.nan
        dir_acc = accuracy_score(y_dir_te, (proba > 0.5).astype(int))
 
        regH, _, _ = fit_best_gbm(Xtr_garch, y_dH_tr, task='reg')
        pred_H = regH.predict(Xte_garch)
        rmse_H = np.sqrt(mean_squared_error(y_dH_te, pred_H))
        rmse_H_base = np.sqrt(mean_squared_error(y_dH_te, np.full(len(y_dH_te), y_dH_tr.mean())))
 
        regL, _, _ = fit_best_gbm(Xtr_garch, y_dL_tr, task='reg')
        pred_L = regL.predict(Xte_garch)
        rmse_L = np.sqrt(mean_squared_error(y_dL_te, pred_L))
        rmse_L_base = np.sqrt(mean_squared_error(y_dL_te, np.full(len(y_dL_te), y_dL_tr.mean())))
 
        regGap, _, _ = fit_best_gbm(Xtr, y_gap_tr, task='reg')
        pred_gap = regGap.predict(Xte)
        rmse_gap = np.sqrt(mean_squared_error(y_gap_te, pred_gap))
        rmse_gap_base = np.sqrt(mean_squared_error(y_gap_te, np.full(len(y_gap_te), y_gap_tr.mean())))
 
        seq_tr, seq_te = seqs[tr_idx].copy(), seqs[te_idx].copy()
        mu = np.nanmean(seq_tr.reshape(-1, seq_tr.shape[-1]), axis=0)
        sd = np.nanstd(seq_tr.reshape(-1, seq_tr.shape[-1]), axis=0) + 1e-8
        seq_tr = (seq_tr - mu) / sd
        seq_te = (seq_te - mu) / sd
 
        lstm_clf = NumpyLSTM(n_features=len(feat_cols), hidden=cfg['lstm_hidden'], task='classification', lr=0.01)
        lstm_clf.fit(seq_tr, y_dir_tr.astype(float), epochs=cfg['lstm_epochs'], batch_size=64)
        lstm_proba = lstm_clf.predict(seq_te)
        lstm_auc = roc_auc_score(y_dir_te, lstm_proba) if len(np.unique(y_dir_te)) > 1 else np.nan
        lstm_acc = accuracy_score(y_dir_te, (lstm_proba > 0.5).astype(int))
 
        lstm_reg = NumpyLSTM(n_features=len(feat_cols), hidden=cfg['lstm_hidden'], task='regression', lr=0.01)
        lstm_reg.fit(seq_tr, y_dH_tr, epochs=cfg['lstm_epochs'], batch_size=64)
        lstm_pred_H = lstm_reg.predict(seq_te)
        lstm_rmse_H = np.sqrt(mean_squared_error(y_dH_te, lstm_pred_H))
 
        results.append(dict(
            fold=fold, test_start=data.iloc[te_idx]['Date'].iloc[0], test_end=data.iloc[te_idx]['Date'].iloc[-1],
            gbm_backend=backend, auc=auc, dir_acc=dir_acc,
            rmse_H=rmse_H, rmse_H_base=rmse_H_base, rmse_L=rmse_L, rmse_L_base=rmse_L_base,
            rmse_gap=rmse_gap, rmse_gap_base=rmse_gap_base,
            garch_persistence=alpha + beta,
            lstm_auc=lstm_auc, lstm_dir_acc=lstm_acc, lstm_rmse_H=lstm_rmse_H,
        ))
        print(f"fold {fold:2d} [{data.iloc[te_idx]['Date'].iloc[0].date()} - {data.iloc[te_idx]['Date'].iloc[-1].date()}] "
              f"GBM AUC={auc:.3f} acc={dir_acc:.3f} | LSTM AUC={lstm_auc:.3f} acc={lstm_acc:.3f} | "
              f"RMSE_gap={rmse_gap:.4f}(base {rmse_gap_base:.4f}) RMSE_H={rmse_H:.4f} RMSE_L={rmse_L:.4f}")
 
        fold += 1
        start += test_win
 
    return pd.DataFrame(results)
 
 
def final_forecast(df, data, feat_cols, cfg):
    """Single 'tomorrow' forecast trained on ALL available history up to the cutoff.
    (Restored from nifty_pipeline_v2.py -- was dropped from main() in the merge.)"""
    X = data[feat_cols].values
    r_pct = data['daily_return'].values * 100
    omega, alpha, beta = fit_garch11(r_pct)
    sigma2 = garch_sigma2_series(r_pct, omega, alpha, beta)
    X_garch = np.column_stack([X, sigma2])
 
    clf, _, backend = fit_best_gbm(X, data['target_dir'].values, task='clf')
    regH, _, _ = fit_best_gbm(X_garch, data['target_dH'].values, task='reg')
    regL, _, _ = fit_best_gbm(X_garch, data['target_dL'].values, task='reg')
    regGap, _, _ = fit_best_gbm(X, data['target_gap'].values, task='reg')
 
    last_row = data[feat_cols].iloc[[-1]].values
    last_sigma2_next = garch_one_step_forecast(r_pct[-1], sigma2[-1], omega, alpha, beta)
    last_row_garch = np.column_stack([last_row, [last_sigma2_next]])
 
    prob_up = clf.predict_proba(last_row)[0, 1]
    pred_dH = regH.predict(last_row_garch)[0]
    pred_dL = regL.predict(last_row_garch)[0]
    pred_gap = regGap.predict(last_row)[0]  # NOTE: uses TODAY's GIFT/US cols as a stand-in;
    # in live use, replace the GIFT_*/US_* columns of `last_row` with tomorrow's actual
    # pre-open GIFT Nifty print + US overnight session data once available.
 
    last_close = df['C'].iloc[-1]
    last_date = df['Date'].iloc[-1].date()
 
    O_central = last_close * np.exp(pred_gap)
    H_central = O_central * np.exp(pred_dH)
    L_central = O_central * np.exp(pred_dL)
 
    intraday = (np.log(df['C'] / df['O'])).tail(250)
    up_avg = intraday[intraday > 0].mean()
    down_avg = intraday[intraday <= 0].mean()
    exp_intraday_ret = prob_up * up_avg + (1 - prob_up) * down_avg
    C_central = O_central * np.exp(exp_intraday_ret)
 
    sigma_next_pct = np.sqrt(last_sigma2_next)
    sigma = sigma_next_pct / 100
 
    return dict(
        last_date=last_date, last_close=last_close,
        prob_up=prob_up, pred_dH=pred_dH, pred_dL=pred_dL, pred_gap=pred_gap,
        garch_sigma_next_pct=sigma_next_pct, backend=backend, garch_params=(omega, alpha, beta),
        Open=O_central, High=H_central, Low=L_central, Close=C_central,
        Close_1sigma_band=(last_close * (1 - sigma), last_close * (1 + sigma)),
        Close_2sigma_band=(last_close * (1 - 2 * sigma), last_close * (1 + 2 * sigma)),
    )
 
# ANSI Color Codes
RED = "\033[91m"
RESET = "\033[0m"

def tompred(pred_clse, tod_close, pred_high, pred_low):
    if pred_clse > tod_close:
        print(f"{RED}tomorrows closing price likely to be above todays price with a high around {pred_high}{RESET}")
    if pred_clse < tod_close:
        print(f"{RED}tomorrows closing price likely to be below todays price with a low of around {pred_low}{RESET}")

def print_forecast_table(fc):
    print(f"{RED}\nAs of {fc['last_date']} (last Close = {fc['last_close']:,.1f}), forecast for the next session:\n{RESET}")
    print(f"{RED}{'Close':<8}{fc['Close']:>12,.1f}   Probability of Green Candle = {fc['prob_up']:.1%}){RESET}")
    tompred(fc['Close'], fc['last_close'], fc['High'], fc['Low'])
    lo1, hi1 = fc['Close_1sigma_band']
    lo2, hi2 = fc['Close_2sigma_band']
    print(f"{RED}\nGARCH(1,1) next-day volatility: {fc['garch_sigma_next_pct']:.3f}%{RESET}")
    print(f"{RED}Predicted range for tomorrow: {lo2:,.0f} - {hi2:,.0f}{RESET}")
    print(f"\n(GBM backend used: {fc['backend']})")
    print("NOTE: Open/High/Low/Close above use TODAY's GIFT Nifty & US overnight columns as a stand-in "
          "for tomorrow's (not yet available in the uploaded files) -- re-run once tomorrow's pre-open "
          "GIFT print and the actual overnight US/SGX session are in the data.")
 
 
def save_forecast_csv(fc, out_path):
    lo1, hi1 = fc['Close_1sigma_band']
    lo2, hi2 = fc['Close_2sigma_band']
    omega, alpha, beta = fc['garch_params']
    row = dict(
        last_date=fc['last_date'], last_close=fc['last_close'],
        forecast_Open=fc['Open'], forecast_High=fc['High'], forecast_Low=fc['Low'], forecast_Close=fc['Close'],
        prob_close_gt_open=fc['prob_up'], garch_sigma_next_pct=fc['garch_sigma_next_pct'],
        garch_omega=omega, garch_alpha=alpha, garch_beta=beta,
        Close_1sigma_low=lo1, Close_1sigma_high=hi1, Close_2sigma_low=lo2, Close_2sigma_high=hi2,
        gbm_backend=fc['backend'],
    )
    pd.DataFrame([row]).to_csv(out_path, index=False)
 
 
# =============================================================================
# 5. direction.py -- the N-day backtest "prediction report" builder
# =============================================================================
def build_report(df, data, feat_cols, cfg, n_days):
    n = len(data)
    train_win, purge = cfg['train_window'], cfg['purge']
 
    if n_days >= n:
        raise ValueError(f"Only {n} usable feature rows available, cannot report {n_days} days.")
 
    test_idx = np.arange(n - n_days, n)
    train_end = test_idx[0] - purge
    train_start = max(0, train_end - train_win)
    if train_end - train_start < 100:
        raise ValueError("Not enough history before the report window to train a model "
                          f"(only {train_end - train_start} rows). Reduce n_days or train_window.")
    tr_idx = np.arange(train_start, train_end)
 
    print(f"[report] Training window: rows {train_start}-{train_end - 1} "
          f"({data['Date'].iloc[train_start].date()} -> {data['Date'].iloc[train_end - 1].date()}), "
          f"{len(tr_idx)} rows")
    print(f"[report] Test window ({n_days} days): "
          f"{data['Date'].iloc[test_idx[0]].date()} -> {data['Date'].iloc[test_idx[-1]].date()}")
 
    Xtr, Xte = data.iloc[tr_idx][feat_cols].values, data.iloc[test_idx][feat_cols].values
    y_dir_tr = data.iloc[tr_idx]['target_dir'].values
    y_dH_tr = data.iloc[tr_idx]['target_dH'].values
    y_dL_tr = data.iloc[tr_idx]['target_dL'].values
    y_gap_tr = data.iloc[tr_idx]['target_gap'].values
 
    # ---- GARCH(1,1) fit on TRAIN returns only; roll sigma2 forward through the test window
    #      using realized returns available at t-1 (no leakage) ----
    r_pct_tr = data.iloc[tr_idx]['daily_return'].values * 100
    omega, alpha, beta = fit_garch11(r_pct_tr)
    sigma2_tr = garch_sigma2_series(r_pct_tr, omega, alpha, beta)
    r_pct_te = data.iloc[test_idx]['daily_return'].values * 100
    sigma2_te = np.zeros(len(test_idx))
    last_sigma2, last_r = sigma2_tr[-1], r_pct_tr[-1]
    for i in range(len(test_idx)):
        s2 = garch_one_step_forecast(last_r, last_sigma2, omega, alpha, beta)
        sigma2_te[i] = s2
        last_sigma2, last_r = s2, r_pct_te[i]
 
    Xtr_garch = np.column_stack([Xtr, sigma2_tr])
    Xte_garch = np.column_stack([Xte, sigma2_te])
 
    # ---- Fit the model set ONCE on the training window ----
    clf, _, backend = fit_best_gbm(Xtr, y_dir_tr, task='clf')
    regH, _, _ = fit_best_gbm(Xtr_garch, y_dH_tr, task='reg')
    regL, _, _ = fit_best_gbm(Xtr_garch, y_dL_tr, task='reg')
    regGap, _, _ = fit_best_gbm(Xtr, y_gap_tr, task='reg')
 
    p_up = clf.predict_proba(Xte)[:, 1]
    pred_dH = regH.predict(Xte_garch)
    pred_dL = regL.predict(Xte_garch)
    pred_gap = regGap.predict(Xte)
 
    # ---- Expected Close: prob-weighted trailing up/down intraday return, computed from
    #      TRAIN data only (last 250 train rows), exactly as in final_forecast() ----
    train_dates = data.iloc[tr_idx]['Date']
    df_tr = df[df['Date'].isin(train_dates)]
    intraday_tr = (np.log(df_tr['C'] / df_tr['O'])).tail(250)
    up_avg = intraday_tr[intraday_tr > 0].mean()
    down_avg = intraday_tr[intraday_tr <= 0].mean()
 
    # ---- Actual OHLC + anchor (prior actual close) per test date, pulled from df ----
    df_idx = df.set_index('Date')
    test_dates = data.iloc[test_idx]['Date'].values
 
    rows = []
    for i, date in enumerate(test_dates):
        date = pd.Timestamp(date)
        anchor_close = df_idx.loc[date, 'prev_C']          # actual close AS OF the day being predicted
        actual_O = df_idx.loc[date, 'O']
        actual_H = df_idx.loc[date, 'H']
        actual_L = df_idx.loc[date, 'L']
        actual_next_close = df_idx.loc[date, 'C']           # actual realized close for that day
 
        O_pred = anchor_close * np.exp(pred_gap[i])
        H_pred = O_pred * np.exp(pred_dH[i])
        L_pred = O_pred * np.exp(pred_dL[i])
        exp_intraday_ret = p_up[i] * up_avg + (1 - p_up[i]) * down_avg
        C_pred = O_pred * np.exp(exp_intraday_ret)
 
        predicted_up = C_pred > anchor_close
        predicted_down = C_pred < anchor_close
        actual_up = actual_next_close > anchor_close
        actual_down = actual_next_close < anchor_close
 
        correct = "Yes" if (predicted_down and actual_down) or (predicted_up and actual_up) else "No"
 
        if predicted_up:
            confidence = "High" if p_up[i] > 0.55 else "Low"
        elif predicted_down:
            confidence = "High" if p_up[i] < 0.45 else "Low"
        else:
            confidence = "Low"
 
        rows.append(dict(
            Date=date.date(),
            Anchor_Close=round(anchor_close, 2),
            Predicted_Open=round(O_pred, 2),
            Predicted_High=round(H_pred, 2),
            Predicted_Low=round(L_pred, 2),
            Predicted_Close=round(C_pred, 2),
            Actual_Open=round(actual_O, 2),
            Actual_High=round(actual_H, 2),
            Actual_Low=round(actual_L, 2),
            Actual_Close=round(actual_next_close, 2),
            P_Close_gt_Open=round(p_up[i], 4),
            Predicted_Direction="Up" if predicted_up else ("Down" if predicted_down else "Flat"),
            Actual_Direction="Up" if actual_up else ("Down" if actual_down else "Flat"),
            Correct=correct,
            Confidence=confidence,
        ))
 
    report = pd.DataFrame(rows)
    return report, dict(backend=backend, omega=omega, alpha=alpha, beta=beta,
                         up_avg=up_avg, down_avg=down_avg,
                         train_start_date=data.iloc[train_start]['Date'].date(),
                         train_end_date=data.iloc[train_end - 1]['Date'].date())
 
 
def summarize(report):
    n = len(report)
    acc = (report['Correct'] == 'Yes').mean()
    print(f"\n[report] {n} days | directional accuracy: {acc:.1%}")
    for conf in ['High', 'Low']:
        sub = report[report['Confidence'] == conf]
        if len(sub):
            sub_acc = (sub['Correct'] == 'Yes').mean()
            print(f"  Confidence={conf:<5} n={len(sub):3d}  accuracy={sub_acc:.1%}")
 
 
def write_xlsx(report, meta, cfg, out_path):
    wb = Workbook()
    ws = wb.active
    ws.title = 'Prediction Report'
 
    header_font = Font(name='Arial', bold=True, color='FFFFFF')
    header_fill = PatternFill('solid', fgColor='1F4E78')
    body_font = Font(name='Arial', size=10)
    center = Alignment(horizontal='center')
 
    yes_fill = PatternFill('solid', fgColor='C6EFCE')  # green
    no_fill = PatternFill('solid', fgColor='FFC7CE')   # red
    high_fill = PatternFill('solid', fgColor='FFEB9C')  # amber
    yes_font = Font(name='Arial', size=10, color='006100')
    no_font = Font(name='Arial', size=10, color='9C0006')
 
    ws['A1'] = f"Nifty {len(report)}-Day Prediction Report"
    ws['A1'].font = Font(name='Arial', bold=True, size=14)
    ws['A2'] = f"Cutoff date: {cfg.get('end_date')}   |   Report window: {report['Date'].iloc[0]} -> {report['Date'].iloc[-1]}"
    ws['A2'].font = Font(name='Arial', italic=True, size=10)
    ws['A3'] = (f"Model trained once on {meta['train_start_date']} -> {meta['train_end_date']} "
                f"({cfg['train_window']} rows, GBM backend: {meta['backend']}); applied unchanged to each test day.")
    ws['A3'].font = Font(name='Arial', italic=True, size=9)
    ws.merge_cells('A1:O1')
    ws.merge_cells('A2:O2')
    ws.merge_cells('A3:O3')
 
    header_row = 5
    cols = list(report.columns)
    for j, col in enumerate(cols, start=1):
        c = ws.cell(row=header_row, column=j, value=col.replace('_', ' '))
        c.font = header_font
        c.fill = header_fill
        c.alignment = center
 
    for i, row in report.iterrows():
        r = header_row + 1 + i
        for j, col in enumerate(cols, start=1):
            val = row[col]
            c = ws.cell(row=r, column=j, value=val)
            c.font = body_font
            c.alignment = center
            if col == 'P_Close_gt_Open':
                c.number_format = '0.0%'
            if col == 'Correct':
                if val == 'Yes':
                    c.fill, c.font = yes_fill, yes_font
                else:
                    c.fill, c.font = no_fill, no_font
            if col == 'Confidence' and val == 'High':
                c.fill = high_fill
 
    sum_row = header_row + len(report) + 2
    n = len(report)
    acc = (report['Correct'] == 'Yes').mean()
    ws.cell(row=sum_row, column=1, value='Summary').font = Font(name='Arial', bold=True, size=12)
    ws.cell(row=sum_row + 1, column=1,
            value=f"Overall directional accuracy: {acc:.1%} ({(report['Correct'] == 'Yes').sum()}/{n})").font = body_font
    r = sum_row + 2
    for conf in ['High', 'Low']:
        sub = report[report['Confidence'] == conf]
        if len(sub):
            sub_acc = (sub['Correct'] == 'Yes').mean()
            ws.cell(row=r, column=1,
                    value=f"Confidence={conf}: n={len(sub)}, accuracy={sub_acc:.1%}").font = body_font
            r += 1
    ws.cell(row=r + 1, column=1,
            value=("Note: Predicted_Close is derived using TODAY's GIFT Nifty / US overnight "
                   "columns as a stand-in for the actual forecast day's pre-open prints (same "
                   "convention as final_forecast()); this is a like-for-like backtest of the "
                   "pipeline's methodology, not a live point-in-time forecast archive.")
            ).font = Font(name='Arial', italic=True, size=9)
    ws.merge_cells(start_row=r + 1, start_column=1, end_row=r + 1, end_column=15)
    ws.cell(row=r + 1, column=1).alignment = Alignment(wrap_text=True)
 
    widths = [12, 13, 15, 14, 13, 15, 12, 12, 11, 13, 15, 18, 15, 9, 11]
    for j, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = ws.cell(row=header_row + 1, column=1)
 
    wb.save(out_path)
 
 
# =============================================================================
# 6. CLI entry point -- run with: python3 nifty_predict_report.py --cutoff-date YYYY-MM-DD
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Backtest Nifty next-day prediction report for the N trading days up to a cutoff date.")
    p.add_argument('--cutoff-date', '-c', required=True,
                    help="Cutoff date (YYYY-MM-DD). Data is capped at <= this date; the report covers the "
                         "N trading days ending on it.")
    p.add_argument('--n-days', '-n', type=int, default=60,
                    help="Number of trading days to report, counted back from the cutoff date (default: 60).")
    p.add_argument('--nifty-xlsx', default=DEFAULT_CONFIG['nifty_xlsx'])
    p.add_argument('--vix-xlsx', default=DEFAULT_CONFIG['vix_xlsx'])
    p.add_argument('--us-csv', default=DEFAULT_CONFIG['us_csv'])
    p.add_argument('--gift-csv', default=DEFAULT_CONFIG['gift_csv'])
    p.add_argument('--train-window', type=int, default=DEFAULT_CONFIG['train_window'])
    p.add_argument('--purge', type=int, default=DEFAULT_CONFIG['purge'])
    p.add_argument('--out-prefix', default=None,
                    help="Output filename prefix (default: 'Nifty_<n_days>Day_Prediction_Report_<cutoff-date>').")
    p.add_argument('--skip-final-forecast', action='store_true',
                    help="Skip the single 'tomorrow' forecast (final_forecast() from nifty_pipeline_v2.py). "
                         "Runs by default -- it's a single fit, cheap.")
    p.add_argument('--run-cv', action='store_true',
                    help="Also run the full purged walk-forward CV over the ENTIRE history up to the cutoff "
                         "(run_walkforward() from nifty_pipeline_v2.py) and save "
                         "Nifty_WalkForward_CV_Results_v2.csv. Off by default -- retrains models across "
                         "every historical fold, so it is much slower than the report or final forecast.")
    return p.parse_args()
 
 
def main():
    args = parse_args()
 
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(
        nifty_xlsx=args.nifty_xlsx, vix_xlsx=args.vix_xlsx, us_csv=args.us_csv, gift_csv=args.gift_csv,
        end_date=args.cutoff_date, train_window=args.train_window, purge=args.purge,
    )
 
    print(f"[gbm backend] {backend_name()}")
 
    df = load_and_merge(cfg)
    data, feat_cols = engineer_features(df)
    print(f"[features] {len(feat_cols)} features, {len(data)} usable rows, "
          f"{data['Date'].min().date()} -> {data['Date'].max().date()}")
 
    report, meta = build_report(df, data, feat_cols, cfg, n_days=args.n_days)
    summarize(report)
 
    prefix = args.out_prefix or f"Nifty_{args.n_days}Day_Prediction_Report_{args.cutoff_date}"
    out_csv = f"{prefix}.csv"
    
 
    report.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")
 
 
    # ---- nifty_pipeline_v2.py output #2: single "tomorrow" forecast (cheap, on by default) ----
    if not args.skip_final_forecast:
        print("\n=== Tomorrow's forecast (final models trained on full history up to cutoff) ===")
        fc = final_forecast(df, data, feat_cols, cfg)
        print_forecast_table(fc)
        
 
    # ---- nifty_pipeline_v2.py output #1: full walk-forward CV (expensive, opt-in) ----
    if args.run_cv:
        print("\n=== Running full purged walk-forward CV over entire history (this will take a while) ===")
        cv_results = run_walkforward(data, feat_cols, cfg)
        cv_out = f"Nifty_WalkForward_CV_Results_v2_{args.cutoff_date}.csv"
        cv_results.to_csv(cv_out, index=False)
        print(f"\n[saved] {cv_out}")
        print("\n=== Aggregate (GBM tabular) ===")
        print("Mean AUC:", cv_results['auc'].mean(), " | Mean dir acc:", cv_results['dir_acc'].mean())
        print("Mean RMSE gap (model vs base):", cv_results['rmse_gap'].mean(), cv_results['rmse_gap_base'].mean())
        print("Mean RMSE dH  (model vs base):", cv_results['rmse_H'].mean(), cv_results['rmse_H_base'].mean())
        print("Mean RMSE dL  (model vs base):", cv_results['rmse_L'].mean(), cv_results['rmse_L_base'].mean())
        print("\n=== Aggregate (numpy LSTM, 30-day sequences) ===")
        print("Mean AUC:", cv_results['lstm_auc'].mean(), " | Mean dir acc:", cv_results['lstm_dir_acc'].mean())
        print("Mean RMSE dH:", cv_results['lstm_rmse_H'].mean())
 
#%run nifty_predict_report.py --cutoff-date 2026-08-02 
if __name__ == '__main__':
    sys.argv = ['nifty_predict_report.py', '--cutoff-date', '2026-08-14']
    main()