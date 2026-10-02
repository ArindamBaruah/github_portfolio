"""
Nifty Next-Day OHLC Forecast Pipeline -- v2
=============================================
Adds, on top of the v1 pipeline:
  1. GIFT Nifty overnight futures signal + US/SGX overnight moves as features
     that directly target the Open (the weakest-modeled variable in v1).
  2. Real XGBoost / LightGBM via gbm_factory.py, with a small hyperparameter
     search inside every purged walk-forward fold (falls back to sklearn
     HistGradientBoosting only because this sandbox has no internet access
     to install xgboost/lightgbm -- swap-free on a machine that has them).
  3. A from-scratch numpy LSTM (numpy_lstm.py, manual BPTT + Adam -- no
     torch/tensorflow available offline) trained on 30-day rolling windows
     of the engineered feature vectors, benchmarked against the tabular GBM.
  4. GARCH(1,1) conditional variance fed in as an explicit feature to the
     High/Low range regressors (previously GARCH ran only as a separate
     benchmark).
  5. Per user instruction: ALL data is restricted to 2021-08-16 onward --
     the first date common to Nifty, India VIX, GIFT Nifty and the US
     overnight feature file.

Run: python3 nifty_pipeline_v2.py
Requires the four raw files at the paths in CONFIG below.
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score

from gbm_factory import fit_best_gbm, backend_name
from garch_module import fit_garch11, garch_sigma2_series, garch_one_step_forecast
from numpy_lstm import NumpyLSTM

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
CONFIG = dict(
    nifty_xlsx='Nifty_OHLC_Direction_Indicators.xlsx',
    vix_xlsx='INDIAVIX_OHLC_Direction_Indicators.xlsx',
    us_csv='us_overnight_market_features.csv',
    gift_csv='Gift_Nifty_50_Futures_Historical_Data.csv',
    end_date='2026-08-02',   # <- data cutoff; set to None to use all available data
    train_window=500,   # ~2 trading years
    test_window=63,     # ~3 months
    purge=1,
    seq_len=30,          # LSTM window
    lstm_hidden=16,
    lstm_epochs=25,
)


# ----------------------------------------------------------------------------
# 1. LOAD & MERGE (restricted to the common 2021+ window, per user instruction)
# ----------------------------------------------------------------------------
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
        print(f"[load] Also capping data at <= {end_date.date()} (per end_date in CONFIG)")

    return df


# ----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------------
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

    # ---- NEW: GIFT Nifty overnight indicator ----
    # GIFT Nifty trades near-24h; its Open on day t is observable BEFORE the NSE
    # Nifty open on day t -> legitimate pre-open (contemporaneous, not lagged) signal.
    df['gift_gap_indicator'] = np.log(df['GIFT_Open'] / df['prev_C'])
    df['gift_range_pct'] = (df['GIFT_High'] - df['GIFT_Low']) / df['GIFT_Open']
    df['gift_own_change_lag1'] = (df['GIFT_Close'] / df['GIFT_Close'].shift(1) - 1).shift(1)

    # ---- NEW: US/SGX overnight moves (already dated as the overnight session before day t) ----
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
    Xcontemp = df[contemp_feats].copy()  # pre-open info, used as-is (see note above)

    data = pd.concat([df[['Date', 'target_dir', 'target_dH', 'target_dL', 'target_gap', 'daily_return']],
                       Xlag, Xcontemp], axis=1)
    data = data.dropna().reset_index(drop=True)
    feat_cols = list(Xlag.columns) + contemp_feats
    return data, feat_cols


# ----------------------------------------------------------------------------
# 3. LSTM sequence builder
# ----------------------------------------------------------------------------
def build_sequences(data, feat_cols, seq_len):
    """Row i's sequence = feat_cols values for rows [i-seq_len+1 .. i] -> predicts target at row i."""
    X = data[feat_cols].values.astype(float)
    # standardize globally is unsafe (leakage); standardize per-fold instead -> done inside run_fold
    n = len(data)
    seqs = np.full((n, seq_len, len(feat_cols)), np.nan)
    for i in range(seq_len - 1, n):
        seqs[i] = X[i - seq_len + 1:i + 1]
    valid_from = seq_len - 1
    return seqs, valid_from


# ----------------------------------------------------------------------------
# 4. Purged walk-forward CV
# ----------------------------------------------------------------------------
def run_walkforward(data, feat_cols, cfg):
    seqs, valid_from = build_sequences(data, feat_cols, cfg['seq_len'])
    n = len(data)
    train_win, test_win, purge, seq_len = cfg['train_window'], cfg['test_window'], cfg['purge'], cfg['seq_len']

    results = []
    start = valid_from  # need full sequence history before first train row
    fold = 0
    while start + train_win + purge + test_win <= n:
        tr_idx = np.arange(start, start + train_win)
        te_idx = np.arange(start + train_win + purge, start + train_win + purge + test_win)

        Xtr, Xte = data.iloc[tr_idx][feat_cols].values, data.iloc[te_idx][feat_cols].values
        y_dir_tr, y_dir_te = data.iloc[tr_idx]['target_dir'].values, data.iloc[te_idx]['target_dir'].values
        y_dH_tr, y_dH_te = data.iloc[tr_idx]['target_dH'].values, data.iloc[te_idx]['target_dH'].values
        y_dL_tr, y_dL_te = data.iloc[tr_idx]['target_dL'].values, data.iloc[te_idx]['target_dL'].values
        y_gap_tr, y_gap_te = data.iloc[tr_idx]['target_gap'].values, data.iloc[te_idx]['target_gap'].values

        # ---- GARCH(1,1) fit on TRAIN daily returns only, feed conditional variance as a feature ----
        r_pct_tr = data.iloc[tr_idx]['daily_return'].values * 100
        omega, alpha, beta = fit_garch11(r_pct_tr)
        sigma2_tr = garch_sigma2_series(r_pct_tr, omega, alpha, beta)
        # roll forward through test period recursively using realized returns (available at t-1, no leakage)
        r_pct_te = data.iloc[te_idx]['daily_return'].values * 100
        sigma2_te = np.zeros(len(te_idx))
        last_sigma2, last_r = sigma2_tr[-1], r_pct_tr[-1]
        for i in range(len(te_idx)):
            s2 = garch_one_step_forecast(last_r, last_sigma2, omega, alpha, beta)
            sigma2_te[i] = s2
            last_sigma2, last_r = s2, r_pct_te[i]

        Xtr_garch = np.column_stack([Xtr, sigma2_tr])
        Xte_garch = np.column_stack([Xte, sigma2_te])

        # ---- Direction classifier (GBM, w/ GIFT+US contemporaneous features already in feat_cols) ----
        clf, clf_params, backend = fit_best_gbm(Xtr, y_dir_tr, task='clf')
        proba = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y_dir_te, proba) if len(np.unique(y_dir_te)) > 1 else np.nan
        dir_acc = accuracy_score(y_dir_te, (proba > 0.5).astype(int))

        # ---- Range regressors WITH explicit GARCH sigma2 feature ----
        regH, _, _ = fit_best_gbm(Xtr_garch, y_dH_tr, task='reg')
        pred_H = regH.predict(Xte_garch)
        rmse_H = np.sqrt(mean_squared_error(y_dH_te, pred_H))
        rmse_H_base = np.sqrt(mean_squared_error(y_dH_te, np.full(len(y_dH_te), y_dH_tr.mean())))

        regL, _, _ = fit_best_gbm(Xtr_garch, y_dL_tr, task='reg')
        pred_L = regL.predict(Xte_garch)
        rmse_L = np.sqrt(mean_squared_error(y_dL_te, pred_L))
        rmse_L_base = np.sqrt(mean_squared_error(y_dL_te, np.full(len(y_dL_te), y_dL_tr.mean())))

        # ---- Open (gap) regressor -- the new target directly using GIFT + US overnight features ----
        regGap, _, _ = fit_best_gbm(Xtr, y_gap_tr, task='reg')
        pred_gap = regGap.predict(Xte)
        rmse_gap = np.sqrt(mean_squared_error(y_gap_te, pred_gap))
        rmse_gap_base = np.sqrt(mean_squared_error(y_gap_te, np.full(len(y_gap_te), y_gap_tr.mean())))

        # ---- LSTM on 30-day sequence windows (direction + dH), standardized per-fold ----
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


# ----------------------------------------------------------------------------
# 5. Final "tomorrow" forecast using ALL available history
# ----------------------------------------------------------------------------
def final_forecast(df, data, feat_cols, cfg):
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

    # ---- Convert log-return targets back into actual price levels ----
    last_close = df['C'].iloc[-1]
    last_date = df['Date'].iloc[-1].date()

    O_central = last_close * np.exp(pred_gap)
    H_central = O_central * np.exp(pred_dH)
    L_central = O_central * np.exp(pred_dL)

    # Expected Close: probability-weighted average of typical up-day / down-day intraday moves
    # (computed from the trailing 250 trading days of actual intraday returns)
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


def print_forecast_table(fc):
    print(f"\nAs of {fc['last_date']} (last Close = {fc['last_close']:,.1f}), forecast for the next session:\n")
    print(f"{'Open':<8}{fc['Open']:>12,.1f}")
    print(f"{'High':<8}{fc['High']:>12,.1f}")
    print(f"{'Low':<8}{fc['Low']:>12,.1f}")
    print(f"{'Close':<8}{fc['Close']:>12,.1f}   (P(Close>Open) = {fc['prob_up']:.1%})")
    lo1, hi1 = fc['Close_1sigma_band']
    lo2, hi2 = fc['Close_2sigma_band']
    print(f"\nGARCH(1,1) next-day volatility: {fc['garch_sigma_next_pct']:.3f}%")
    print(f"Close  ±1σ band: {lo1:,.0f} - {hi1:,.0f}")
    print(f"Close  ±2σ band: {lo2:,.0f} - {hi2:,.0f}")
    print(f"\n(GBM backend used: {fc['backend']})")
    print("NOTE: Open/High/Low/Close above use TODAY's GIFT Nifty & US overnight columns as a stand-in "
          "for tomorrow's (not yet available in the uploaded files) -- re-run once tomorrow's pre-open "
          "GIFT print and the actual overnight US/SGX session are in the data.")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    cfg = CONFIG
    print(f"[gbm backend] {backend_name()}")

    df = load_and_merge(cfg)
    data, feat_cols = engineer_features(df)
    print(f"[features] {len(feat_cols)} features, {len(data)} usable rows, "
          f"{data['Date'].min().date()} -> {data['Date'].max().date()}")

    cv_results = run_walkforward(data, feat_cols, cfg)
    cv_results.to_csv('Nifty_WalkForward_CV_Results_v2.csv', index=False)

    print("\n=== Aggregate (GBM tabular) ===")
    print("Mean AUC:", cv_results['auc'].mean(), " | Mean dir acc:", cv_results['dir_acc'].mean())
    print("Mean RMSE gap (model vs base):", cv_results['rmse_gap'].mean(), cv_results['rmse_gap_base'].mean())
    print("Mean RMSE dH  (model vs base):", cv_results['rmse_H'].mean(), cv_results['rmse_H_base'].mean())
    print("Mean RMSE dL  (model vs base):", cv_results['rmse_L'].mean(), cv_results['rmse_L_base'].mean())

    print("\n=== Aggregate (numpy LSTM, 30-day sequences) ===")
    print("Mean AUC:", cv_results['lstm_auc'].mean(), " | Mean dir acc:", cv_results['lstm_dir_acc'].mean())
    print("Mean RMSE dH:", cv_results['lstm_rmse_H'].mean())

    fc = final_forecast(df, data, feat_cols, cfg)
    print("\n=== Tomorrow's forecast (final models trained on full history) ===")
    print_forecast_table(fc)
