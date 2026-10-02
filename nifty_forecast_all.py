"""
nifty_forecast.py
==================
Live next-session forecast pipeline for the Nifty 50 index.

Given historical OHLC/indicator data (Nifty, India VIX, US overnight
markets, GIFT Nifty futures) up to a cutoff date, this script:

  1. Loads and merges all four data sources into one daily feature table.
  2. Fits (or loads a previously saved) forecasting model set:
       - a GBM classifier for next-session direction (Close > Open),
       - GBM regressors for the next session's High/Low/Open-gap,
       - a GARCH(1,1) volatility model for the confidence bands.
  3. Looks up the next trading session's actual 09:15-09:45 print from a
     separate "opening session" workbook, folds it into the dataset (so
     the historical record reflects the real print instead of the
     GIFT-Nifty/US-overnight stand-in used pre-open), and produces a
     forecast for that session's Close, anchored to the real Open.
  4. Prints and saves the forecast.

Run with:
    python3 nifty_forecast.py --cutoff-date YYYY-MM-DD

Or to just train once and cache the model for later reuse:
    python3 nifty_forecast.py --cutoff-date YYYY-MM-DD --train-and-save

This is a single, de-duplicated script consolidating the two near-identical
source files it replaces (niftyprediction.py and pred1.py). Everything
related to historical backtesting (the N-day backtest report, the
purged walk-forward cross-validation, and the numpy-only LSTM benchmark
used solely for that CV) has been removed -- this script only ever
produces a single, live, next-session forecast.

Universe extension (NSE/BSE indices + NIFTY 50 stocks)
--------------------------------------------------------
The pipeline above (load_and_merge -> engineer_features -> fit/forecast ->
fold in the real opening print) is entity-agnostic: it only ever reads
cfg['nifty_xlsx'] (whatever entity's OHLC+indicators workbook that points to)
and folds in the SAME shared macro data (India VIX, US overnight features,
GIFT Nifty) used for NIFTY 50 itself, since those describe market-wide
conditions rather than anything NIFTY-50-specific.

--all runs this exact pipeline once per entity in entities_universe.py
(every major NSE/BSE index + every NIFTY 50 stock + NIFTY 50 itself), each
entity getting its own trained model, its own <STEM>_LiveForecast_<date>.csv,
and a combined All_Entities_LiveForecast_<date>.csv summary at the end. One
entity failing (e.g. its opening-session workbook hasn't been downloaded, or
doesn't have enough history yet) is caught and logged -- it does not stop the
rest of the universe from being forecast.
"""
import argparse
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, mean_squared_error

from entities_universe import NIFTY_50_LABEL, all_forecast_entities, file_stem

warnings.filterwarnings('ignore', category=FutureWarning)

# ANSI color codes used for the console forecast summary
RED = "\033[91m"
RESET = "\033[0m"


# =============================================================================
# GARCH(1,1) volatility model
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
# Gradient-boosting model factory (xgboost -> lightgbm -> sklearn fallback)
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
            "xgboost/lightgbm not installed -- falling back to sklearn HistGradientBoosting, "
            "LightGBM's closest algorithmic cousin in scikit-learn. Install xgboost/lightgbm "
            "and this module will use them automatically, no code changes needed.", RuntimeWarning)
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
    """Small time-respecting hyperparameter search (last 20% of the training fold held
    out, in time order, as validation) + final fit on the full training fold."""
    backend = backend_name()
    n = len(X_train)
    cut = int(n * 0.8)
    Xtr, Xval = X_train[:cut], X_train[cut:]
    ytr, yval = y_train[:cut], y_train[cut:]

    best_score, best_params = (-np.inf, PARAM_GRID[0]) if task == 'clf' else (np.inf, PARAM_GRID[0])
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
# Configuration
# =============================================================================
DEFAULT_CONFIG = dict(
    nifty_xlsx='Nifty_OHLC_Direction_Indicators.xlsx',
    vix_xlsx='INDIAVIX_OHLC_Direction_Indicators.xlsx',
    us_csv='us_overnight_market_features.csv',
    gift_csv='Gift_Nifty_50_Futures_Historical_Data.csv',
    end_date=None,  # <- set from --cutoff-date on the command line
)

# Where the one-time-trained forecast model set (clf + regH/regL/regGap + GARCH
# params) gets pickled to by --train-and-save, and loaded from on later runs.
DEFAULT_MODEL_FILE = 'nifty_trained_model.pkl'


# =============================================================================
# Data loading and feature engineering
# =============================================================================
def load_and_merge(cfg):
    nifty = pd.read_excel(cfg['nifty_xlsx'], sheet_name='OHLC Data')
    nifty['Date'] = pd.to_datetime(nifty['Date'])
    nifty = nifty.sort_values('Date').reset_index(drop=True)
    nifty = nifty.rename(columns={'Open': 'O', 'High': 'H', 'Low': 'L', 'Close': 'C', 'Volume': 'V'})

    vix = pd.read_excel(cfg['vix_xlsx'], sheet_name='OHLC Data')
    vix['Date'] = pd.to_datetime(vix['Date'])
    vix = vix[['Date', 'Close']].rename(columns={'Close': 'VIX_Close'})

    us = pd.read_csv(cfg['us_csv'])
    us['Date'] = pd.to_datetime(us['Date'], dayfirst=True)

    gift = pd.read_csv(cfg['gift_csv'])
    gift['Date'] = pd.to_datetime(gift['Date'], format='mixed', dayfirst=True)
    for c in ['Price', 'Open', 'High', 'Low']:
        gift[c] = gift[c].astype(str).str.replace(',', '').astype(float)
    gift = gift.sort_values('Date').reset_index(drop=True)
    gift = gift.rename(columns={'Price': 'GIFT_Close', 'Open': 'GIFT_Open',
                                 'High': 'GIFT_High', 'Low': 'GIFT_Low'})
    gift = gift[['Date', 'GIFT_Open', 'GIFT_High', 'GIFT_Low', 'GIFT_Close']]

    common_start = max(nifty['Date'].min(), vix['Date'].min(), us['Date'].min(), gift['Date'].min())
    print(f"[load] Restricting all data to >= {common_start.date()} "
          f"(first date common to all 4 sources)")

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


# =============================================================================
# Model training and forecasting
# =============================================================================
def _fit_forecast_bundle(df, data, feat_cols):
    """One-time TRAIN step: fits clf/regH/regL/regGap + GARCH(1,1) on ALL rows of
    `data` (i.e. the full history up to the cutoff date). Pulled out into its own
    function so it can be done once and pickled (see train_and_save_forecast_model()),
    instead of being repeated every time a forecast is needed."""
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

    intraday = (np.log(df['C'] / df['O'])).tail(250)
    up_avg = intraday[intraday > 0].mean()
    down_avg = intraday[intraday <= 0].mean()

    return dict(
        clf=clf, regH=regH, regL=regL, regGap=regGap,
        feat_cols=feat_cols, backend=backend,
        garch_params=(omega, alpha, beta),
        last_sigma2_next=last_sigma2_next,
        last_row=last_row,  # pre-open feature row as of last_date (fallback)
        last_close=df['C'].iloc[-1],
        last_date=df['Date'].iloc[-1].date(),
        up_avg=up_avg, down_avg=down_avg,
        train_end_date=df['Date'].iloc[-1].date(),
    )


def _forecast_from_bundle(bundle, feature_row=None, actual_open=None):
    """PREDICT step: produces the next-session forecast dict from an already-fitted
    bundle -- no retraining happens here, so this is cheap enough to run live.

    feature_row : optional 1 x n_features array to use INSTEAD of the bundle's
        stored last_row (e.g. once the forecast day's real pre-open GIFT/US
        columns are available). Defaults to the bundle's stored row.
    actual_open : optional known/actual session Open (e.g. the real 09:15-09:45
        print). When given, this REPLACES the model's own predicted Open
        (anchor_close * exp(pred_gap)) as the base that High/Low/Close are
        built from, so the forecast is anchored to the real print instead of
        a modeled gap.
    """
    row = bundle['last_row'] if feature_row is None else feature_row
    omega, alpha, beta = bundle['garch_params']

    prob_up = bundle['clf'].predict_proba(row)[0, 1]
    row_garch = np.column_stack([row, [bundle['last_sigma2_next']]])
    pred_dH = bundle['regH'].predict(row_garch)[0]
    pred_dL = bundle['regL'].predict(row_garch)[0]
    pred_gap = bundle['regGap'].predict(row)[0]  # only used when actual_open isn't supplied

    last_close = bundle['last_close']
    last_date = bundle['last_date']

    O_central = last_close * np.exp(pred_gap) if actual_open is None else actual_open
    H_central = O_central * np.exp(pred_dH)
    L_central = O_central * np.exp(pred_dL)

    exp_intraday_ret = prob_up * bundle['up_avg'] + (1 - prob_up) * bundle['down_avg']
    C_central = O_central * np.exp(exp_intraday_ret)

    sigma_next_pct = np.sqrt(bundle['last_sigma2_next'])
    sigma = sigma_next_pct / 100

    return dict(
        last_date=last_date, last_close=last_close, original_close=last_close,
        prob_up=prob_up, pred_dH=pred_dH, pred_dL=pred_dL, pred_gap=pred_gap,
        garch_sigma_next_pct=sigma_next_pct, backend=bundle['backend'], garch_params=(omega, alpha, beta),
        Open=O_central, High=H_central, Low=L_central, Close=C_central,
        Close_1sigma_band=(last_close * (1 - sigma), last_close * (1 + sigma)),
        Close_2sigma_band=(last_close * (1 - 2 * sigma), last_close * (1 + 2 * sigma)),
    )


def train_and_save_forecast_model(cfg, model_path=DEFAULT_MODEL_FILE):
    """One-time training step: loads data up to cfg['end_date'] (the cutoff), fits the
    forecast model set, and pickles it to `model_path`. A later run can then call
    load_forecast_model() + _forecast_from_bundle() to predict a future session's
    close WITHOUT retraining -- only a live pre-open OHLC print is needed at predict
    time."""
    print(f"[train] Loading data and training the forecast model set (cutoff={cfg.get('end_date')}) ...")
    df = load_and_merge(cfg)
    data, feat_cols = engineer_features(df)
    print(f"[train] {len(feat_cols)} features, {len(data)} usable rows, "
          f"{data['Date'].min().date()} -> {data['Date'].max().date()}")
    bundle = _fit_forecast_bundle(df, data, feat_cols)
    with open(model_path, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"[train] Model trained on data through {bundle['last_date']} "
          f"(GBM backend: {bundle['backend']}) and saved -> {model_path}")
    return bundle


def load_forecast_model(model_path=DEFAULT_MODEL_FILE):
    """Loads a model bundle previously saved by train_and_save_forecast_model()."""
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    print(f"[model] Loaded saved model (trained through {bundle['last_date']}) from {model_path}")
    return bundle


# =============================================================================
# Live next-session workflow
# =============================================================================
def _find_opening_session_header_row(opening_xlsx):
    raw = pd.read_excel(opening_xlsx, header=None, nrows=10)
    for i in range(len(raw)):
        if str(raw.iloc[i, 0]).strip().lower() == 'date':
            return i
    raise ValueError(f"Could not find a 'Date' header row in {opening_xlsx} (checked first 10 rows).")


def find_next_session(opening_xlsx, cutoff_date):
    """Looks up the first recorded 09:15-09:45 opening print after cutoff_date in the
    opening-session tracking workbook."""
    header_row = _find_opening_session_header_row(opening_xlsx)
    sess = pd.read_excel(opening_xlsx, header=header_row)
    sess.columns = [str(c).strip() for c in sess.columns]
    sess['Date'] = pd.to_datetime(sess['Date'])
    sess = sess.sort_values('Date').reset_index(drop=True)

    cutoff_ts = pd.Timestamp(cutoff_date)
    nxt = sess[sess['Date'] == cutoff_ts]
    if nxt.empty:
        raise ValueError(f"No session after {cutoff_ts.date()} found in {opening_xlsx} yet -- "
                          f"the next trading day's 09:45 print hasn't been added to the workbook.")
    row = nxt.iloc[0]
    return dict(
        date=row['Date'],
        Open=float(row['Open (09:15)']),
        High=float(row['High (09:15-09:45)']),
        Low=float(row['Low (09:15-09:45)']),
        Close=float(row['Close (09:45)']),
    )


def upsert_session_row(nifty_xlsx, session):
    """Folds the next session's real 09:15-09:45 print into the main Nifty workbook,
    keyed on the SESSION'S OWN date: updates that row if it already exists (e.g. a
    prior placeholder), or appends it as a new row otherwise. This never touches a
    different day's row, so a completed trading day's real OHLC is never overwritten.

    NOTE: this permanently writes to `nifty_xlsx` on disk. Keep a backup of the
    workbook if you want to preserve its prior state.
    """
    sheets = pd.read_excel(nifty_xlsx, sheet_name=None)
    ohlc = sheets['OHLC Data']
    ohlc['Date'] = pd.to_datetime(ohlc['Date'])
    session_ts = pd.Timestamp(session['date'])

    match = ohlc.index[ohlc['Date'] == session_ts]
    if len(match):
        idx = match[0]
        ohlc.loc[idx, ['Open', 'High', 'Low', 'Close']] = [
            session['Open'], session['High'], session['Low'], session['Close']]
        action = 'Updated'
    else:
        new_row = {col: np.nan for col in ohlc.columns}
        new_row.update(Date=session_ts, Open=session['Open'], High=session['High'],
                        Low=session['Low'], Close=session['Close'])
        ohlc = pd.concat([ohlc, pd.DataFrame([new_row])], ignore_index=True)
        action = 'Added'
    sheets['OHLC Data'] = ohlc.sort_values('Date').reset_index(drop=True)

    with pd.ExcelWriter(nifty_xlsx, engine='openpyxl') as writer:
        for name, sdf in sheets.items():
            sdf.to_excel(writer, sheet_name=name, index=False)

    print(f"[data] {action} {session_ts.date()} row in {nifty_xlsx} with its 09:45 print: "
          f"O={session['Open']:.2f} H={session['High']:.2f} L={session['Low']:.2f} C={session['Close']:.2f}")


def build_cfg_for_entity(label, cutoff_date):
    """Returns (cfg, opening_xlsx, model_path) for one entity, reusing the same
    shared macro files (VIX/US/GIFT) as NIFTY 50 but pointing nifty_xlsx /
    opening_xlsx / model_path at that entity's own files. NIFTY 50 itself keeps
    its original filenames; every other entity uses entities_universe.file_stem()."""
    if label == NIFTY_50_LABEL:
        nifty_xlsx = DEFAULT_CONFIG['nifty_xlsx']
        opening_xlsx = 'nifty_opening_session_60d.xlsx'
        model_path = DEFAULT_MODEL_FILE
    else:
        stem = file_stem(label)
        nifty_xlsx = f"{stem}_OHLC_Direction_Indicators.xlsx"
        opening_xlsx = f"{stem}_opening_session_60d.xlsx"
        model_path = f"{stem}_trained_model.pkl"

    cfg = dict(DEFAULT_CONFIG)
    cfg.update(nifty_xlsx=nifty_xlsx, end_date=cutoff_date)
    return cfg, opening_xlsx, model_path


def run_all_entities(cutoff_date, train_and_save=False, entities=None):
    """Runs the full live-forecast pipeline once per entity (every NSE/BSE index +
    every NIFTY 50 stock + NIFTY 50 itself, unless `entities` restricts it to a
    subset of labels). Returns {label: forecast_dict} for every entity that
    succeeded, and also writes a combined All_Entities_LiveForecast_<date>.csv."""
    universe = all_forecast_entities()
    if entities:
        universe = {label: t for label, t in universe.items() if label in entities}

    all_results = {}
    forecast_rows = []
    for label, ticker in universe.items():
        stem = file_stem(label) if label != NIFTY_50_LABEL else 'NIFTY'
        cfg, opening_xlsx, model_path = build_cfg_for_entity(label, cutoff_date)
        print(f"\n{'=' * 70}\n[{label}]  ({ticker})\n{'=' * 70}")
        try:
            if train_and_save:
                train_and_save_forecast_model(cfg, model_path=model_path)
                continue

            fc = run_live_next_session(cfg, cutoff_date, model_path, opening_xlsx)
            print(f"\n=== Forecast for {fc['target_date']} "
                  f"(next session after cutoff {cutoff_date}) ===")
            print_forecast_table(fc)

            out_csv = f"{stem}_LiveForecast_{fc['target_date']}.csv"
            save_forecast_csv(fc, out_csv)
            print(f"\n[saved] {out_csv}")

            all_results[label] = fc
            ref_close = fc.get('original_close', fc['last_close'])
            forecast_rows.append(dict(
                Entity=label, Ticker=ticker, Target_Date=fc['target_date'],
                Ref_Close=ref_close,
                Forecast_Open=fc['Open'], Forecast_High=fc['High'],
                Forecast_Low=fc['Low'], Forecast_Close=fc['Close'],
                Prob_Up=fc['prob_up'],
                GARCH_Sigma_Next_Pct=fc['garch_sigma_next_pct'],
                GBM_Backend=fc['backend'],
            ))
        except Exception as e:
            print(f"[FAILED] {label} ({ticker}): {type(e).__name__}: {e}")
            continue

    if forecast_rows and not train_and_save:
        summary_df = pd.DataFrame(forecast_rows)
        summary_path = f"All_Entities_LiveForecast_{cutoff_date}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'=' * 70}\n[saved] Combined summary for {len(forecast_rows)}/"
              f"{len(universe)} entities -> {summary_path}\n{'=' * 70}")

    return all_results


def run_live_next_session(cfg, cutoff_date, model_path, opening_xlsx):
    """Full live workflow:
      1. Train (or load a cached) forecast model on data through cutoff_date.
      2. Keep cutoff_date's close for comparison (this is unaffected by step 3).
      3. Look up the next session's real 09:45 print and fold it into the workbook.
      4. Predict the session Close anchored to the real Open and compare vs cutoff_date's close.
    """
    try:
        bundle = load_forecast_model(model_path)
    except FileNotFoundError:
        print(f"[model] {model_path} not found -- training fresh on data through {cutoff_date} instead.")
        cfg_train = dict(cfg, end_date=cutoff_date)
        df = load_and_merge(cfg_train)
        data, feat_cols = engineer_features(df)
        bundle = _fit_forecast_bundle(df, data, feat_cols)

    original_close = bundle['last_close']

    session = find_next_session(opening_xlsx, cutoff_date)
    upsert_session_row(cfg['nifty_xlsx'], session)

    fc = _forecast_from_bundle(bundle, actual_open=session['Open'])
    fc['original_close'] = original_close
    fc['target_date'] = session['date'].date()
    return fc


# =============================================================================
# Reporting
# =============================================================================
def tompred(pred_close, ref_close, pred_high, pred_low):
    if pred_close > ref_close:
        print(f"{RED}Next session's closing price is likely to be above the reference close, "
              f"with a high around {pred_high:,.1f}{RESET}")
    elif pred_close < ref_close:
        print(f"{RED}Next session's closing price is likely to be below the reference close, "
              f"with a low around {pred_low:,.1f}{RESET}")
    else:
        print(f"{RED}Next session's closing price is expected to be roughly flat vs. the reference close.{RESET}")


def print_forecast_table(fc):
    ref_close = fc.get('original_close', fc['last_close'])
    lo2, hi2 = fc['Close_2sigma_band']
    print(f"{RED}\nAs of {fc['last_date']} (reference Close = {ref_close:,.1f}), forecast for the next session:\n{RESET}")
    print(f"{RED}{'Close':<8}{fc['Close']:>12,.1f}   Probability of Green Candle = {fc['prob_up']:.1%}{RESET}")
    tompred(fc['Close'], ref_close, fc['High'], fc['Low'])
    print(f"{RED}\nGARCH(1,1) next-day volatility: {fc['garch_sigma_next_pct']:.3f}%{RESET}")
    print(f"{RED}Predicted range for the next session: {lo2:,.0f} - {hi2:,.0f}{RESET}")
    print(f"\n(GBM backend used: {fc['backend']})")


def save_forecast_csv(fc, out_path):
    lo1, hi1 = fc['Close_1sigma_band']
    lo2, hi2 = fc['Close_2sigma_band']
    omega, alpha, beta = fc['garch_params']
    ref_close = fc.get('original_close', fc['last_close'])
    row = dict(
        last_date=fc['last_date'], last_close=ref_close,
        forecast_Open=fc['Open'], forecast_High=fc['High'], forecast_Low=fc['Low'], forecast_Close=fc['Close'],
        prob_close_gt_open=fc['prob_up'], garch_sigma_next_pct=fc['garch_sigma_next_pct'],
        garch_omega=omega, garch_alpha=alpha, garch_beta=beta,
        Close_1sigma_low=lo1, Close_1sigma_high=hi1, Close_2sigma_low=lo2, Close_2sigma_high=hi2,
        gbm_backend=fc['backend'],
    )
    pd.DataFrame([row]).to_csv(out_path, index=False)


# =============================================================================
# CLI entry point
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch the next Nifty session's live 09:15-09:45 print, fold it into the "
                    "dataset, and forecast that session's Close (probability up, predicted OHLC, "
                    "GARCH volatility bands).")
    p.add_argument('cutoff_date_pos', nargs='?', default=None, metavar='CUTOFF_DATE',
                    help="Cutoff date (YYYY-MM-DD), given positionally, e.g. "
                         "'nifty_forecast.py 2026-08-17'. Equivalent to --cutoff-date; "
                         "if both are given, --cutoff-date wins.")
    p.add_argument('--cutoff-date', '-c', default=None,
                    help="Cutoff date (YYYY-MM-DD). The model is trained on data <= this date; "
                         "the forecast is for the first session found after it. Same effect as "
                         "the positional CUTOFF_DATE argument.")
    p.add_argument('--nifty-xlsx', default=DEFAULT_CONFIG['nifty_xlsx'])
    p.add_argument('--vix-xlsx', default=DEFAULT_CONFIG['vix_xlsx'])
    p.add_argument('--us-csv', default=DEFAULT_CONFIG['us_csv'])
    p.add_argument('--gift-csv', default=DEFAULT_CONFIG['gift_csv'])
    p.add_argument('--opening-session-xlsx', default='nifty_opening_session_60d.xlsx',
                    help="Workbook tracking each session's real 09:15-09:45 opening print.")
    p.add_argument('--model-file', default=DEFAULT_MODEL_FILE,
                    help=f"Path to load a cached forecast model from, or (with --train-and-save) "
                         f"to save one to (default: {DEFAULT_MODEL_FILE}).")
    p.add_argument('--train-and-save', action='store_true',
                    help="Train the forecast model set ONE TIME on data up to --cutoff-date and "
                         "pickle it to --model-file, instead of running the live forecast.")
    p.add_argument('--all', action='store_true',
                    help="Run the live forecast for EVERY entity in entities_universe.py "
                         "(all major NSE/BSE indices + all NIFTY 50 stocks + NIFTY 50 itself), "
                         "instead of just NIFTY 50. Ignores --nifty-xlsx/--opening-session-xlsx/"
                         "--model-file (each entity uses its own files).")
    p.add_argument('--only', default=None,
                    help="With --all, comma-separated list of entity labels to restrict the "
                         "run to, e.g. --only 'NIFTY BANK,Reliance Industries'.")
    args = p.parse_args()

    args.cutoff_date = args.cutoff_date or args.cutoff_date_pos
    if not args.cutoff_date:
        p.error("a cutoff date is required, either positionally (e.g. 'nifty_forecast.py "
                 "2026-08-17') or via --cutoff-date/-c")
    return args


def main(cutoff_date=None, nifty_xlsx=None, vix_xlsx=None, us_csv=None, gift_csv=None,
         opening_session_xlsx=None, model_file=None, train_and_save=False,
         all_entities=False, only=None):
    """Runs the forecast pipeline.

    Called with no arguments, this parses the command line as usual (see parse_args()).
    Called with `cutoff_date` set directly, it skips argparse entirely -- useful for
    importing this module and calling main("2026-08-17") from other code. Any other
    parameter left as None falls back to DEFAULT_CONFIG / DEFAULT_MODEL_FILE.

    Examples:
        main()                          # normal CLI usage: reads sys.argv
        main("2026-08-17")              # direct call, single entity (NIFTY 50), all other settings default
        main("2026-08-17", train_and_save=True, model_file="my_model.pkl")
        main("2026-08-17", all_entities=True)   # forecast every NSE/BSE index + NIFTY 50 stock
    """
    if cutoff_date is None:
        args = parse_args()
        cutoff_date = args.cutoff_date
        nifty_xlsx = args.nifty_xlsx
        vix_xlsx = args.vix_xlsx
        us_csv = args.us_csv
        gift_csv = args.gift_csv
        opening_session_xlsx = args.opening_session_xlsx
        model_file = args.model_file
        train_and_save = args.train_and_save
        all_entities = args.all
        only = [s.strip() for s in args.only.split(",")] if args.only else None
    else:
        nifty_xlsx = nifty_xlsx or DEFAULT_CONFIG['nifty_xlsx']
        vix_xlsx = vix_xlsx or DEFAULT_CONFIG['vix_xlsx']
        us_csv = us_csv or DEFAULT_CONFIG['us_csv']
        gift_csv = gift_csv or DEFAULT_CONFIG['gift_csv']
        opening_session_xlsx = opening_session_xlsx or 'nifty_opening_session_60d.xlsx'
        model_file = model_file or DEFAULT_MODEL_FILE

    print(f"[gbm backend] {backend_name()}")

    if all_entities:
        run_all_entities(cutoff_date, train_and_save=train_and_save, entities=only)
        return

    cfg = dict(DEFAULT_CONFIG)
    cfg.update(nifty_xlsx=nifty_xlsx, vix_xlsx=vix_xlsx, us_csv=us_csv,
               gift_csv=gift_csv, end_date=cutoff_date)

    if train_and_save:
        train_and_save_forecast_model(cfg, model_path=model_file)
        return

    fc = run_live_next_session(cfg, cutoff_date, model_file, opening_session_xlsx)
    print(f"\n=== Forecast for {fc['target_date']} (next session after cutoff {cutoff_date}) ===")
    print_forecast_table(fc)

    out_csv = f"Nifty_LiveForecast_{fc['target_date']}.csv"
    save_forecast_csv(fc, out_csv)
    print(f"\n[saved] {out_csv}")

#%run nifty_orb_retest_strategy.py --backtest
if __name__ == '__main__':
    # NOTE: all_entities=True forecasts every NSE/BSE index + every NIFTY 50
    # stock + NIFTY 50 itself (see entities_universe.py), each needing its
    # download_all_data.py files already present. Set all_entities=False (or
    # just main("2026-08-10")) for the original NIFTY-50-only behavior.
    main("2026-08-03", all_entities=True)