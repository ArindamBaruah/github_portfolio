import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


# ==========================================
# 1. DATA INGESTION & ROBUST PARSING
# ==========================================
def parse_volume(vol_str):
    """Parses volume strings with K, M, B suffixes into floats."""
    if pd.isna(vol_str) or vol_str == "":
        return 0.0
    v = str(vol_str).strip().upper()
    if v.endswith("K"):
        return float(v[:-1].replace(",", "")) * 1e3
    elif v.endswith("M"):
        return float(v[:-1].replace(",", "")) * 1e6
    elif v.endswith("B"):
        return float(v[:-1].replace(",", "")) * 1e9
    try:
        return float(v.replace(",", ""))
    except ValueError:
        return 0.0


def load_and_prepare_dataset():
    """Loads and synchronizes NIFTY, INDIA VIX, GIFT NIFTY, and US overnight data."""
    required_files = [
        "Nifty_OHLC_Direction_Indicators.xlsx",
        "INDIAVIX_OHLC_Direction_Indicators.xlsx",
        "Gift_Nifty_50_Futures_Historical_Data.csv",
        "us_overnight_market_features.csv",
    ]
    for file in required_files:
        if not os.path.exists(file):
            sys.exit(f"Error: Required data file '{file}' not found in current directory.")

    # 1. Nifty Spot & Pre-computed Technical Indicators
    nifty_ohlc = pd.read_excel("Nifty_OHLC_Direction_Indicators.xlsx", sheet_name="OHLC Data")
    nifty_ind = pd.read_excel("Nifty_OHLC_Direction_Indicators.xlsx", sheet_name="Indicators")
    nifty_ohlc["Date"] = pd.to_datetime(nifty_ohlc["Date"])
    nifty_ind["Date"] = pd.to_datetime(nifty_ind["Date"])

    nifty_ind_renamed = nifty_ind.drop(
        columns=["Open", "High", "Low", "Close", "Volume"], errors="ignore"
    ).rename(
        columns=lambda c: f"NIFTY_{c}" if c != "Date" else c
    )

    # 2. India VIX
    vix_ohlc = pd.read_excel("INDIAVIX_OHLC_Direction_Indicators.xlsx", sheet_name="OHLC Data")
    vix_ind = pd.read_excel("INDIAVIX_OHLC_Direction_Indicators.xlsx", sheet_name="Indicators")
    vix_ohlc["Date"] = pd.to_datetime(vix_ohlc["Date"])
    vix_ind["Date"] = pd.to_datetime(vix_ind["Date"])

    vix_ohlc_renamed = vix_ohlc.rename(
        columns={
            "Open": "VIX_Open",
            "High": "VIX_High",
            "Low": "VIX_Low",
            "Close": "VIX_Close",
            "Volume": "VIX_Volume",
        }
    )
    vix_ind_renamed = vix_ind.drop(
        columns=["Open", "High", "Low", "Close", "Volume"], errors="ignore"
    ).rename(
        columns=lambda c: f"VIX_{c}" if c != "Date" else c
    )

    # 3. GIFT Nifty Futures
    gift_df = pd.read_csv("Gift_Nifty_50_Futures_Historical_Data.csv").dropna(subset=["Date"]).copy()
    gift_df["Date"] = pd.to_datetime(gift_df["Date"], format="mixed")
    for col in ["Price", "Open", "High", "Low"]:
        if col in gift_df.columns:
            gift_df[col] = gift_df[col].astype(str).str.replace(",", "").astype(float)
    gift_df["Change %"] = gift_df["Change %"].astype(str).str.replace("%", "").astype(float) / 100.0
    gift_df["Vol_parsed"] = gift_df["Vol."].apply(parse_volume)

    gift_renamed = gift_df[["Date", "Price", "Open", "High", "Low", "Vol_parsed", "Change %"]].rename(
        columns={
            "Price": "GIFT_Close",
            "Open": "GIFT_Open",
            "High": "GIFT_High",
            "Low": "GIFT_Low",
            "Vol_parsed": "GIFT_Volume",
            "Change %": "GIFT_Change_Pct",
        }
    )

    # 4. US Overnight Features
    us_df = pd.read_csv("us_overnight_market_features.csv")
    us_df["Date"] = pd.to_datetime(us_df["Date"])

    # Master Chronological Merge
    merged = nifty_ohlc.merge(nifty_ind_renamed, on="Date", how="left")
    merged = merged.merge(vix_ohlc_renamed, on="Date", how="left")
    merged = merged.merge(vix_ind_renamed, on="Date", how="left")
    merged = merged.merge(us_df, on="Date", how="left")
    merged = merged.merge(gift_renamed, on="Date", how="left")
    merged = merged.sort_values("Date").reset_index(drop=True)

    return merged


# ==========================================
# 2. FEATURE ENGINEERING (DIVERGENCE & REGIME)
# ==========================================
def engineer_features(df):
    """Extracts leading divergences, price-action rejection, and macro shock metrics."""
    data = df.copy()

    # Targets (T+1 Forward Returns & Directions)
    data["NIFTY_Fwd_Ret"] = data["Close"].shift(-1) / data["Close"] - 1.0
    data["NIFTY_Target_Dir"] = (data["NIFTY_Fwd_Ret"] > 0).astype(int)

    data["VIX_Fwd_Ret"] = data["VIX_Close"].shift(-1) / data["VIX_Close"] - 1.0
    data["VIX_Target_Dir"] = (data["VIX_Fwd_Ret"] > 0).astype(int)

    data["GIFT_Fwd_Ret"] = data["GIFT_Close"].shift(-1) / data["GIFT_Close"] - 1.0
    data["GIFT_Target_Dir"] = (data["GIFT_Fwd_Ret"] > 0).astype(int)

    # Momentum Windows
    for k in [1, 2, 3, 5, 10, 20]:
        data[f"NIFTY_Ret_{k}d"] = data["Close"].pct_change(k)
        data[f"VIX_Ret_{k}d"] = data["VIX_Close"].pct_change(k)
        data[f"GIFT_Ret_{k}d"] = data["GIFT_Close"].pct_change(k)

    # Leading Price Action & Candle Rejection Features
    candle_range = data["High"] - data["Low"] + 1e-6
    data["NIFTY_Body_to_Range"] = (data["Close"] - data["Open"]) / candle_range
    data["NIFTY_Upper_Wick"] = (data["High"] - np.maximum(data["Open"], data["Close"])) / candle_range
    data["NIFTY_Lower_Wick"] = (np.minimum(data["Open"], data["Close"]) - data["Low"]) / candle_range
    data["Close_Off_High"] = (data["Close"] - data["High"]) / candle_range
    data["NIFTY_Gap"] = (data["Open"] - data["Close"].shift(1)) / data["Close"].shift(1)
    data["Intraday_Ret"] = (data["Close"] - data["Open"]) / data["Open"]

    # Garman-Klass Volatility Estimators
    data["NIFTY_Garman_Klass"] = np.sqrt(
        0.5 * (np.log(data["High"] / data["Low"])) ** 2
        - (2 * np.log(2) - 1) * (np.log(data["Close"] / data["Open"])) ** 2
    )
    data["VIX_Garman_Klass"] = np.sqrt(
        0.5 * (np.log(data["VIX_High"] / data["VIX_Low"])) ** 2
        - (2 * np.log(2) - 1) * (np.log(data["VIX_Close"] / data["VIX_Open"])) ** 2
    )

    # Intermarket Spreads & Relative Strength
    data["Gift_Basis_Pct"] = (data["GIFT_Close"] - data["Close"]) / data["Close"]
    data["Gift_Momentum_Divergence"] = data["GIFT_Change_Pct"] - data["NIFTY_Ret_1d"]
    data["NIFTY_Vol_SMA20"] = data["Volume"] / (data["Volume"].rolling(20).mean() + 1e-6)
    data["GIFT_Vol_SMA20"] = data["GIFT_Volume"] / (data["GIFT_Volume"].rolling(20).mean() + 1e-6)

    # Overextension & Complacency Penalties
    data["Dist_from_EMA20"] = (data["Close"] - data["NIFTY_EMA20"]) / data["NIFTY_EMA20"]
    data["Dist_from_SMA20"] = (data["Close"] - data["NIFTY_SMA20"]) / data["NIFTY_SMA20"]
    data["RSI_Overbought"] = (data["NIFTY_RSI14"] - 60).clip(lower=0) / 40.0
    data["VIX_Complacency"] = (12.0 - data["VIX_Close"]).clip(lower=0)

    # Normalized Technical Oscillators
    data["NIFTY_RSI_norm"] = (data["NIFTY_RSI14"] - 50.0) / 50.0
    data["NIFTY_MACD_Hist_norm"] = data["NIFTY_MACD_Hist"] / data["Close"]
    data["NIFTY_SMA_Diff"] = (data["NIFTY_SMA20"] - data["NIFTY_SMA50"]) / data["Close"]
    data["NIFTY_EMA_Diff"] = (data["NIFTY_EMA20"] - data["NIFTY_EMA50"]) / data["Close"]
    data["NIFTY_ADX_Trend"] = (data["NIFTY_Plus_DI"] - data["NIFTY_Minus_DI"]) * (data["NIFTY_ADX14"] / 100.0)

    data["VIX_RSI_norm"] = (data["VIX_RSI14"] - 50.0) / 50.0
    data["VIX_SMA_Diff"] = (data["VIX_SMA20"] - data["VIX_SMA50"]) / (data["VIX_Close"] + 1e-6)
    data["VIX_EMA_Diff"] = (data["VIX_EMA20"] - data["VIX_EMA50"]) / (data["VIX_Close"] + 1e-6)
    data["VIX_ADX_Trend"] = (data["VIX_Plus_DI"] - data["VIX_Minus_DI"]) * (data["VIX_ADX14"] / 100.0)

    # Forward fill macro features
    data["SP500_Return"] = data["SP500_Return"].ffill().fillna(0)
    data["NASDAQ_Return"] = data["NASDAQ_Return"].ffill().fillna(0)
    data["Crude_Change"] = data["Crude_Change"].ffill().fillna(0)
    data["DXY_Change"] = data["DXY_Change"].ffill().fillna(0)
    data["US_VIX_Change"] = data["US_VIX_Change"].ffill().fillna(0)

    
    # Update exclude_cols inside engineer_features() in quant_direction_engine.py
    exclude_cols = [
        "Date", "Open", "High", "Low", "Close", "Volume",
        "VIX_Open", "VIX_High", "VIX_Low", "VIX_Close", "VIX_Volume",
        "GIFT_Close", "GIFT_Open", "GIFT_High", "GIFT_Low", "GIFT_Volume",
        "NIFTY_Fwd_Ret", "NIFTY_Target_Dir",
        "VIX_Fwd_Ret", "VIX_Target_Dir",
        "GIFT_Fwd_Ret", "GIFT_Target_Dir",
        "NIFTY_SMA_Direction", "NIFTY_EMA_Direction", "NIFTY_MACD_Direction",
        "NIFTY_RSI_Direction", "NIFTY_ADX_Direction",
        "VIX_SMA_Direction", "VIX_EMA_Direction", "VIX_MACD_Direction",
        "VIX_RSI_Direction", "VIX_ADX_Direction",
        
        # --- ADD THESE NON-STATIONARY RAW PRICE LEVELS TO EXCLUDE ---
        "NIFTY_SMA20", "NIFTY_SMA50", "NIFTY_EMA20", "NIFTY_EMA50",
        "VIX_SMA20", "VIX_SMA50", "VIX_EMA20", "VIX_EMA50",
        "NIFTY_MACD", "NIFTY_MACD_Signal", "NIFTY_MACD_Hist",
        "VIX_MACD", "VIX_MACD_Signal", "VIX_MACD_Hist",
        "NIFTY_Plus_DI", "NIFTY_Minus_DI", "NIFTY_ADX14",
        "VIX_Plus_DI", "VIX_Minus_DI", "VIX_ADX14",
        "NIFTY_RSI14", "VIX_RSI14"
    ]
    feature_cols = [c for c in data.columns if c not in exclude_cols]
    return data, feature_cols


# ==========================================
# 3. QUANT PREDICTION & CONVICTION ENGINE
# ==========================================
def predict_market_direction(df, feature_cols, cutoff_date_str=None):
    """Trains on data strictly <= cutoff_date and forecasts next session movement."""
    # Warmup period (drop first 50 bars for indicator stabilization)
    clean_df = df.iloc[50:].reset_index(drop=True)

    if cutoff_date_str:
        cutoff_date = pd.to_datetime(cutoff_date_str, format="mixed")
        if cutoff_date not in clean_df["Date"].values:
            # Fallback to closest available prior date
            available_dates = clean_df[clean_df["Date"] <= cutoff_date]["Date"]
            if available_dates.empty:
                sys.exit(f"Error: Cutoff date {cutoff_date_str} is before available dataset history.")
            cutoff_date = available_dates.max()
    else:
        cutoff_date = clean_df["Date"].iloc[-1]

    # Training set: All bars strictly before cutoff date with realized forward returns
    train_data = clean_df[clean_df["Date"] < cutoff_date].dropna(subset=["NIFTY_Fwd_Ret"]).reset_index(drop=True)
    pred_row = clean_df[clean_df["Date"] == cutoff_date].reset_index(drop=True)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_data[feature_cols].values)
    X_pred = imputer.transform(pred_row[feature_cols].values)

    results = {}
    entities = [
        ("NIFTY 50", "NIFTY_Fwd_Ret", "NIFTY_Target_Dir"),
        ("INDIA VIX", "VIX_Fwd_Ret", "VIX_Target_Dir"),
        ("GIFT NIFTY", "GIFT_Fwd_Ret", "GIFT_Target_Dir"),
    ]

    for name, ret_col, dir_col in entities:
        y_ret = train_data[ret_col].values
        valid_mask = ~np.isnan(y_ret)

        X_tr = X_train[valid_mask]
        y_tr_ret = y_ret[valid_mask]
        y_tr_dir = train_data[dir_col].values[valid_mask]

        # 1. Regressors for continuous expected return E[R]
        hist_reg = HistGradientBoostingRegressor(learning_rate=0.03, max_iter=150, max_leaf_nodes=12, min_samples_leaf=15, random_state=42)
        rf_reg = RandomForestRegressor(n_estimators=250, max_depth=5, min_samples_leaf=15, random_state=42)
        et_reg = ExtraTreesRegressor(n_estimators=250, max_depth=5, min_samples_leaf=15, random_state=42)

        hist_reg.fit(X_tr, y_tr_ret)
        rf_reg.fit(X_tr, y_tr_ret)
        et_reg.fit(X_tr, y_tr_ret)

        e_hist = hist_reg.predict(X_pred)[0]
        e_rf = rf_reg.predict(X_pred)[0]
        e_et = et_reg.predict(X_pred)[0]
        e_ensemble = np.mean([e_hist, e_rf, e_et])

        # 2. Classifiers for directional probability P(Bullish)
        hist_clf = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=150, max_leaf_nodes=12, min_samples_leaf=15, random_state=42)
        rf_clf = RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=15, random_state=42)
        et_clf = ExtraTreesClassifier(n_estimators=250, max_depth=5, min_samples_leaf=15, random_state=42)

        hist_clf.fit(X_tr, y_tr_dir)
        rf_clf.fit(X_tr, y_tr_dir)
        et_clf.fit(X_tr, y_tr_dir)

        p_hist = hist_clf.predict_proba(X_pred)[:, 1][0]
        p_rf = rf_clf.predict_proba(X_pred)[:, 1][0]
        p_et = et_clf.predict_proba(X_pred)[:, 1][0]
        p_ensemble = np.mean([p_hist, p_rf, p_et])

        # Directional Verdict & Conviction Rule
        if p_ensemble >= 0.50:
            direction = "BULLISH (UP)"
        else:
            direction = "BEARISH (DOWN)"

        # Alpha Hurdle / Conviction State
        if abs(e_ensemble) >= 0.0015 or abs(p_ensemble - 0.50) >= 0.05:
            conviction = "TRADEABLE SIGNAL"
        else:
            conviction = "NEUTRAL / NO-TRADE BAND"

        results[name] = {
            "Direction": direction,
            "Bullish_Prob_HistGB": p_hist,
            "Bullish_Prob_RF": p_rf,
            "Bullish_Prob_ET": p_et,
            "Bullish_Prob_Ensemble": p_ensemble,
            "Bearish_Prob_Ensemble": 1.0 - p_ensemble,
            "Exp_Return_HistGB": e_hist,
            "Exp_Return_RF": e_rf,
            "Exp_Return_ET": e_et,
            "Exp_Return_Ensemble": e_ensemble,
            "Conviction": conviction
        }

    # Display Results
    print("=" * 75)
    print(f" QUANT DIRECTION FORECAST (Input Data As Of: {cutoff_date.strftime('%d-%m-%Y')})")
    print(f" Target Horizon: Next Trading Session Following {cutoff_date.strftime('%d-%m-%Y')}")
    print("=" * 75)
    
    for name, res in results.items():
        print(f"\n[{name}] -> Verdict: {res['Direction']} ({res['Conviction']})")
        print("-" * 65)
        print("  CLASSIFIER PROBABILITIES (P_Bullish):")
        print(f"    • HistGradientBoosting Classifier : {res['Bullish_Prob_HistGB']*100:.2f}%")
        print(f"    • RandomForest Classifier         : {res['Bullish_Prob_RF']*100:.2f}%")
        print(f"    • ExtraTrees Classifier           : {res['Bullish_Prob_ET']*100:.2f}%")
        print(f"    • Ensemble Consensus Prob         : {res['Bullish_Prob_Ensemble']*100:.2f}% Bullish | {res['Bearish_Prob_Ensemble']*100:.2f}% Bearish")
        print("\n  REGRESSION EXPECTED RETURNS (E[R]):")
        print(f"    • HistGradientBoosting Regressor  : {res['Exp_Return_HistGB']*100:+.3f}%")
        print(f"    • RandomForest Regressor          : {res['Exp_Return_RF']*100:+.3f}%")
        print(f"    • ExtraTrees Regressor            : {res['Exp_Return_ET']*100:+.3f}%")
        print(f"    • Ensemble Expected Return        : {res['Exp_Return_Ensemble']*100:+.3f}%")
        
    print("\n" + "=" * 75)


# ==========================================
# 4. SCRIPT ENTRY POINT
# ==========================================
#%run quant_direction_engine.py --target-date 07-07-2026
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Entity Quant Direction Predictor")
    parser.add_argument(
        "--target-date",
        type=str,
        default=None,
        help="Date as of close to make prediction from (e.g., '07-07-2026' or '2026-07-07'). Default is latest date.",
    )
    args = parser.parse_args()

    raw_data = load_and_prepare_dataset()
    featured_data, features = engineer_features(raw_data)
    predict_market_direction(featured_data, features, cutoff_date_str=args.target_date)