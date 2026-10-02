"""
Quant Direction Engine v2
==========================
Changes from v1, based on a walk-forward backtest (retrain daily, expanding window,
2023-11-15 -> 2026-03-06, n=570 out-of-sample days):

  1. DROPPED GIFT NIFTY FEATURES.
     Diagnosis: GIFT_Change_Pct(date D) correlates 0.96 with NIFTY's OWN same-day
     return, and ~0.00 with the actual T+1 target. As sourced (daily OHLC from
     investing.com), "GIFT Nifty on date D" is contemporaneous with Nifty's own
     session on D, not a preview of D+1 -- so it carries no real leading information,
     only redundant noise.
     Backtest evidence (identical 570-day out-of-sample window, same harness):
         WITH GIFT    (49 feats): 52.98% acc vs 53.16% majority baseline  (edge: -0.18pp)
         WITHOUT GIFT (39 feats): 56.32% acc vs 53.16% majority baseline  (edge: +3.16pp)
     Removing GIFT was a clear, measured improvement, not just a hunch.

  2. ADDED A WALK-FORWARD BACKTEST (--backtest flag).
     Retrains on an expanding window and predicts one unseen day at a time, so you
     get a real out-of-sample accuracy number instead of trusting a single day's
     probability. This is now the primary way to evaluate any change to this engine
     -- if a change doesn't improve the backtest number, don't ship it.

  3. FEATURE PRUNING: tested, evidence says DON'T naively cut to "top-K by importance".
     A naive top-15-by-importance cut UNDERPERFORMED the full 39-feature (no-GIFT)
     set on identical out-of-sample days (53.51% vs 56.32%). Static importance
     ranking is itself computed in-sample per fold and is not a safe pruning
     criterion on its own. The full feature set is kept as default. If you want to
     try trimming further, do it by running --backtest on the candidate subset and
     comparing the resulting accuracy -- never by importance rank alone.

Reality check: even after these fixes, next-day direction of a liquid, heavily
arbitraged index has a low ceiling. The edge found here (see --backtest output)
is real vs. a naive baseline but modest, and its significance vs. the majority-class
baseline (not just vs. a coin flip) should be checked before trusting it with money.
"""
import argparse
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

warnings.filterwarnings("ignore")


# ==========================================
# 1. DATA INGESTION (GIFT NIFTY DROPPED)
# ==========================================
def load_and_prepare_dataset(equity_file="Nifty_OHLC_Direction_Indicators.xlsx"):
    """Loads and synchronizes the primary equity/index, INDIA VIX, and US overnight data.
    GIFT NIFTY is intentionally NOT merged in -- see module docstring.
    `equity_file` can point at any instrument's OHLC+Indicators workbook (same schema:
    'OHLC Data' and 'Indicators' sheets) -- e.g. an index like Nifty or a single stock."""
    required_files = [
        equity_file,
        "INDIAVIX_OHLC_Direction_Indicators.xlsx",
        "us_overnight_market_features.csv",
    ]
    for file in required_files:
        if not os.path.exists(file):
            sys.exit(f"Error: Required data file '{file}' not found in current directory.")

    nifty_ohlc = pd.read_excel(equity_file, sheet_name="OHLC Data")
    nifty_ind = pd.read_excel(equity_file, sheet_name="Indicators")
    nifty_ohlc["Date"] = pd.to_datetime(nifty_ohlc["Date"])
    nifty_ind["Date"] = pd.to_datetime(nifty_ind["Date"])

    nifty_ind_renamed = nifty_ind.drop(
        columns=["Open", "High", "Low", "Close", "Volume"], errors="ignore"
    ).rename(columns=lambda c: f"NIFTY_{c}" if c != "Date" else c)

    vix_ohlc = pd.read_excel("INDIAVIX_OHLC_Direction_Indicators.xlsx", sheet_name="OHLC Data")
    vix_ind = pd.read_excel("INDIAVIX_OHLC_Direction_Indicators.xlsx", sheet_name="Indicators")
    vix_ohlc["Date"] = pd.to_datetime(vix_ohlc["Date"])
    vix_ind["Date"] = pd.to_datetime(vix_ind["Date"])

    vix_ohlc_renamed = vix_ohlc.rename(
        columns={"Open": "VIX_Open", "High": "VIX_High", "Low": "VIX_Low",
                 "Close": "VIX_Close", "Volume": "VIX_Volume"}
    )
    vix_ind_renamed = vix_ind.drop(
        columns=["Open", "High", "Low", "Close", "Volume"], errors="ignore"
    ).rename(columns=lambda c: f"VIX_{c}" if c != "Date" else c)

    us_df = pd.read_csv("us_overnight_market_features.csv")
    us_df["Date"] = pd.to_datetime(us_df["Date"])

    merged = nifty_ohlc.merge(nifty_ind_renamed, on="Date", how="left")
    merged = merged.merge(vix_ohlc_renamed, on="Date", how="left")
    merged = merged.merge(vix_ind_renamed, on="Date", how="left")
    merged = merged.merge(us_df, on="Date", how="left")
    merged = merged.sort_values("Date").reset_index(drop=True)
    return merged


# ==========================================
# 2. FEATURE ENGINEERING (GIFT NIFTY DROPPED)
# ==========================================
def engineer_features(df):
    data = df.copy()

    data["NIFTY_Fwd_Ret"] = data["Close"].shift(-1) / data["Close"] - 1.0
    data["NIFTY_Target_Dir"] = (data["NIFTY_Fwd_Ret"] > 0).astype(int)
    data["VIX_Fwd_Ret"] = data["VIX_Close"].shift(-1) / data["VIX_Close"] - 1.0
    data["VIX_Target_Dir"] = (data["VIX_Fwd_Ret"] > 0).astype(int)

    for k in [1, 2, 3, 5, 10, 20]:
        data[f"NIFTY_Ret_{k}d"] = data["Close"].pct_change(k)
        data[f"VIX_Ret_{k}d"] = data["VIX_Close"].pct_change(k)

    candle_range = data["High"] - data["Low"] + 1e-6
    data["NIFTY_Body_to_Range"] = (data["Close"] - data["Open"]) / candle_range
    data["NIFTY_Upper_Wick"] = (data["High"] - np.maximum(data["Open"], data["Close"])) / candle_range
    data["NIFTY_Lower_Wick"] = (np.minimum(data["Open"], data["Close"]) - data["Low"]) / candle_range
    data["Close_Off_High"] = (data["Close"] - data["High"]) / candle_range
    data["NIFTY_Gap"] = (data["Open"] - data["Close"].shift(1)) / data["Close"].shift(1)
    data["Intraday_Ret"] = (data["Close"] - data["Open"]) / data["Open"]

    data["NIFTY_Garman_Klass"] = np.sqrt(
        0.5 * (np.log(data["High"] / data["Low"])) ** 2
        - (2 * np.log(2) - 1) * (np.log(data["Close"] / data["Open"])) ** 2
    )
    data["VIX_Garman_Klass"] = np.sqrt(
        0.5 * (np.log(data["VIX_High"] / data["VIX_Low"])) ** 2
        - (2 * np.log(2) - 1) * (np.log(data["VIX_Close"] / data["VIX_Open"])) ** 2
    )

    data["NIFTY_Vol_SMA20"] = data["Volume"] / (data["Volume"].rolling(20).mean() + 1e-6)

    data["Dist_from_EMA20"] = (data["Close"] - data["NIFTY_EMA20"]) / data["NIFTY_EMA20"]
    data["Dist_from_SMA20"] = (data["Close"] - data["NIFTY_SMA20"]) / data["NIFTY_SMA20"]
    data["RSI_Overbought"] = (data["NIFTY_RSI14"] - 60).clip(lower=0) / 40.0
    data["VIX_Complacency"] = (12.0 - data["VIX_Close"]).clip(lower=0)

    data["NIFTY_RSI_norm"] = (data["NIFTY_RSI14"] - 50.0) / 50.0
    data["NIFTY_MACD_Hist_norm"] = data["NIFTY_MACD_Hist"] / data["Close"]
    data["NIFTY_SMA_Diff"] = (data["NIFTY_SMA20"] - data["NIFTY_SMA50"]) / data["Close"]
    data["NIFTY_EMA_Diff"] = (data["NIFTY_EMA20"] - data["NIFTY_EMA50"]) / data["Close"]
    data["NIFTY_ADX_Trend"] = (data["NIFTY_Plus_DI"] - data["NIFTY_Minus_DI"]) * (data["NIFTY_ADX14"] / 100.0)

    data["VIX_RSI_norm"] = (data["VIX_RSI14"] - 50.0) / 50.0
    data["VIX_SMA_Diff"] = (data["VIX_SMA20"] - data["VIX_SMA50"]) / (data["VIX_Close"] + 1e-6)
    data["VIX_EMA_Diff"] = (data["VIX_EMA20"] - data["VIX_EMA50"]) / (data["VIX_Close"] + 1e-6)
    data["VIX_ADX_Trend"] = (data["VIX_Plus_DI"] - data["VIX_Minus_DI"]) * (data["VIX_ADX14"] / 100.0)

    for c in ["SP500_Return", "NASDAQ_Return", "Crude_Change", "DXY_Change", "US_VIX_Change"]:
        data[c] = data[c].ffill().fillna(0)

    exclude_cols = [
        "Date", "Open", "High", "Low", "Close", "Volume",
        "VIX_Open", "VIX_High", "VIX_Low", "VIX_Close", "VIX_Volume",
        "NIFTY_Fwd_Ret", "NIFTY_Target_Dir", "VIX_Fwd_Ret", "VIX_Target_Dir",
        "NIFTY_SMA_Direction", "NIFTY_EMA_Direction", "NIFTY_MACD_Direction",
        "NIFTY_RSI_Direction", "NIFTY_ADX_Direction",
        "VIX_SMA_Direction", "VIX_EMA_Direction", "VIX_MACD_Direction",
        "VIX_RSI_Direction", "VIX_ADX_Direction",
        "NIFTY_SMA20", "NIFTY_SMA50", "NIFTY_EMA20", "NIFTY_EMA50",
        "VIX_SMA20", "VIX_SMA50", "VIX_EMA20", "VIX_EMA50",
        "NIFTY_MACD", "NIFTY_MACD_Signal", "NIFTY_MACD_Hist",
        "VIX_MACD", "VIX_MACD_Signal", "VIX_MACD_Hist",
        "NIFTY_Plus_DI", "NIFTY_Minus_DI", "NIFTY_ADX14",
        "VIX_Plus_DI", "VIX_Minus_DI", "VIX_ADX14",
        "NIFTY_RSI14", "VIX_RSI14",
    ]
    feature_cols = [c for c in data.columns if c not in exclude_cols]
    return data, feature_cols


# ==========================================
# 3. WALK-FORWARD BACKTEST
# ==========================================
def walk_forward_backtest(data, feature_cols, target_col="NIFTY_Target_Dir", ret_col="NIFTY_Fwd_Ret",
                           min_train=500, n_estimators=80, step=1,
                           deadline_seconds=None, verbose_every=50, test_days=None):
    """Expanding-window walk-forward test: retrain on all data strictly before day i,
    predict day i, move on. This is the only honest way to know current accuracy --
    a single day's probability tells you nothing about whether the model has real skill.

    Each fold trains both a classifier (direction) and a regressor (expected return) for
    HGB and XGB. Model agreement is FULL CONSENSUS: the two classifiers must pick the same
    direction AND the two regressors' expected returns must have the same sign -- same
    definition used in the single-date forecast. Only full-consensus days get a graded
    CORRECT/INCORRECT call; everything else is NO_CONSENSUS.

    If test_days is given, min_train is set so the backtest evaluates exactly the last
    `test_days` trading days (e.g. test_days=30 for a 30-day report) instead of the
    default full-history expanding window."""
    clean = data.iloc[50:].reset_index(drop=True)
    clean = clean.dropna(subset=[target_col]).reset_index(drop=True)

    X_all = clean[feature_cols].values
    y_all = clean[target_col].values
    yret_all = clean[ret_col].values
    dates = clean["Date"].values
    n = len(clean)

    if test_days is not None:
        min_train = max(n - test_days, 1)
        if min_train < 100:
            print(f"  [warning: only {min_train} rows available to train on before "
                  f"the {test_days}-day test window -- results may be noisy]")

    hgb_preds, hgb_probs, hgb_rets = [], [], []
    xgb_preds, xgb_probs, xgb_rets = [], [], []
    actuals, test_dates = [], []
    importances = np.zeros(len(feature_cols))
    n_folds = 0
    t0 = time.time()

    test_idx = list(range(min_train, n, step))
    for i in test_idx:
        if deadline_seconds and (time.time() - t0) > deadline_seconds:
            print(f"  [time budget reached after {n_folds} folds, stopping early]")
            break

        imputer = SimpleImputer(strategy="median")
        Xtr = imputer.fit_transform(X_all[:i])
        Xte = imputer.transform(X_all[i:i + 1])
        y_train = y_all[:i]
        yret_train = yret_all[:i]

        hg_clf = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=100, max_leaf_nodes=12, min_samples_leaf=15, random_state=42)
        xgb_clf = XGBClassifier(n_estimators=n_estimators, max_depth=5, learning_rate=0.03, min_child_weight=15,
                                 random_state=42, eval_metric="logloss")
        hg_reg = HistGradientBoostingRegressor(learning_rate=0.03, max_iter=100, max_leaf_nodes=12, min_samples_leaf=15, random_state=42)
        xgb_reg = XGBRegressor(n_estimators=n_estimators, max_depth=5, learning_rate=0.03, min_child_weight=15, random_state=42)

        hg_clf.fit(Xtr, y_train)
        xgb_clf.fit(Xtr, y_train)
        hg_reg.fit(Xtr, yret_train)
        xgb_reg.fit(Xtr, yret_train)

        p_hg = hg_clf.predict_proba(Xte)[:, 1][0]
        p_xgb = xgb_clf.predict_proba(Xte)[:, 1][0]
        e_hg = hg_reg.predict(Xte)[0]
        e_xgb = xgb_reg.predict(Xte)[0]

        hgb_probs.append(p_hg)
        hgb_preds.append(int(p_hg >= 0.5))
        hgb_rets.append(e_hg)
        xgb_probs.append(p_xgb)
        xgb_preds.append(int(p_xgb >= 0.5))
        xgb_rets.append(e_xgb)
        actuals.append(y_all[i])
        test_dates.append(dates[i])
        importances += xgb_clf.feature_importances_
        n_folds += 1

        if verbose_every and n_folds % verbose_every == 0:
            print(f"  fold {n_folds}/{len(test_idx)} | elapsed {time.time()-t0:.1f}s "
                  f"| running acc (HGB): {accuracy_score(actuals, hgb_preds):.4f} "
                  f"| running acc (XGB): {accuracy_score(actuals, xgb_preds):.4f}")

    importances /= max(n_folds, 1)
    hgb_preds, xgb_preds = np.array(hgb_preds), np.array(xgb_preds)
    hgb_rets, xgb_rets = np.array(hgb_rets), np.array(xgb_rets)
    actuals_arr = np.array(actuals)

    classifiers_agree = hgb_preds == xgb_preds
    regressors_agree = np.sign(hgb_rets) == np.sign(xgb_rets)
    full_consensus = classifiers_agree & regressors_agree

    consensus_pred = np.where(full_consensus, hgb_preds, -1)  # -1 = no full-consensus call
    consensus_result = np.select(
        [~full_consensus, consensus_pred == actuals_arr, consensus_pred != actuals_arr],
        ["NO_CONSENSUS", "CORRECT", "INCORRECT"],
        default="",
    )

    result = pd.DataFrame({
        "Date": test_dates,
        "Actual_Direction": np.where(actuals_arr == 1, "UP", "DOWN"),
        "HGB_Direction": np.where(hgb_preds == 1, "UP", "DOWN"),
        "HGB_Prob_Bullish": hgb_probs,
        "HGB_Exp_Return": hgb_rets,
        "XGB_Direction": np.where(xgb_preds == 1, "UP", "DOWN"),
        "XGB_Prob_Bullish": xgb_probs,
        "XGB_Exp_Return": xgb_rets,
        "Classifiers_Agree": classifiers_agree,
        "Regressors_Agree": regressors_agree,
        "Models_Agree": full_consensus,
        "Consensus_Direction": np.where(full_consensus, np.where(consensus_pred == 1, "UP", "DOWN"), "NO_CONSENSUS"),
        "Result": consensus_result,
        # raw 0/1 columns kept for programmatic use (metrics, plotting, etc.)
        "actual": actuals, "hgb_pred": hgb_preds, "hgb_prob": hgb_probs,
        "xgb_pred": xgb_preds, "xgb_prob": xgb_probs,
    })
    return result, pd.Series(importances, index=feature_cols).sort_values(ascending=False)


def report_backtest(result, label=""):
    majority = max(result["actual"].mean(), 1 - result["actual"].mean())
    print("=" * 70)
    print(f" WALK-FORWARD BACKTEST RESULTS {label}")
    print("=" * 70)
    print(f" Test period      : {pd.to_datetime(result['Date'].min()).date()} -> "
          f"{pd.to_datetime(result['Date'].max()).date()}  (n={len(result)})")
    print(f" Majority baseline: {majority:.4f}")
    print("-" * 70)
    for name, pred_col, prob_col in [
        ("HistGradientBoosting", "hgb_pred", "hgb_prob"),
        ("XGBoost", "xgb_pred", "xgb_prob"),
    ]:
        acc = accuracy_score(result["actual"], result[pred_col])
        ll = log_loss(result["actual"], result[prob_col])
        brier = brier_score_loss(result["actual"], result[prob_col])
        print(f" [{name}]")
        print(f"   Model accuracy : {acc:.4f}   (edge vs majority: {acc - majority:+.4f})")
        print(f"   Log loss       : {ll:.4f}")
        print(f"   Brier score    : {brier:.4f}")
    print("-" * 70)

    n_total = len(result)
    n_clf_agree = int(result["Classifiers_Agree"].sum())
    n_full = int(result["Models_Agree"].sum())
    consensus_rows = result[result["Models_Agree"]]
    if len(consensus_rows) > 0:
        consensus_acc = (consensus_rows["Result"] == "CORRECT").mean()
    else:
        consensus_acc = float("nan")
    print(f" [Model Agreement]")
    print(f"   Classifiers agreed (direction only)      : {n_clf_agree}/{n_total} ({n_clf_agree / n_total * 100:.1f}%)")
    print(f"   FULL CONSENSUS (classifiers + regressors) : {n_full}/{n_total} ({n_full / n_total * 100:.1f}%)")
    print(f"   Accuracy on full-consensus days           : {consensus_acc:.4f}   (edge vs majority: {consensus_acc - majority:+.4f})")
    print("=" * 70)


# ==========================================
# 4. SINGLE-DATE PREDICTION (NIFTY + VIX ONLY -- GIFT ENTITY DROPPED)
# ==========================================
def predict_market_direction(df, feature_cols, cutoff_date_str=None, symbol="NIFTY 50"):
    clean_df = df.iloc[50:].reset_index(drop=True)

    if cutoff_date_str:
        cutoff_date = pd.to_datetime(cutoff_date_str, dayfirst=True)
        if cutoff_date not in clean_df["Date"].values:
            available_dates = clean_df[clean_df["Date"] <= cutoff_date]["Date"]
            if available_dates.empty:
                sys.exit(f"Error: Cutoff date {cutoff_date_str} is before available dataset history.")
            cutoff_date = available_dates.max()
    else:
        cutoff_date = clean_df["Date"].iloc[-1]

    train_data = clean_df[clean_df["Date"] < cutoff_date].dropna(subset=["NIFTY_Fwd_Ret"]).reset_index(drop=True)
    pred_row = clean_df[clean_df["Date"] == cutoff_date].reset_index(drop=True)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_data[feature_cols].values)
    X_pred = imputer.transform(pred_row[feature_cols].values)

    results = {}
    entities = [(symbol, "NIFTY_Fwd_Ret", "NIFTY_Target_Dir"),
                ("INDIA VIX", "VIX_Fwd_Ret", "VIX_Target_Dir")]

    models = [
        ("HistGradientBoosting",
         HistGradientBoostingRegressor(learning_rate=0.03, max_iter=150, max_leaf_nodes=12, min_samples_leaf=15, random_state=42),
         HistGradientBoostingClassifier(learning_rate=0.03, max_iter=150, max_leaf_nodes=12, min_samples_leaf=15, random_state=42)),
        ("XGBoost",
         XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.03, min_child_weight=15, random_state=42),
         XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.03, min_child_weight=15, random_state=42, eval_metric="logloss")),
    ]

    for name, ret_col, dir_col in entities:
        y_ret = train_data[ret_col].values
        valid_mask = ~np.isnan(y_ret)
        X_tr = X_train[valid_mask]
        y_tr_ret = y_ret[valid_mask]
        y_tr_dir = train_data[dir_col].values[valid_mask]

        entity_results = {}
        for model_name, reg, clf in models:
            reg.fit(X_tr, y_tr_ret)
            e_pred = reg.predict(X_pred)[0]

            clf.fit(X_tr, y_tr_dir)
            p_pred = clf.predict_proba(X_pred)[:, 1][0]

            direction = "BULLISH (UP)" if p_pred >= 0.50 else "BEARISH (DOWN)"
            conviction = "TRADEABLE SIGNAL" if (abs(e_pred) >= 0.0015 or abs(p_pred - 0.50) >= 0.05) else "NEUTRAL / NO-TRADE BAND"

            entity_results[model_name] = {"Direction": direction, "Bullish_Prob": p_pred,
                                           "Exp_Return": e_pred, "Conviction": conviction}

        # Model agreement: do the two classifiers' verdicts match, AND do the two
        # regressors' expected returns point the same way? Both is the strongest signal.
        clf_dirs = {m: r["Direction"] for m, r in entity_results.items()}
        reg_signs = {m: np.sign(r["Exp_Return"]) for m, r in entity_results.items()}
        classifiers_agree = len(set(clf_dirs.values())) == 1
        regressors_agree = len(set(reg_signs.values())) == 1
        if classifiers_agree and regressors_agree:
            agreement = "FULL CONSENSUS (both models, both signals agree)"
        elif classifiers_agree:
            agreement = "PARTIAL -- classifiers agree, but expected-return sign differs"
        else:
            agreement = "NO CONSENSUS -- models disagree on direction"

        results[name] = {"models": entity_results, "agreement": agreement}

    print("=" * 75)
    print(f" QUANT DIRECTION FORECAST (Input Data As Of: {cutoff_date.strftime('%d-%m-%Y')})")
    print(f" Target Horizon: Next Trading Session Following {cutoff_date.strftime('%d-%m-%Y')}")
    print(" (Reminder: run --backtest periodically to confirm this still beats baseline)")
    print("=" * 75)
    for name, entry in results.items():
        print(f"\n[{name}]")
        for model_name, res in entry["models"].items():
            print(f"  {model_name:<22} -> Verdict: {res['Direction']} ({res['Conviction']})")
            print(f"    {'':<22}    P(Bullish)      : {res['Bullish_Prob']*100:.2f}%")
            print(f"    {'':<22}    Expected Return : {res['Exp_Return']*100:+.3f}%")
        print(f"  Model Agreement        : {entry['agreement']}")
    print("\n" + "=" * 75)


# ==========================================
# 5. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Entity Quant Direction Predictor v2 (no GIFT, backtest-checked)")
    parser.add_argument("--equity-file", type=str, default="Nifty_OHLC_Direction_Indicators.xlsx",
                         help="OHLC+Indicators workbook for the primary instrument (index or single stock).")
    parser.add_argument("--symbol", type=str, default="NIFTY 50",
                         help="Display label for the primary instrument in printed output/reports.")
    parser.add_argument("--target-date", type=str, default=None,
                         help="Date as of close to make prediction from, e.g. '07-07-2026'. Default is latest date.")
    parser.add_argument("--backtest", action="store_true",
                         help="Run walk-forward backtest instead of a single prediction.")
    parser.add_argument("--min-train", type=int, default=500, help="Minimum training rows before first backtest fold.")
    parser.add_argument("--step", type=int, default=1, help="Test every Nth day in the backtest (1 = every day).")
    parser.add_argument("--deadline-seconds", type=float, default=None, help="Stop the backtest early after this many seconds.")
    parser.add_argument("--report", type=int, default=None,
                         help="Run a walk-forward backtest over just the last N trading days and save a CSV "
                              "report (with per-day model-agreement/correctness columns) to "
                              "backtest_report_last_<N>days.csv. E.g. --report 30")
    args = parser.parse_args()

    raw_data = load_and_prepare_dataset(args.equity_file)
    featured_data, features = engineer_features(raw_data)

    if args.report is not None:
        result, importances = walk_forward_backtest(
            featured_data, features, test_days=args.report, step=args.step,
            deadline_seconds=args.deadline_seconds)
        report_backtest(result, label=f"(last {args.report} trading days, {args.symbol}, no-GIFT feature set)")
        out_path = f"backtest_report_last_{args.report}days.csv"
        result.to_csv(out_path, index=False)
        print(f"\nSaved report ({len(result)} rows) to {out_path}")
    elif args.backtest:
        result, importances = walk_forward_backtest(
            featured_data, features, min_train=args.min_train, step=args.step,
            deadline_seconds=args.deadline_seconds)
        report_backtest(result, label=f"({args.symbol}, no-GIFT feature set)")
        print("\nTop 15 features by walk-forward-averaged importance:")
        print(importances.head(15))
    else:
        predict_market_direction(featured_data, features, cutoff_date_str=args.target_date, symbol=args.symbol)
        #%run quant_all.py --target-date 06-07-2026
        #%run quant_all_v2.py --all-stocks --target-date 22-06-2026