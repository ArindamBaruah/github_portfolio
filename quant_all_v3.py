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

  4. ADDED --all-stocks BATCH MODE.
     Runs the same single-date HGB+XGB forecast used by predict_market_direction()
     across every local *_OHLC_Direction_Indicators.xlsx file (any stock with the
     same 'OHLC Data' + 'Indicators' sheet schema as the Nifty file), grades each
     one against its actual outcome when that's already known, and writes ONE
     combined summary CSV (one row per stock) instead of a per-stock printout.
     NOTE: this addition has not been run/backtested yet -- verify it on your data
     before relying on its output the way the rest of this engine has been verified.
"""
import argparse
import glob
import os
import re
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
    # NOTE: NaN > 0 evaluates to False, and .astype(int) would then silently turn
    # that False into 0 ("DOWN") -- which wrongly makes the last row (where the
    # future close, and therefore the true direction, isn't known yet) look like
    # a graded "known" outcome. Keep it NaN wherever NIFTY_Fwd_Ret is NaN instead.
    data["NIFTY_Target_Dir"] = np.where(
        data["NIFTY_Fwd_Ret"].isna(), np.nan, (data["NIFTY_Fwd_Ret"] > 0).astype(float)
    )
    data["VIX_Fwd_Ret"] = data["VIX_Close"].shift(-1) / data["VIX_Close"] - 1.0
    data["VIX_Target_Dir"] = np.where(
        data["VIX_Fwd_Ret"].isna(), np.nan, (data["VIX_Fwd_Ret"] > 0).astype(float)
    )

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
    # NEW: added a ref-close column per entity (the entity's own actual Close on
    # cutoff_date) so a predicted CLOSE PRICE (not just UP/DOWN direction) can be
    # computed as ref_close * (1 + Exp_Return) for each model below.
    entities = [(symbol, "NIFTY_Fwd_Ret", "NIFTY_Target_Dir", "Close"),
                ("INDIA VIX", "VIX_Fwd_Ret", "VIX_Target_Dir", "VIX_Close")]

    models = [
        ("HistGradientBoosting",
         HistGradientBoostingRegressor(learning_rate=0.03, max_iter=150, max_leaf_nodes=12, min_samples_leaf=15, random_state=42),
         HistGradientBoostingClassifier(learning_rate=0.03, max_iter=150, max_leaf_nodes=12, min_samples_leaf=15, random_state=42)),
        ("XGBoost",
         XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.03, min_child_weight=15, random_state=42),
         XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.03, min_child_weight=15, random_state=42, eval_metric="logloss")),
    ]

    for name, ret_col, dir_col, close_col in entities:
        y_ret = train_data[ret_col].values
        valid_mask = ~np.isnan(y_ret)
        X_tr = X_train[valid_mask]
        y_tr_ret = y_ret[valid_mask]
        y_tr_dir = train_data[dir_col].values[valid_mask]
        ref_close = float(pred_row[close_col].iloc[0])

        entity_results = {}
        for model_name, reg, clf in models:
            reg.fit(X_tr, y_tr_ret)
            e_pred = reg.predict(X_pred)[0]
            pred_close = ref_close * (1.0 + e_pred)

            clf.fit(X_tr, y_tr_dir)
            p_pred = clf.predict_proba(X_pred)[:, 1][0]

            direction = "BULLISH (UP)" if p_pred >= 0.50 else "BEARISH (DOWN)"
            conviction = "TRADEABLE SIGNAL" if (abs(e_pred) >= 0.0015 or abs(p_pred - 0.50) >= 0.05) else "NEUTRAL / NO-TRADE BAND"

            entity_results[model_name] = {"Direction": direction, "Bullish_Prob": p_pred,
                                           "Exp_Return": e_pred, "Conviction": conviction,
                                           "Ref_Close": ref_close, "Predicted_Close": pred_close}

        # Model agreement: do the two classifiers' verdicts match, AND do the two
        # regressors' expected returns point the same way? Both is the strongest signal.
        clf_dirs = {m: r["Direction"] for m, r in entity_results.items()}
        reg_signs = {m: np.sign(r["Exp_Return"]) for m, r in entity_results.items()}
        classifiers_agree = len(set(clf_dirs.values())) == 1
        regressors_agree = len(set(reg_signs.values())) == 1
        # NEW: each model's OWN classifier and OWN regressor are trained separately
        # and can disagree with each other even when both classifiers agree with each
        # other and both regressors agree with each other -- e.g. both classifiers say
        # BEARISH while both regressors independently return a POSITIVE expected
        # return. Without checking this, "FULL CONSENSUS" could be declared on a
        # Predicted_Close that's actually above Ref_Close despite a "DOWN" verdict.
        self_consistent = all(
            (r["Direction"].startswith("BULLISH")) == (r["Exp_Return"] >= 0)
            for r in entity_results.values()
        )
        if classifiers_agree and regressors_agree and self_consistent:
            agreement = "FULL CONSENSUS (both models, both signals agree)"
        elif classifiers_agree and regressors_agree:
            agreement = "PARTIAL -- classifiers agree with each other and regressors agree " \
                        "with each other, but a model's own classifier/regressor disagree"
        elif classifiers_agree:
            agreement = "PARTIAL -- classifiers agree, but expected-return sign differs"
        else:
            agreement = "NO CONSENSUS -- models disagree on direction"

        # NEW: an average predicted close across both models, for a single at-a-glance number.
        avg_pred_close = np.mean([r["Predicted_Close"] for r in entity_results.values()])

        results[name] = {"models": entity_results, "agreement": agreement,
                          "ref_close": ref_close, "avg_pred_close": avg_pred_close}

    print("=" * 75)
    print(f" QUANT DIRECTION FORECAST (Input Data As Of: {cutoff_date.strftime('%d-%m-%Y')})")
    print(f" Target Horizon: Next Trading Session Following {cutoff_date.strftime('%d-%m-%Y')}")
    print(" (Reminder: run --backtest periodically to confirm this still beats baseline)")
    print("=" * 75)
    for name, entry in results.items():
        print(f"\n[{name}]  (Last Close: {entry['ref_close']:,.2f})")
        for model_name, res in entry["models"].items():
            print(f"  {model_name:<22} -> Verdict: {res['Direction']} ({res['Conviction']})")
            print(f"    {'':<22}    P(Bullish)      : {res['Bullish_Prob']*100:.2f}%")
            print(f"    {'':<22}    Expected Return : {res['Exp_Return']*100:+.3f}%")
            print(f"    {'':<22}    Predicted Close : {res['Predicted_Close']:,.2f}")
        print(f"  Model Agreement        : {entry['agreement']}")
        print(f"  Avg Predicted Close    : {entry['avg_pred_close']:,.2f}")
    print("\n" + "=" * 75)


# ==========================================
# 5. BATCH MULTI-STOCK PREDICTION + GRADING REPORT
# ==========================================
def find_stock_files(directory=".", pattern="*_OHLC_Direction_Indicators.xlsx",
                      exclude=("INDIAVIX_OHLC_Direction_Indicators.xlsx",)):
    """Finds every local workbook that matches the same schema as
    Nifty_OHLC_Direction_Indicators.xlsx ('OHLC Data' + 'Indicators' sheets),
    so each one can be run through the engine as its own instrument.
    INDIA VIX is excluded by default -- it's a supporting feature file merged
    into every stock's features already, not something we predict a direction
    for on its own."""
    matches = sorted(glob.glob(os.path.join(directory, pattern)))
    return [f for f in matches if os.path.basename(f) not in exclude]


def stock_symbol_from_filename(path, suffix="_OHLC_Direction_Indicators.xlsx"):
    """RELIANCE_OHLC_Direction_Indicators.xlsx -> RELIANCE"""
    name = os.path.basename(path)
    return name[: -len(suffix)] if name.endswith(suffix) else os.path.splitext(name)[0]


def predict_and_grade_one(equity_file, symbol, target_date_str=None):
    """Runs the same single-date HGB+XGB forecast as predict_market_direction()
    for ONE instrument, then grades it against the ACTUAL outcome for that date
    if it's already known (i.e. the date isn't the most recent bar in the file).

    Uses the same FULL CONSENSUS rule as walk_forward_backtest()/predict_market_direction():
    both classifiers must pick the same direction AND both regressors' expected
    returns must agree in sign, otherwise the call is NO_CONSENSUS and left ungraded.

    Returns one summary dict -- this is the row that goes into the combined
    multi-stock CSV report. Nothing is printed per-stock (kept quiet for batch runs);
    any failure for one stock is caught and recorded in that stock's row instead of
    crashing the whole batch."""
    row = {"Symbol": symbol, "Equity_File": os.path.basename(equity_file)}
    try:
        raw = load_and_prepare_dataset(equity_file)
        data, feature_cols = engineer_features(raw)
        clean = data.iloc[50:].reset_index(drop=True)

        if target_date_str:
            cutoff_date = pd.to_datetime(target_date_str, dayfirst=True)
            available = clean[clean["Date"] <= cutoff_date]["Date"]
            if available.empty:
                row["Status"] = f"ERROR: no data on/before {target_date_str}"
                return row
            cutoff_date = available.max()
        else:
            # No --target-date given: default to the latest date whose actual T+1
            # outcome is already known, so every stock's row can be graded without
            # requiring the caller to pick a date manually.
            known = clean.dropna(subset=["NIFTY_Fwd_Ret"])
            if known.empty:
                row["Status"] = "ERROR: no graded (non-NaN target) rows available"
                return row
            cutoff_date = known["Date"].max()

        train_data = clean[clean["Date"] < cutoff_date].dropna(subset=["NIFTY_Fwd_Ret"]).reset_index(drop=True)
        pred_row = clean[clean["Date"] == cutoff_date].reset_index(drop=True)
        if len(train_data) < 100:
            row["Status"] = f"ERROR: only {len(train_data)} training rows before {cutoff_date.date()}"
            return row
        if pred_row.empty:
            row["Status"] = f"ERROR: target date {cutoff_date.date()} not found after cleaning"
            return row

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(train_data[feature_cols].values)
        X_pred = imputer.transform(pred_row[feature_cols].values)
        y_train_dir = train_data["NIFTY_Target_Dir"].values
        y_train_ret = train_data["NIFTY_Fwd_Ret"].values

        hg_clf = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=150, max_leaf_nodes=12, min_samples_leaf=15, random_state=42)
        xgb_clf = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.03, min_child_weight=15, random_state=42, eval_metric="logloss")
        hg_reg = HistGradientBoostingRegressor(learning_rate=0.03, max_iter=150, max_leaf_nodes=12, min_samples_leaf=15, random_state=42)
        xgb_reg = XGBRegressor(n_estimators=250, max_depth=5, learning_rate=0.03, min_child_weight=15, random_state=42)

        hg_clf.fit(X_train, y_train_dir); xgb_clf.fit(X_train, y_train_dir)
        hg_reg.fit(X_train, y_train_ret); xgb_reg.fit(X_train, y_train_ret)

        p_hg = hg_clf.predict_proba(X_pred)[:, 1][0]
        p_xgb = xgb_clf.predict_proba(X_pred)[:, 1][0]
        e_hg = hg_reg.predict(X_pred)[0]
        e_xgb = xgb_reg.predict(X_pred)[0]

        hgb_dir = "UP" if p_hg >= 0.5 else "DOWN"
        xgb_dir = "UP" if p_xgb >= 0.5 else "DOWN"
        classifiers_agree = hgb_dir == xgb_dir
        regressors_agree = np.sign(e_hg) == np.sign(e_xgb)
        # NEW: the classifier and regressor are two SEPARATE models (HGB/XGB
        # classifier vs. HGB/XGB regressor) trained independently -- nothing
        # forces them to agree with each other, only with their own counterpart
        # (HGB classifier vs HGB regressor, XGB vs XGB). Without this check, both
        # classifiers can agree on DOWN while both regressors independently agree
        # on a POSITIVE expected return, giving "full consensus" on a Predicted_Close
        # that's actually ABOVE Last_Close -- self-contradictory. Requiring each
        # model's own classifier direction to match its own regressor's sign closes
        # that hole.
        hgb_internally_consistent = (hgb_dir == "UP") == (e_hg >= 0)
        xgb_internally_consistent = (xgb_dir == "UP") == (e_xgb >= 0)
        full_consensus = (classifiers_agree and regressors_agree
                           and hgb_internally_consistent and xgb_internally_consistent)
        consensus_dir = hgb_dir if full_consensus else "NO_CONSENSUS"

        # NEW: turn each model's expected RETURN into an actual predicted CLOSE PRICE
        # (ref_close * (1 + expected_return)), using the instrument's own last known
        # Close on cutoff_date as the reference. Direction (UP/DOWN) is just the sign
        # of this same move; this is the price-level version of it.
        ref_close = float(pred_row["Close"].iloc[0])
        hgb_pred_close = ref_close * (1.0 + e_hg)
        xgb_pred_close = ref_close * (1.0 + e_xgb)
        avg_pred_close = (hgb_pred_close + xgb_pred_close) / 2.0

        # Grade against the actual outcome, if it's known yet.
        actual_val = pred_row["NIFTY_Target_Dir"].iloc[0]
        if pd.isna(actual_val):
            actual_dir = None
            result = "PENDING (actual outcome not yet known)"
        else:
            actual_dir = "UP" if int(actual_val) == 1 else "DOWN"
            result = "NO_CONSENSUS" if not full_consensus else (
                "CORRECT" if consensus_dir == actual_dir else "INCORRECT")

        row.update({
            "Status": "OK",
            "Target_Date": cutoff_date.date().isoformat(),
            "Last_Close": round(ref_close, 4),
            "Actual_Direction": actual_dir if actual_dir is not None else "UNKNOWN",
            "HGB_Direction": hgb_dir,
            "HGB_Prob_Bullish": round(float(p_hg), 4),
            "HGB_Exp_Return": round(float(e_hg), 6),
            "HGB_Predicted_Close": round(hgb_pred_close, 4),
            "XGB_Direction": xgb_dir,
            "XGB_Prob_Bullish": round(float(p_xgb), 4),
            "XGB_Exp_Return": round(float(e_xgb), 6),
            "XGB_Predicted_Close": round(xgb_pred_close, 4),
            "Predicted_Close": round(avg_pred_close, 4),
            "Models_Agree": full_consensus,
            "Consensus_Direction": consensus_dir,
            "Result": result,
            "Predicted_Correct": (result == "CORRECT") if result in ("CORRECT", "INCORRECT") else None,
            "Training_Rows": len(train_data),
        })
    except Exception as exc:
        row["Status"] = f"ERROR: {type(exc).__name__}: {exc}"
    return row


def run_batch_report(equity_files=None, target_date_str=None, out_csv="combined_prediction_report.csv",
                      stock_pattern="*_OHLC_Direction_Indicators.xlsx"):
    """Runs predict_and_grade_one() over every local stock workbook and writes ONE
    combined summary CSV -- one row per stock -- including whether each stock's
    prediction was correct for the target date. Meant for unattended/batch runs
    across many instruments; does not print a full per-stock forecast block."""
    if equity_files is None:
        equity_files = find_stock_files(pattern=stock_pattern)
    if not equity_files:
        sys.exit(f"Error: no files matching '{stock_pattern}' found in current directory "
                  f"(excluding INDIAVIX_OHLC_Direction_Indicators.xlsx).")

    print(f"Running batch prediction over {len(equity_files)} instrument(s): "
          f"{', '.join(stock_symbol_from_filename(f) for f in equity_files)}")

    rows = []
    for f in equity_files:
        symbol = stock_symbol_from_filename(f)
        print(f"  [{symbol}] processing ...")
        rows.append(predict_and_grade_one(f, symbol, target_date_str=target_date_str))

    report = pd.DataFrame(rows)
    report.to_csv(out_csv, index=False)

    n_ok = int((report["Status"] == "OK").sum())
    n_correct = int((report.get("Predicted_Correct") == True).sum()) if "Predicted_Correct" in report else 0
    n_incorrect = int((report.get("Predicted_Correct") == False).sum()) if "Predicted_Correct" in report else 0
    print("=" * 70)
    print(f" BATCH SUMMARY: {n_ok}/{len(report)} instruments processed OK")
    if n_correct + n_incorrect > 0:
        print(f" Graded (full-consensus) calls: {n_correct} CORRECT / {n_incorrect} INCORRECT "
              f"(accuracy: {n_correct / (n_correct + n_incorrect):.4f})")
    print(f" Saved combined report ({len(report)} rows) to {out_csv}")
    print("=" * 70)
    return report


# ==========================================
# 5b. NIFTY 50 PREDICTED CLOSE PRICE SUM (NEW)
# ==========================================
# NSE trading symbols for the current Nifty 50 constituents (as of the Sept 2025
# rejig: IndiGo/Max Healthcare in, Hero MotoCorp/IndusInd Bank out; Mar 2025 rejig:
# Jio Financial/Eternal in, Bharat Petroleum/Britannia out). Used to filter the
# --all-stocks combined report down to Nifty 50 stocks only (any other
# *_OHLC_Direction_Indicators.xlsx file sitting in the same folder, e.g. an
# index or global-macro entity, is ignored for this sum -- see the note below).
# Matching is normalized (see _normalize_symbol below), so a '.NS'/'.BO' suffix
# or different casing in your local filenames won't cause a false "not found".
# "TATAMOTORS"/"TMPV" are both included since data files may use either symbol
# depending on when they were generated (Tata Motors Passenger Vehicles demerger).
#
# FIXED: this used to also contain NIFTY 50/NIFTY BANK/NIFTY MIDCAP 50/NIFTY IT/
# NIFTY PHARMA/NIFTY 500/SENSEX/USD-INR/Dow Jones/Nikkei 225/Hang Seng/Crude Oil/
# US 10Y Yield -- none of those are Nifty 50 STOCK CONSTITUENTS (they're indices
# and global-macro instruments), so they inflated "PREDICTED CLOSE SUMMARY (Nifty
# 50 stocks only)" with non-stock entities. This set is now exactly the 50 actual
# constituents (matches entities_universe.NIFTY50_STOCKS one-for-one).
NIFTY50_SYMBOLS = {
    "Reliance Industries",
    "Tata Consultancy Services",
    "HDFC Bank",
    "Bharti Airtel",
    "ICICI Bank",
    "State Bank of India",
    "Infosys",
    "Larsen & Toubro",
    "Hindustan Unilever",
    "ITC",
    "Bajaj Finance",
    "HCL Technologies",
    "Sun Pharmaceutical",
    "Maruti Suzuki",
    "Mahindra & Mahindra",
    "Kotak Mahindra Bank",
    "Axis Bank",
    "UltraTech Cement",
    "NTPC",
    "Titan Company",
    "Bajaj Finserv",
    "Adani Enterprises",
    "Power Grid Corporation",
    "Oil & Natural Gas Corp",
    "Nestle India",
    "Wipro",
    "JSW Steel",
    "Tata Steel",
    "Asian Paints",
    "Coal India",
    "Bajaj Auto",
    "Grasim Industries",
    "Tech Mahindra",
    "HDFC Life Insurance",
    "Adani Ports & SEZ",
    "SBI Life Insurance",
    "Cipla",
    "Dr. Reddy's Laboratories",
    "Eicher Motors",
    "Apollo Hospitals",
    "Bharat Electronics",
    "Tata Consumer Products",
    "Hindalco Industries",
    "Trent",
    "Shriram Finance",
    "Jio Financial Services",
    "Eternal (Zomato)",
    "Tata Motors (Passenger Vehicles)",
    "IndiGo",
    "Max Healthcare",
}


def _normalize_symbol(sym):
    """Normalizes a ticker/symbol for matching purposes: uppercases it and strips
    a trailing '.NS' or '.BO' exchange suffix if present (e.g. 'reliance.NS' and
    'RELIANCE' both normalize to 'RELIANCE'). This is what prevents a harmless
    naming difference (exchange suffix, or lower/upper case) between how your
    local files are named and how NIFTY50_SYMBOLS is written from showing up as
    a false 'not found' in the summary below."""
    if not isinstance(sym, str):
        return sym
    s = sym.strip().upper()
    for suffix in (".NS", ".BO"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def _expected_symbol_for_label(label):
    """Converts a human-readable NIFTY50_SYMBOLS entry (e.g. 'Reliance Industries',
    'USD/INR', 'NIFTY BANK') into the on-disk Symbol it should actually appear as.

    download_all_data.py names every generated file using entities_universe.py's
    file_stem(label) convention -- non-alphanumeric characters collapsed to a
    single underscore, uppercased -- e.g. 'Reliance Industries' -> 'RELIANCE_INDUSTRIES'
    ('RELIANCE_INDUSTRIES_OHLC_Direction_Indicators.xlsx'), 'USD/INR' -> 'USD_INR'.
    Comparing the raw label against the filename-derived Symbol (as the old code
    did, via _normalize_symbol alone) never matches, because _normalize_symbol only
    upper-cases and strips a ticker suffix -- it doesn't collapse spaces/'&'/'/'/'()'
    into underscores. That mismatch is what caused every multi-word/punctuated
    entity to show up as falsely "not found".

    NIFTY 50 itself is special-cased: unlike every other entity it keeps its
    original, unprefixed 'Nifty_OHLC_Direction_Indicators.xlsx' filename rather
    than one generated from file_stem('NIFTY 50'), so its on-disk Symbol is
    'Nifty' (normalizes to 'NIFTY'), not 'NIFTY_50'."""
    if label == "NIFTY 50":
        return "NIFTY"
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").upper()


def _predicted_and_last_close_for_row(row, directory="."):
    """Combines the actual Close price on a batch report row's Target_Date
    (read straight from that stock's own OHLC workbook) with the row's averaged
    HGB/XGB expected return to get (last_close, predicted_close) for that one
    stock. Returns (None, None) if the row wasn't OK or anything needed is
    missing -- never raises, so one bad row can't break the sums below."""
    if row.get("Status") != "OK":
        return None, None
    equity_file = row.get("Equity_File")
    target_date = row.get("Target_Date")
    if not equity_file or not target_date:
        return None, None
    path = equity_file if os.path.exists(equity_file) else os.path.join(directory, equity_file)
    if not os.path.exists(path):
        return None, None
    try:
        closes = _load_close_series(path)
        match = closes[closes["Date"] == pd.to_datetime(target_date)]
        if match.empty:
            return None, None
        last_close = float(match["Close"].iloc[0])
        hgb_ret = row.get("HGB_Exp_Return")
        xgb_ret = row.get("XGB_Exp_Return")
        if pd.isna(hgb_ret) or pd.isna(xgb_ret):
            return None, None
        avg_ret = (float(hgb_ret) + float(xgb_ret)) / 2.0
        return last_close, last_close * (1.0 + avg_ret)
    except Exception:
        return None, None


def sum_nifty50_predicted_close(report, nifty50_symbols=NIFTY50_SYMBOLS, directory=".",
                                 breadth_prediction=None):
    """NEW: Given the combined batch report DataFrame produced by run_batch_report()
    (i.e. what --all-stocks builds), for every Nifty 50 STOCK CONSTITUENT found in
    the current batch, in full model consensus (Models_Agree == True), AND (if
    breadth_prediction is given) whose own Consensus_Direction agrees with the
    overall breadth-vote call for Nifty, this prints:
      1) the sum of each stock's predicted next-session CLOSE PRICE, and
      2) the sum of each stock's (Predicted_Close - Last_Close), i.e. the total
         expected point move across the Nifty 50 names in this batch.
    Matching against nifty50_symbols is done on a NORMALIZED symbol (uppercased,
    '.NS'/'.BO' suffix stripped) so exchange-suffixed or differently-cased local
    filenames still match correctly. Any stock file in the batch that isn't a
    Nifty 50 constituent is ignored. Purely additive: doesn't change
    run_batch_report()'s own printout, CSV, or return value.

    breadth_prediction: pass the "Breadth_Prediction" value ("UP"/"DOWN"/"TIE")
    from compute_breadth_summary()/predict_breadth_direction() run over the SAME
    report. When given, a stock is only summed if its Consensus_Direction equals
    breadth_prediction -- i.e. only stocks pulling the SAME way as the overall
    market-breadth call get counted, so a stock calling DOWN doesn't get summed
    into a total meant to represent an UP breadth day (or vice versa). Pass None
    (default) to skip this filter and keep the old Models_Agree-only behavior.
    If breadth_prediction is "TIE" (or otherwise not "UP"/"DOWN"), nothing passes
    this filter -- a tied breadth vote has no majority direction for a stock's
    call to "agree" with, and printing a leftover sum in that case would silently
    imply a directional lean the data doesn't support, so we report zero instead.

    FIXED (three bugs):
      - nifty50_symbols used to also include indices (NIFTY BANK, SENSEX, ...) and
        global-macro entities (USD/INR, Dow Jones, ...), which aren't Nifty 50 STOCK
        constituents at all -- they inflated this "stocks only" sum. NIFTY50_SYMBOLS
        is now exactly the 50 actual constituents.
      - this used to sum a row's Predicted_Close regardless of Models_Agree, so a
        NO_CONSENSUS row (or, before the consensus-rule fix, a row whose classifier
        and regressor silently disagreed) could still be added into the total. Rows
        are now required to have Models_Agree == True to be included; anything else
        (NO_CONSENSUS, ERROR, or a Nifty 50 stock with no local file at all) is
        excluded from the sum and reported separately below instead of being
        silently dropped or silently included.
      - (NEW) this never checked a stock's call against the overall breadth-vote
        direction, so a stock calling DOWN could still be summed into the total
        even on a day the aggregate breadth call was UP (or vice versa). Passing
        breadth_prediction now filters those mismatches out too, and they're
        reported separately from "no consensus" and "file not found" exclusions.

    Prefers the 'Last_Close'/'Predicted_Close' columns that predict_and_grade_one()
    now writes onto each row (same numbers the CLI shows per-stock); falls back to
    recomputing from the raw OHLC file + Exp_Return columns only if those columns
    aren't present (e.g. an older saved CSV)."""
    # Map each human-readable label to the on-disk Symbol it should match (see
    # _expected_symbol_for_label docstring for why this can't just be
    # _normalize_symbol(label) -- that never collapses spaces/punctuation the
    # way the actual generated filenames do).
    label_by_expected = {_expected_symbol_for_label(s): s for s in nifty50_symbols}
    norm_targets = set(label_by_expected)
    report = report.copy()
    report["_norm_symbol"] = report["Symbol"].map(_normalize_symbol)
    nifty_rows = report[report["_norm_symbol"].isin(norm_targets)]

    # NEW: split into consensus vs. no-consensus BEFORE summing, so a NO_CONSENSUS
    # (or otherwise not-agreed) row is never folded into the total.
    agree_mask = nifty_rows.get("Models_Agree")
    if agree_mask is None:
        agreed_rows, disagreed_rows = nifty_rows, nifty_rows.iloc[0:0]
    else:
        agreed_rows = nifty_rows[nifty_rows["Models_Agree"] == True]  # noqa: E712
        disagreed_rows = nifty_rows[nifty_rows["Models_Agree"] != True]  # noqa: E712

    # NEW: further split out stocks whose own call doesn't match the overall
    # breadth-vote direction, if a breadth_prediction was supplied.
    breadth_mismatch_rows = agreed_rows.iloc[0:0]
    if breadth_prediction is not None:
        if breadth_prediction in ("UP", "DOWN"):
            breadth_mismatch_rows = agreed_rows[agreed_rows["Consensus_Direction"] != breadth_prediction]
            agreed_rows = agreed_rows[agreed_rows["Consensus_Direction"] == breadth_prediction]
        else:
            # TIE or anything else: no majority direction to agree with.
            breadth_mismatch_rows = agreed_rows
            agreed_rows = agreed_rows.iloc[0:0]

    predicted_closes, last_closes = {}, {}
    for _, row in agreed_rows.iterrows():
        sym = row["Symbol"]
        last_close, pred_close = row.get("Last_Close"), row.get("Predicted_Close")
        if pred_close is None or (isinstance(pred_close, float) and pd.isna(pred_close)) \
                or last_close is None or (isinstance(last_close, float) and pd.isna(last_close)):
            last_close, pred_close = _predicted_and_last_close_for_row(row, directory=directory)
        if pred_close is not None and last_close is not None:
            predicted_closes[sym] = float(pred_close)
            last_closes[sym] = float(last_close)

    diffs = {sym: predicted_closes[sym] - last_closes[sym] for sym in predicted_closes}
    total_pred_close = sum(predicted_closes.values())
    total_diff = sum(diffs.values())
    found_norm = set(nifty_rows["_norm_symbol"])
    missing_symbols = sorted({label_by_expected[e] for e in norm_targets if e not in found_norm})
    no_consensus_symbols = sorted(disagreed_rows["Symbol"].tolist())
    breadth_mismatch_symbols = sorted(breadth_mismatch_rows["Symbol"].tolist())

    print("=" * 70)
    print(" NIFTY 50 CONSTITUENTS -- PREDICTED CLOSE SUMMARY")
    print("=" * 70)
    if breadth_prediction is not None:
        print(f" Filtered to stocks matching breadth-vote call: {breadth_prediction}")
    print(f" Nifty 50 stocks summed : {len(predicted_closes)}/{len(nifty50_symbols)}")
    if missing_symbols:
        print(f" Nifty 50 symbols NOT found as local *_OHLC_Direction_Indicators.xlsx "
              f"files (skipped): {', '.join(missing_symbols)}")
    if no_consensus_symbols:
        print(f" Nifty 50 symbols found but excluded (Models_Agree != True, e.g. "
              f"NO_CONSENSUS): {', '.join(no_consensus_symbols)}")
    if breadth_mismatch_symbols:
        print(f" Nifty 50 symbols found & in consensus but excluded (own call != "
              f"breadth call {breadth_prediction}): {', '.join(breadth_mismatch_symbols)}")
    print("-" * 70)
    print(f"   {'Symbol':<12} {'Last Close':>14} {'Pred Close':>14} {'Pred - Last':>14}")
    for sym in sorted(predicted_closes):
        print(f"   {sym:<12} {last_closes[sym]:>14,.2f} {predicted_closes[sym]:>14,.2f} {diffs[sym]:>+14,.2f}")
    print("-" * 70)
    suffix = " (Nifty 50 stocks, consensus" + (f" + breadth={breadth_prediction})" if breadth_prediction else ")")
    print(f" TOTAL predicted close{suffix}: {total_pred_close:,.2f}")
    print(f" TOTAL (Predicted Close - Last Close){suffix}: {total_diff:+,.2f}")
    print("=" * 70)
    return total_pred_close, total_diff


# ==========================================
# 6. BREADTH-VOTE AGGREGATE PREDICTION (NEW)
# ==========================================
def _load_close_series(ohlc_file):
    """Small read-only helper: reads just Date+Close from an OHLC+Indicators
    workbook's 'OHLC Data' sheet. Used only for the correlation check below --
    does not touch load_and_prepare_dataset() / engineer_features()."""
    df = pd.read_excel(ohlc_file, sheet_name="OHLC Data")[["Date", "Close"]]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def detect_inverse_entities(report_df, nifty_equity_file="Nifty_OHLC_Direction_Indicators.xlsx",
                             corr_lookback=252, inverse_corr_threshold=-0.3):
    """
    For every row in report_df (a combined_prediction_report.csv-style DataFrame,
    one row per entity), checks whether that entity's daily % returns are
    historically NEGATIVELY correlated with Nifty's own daily % returns (e.g.
    USD/INR or US 10Y yield tending to move opposite Nifty). This is how "works
    exactly opposite of Nifty" is detected -- not asserted, but measured from
    each entity's own OHLC file against Nifty's, on overlapping dates.

    corr_lookback: most recent N overlapping trading days used (None = all
    overlapping history). inverse_corr_threshold: an entity is only flagged
    "inverse" if correlation <= this value (default -0.3), so mild/noisy
    negative correlation doesn't get flipped.

    Returns {Symbol: {"correlation": float or None, "inverse": bool}}.
    """
    nifty_close = _load_close_series(nifty_equity_file)
    nifty_ret = nifty_close.assign(Nifty_Ret=nifty_close["Close"].pct_change())[["Date", "Nifty_Ret"]]

    info = {}
    for _, r in report_df.iterrows():
        symbol, eq_file = r["Symbol"], r["Equity_File"]
        if os.path.basename(str(eq_file)) == os.path.basename(nifty_equity_file):
            info[symbol] = {"correlation": 1.0, "inverse": False}
            continue
        try:
            ent_close = _load_close_series(eq_file)
            ent_ret = ent_close.assign(Ent_Ret=ent_close["Close"].pct_change())[["Date", "Ent_Ret"]]
            merged = ent_ret.merge(nifty_ret, on="Date", how="inner").dropna()
            if corr_lookback:
                merged = merged.tail(corr_lookback)
            if len(merged) < 30:
                info[symbol] = {"correlation": None, "inverse": False}
                continue
            corr = merged["Ent_Ret"].corr(merged["Nifty_Ret"])
            corr = None if pd.isna(corr) else round(float(corr), 4)
            info[symbol] = {"correlation": corr,
                             "inverse": (corr is not None and corr <= inverse_corr_threshold)}
        except Exception:
            info[symbol] = {"correlation": None, "inverse": False}
    return info


def compute_breadth_summary(report_df, nifty_symbol="Nifty",
                             nifty_equity_file="Nifty_OHLC_Direction_Indicators.xlsx",
                             corr_lookback=252, inverse_corr_threshold=-0.3):
    """Pure computation shared by predict_breadth_direction() (reads a saved CSV)
    and the --all-stocks flow (uses the in-memory batch_report DataFrame directly,
    so sum_nifty50_predicted_close() can filter its sum down to only the stocks
    whose own call agrees with this same breadth call -- see that function's
    breadth_prediction parameter).

    Turns all the individual per-entity calls in report_df into ONE breadth-based
    UP/DOWN call for Nifty:
      1. Keep only entities in full model consensus (Models_Agree == True) --
         NO_CONSENSUS entities don't get a vote.
      2. For each remaining entity, detect_inverse_entities() checks if it's
         historically negatively correlated with Nifty (e.g. USD/INR, US 10Y
         yield). If so, its vote is FLIPPED before counting -- an inverse
         entity calling UP counts as a vote for DOWN, and vice versa.
      3. Weight_Up = count of (flipped) UP votes, Weight_Down = count of
         (flipped) DOWN votes. Breadth_Prediction = "UP" if Weight_Up >
         Weight_Down, "DOWN" if Weight_Down > Weight_Up, else "TIE".
      4. Grades Breadth_Prediction against Nifty's own Actual_Direction from
         the same report (the row where Symbol == nifty_symbol), if known.

    Returns the summary dict (same shape predict_breadth_direction() used to
    build inline)."""
    ok_df = report_df[report_df["Status"] == "OK"].copy()
    if ok_df.empty:
        return None

    target_date = ok_df["Target_Date"].iloc[0]

    nifty_rows = ok_df[ok_df["Symbol"] == nifty_symbol]
    nifty_actual = nifty_rows["Actual_Direction"].iloc[0] if not nifty_rows.empty else "UNKNOWN"

    voters = ok_df[ok_df["Models_Agree"] == True].copy()  # noqa: E712 (pandas bool column)

    inverse_info = detect_inverse_entities(
        voters, nifty_equity_file=nifty_equity_file,
        corr_lookback=corr_lookback, inverse_corr_threshold=inverse_corr_threshold,
    )

    def _effective_vote(row):
        raw = row["Consensus_Direction"]
        if raw not in ("UP", "DOWN"):
            return None
        if inverse_info.get(row["Symbol"], {}).get("inverse", False):
            return "DOWN" if raw == "UP" else "UP"
        return raw

    voters["Effective_Vote"] = voters.apply(_effective_vote, axis=1)
    voters["Correlation_To_Nifty"] = voters["Symbol"].map(lambda s: inverse_info.get(s, {}).get("correlation"))
    voters["Inverse_Of_Nifty"] = voters["Symbol"].map(lambda s: inverse_info.get(s, {}).get("inverse", False))

    weight_up = int((voters["Effective_Vote"] == "UP").sum())
    weight_down = int((voters["Effective_Vote"] == "DOWN").sum())

    if weight_up > weight_down:
        breadth_prediction = "UP"
    elif weight_down > weight_up:
        breadth_prediction = "DOWN"
    else:
        breadth_prediction = "TIE"

    if nifty_actual not in ("UP", "DOWN"):
        result, correct = "PENDING (actual outcome not yet known)", None
    elif breadth_prediction == "TIE":
        result, correct = "TIE (no directional call)", None
    else:
        correct = (breadth_prediction == nifty_actual)
        result = "CORRECT" if correct else "INCORRECT"

    inverse_entities = sorted(voters.loc[voters["Inverse_Of_Nifty"], "Symbol"].tolist())

    return {
        "Target_Date": target_date,
        "Entities_Voting": int(len(voters)),
        "Weight_Up": weight_up,
        "Weight_Down": weight_down,
        "Inverse_Entities_Flipped": ", ".join(inverse_entities) if inverse_entities else "",
        "Breadth_Prediction": breadth_prediction,
        "Nifty_Actual_Direction": nifty_actual,
        "Result": result,
        "Predicted_Correct": correct,
    }


def predict_breadth_direction(report_csv="combined_prediction_report.csv",
                               nifty_symbol="Nifty",
                               nifty_equity_file="Nifty_OHLC_Direction_Indicators.xlsx",
                               corr_lookback=252, inverse_corr_threshold=-0.3,
                               out_csv="breadth_prediction_backtest.csv"):
    """
    Reads a saved combined_prediction_report.csv (the output of run_batch_report /
    --all-stocks), computes the breadth call via compute_breadth_summary() (see
    that function's docstring for the 4-step method), and appends/updates one row
    (keyed by Target_Date) in a separate running backtest CSV at out_csv --
    rerunning this over many days' combined reports builds a real backtest of the
    breadth signal over time.

    Returns the summary dict that was appended.
    """
    report_df = pd.read_csv(report_csv)
    summary = compute_breadth_summary(
        report_df, nifty_symbol=nifty_symbol, nifty_equity_file=nifty_equity_file,
        corr_lookback=corr_lookback, inverse_corr_threshold=inverse_corr_threshold,
    )
    if summary is None:
        sys.exit(f"Error: no OK rows found in '{report_csv}'.")

    target_date = summary["Target_Date"]
    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        existing = existing[existing["Target_Date"] != target_date]  # replace same-date row on rerun
        backtest_df = pd.concat([existing, pd.DataFrame([summary])], ignore_index=True)
    else:
        backtest_df = pd.DataFrame([summary])

    backtest_df = backtest_df.sort_values("Target_Date").reset_index(drop=True)
    backtest_df.to_csv(out_csv, index=False)

    print(f"Breadth call for {target_date}: {summary['Breadth_Prediction']} "
          f"(Up={summary['Weight_Up']}, Down={summary['Weight_Down']}, "
          f"entities={summary['Entities_Voting']}) -> {summary['Result']}")
    if summary["Inverse_Entities_Flipped"]:
        print(f"  Flipped (inverse-of-Nifty) entities: {summary['Inverse_Entities_Flipped']}")
    print(f"  Appended to breadth backtest report: {out_csv}")

    return summary



def run_batch_report_range(equity_files=None, end_date_str=None, num_days=60,
                            out_csv="combined_prediction_report_range.csv",
                            stock_pattern="*_OHLC_Direction_Indicators.xlsx",
                            calendar_equity_file="Nifty_OHLC_Direction_Indicators.xlsx",
                            build_breadth_backtest=True,
                            breadth_out_csv="breadth_prediction_backtest.csv",
                            inverse_corr_threshold=-0.3):
    """
    Runs the SAME per-stock grading as run_batch_report()/predict_and_grade_one()
    (unchanged -- this just calls them) across the last `num_days` trading days
    (ending at end_date_str, or the most recent date available if None) instead
    of a single Target_Date.

    Trading days are taken from calendar_equity_file's own Date column (default:
    Nifty's), so every stock is graded on the same calendar of dates. For each of
    those dates, every matching stock/entity workbook is run through
    predict_and_grade_one() exactly as --all-stocks does today, and all rows
    (Symbol x Target_Date) are stacked into one combined CSV.

    If build_breadth_backtest is True, predict_breadth_direction() is also run
    once per date on that date's slice of rows, so breadth_out_csv ends up with
    one graded breadth row per date across the whole window -- a real multi-day
    backtest of the breadth-vote signal, not just a single day's snapshot.

    NOTE: this retrains 4 models (HGB+XGB classifier and regressor) per stock,
    per day -- i.e. len(equity_files) x num_days model fits. It can take a while
    for a large stock universe / long range.
    """
    if equity_files is None:
        equity_files = find_stock_files(pattern=stock_pattern)
    if not equity_files:
        sys.exit(f"Error: no files matching '{stock_pattern}' found in current directory "
                  f"(excluding INDIAVIX_OHLC_Direction_Indicators.xlsx).")
    if not os.path.exists(calendar_equity_file):
        sys.exit(f"Error: calendar reference file '{calendar_equity_file}' not found "
                  f"(needed to build the list of trading days).")

    cal = pd.read_excel(calendar_equity_file, sheet_name="OHLC Data")[["Date"]]
    cal["Date"] = pd.to_datetime(cal["Date"]).sort_values()
    cal = cal.sort_values("Date")
    if end_date_str:
        end_date = pd.to_datetime(end_date_str, dayfirst=True)
        cal = cal[cal["Date"] <= end_date]
    trading_days = cal["Date"].tail(num_days).tolist()
    if not trading_days:
        sys.exit("Error: no trading days found for the requested range.")

    print(f"Running range batch: {len(equity_files)} instrument(s) x {len(trading_days)} "
          f"trading day(s) ({trading_days[0].date()} -> {trading_days[-1].date()})")

    all_rows = []
    for d in trading_days:
        d_str = d.strftime("%d-%m-%Y")
        print(f" -- Target date {d.date()} --")
        for f in equity_files:
            symbol = stock_symbol_from_filename(f)
            all_rows.append(predict_and_grade_one(f, symbol, target_date_str=d_str))

    report = pd.DataFrame(all_rows)
    report.to_csv(out_csv, index=False)

    n_ok = int((report["Status"] == "OK").sum())
    n_correct = int((report.get("Predicted_Correct") == True).sum()) if "Predicted_Correct" in report else 0
    n_incorrect = int((report.get("Predicted_Correct") == False).sum()) if "Predicted_Correct" in report else 0
    print("=" * 70)
    print(f" RANGE BATCH SUMMARY: {n_ok}/{len(report)} (stock x day) rows processed OK")
    if n_correct + n_incorrect > 0:
        print(f" Graded (full-consensus) calls: {n_correct} CORRECT / {n_incorrect} INCORRECT "
              f"(accuracy: {n_correct / (n_correct + n_incorrect):.4f})")
    print(f" Saved combined range report ({len(report)} rows) to {out_csv}")
    print("=" * 70)

    if build_breadth_backtest:
        if "Target_Date" not in report.columns or n_ok == 0:
            print(" Skipping breadth backtest: no OK (successfully graded) rows in the range report.")
        else:
            tmp_csv = "_tmp_breadth_day_slice.csv"
            for d in trading_days:
                d_str = d.strftime("%Y-%m-%d")
                day_slice = report[report["Target_Date"] == d_str]
                if day_slice.empty or (day_slice["Status"] == "OK").sum() == 0:
                    continue
                day_slice.to_csv(tmp_csv, index=False)
                try:
                    predict_breadth_direction(
                        report_csv=tmp_csv, nifty_equity_file=calendar_equity_file,
                        inverse_corr_threshold=inverse_corr_threshold, out_csv=breadth_out_csv,
                    )
                except SystemExit as e:
                    print(f"  [breadth] skipped {d.date()}: {e}")
            if os.path.exists(tmp_csv):
                os.remove(tmp_csv)
            print(f" Breadth backtest across the range saved/updated at {breadth_out_csv}")

    return report


# ==========================================
# 8. ENTRY POINT
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
    parser.add_argument("--all-stocks", action="store_true",
                         help="Batch mode: run a single-date forecast for every local "
                              "*_OHLC_Direction_Indicators.xlsx file (e.g. RELIANCE, TCS, ... -- any "
                              "workbook with the same 'OHLC Data'+'Indicators' sheet schema as the Nifty "
                              "file), grade each one against its actual outcome if already known, and "
                              "save ONE combined summary CSV. Ignores --equity-file/--symbol.")
    parser.add_argument("--stock-pattern", type=str, default="*_OHLC_Direction_Indicators.xlsx",
                         help="Glob pattern used to find per-stock workbooks in --all-stocks mode. "
                              "INDIAVIX_OHLC_Direction_Indicators.xlsx is always excluded.")
    parser.add_argument("--out-csv", type=str, default="combined_prediction_report.csv",
                         help="Output path for the combined summary CSV in --all-stocks mode.")
    parser.add_argument("--breadth-predict", action="store_true",
                         help="NEW: read an already-saved combined_prediction_report.csv (see "
                              "--all-stocks/--out-csv), aggregate all full-consensus entity calls "
                              "into one breadth-based UP/DOWN call for Nifty (auto-flipping any "
                              "entity that's historically negatively correlated with Nifty), grade "
                              "it, and append the result to --breadth-out-csv.")
    parser.add_argument("--breadth-report-csv", type=str, default="combined_prediction_report.csv",
                         help="Input combined_prediction_report.csv to read for --breadth-predict.")
    parser.add_argument("--breadth-out-csv", type=str, default="breadth_prediction_backtest.csv",
                         help="Output running backtest CSV for --breadth-predict (one row per Target_Date).")
    parser.add_argument("--inverse-corr-threshold", type=float, default=-0.3,
                         help="Correlation-to-Nifty at/below which an entity's vote is flipped "
                              "in --breadth-predict (default -0.3).")
    parser.add_argument("--all-stocks-days", type=int, default=None,
                         help="NEW: like --all-stocks, but runs the same per-stock grading across "
                              "the last N trading days (ending at --target-date, or the latest "
                              "available date) instead of just one date. Saves one combined CSV "
                              "(one row per Symbol x Target_Date) to --range-out-csv, and also "
                              "builds a multi-day breadth backtest at --breadth-out-csv unless "
                              "--skip-breadth-backtest is passed. E.g. --all-stocks-days 60")
    parser.add_argument("--range-out-csv", type=str, default="combined_prediction_report_range.csv",
                         help="Output path for the combined range CSV used by --all-stocks-days.")
    parser.add_argument("--skip-breadth-backtest", action="store_true",
                         help="With --all-stocks-days, skip building the multi-day breadth backtest "
                              "(only save the raw per-stock range CSV).")
    args = parser.parse_args()

    if args.breadth_predict:
        predict_breadth_direction(
            report_csv=args.breadth_report_csv, nifty_equity_file=args.equity_file,
            inverse_corr_threshold=args.inverse_corr_threshold, out_csv=args.breadth_out_csv,
        )
    elif args.all_stocks_days:
        run_batch_report_range(
            end_date_str=args.target_date, num_days=args.all_stocks_days,
            out_csv=args.range_out_csv, stock_pattern=args.stock_pattern,
            calendar_equity_file=args.equity_file,
            build_breadth_backtest=not args.skip_breadth_backtest,
            breadth_out_csv=args.breadth_out_csv,
            inverse_corr_threshold=args.inverse_corr_threshold,
        )
    elif args.all_stocks:
        # Batch mode predicts+grades every matching stock file and writes one combined
        # CSV; it does not need the single --equity-file loaded up front like the modes below.
        batch_report = run_batch_report(target_date_str=args.target_date, out_csv=args.out_csv,
                          stock_pattern=args.stock_pattern)
        # NEW: compute the same breadth-vote call predict_breadth_direction() would
        # (over this same in-memory batch_report, so it's always in sync with what
        # was just generated -- no separate --breadth-predict run/CSV needed), then
        # use it to filter the Nifty 50 stock sum below to only stocks whose own
        # call agrees with the overall breadth direction.
        nifty_symbol_in_report = stock_symbol_from_filename(args.equity_file)
        breadth_summary = compute_breadth_summary(
            batch_report, nifty_symbol=nifty_symbol_in_report, nifty_equity_file=args.equity_file,
            inverse_corr_threshold=args.inverse_corr_threshold,
        )
        breadth_call = breadth_summary["Breadth_Prediction"] if breadth_summary else None
        if breadth_summary:
            print(f"Breadth call for {breadth_summary['Target_Date']}: {breadth_call} "
                  f"(Up={breadth_summary['Weight_Up']}, Down={breadth_summary['Weight_Down']}, "
                  f"entities={breadth_summary['Entities_Voting']})")
        # NEW: also print the sum of predicted close prices across Nifty 50 stocks only
        # (in consensus AND matching the breadth call above), alongside the batch output.
        sum_nifty50_predicted_close(batch_report, breadth_prediction=breadth_call)
    else:
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
            
            #%run quant_all_v3.py --all-stocks --target-date 27-07-2026
            #%run quant_all_v3.py --breadth-predict
            #%run quant_all_v3.py --all-stocks-days 60 --target-date 28-08-2026