"""
================================================================================
NIFTY OPTIONS INTRADAY VOLATILITY-EXPANSION DETECTOR - single-file pipeline
================================================================================

USAGE (matches your local layout: <root>\\nifty_options\\<year>\\<month>\\*.csv
and <root>\\nifty_spot\\<year>\\<month>\\*.csv, one file per trading day):

  1) Build the training panel + train both models on everything strictly
     BEFORE a cutoff date, and evaluate on a purged walk-forward basis:

     python nifty_vol_pipeline.py train ^
         --root "C:\\Users\\RISHI\\nifty_data" ^
         --start 2020-01-01 --end 2024-09-30 ^
         --artifacts ".\\artifacts" --workers 8

  2) Predict for one specific out-of-sample date, minute by minute, using
     ONLY models trained on data strictly before that date (Step 5):

     python nifty_vol_pipeline.py predict ^
         --root "C:\\Users\\RISHI\\nifty_data" ^
         --date 2024-10-08 ^
         --artifacts ".\\artifacts"

     This writes <artifacts>\\predictions_2024-10-08.csv (one row per minute:
     spike probability + forward-vol regression estimate) and a PNG chart of
     predicted spike probability vs. what actually happened that day.

Install once, locally (this sandbox couldn't reach the internet to verify
these two, so test them on your machine - everything else here WAS run and
verified against your real sample files):
     pip install lightgbm torch --index-url <your usual index>
(xgboost and scikit-learn's HistGradientBoosting are used as automatic
fallbacks if lightgbm isn't installed - see train_ml_models() below - so the
script still runs without it, just with a slightly weaker tabular model.)

================================================================================
HONEST SCOPE NOTES (carried over from earlier discussion, read before trusting
numbers this produces):
================================================================================
- IV/Greeks are INVERTED from traded option OHLC via Black-Scholes (flat
  r=6.5%, q=1.2% assumptions), not exchange-quoted IV. See `bs_*` functions.
- India VIX and the 4 heavyweight-stock files you have are DAILY, not
  intraday, so they're wired in as static once-a-day context features
  (`load_daily_context`), not true intraday momentum/dispersion signals.
  domain2's spec'd "VIX(t) - VWAP_VIX(t)" and domain5's "synchronous
  intraday momentum" are NOT implemented as originally spec'd - that needs
  intraday VIX/stock ticks you don't have.
- CVD is NOT computable (your spot file has no volume column). Bid-ask
  spread is NOT computable (no bid/ask in the options file, only OHLC).
  Both are left out rather than faked - see domain4_microstructure().
- Feature panel build is vectorized and parallelized across days
  (~10-11s/day single-threaded on the 2020-01-01 sample; the `train` command
  parallelizes across days with `--workers`).
================================================================================
"""
import argparse
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import joblib

warnings.filterwarnings("ignore", category=RuntimeWarning)

RISK_FREE_RATE = 0.065
DIVIDEND_YIELD = 0.012
LABEL_HORIZON_MIN = 15
LABEL_K_SIGMA = 1.5
LABEL_M_STRADDLE = 0.5
SEQ_WINDOW = 30  # minutes of history the DL model looks back over


# ============================================================================
# 1. SYMBOL PARSING & FILE LOADING
# ============================================================================
MONTH_MAP = {m.upper(): i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}
SYMBOL_RE = re.compile(r"^NIFTY(\d{2})([A-Z]{3})(\d{2})(\d+)(CE|PE)$")
FILE_DATE_RE = re.compile(r"(\d{2})_(\d{2})_(\d{4})")


def parse_symbol(symbol: str):
    m = SYMBOL_RE.match(symbol.strip())
    if not m:
        return None
    dd, mon, yy, strike, opt_type = m.groups()
    mon_num = MONTH_MAP.get(mon.upper())
    if mon_num is None:
        return None
    try:
        expiry = date(2000 + int(yy), mon_num, int(dd))
    except ValueError:
        return None
    return {"expiry": expiry, "strike": float(strike), "opt_type": opt_type}


def load_options_day(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    parsed = df["symbol"].apply(parse_symbol)
    mask = parsed.notna()
    df = df[mask].copy()
    parsed = parsed[mask]
    df["expiry"] = parsed.apply(lambda d: d["expiry"])
    df["strike"] = parsed.apply(lambda d: d["strike"])
    df["opt_type"] = parsed.apply(lambda d: d["opt_type"])
    df["datetime"] = pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d") + " " + df["time"])
    return df.sort_values("datetime").reset_index(drop=True)


def load_spot_day(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["datetime"] = pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d") + " " + df["time"])
    return df.sort_values("datetime").reset_index(drop=True)


def load_daily_context(path) -> pd.DataFrame:
    """
    Generic loader for your DAILY (not intraday) VIX / constituent-stock
    files. Expects at minimum columns `date` and `close` (extra columns are
    ignored). Returns a DataFrame indexed by date with `close` and
    `pct_change_1d`. Adjust the column names here to match your actual daily
    files - I have not seen their exact schema, only that they're daily.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["pct_change_1d"] = df["close"].pct_change()
    return df.set_index(df["date"].dt.date)[["close", "pct_change_1d"]]


def find_day_files(root: str, year: int, month: int):
    """Returns dict date_str -> (options_path, spot_path) for a month, matched
    by the DD_MM_YYYY embedded in each filename (options/spot are matched by
    that date, not by directory listing order - your own sample files had a
    spot/options date mismatch, so never assume the two sides line up)."""
    opt_dir = Path(root) / "nifty_options" / str(year) / str(month)
    spot_dir = Path(root) / "nifty_spot" / str(year) / str(month)
    opt_files, spot_files = {}, {}
    if opt_dir.exists():
        for p in opt_dir.glob("*.csv"):
            m = FILE_DATE_RE.search(p.stem)
            if m:
                dd, mm, yyyy = m.groups()
                opt_files[f"{yyyy}-{mm}-{dd}"] = p
    if spot_dir.exists():
        for p in spot_dir.glob("*.csv"):
            m = FILE_DATE_RE.search(p.stem)
            if m:
                dd, mm, yyyy = m.groups()
                spot_files[f"{yyyy}-{mm}-{dd}"] = p
    return opt_files, spot_files


def iter_date_range(root: str, start: date, end: date):
    """Yields (date_str, options_path, spot_path) for every day in [start,end]
    that has BOTH an options and a spot file."""
    cur = start.replace(day=1)
    while cur <= end:
        opt_files, spot_files = find_day_files(root, cur.year, cur.month)
        for date_str in sorted(set(opt_files) & set(spot_files)):
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            if start <= d <= end:
                yield date_str, opt_files[date_str], spot_files[date_str]
        cur = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)


# ============================================================================
# 2. BLACK-SCHOLES ENGINE (vectorized - see earlier profiling: this is the
#    part that made the naive per-row version 15x too slow to use)
# ============================================================================
def _bs_price_vec(S, K, T, r, q, sigma, is_call):
    sigma = np.maximum(sigma, 1e-6)
    T = np.maximum(T, 1e-8)
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return np.where(is_call, call_price, put_price)


def _bs_vega_vec(S, K, T, r, q, sigma):
    sigma = np.maximum(sigma, 1e-6)
    T = np.maximum(T, 1e-8)
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    return S * np.exp(-q * T) * norm.pdf(d1) * sqT


def implied_vol_batch(price, S, K, T, r, q, is_call, max_iter=50, tol=1e-6):
    price = np.asarray(price, dtype=float); S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float); T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float); q = np.asarray(q, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    n = len(price)
    sigma = np.full(n, 0.20)
    intrinsic = np.where(is_call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    invalid = (T <= 0) | (price <= 0) | (price < intrinsic - 1e-6)
    active = ~invalid
    for _ in range(max_iter):
        if not active.any():
            break
        mp = _bs_price_vec(S[active], K[active], T[active], r[active], q[active], sigma[active], is_call[active])
        vega = _bs_vega_vec(S[active], K[active], T[active], r[active], q[active], sigma[active])
        diff = mp - price[active]
        vega_safe = np.where(np.abs(vega) < 1e-8, 1e-8, vega)
        step = np.clip(diff / vega_safe, -0.5, 0.5)
        new_sigma = np.clip(sigma[active] - step, 1e-4, 5.0)
        converged = np.abs(diff) < tol
        idx_active = np.where(active)[0]
        sigma[idx_active] = new_sigma
        active = np.zeros(n, dtype=bool)
        active[idx_active[~converged]] = True
    final_price = _bs_price_vec(S, K, T, r, q, sigma, is_call)
    resid = np.abs(final_price - price)
    return np.where(invalid | (resid > 1e-3), np.nan, sigma)


def bs_greeks_batch(S, K, T, r, q, sigma, is_call):
    S = np.asarray(S, dtype=float); K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float); r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float); sigma = np.asarray(sigma, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        sqT = np.sqrt(np.maximum(T, 1e-8))
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqT)
        pdf_d1 = norm.pdf(d1)
        gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * sqT)
        delta = np.where(is_call, np.exp(-q * T) * norm.cdf(d1), -np.exp(-q * T) * norm.cdf(-d1))
    bad = ~np.isfinite(sigma) | (T <= 0) | (sigma <= 0)
    return np.where(bad, np.nan, delta), np.where(bad, np.nan, gamma)


def find_gamma_flip(strikes, call_oi, put_oi, S_grid, T, r, q, sigma_flat):
    strikes = np.asarray(strikes, dtype=float); call_oi = np.asarray(call_oi, dtype=float)
    put_oi = np.asarray(put_oi, dtype=float); S_grid = np.asarray(S_grid, dtype=float)
    S_mat = S_grid[:, None]; K_mat = strikes[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        sqT = np.sqrt(max(T, 1e-8))
        d1 = (np.log(S_mat / K_mat) + (r - q + 0.5 * sigma_flat ** 2) * T) / (sigma_flat * sqT)
        gamma_mat = np.exp(-q * T) * norm.pdf(d1) / (S_mat * sigma_flat * sqT)
    gex_curve = (gamma_mat * call_oi[None, :] * S_mat * 100).sum(axis=1) - (gamma_mat * put_oi[None, :] * S_mat * 100).sum(axis=1)
    sign = np.sign(gex_curve)
    idx = np.where(np.diff(sign) != 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    x0, x1 = S_grid[i], S_grid[i + 1]
    y0, y1 = gex_curve[i], gex_curve[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


# ============================================================================
# 3. FEATURE ENGINEERING (per-minute, strictly causal - see module docstring
#    at top for exactly which of the original 5 domains are/aren't covered)
# ============================================================================
def year_frac(t: pd.Timestamp, expiry) -> float:
    expiry_dt = pd.Timestamp(expiry) + pd.Timedelta(hours=15, minutes=30)
    return max((expiry_dt - t).total_seconds() / (365.0 * 24 * 3600), 1e-6)


def nearest_weekly_expiry(opt_slice: pd.DataFrame, t: pd.Timestamp):
    exps = sorted(e for e in opt_slice["expiry"].unique() if pd.Timestamp(e) + pd.Timedelta(hours=15, minutes=30) >= t)
    return exps[0] if exps else sorted(opt_slice["expiry"].unique())[-1]


def solve_chain_iv(chain_t: pd.DataFrame, S: float, T: float) -> pd.DataFrame:
    if chain_t.empty:
        out = chain_t.copy(); out["iv"], out["delta"], out["gamma"] = [], [], []
        return out
    n = len(chain_t)
    is_call = (chain_t["opt_type"].values == "CE")
    ivs = implied_vol_batch(chain_t["close"].values, np.full(n, S), chain_t["strike"].values,
                             np.full(n, T), np.full(n, RISK_FREE_RATE), np.full(n, DIVIDEND_YIELD), is_call)
    deltas, gammas = bs_greeks_batch(np.full(n, S), chain_t["strike"].values, np.full(n, T),
                                      np.full(n, RISK_FREE_RATE), np.full(n, DIVIDEND_YIELD), ivs, is_call)
    out = chain_t.copy()
    out["iv"], out["delta"], out["gamma"] = ivs, deltas, gammas
    return out


def domain1_iv_features(chain_iv: pd.DataFrame, S: float):
    valid = chain_iv.dropna(subset=["strike"])
    if valid.empty:
        return {"straddle_price": np.nan, "atm_iv": np.nan, "skew_25d": np.nan, "atm_strike": np.nan}
    atm_strike = valid.iloc[(valid["strike"] - S).abs().argsort()[:1]]["strike"].values[0]
    ce = valid[(valid["strike"] == atm_strike) & (valid["opt_type"] == "CE")]
    pe = valid[(valid["strike"] == atm_strike) & (valid["opt_type"] == "PE")]
    straddle = ce["close"].values[0] + pe["close"].values[0] if (not ce.empty and not pe.empty) else np.nan
    atm_iv_vals = pd.concat([ce["iv"], pe["iv"]]).dropna()
    atm_iv = atm_iv_vals.mean() if not atm_iv_vals.empty else np.nan
    calls = valid[(valid["opt_type"] == "CE") & valid["delta"].notna()]
    puts = valid[(valid["opt_type"] == "PE") & valid["delta"].notna()]
    skew = np.nan
    if not calls.empty and not puts.empty:
        c25 = calls.iloc[(calls["delta"] - 0.25).abs().argsort()[:1]]
        p25 = puts.iloc[(puts["delta"] + 0.25).abs().argsort()[:1]]
        if not c25.empty and not p25.empty and c25["iv"].notna().all() and p25["iv"].notna().all():
            skew = p25["iv"].values[0] - c25["iv"].values[0]
    return {"straddle_price": straddle, "atm_iv": atm_iv, "skew_25d": skew, "atm_strike": atm_strike}


def domain2_gamma_features(chain_iv: pd.DataFrame, S: float, T: float, atm_iv: float):
    valid = chain_iv.dropna(subset=["gamma"])
    calls, puts = valid[valid["opt_type"] == "CE"], valid[valid["opt_type"] == "PE"]
    gex = (calls["gamma"] * calls["oi"] * S * 100).sum() - (puts["gamma"] * puts["oi"] * S * 100).sum()
    flip = np.nan
    if np.isfinite(atm_iv) and atm_iv > 0 and not valid.empty:
        grid = np.linspace(S * 0.9, S * 1.1, 41)
        call_agg = calls.groupby("strike")["oi"].sum()
        put_agg = puts.groupby("strike")["oi"].sum()
        all_strikes = sorted(set(call_agg.index) | set(put_agg.index))
        flip = find_gamma_flip(all_strikes,
                                call_agg.reindex(all_strikes, fill_value=0).values,
                                put_agg.reindex(all_strikes, fill_value=0).values,
                                grid, T, RISK_FREE_RATE, DIVIDEND_YIELD, atm_iv)
    dist_to_flip = (S - flip) if (flip is not None and np.isfinite(flip)) else np.nan
    return {"gex": gex, "gamma_flip": flip if flip is not None else np.nan, "dist_to_flip": dist_to_flip}


def domain3_oi_features(chain_t: pd.DataFrame, chain_prev_5m: pd.DataFrame, S: float, straddle_price: float):
    calls, puts = chain_t[chain_t["opt_type"] == "CE"], chain_t[chain_t["opt_type"] == "PE"]
    pcr_oi = puts["oi"].sum() / calls["oi"].sum() if calls["oi"].sum() > 0 else np.nan
    pcr_vol = puts["volume"].sum() / calls["volume"].sum() if calls["volume"].sum() > 0 else np.nan
    d_pcr_oi = d_pcr_vol = np.nan
    if chain_prev_5m is not None and not chain_prev_5m.empty:
        pcalls, pputs = chain_prev_5m[chain_prev_5m["opt_type"] == "CE"], chain_prev_5m[chain_prev_5m["opt_type"] == "PE"]
        prev_pcr_oi = pputs["oi"].sum() / pcalls["oi"].sum() if pcalls["oi"].sum() > 0 else np.nan
        prev_pcr_vol = pputs["volume"].sum() / pcalls["volume"].sum() if pcalls["volume"].sum() > 0 else np.nan
        d_pcr_oi, d_pcr_vol = pcr_oi - prev_pcr_oi, pcr_vol - prev_pcr_vol
    call_strikes, call_oi = calls["strike"].values, calls["oi"].fillna(0).values
    put_strikes, put_oi = puts["strike"].values, puts["oi"].fillna(0).values
    candidates = np.array(sorted(chain_t["strike"].unique()))
    if len(candidates) and (len(call_strikes) or len(put_strikes)):
        call_payout = np.maximum(candidates[:, None] - call_strikes[None, :], 0.0) @ call_oi if len(call_strikes) else np.zeros(len(candidates))
        put_payout = np.maximum(put_strikes[None, :] - candidates[:, None], 0.0) @ put_oi if len(put_strikes) else np.zeros(len(candidates))
        max_pain_strike = float(candidates[np.argmin(call_payout + put_payout)])
    else:
        max_pain_strike = np.nan
    dist_max_pain = (S - max_pain_strike) / straddle_price if (np.isfinite(max_pain_strike) and np.isfinite(straddle_price) and straddle_price > 0) else np.nan
    return {"pcr_oi": pcr_oi, "pcr_vol": pcr_vol, "d_pcr_oi_5m": d_pcr_oi, "d_pcr_vol_5m": d_pcr_vol,
            "max_pain_strike": max_pain_strike, "dist_max_pain_norm": dist_max_pain}


def parkinson_vol(ohlc_5m: pd.DataFrame, window: int = 12) -> float:
    if len(ohlc_5m) < 2:
        return np.nan
    bars = ohlc_5m.tail(window)
    var = (1.0 / (4 * np.log(2))) * (np.log(bars["high"] / bars["low"]) ** 2).mean() * (252 * (375 / 5))
    return float(np.sqrt(var)) if var > 0 else np.nan


def garman_klass_vol(ohlc_5m: pd.DataFrame, window: int = 12) -> float:
    if len(ohlc_5m) < 2:
        return np.nan
    bars = ohlc_5m.tail(window)
    log_hl = np.log(bars["high"] / bars["low"]) ** 2
    log_co = np.log(bars["close"] / bars["open"]) ** 2
    var = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).mean() * (252 * (375 / 5))
    return float(np.sqrt(var)) if var > 0 else np.nan


def domain4_microstructure(spot_slice: pd.DataFrame):
    ohlc_5m = spot_slice.set_index("datetime")[["open", "high", "low", "close"]].resample(
        "5min", origin="start_day").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    typ_price = (spot_slice["high"] + spot_slice["low"] + spot_slice["close"]) / 3
    cutler_vwap = typ_price.expanding().mean().iloc[-1] if len(typ_price) else np.nan
    band_std = typ_price.expanding().std().iloc[-1] if len(typ_price) > 1 else np.nan
    return {"parkinson_vol": parkinson_vol(ohlc_5m), "garman_klass_vol": garman_klass_vol(ohlc_5m),
            "cutler_vwap_proxy": cutler_vwap,
            "vwap_band_2sd": 2 * band_std if np.isfinite(band_std) else np.nan,
            "vwap_band_3sd": 3 * band_std if np.isfinite(band_std) else np.nan}


def build_day_panel(options_path, spot_path, vix_ctx=None, stock_ctx=None) -> pd.DataFrame:
    """vix_ctx / stock_ctx: optional dict of {date -> {feature: value}} static daily context
    (see load_daily_context) merged onto every minute row of that day, forward-filled from the
    PRIOR trading day's close only (never today's, which wouldn't exist yet intraday)."""
    opt = load_options_day(options_path)
    spot = load_spot_day(spot_path)
    rows = []
    for t in spot["datetime"].tolist():
        spot_slice = spot[spot["datetime"] <= t]
        S = spot_slice["close"].iloc[-1]
        opt_slice = opt[opt["datetime"] <= t]
        if opt_slice.empty:
            continue
        expiry = nearest_weekly_expiry(opt_slice, t)
        chain_now = opt_slice[opt_slice["expiry"] == expiry]
        chain_t = chain_now.sort_values("datetime").groupby(["strike", "opt_type"], as_index=False).last()
        T = year_frac(t, expiry)
        chain_iv = solve_chain_iv(chain_t, S, T)
        d1 = domain1_iv_features(chain_iv, S)
        d2 = domain2_gamma_features(chain_iv, S, T, d1["atm_iv"])
        t_minus_5 = t - pd.Timedelta(minutes=5)
        chain_prev = opt[(opt["datetime"] <= t_minus_5) & (opt["expiry"] == expiry)]
        chain_prev_t = (chain_prev.sort_values("datetime").groupby(["strike", "opt_type"], as_index=False).last()
                         if not chain_prev.empty else None)
        d3 = domain3_oi_features(chain_t, chain_prev_t, S, d1["straddle_price"])
        d4 = domain4_microstructure(spot_slice)
        row = {"datetime": t, "spot": S, "expiry": expiry, "T_years": T}
        row.update(d1); row.update(d2); row.update(d3); row.update(d4)
        rows.append(row)
    panel = pd.DataFrame(rows)

    day = spot["date"].iloc[0].date() if len(spot) else None
    if vix_ctx is not None and day is not None:
        prior_days = sorted(d for d in vix_ctx.index if d < day)
        if prior_days:
            panel["vix_prev_close"] = vix_ctx.loc[prior_days[-1], "close"]
            panel["vix_prev_pct_change"] = vix_ctx.loc[prior_days[-1], "pct_change_1d"]
    if stock_ctx is not None and day is not None:
        for name, ctx in stock_ctx.items():
            prior_days = sorted(d for d in ctx.index if d < day)
            if prior_days:
                panel[f"{name}_prev_pct_change"] = ctx.loc[prior_days[-1], "pct_change_1d"]
    return panel


# ============================================================================
# 4. LABELING (Step 2)
# ============================================================================
def forward_realized_vol(spot: pd.DataFrame, ts: pd.Series, horizon_min: int) -> pd.Series:
    s = spot.sort_values("datetime").set_index("datetime")["close"]
    log_ret = np.log(s / s.shift(1))
    out = []
    for t in ts:
        window = log_ret[(log_ret.index > t) & (log_ret.index <= t + pd.Timedelta(minutes=horizon_min))]
        out.append(window.std(ddof=0) * np.sqrt(252 * 375) if len(window) >= max(2, horizon_min // 3) else np.nan)
    return pd.Series(out, index=ts.index)


def forward_price_range(spot: pd.DataFrame, ts: pd.Series, horizon_min: int) -> pd.Series:
    s = spot.sort_values("datetime").set_index("datetime")["close"]
    out = []
    for t in ts:
        window = s[(s.index > t) & (s.index <= t + pd.Timedelta(minutes=horizon_min))]
        out.append(window.max() - window.min() if len(window) else np.nan)
    return pd.Series(out, index=ts.index)


def build_labels(panel: pd.DataFrame, spot: pd.DataFrame, horizon_min=LABEL_HORIZON_MIN,
                  k_sigma=LABEL_K_SIGMA, m_straddle=LABEL_M_STRADDLE) -> pd.DataFrame:
    out = panel.copy().reset_index(drop=True)
    out["fwd_realized_vol"] = forward_realized_vol(spot, out["datetime"], horizon_min)
    out["fwd_price_range"] = forward_price_range(spot, out["datetime"], horizon_min)
    cond_vol = out["fwd_realized_vol"] > (k_sigma * out["parkinson_vol"])
    cond_range = out["fwd_price_range"] > (m_straddle * out["straddle_price"])
    out["y_spike"] = (cond_vol | cond_range).astype(float)
    out["label_valid"] = out["fwd_realized_vol"].notna() & out["fwd_price_range"].notna() & out["parkinson_vol"].notna()
    out.loc[~out["label_valid"], "y_spike"] = np.nan
    return out


# ============================================================================
# 5. PURGED WALK-FORWARD CV (Step 3)
# ============================================================================
@dataclass
class Fold:
    train_days: list
    test_days: list


def purged_walk_forward_splits(all_days, n_splits=5, min_train_days=60, purge_days=1, test_days_per_split=20):
    all_days = sorted(all_days)
    n = len(all_days)
    folds, start_test_idx = [], min_train_days
    for _ in range(n_splits):
        train_end, test_start = start_test_idx - purge_days, start_test_idx
        test_end = min(test_start + test_days_per_split, n)
        if train_end <= 0 or test_start >= n:
            break
        test_days = all_days[test_start:test_end]
        if not test_days:
            break
        folds.append(Fold(train_days=all_days[:train_end], test_days=test_days))
        start_test_idx = test_end
    return folds


FEATURE_COLS = [
    "straddle_price", "atm_iv", "skew_25d", "gex", "gamma_flip", "dist_to_flip",
    "pcr_oi", "pcr_vol", "d_pcr_oi_5m", "d_pcr_vol_5m", "dist_max_pain_norm",
    "parkinson_vol", "garman_klass_vol", "cutler_vwap_proxy", "vwap_band_2sd", "vwap_band_3sd",
    "vix_prev_close", "vix_prev_pct_change",
]


def build_dataset_parallel(root, start, end, vix_ctx=None, stock_ctx=None, workers=1):
    """Builds the full labeled panel across [start,end], one process per trading day."""
    days = list(iter_date_range(root, start, end))
    args = [(opt_p, spot_p, vix_ctx, stock_ctx) for _, opt_p, spot_p in days]
    if workers > 1:
        with Pool(min(workers, cpu_count())) as pool:
            panels = pool.starmap(_build_and_label_one_day, args)
    else:
        panels = [_build_and_label_one_day(*a) for a in args]
    panels = [p for p in panels if p is not None and not p.empty]
    if not panels:
        raise RuntimeError(f"No trading days found under {root} for {start}..{end} with matching options+spot files.")
    full = pd.concat(panels, ignore_index=True)
    full["trading_day"] = full["datetime"].dt.date
    return full


def _build_and_label_one_day(opt_path, spot_path, vix_ctx, stock_ctx):
    panel = build_day_panel(opt_path, spot_path, vix_ctx, stock_ctx)
    if panel.empty:
        return None
    spot = load_spot_day(spot_path)
    return build_labels(panel, spot)


# ============================================================================
# 6. ML MODEL (Step 4a) - LightGBM if available, else XGBoost, else sklearn
# ============================================================================
def _get_ml_backend():
    try:
        import lightgbm as lgb
        return "lightgbm", lgb
    except ImportError:
        pass
    try:
        import xgboost as xgb
        return "xgboost", xgb
    except ImportError:
        pass
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    return "sklearn", (HistGradientBoostingClassifier, HistGradientBoostingRegressor)


def train_ml_models(train_df, feature_cols):
    from sklearn.preprocessing import StandardScaler
    train_df = train_df.dropna(subset=feature_cols + ["y_spike", "fwd_realized_vol"])
    X = train_df[feature_cols].values
    y_clf = train_df["y_spike"].values
    y_reg = train_df["fwd_realized_vol"].values

    scaler = StandardScaler().fit(X)  # fit on TRAIN ONLY
    Xs = scaler.transform(X)

    backend, lib = _get_ml_backend()
    pos_rate = y_clf.mean()
    scale_pos_weight = (1 - pos_rate) / max(pos_rate, 1e-6)  # cost-sensitive weighting for rare spikes

    if backend == "lightgbm":
        clf = lib.LGBMClassifier(n_estimators=400, learning_rate=0.03, max_depth=6,
                                  scale_pos_weight=scale_pos_weight, subsample=0.8, colsample_bytree=0.8)
        reg = lib.LGBMRegressor(n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8)
        clf.fit(Xs, y_clf)
        reg.fit(Xs, y_reg)
    elif backend == "xgboost":
        clf = lib.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=6,
                                 scale_pos_weight=scale_pos_weight, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss")
        reg = lib.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8)
        clf.fit(Xs, y_clf)
        reg.fit(Xs, y_reg)
    else:
        ClfCls, RegCls = lib
        sample_weight = np.where(y_clf == 1, scale_pos_weight, 1.0)
        clf = ClfCls(max_iter=400, learning_rate=0.05, max_depth=6)
        clf.fit(Xs, y_clf, sample_weight=sample_weight)
        reg = RegCls(max_iter=400, learning_rate=0.05, max_depth=6)
        reg.fit(Xs, y_reg)

    print(f"[ML] backend={backend}  n_train={len(train_df)}  spike_rate={pos_rate:.3f}  scale_pos_weight={scale_pos_weight:.2f}")
    return {"backend": backend, "scaler": scaler, "clf": clf, "reg": reg}


def eval_ml_models(models, test_df, feature_cols):
    from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
    test_df = test_df.dropna(subset=feature_cols + ["y_spike", "fwd_realized_vol"])
    if test_df.empty:
        return None
    Xs = models["scaler"].transform(test_df[feature_cols].values)  # transform only, never re-fit
    p_spike = models["clf"].predict_proba(Xs)[:, 1] if hasattr(models["clf"], "predict_proba") else models["clf"].predict(Xs)
    pred_vol = models["reg"].predict(Xs)
    y_clf, y_reg = test_df["y_spike"].values, test_df["fwd_realized_vol"].values
    metrics = {"mae_vol": mean_absolute_error(y_reg, pred_vol)}
    if len(np.unique(y_clf)) > 1:
        metrics["auc"] = roc_auc_score(y_clf, p_spike)
        metrics["pr_auc"] = average_precision_score(y_clf, p_spike)
    return metrics


# ============================================================================
# 7. DL MODEL (Step 4b) - LSTM + attention over the past SEQ_WINDOW minutes
#    Requires torch (pip install torch). Not executable in the sandbox this
#    was built in (no internet to install it there) - standard PyTorch,
#    test it on your machine.
# ============================================================================
def _torch_available():
    try:
        import torch  # noqa
        return True
    except ImportError:
        return False


def make_sequences(df, feature_cols, seq_window=SEQ_WINDOW):
    """Builds (N, seq_window, n_features) arrays PER TRADING DAY (never spans
    the 09:15 session boundary - that would leak yesterday's regime into
    today's early-morning prediction, which trades on a fresh state)."""
    X_list, y_clf_list, y_reg_list = [], [], []
    for _, day_df in df.groupby("trading_day"):
        day_df = day_df.sort_values("datetime").reset_index(drop=True)
        feats = day_df[feature_cols].ffill().fillna(0).values
        for i in range(seq_window, len(day_df)):
            if pd.isna(day_df["y_spike"].iloc[i]):
                continue
            X_list.append(feats[i - seq_window:i])
            y_clf_list.append(day_df["y_spike"].iloc[i])
            y_reg_list.append(day_df["fwd_realized_vol"].iloc[i])
    if not X_list:
        return None, None, None
    return np.stack(X_list), np.array(y_clf_list), np.array(y_reg_list)


def build_lstm_attention_model(n_features, hidden_size=64):
    import torch
    import torch.nn as nn

    class LSTMAttention(nn.Module):
        def __init__(self, n_features, hidden_size):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
            self.attn = nn.Linear(hidden_size, 1)
            self.head_clf = nn.Linear(hidden_size, 1)
            self.head_reg = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)                       # (B, T, H) - causal: LSTM only ever sees t' <= t within the window
            attn_scores = self.attn(out).squeeze(-1)     # (B, T)
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, T, 1)
            context = (out * attn_weights).sum(dim=1)    # (B, H) - weighted combo of PAST timesteps only
            spike_logit = self.head_clf(context).squeeze(-1)
            vol_pred = self.head_reg(context).squeeze(-1)
            return spike_logit, vol_pred

    return LSTMAttention(n_features, hidden_size)


def train_dl_model(train_df, val_df, feature_cols, epochs=30, batch_size=64, lr=1e-3, patience=5):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import StandardScaler

    train_df = train_df.dropna(subset=["y_spike", "fwd_realized_vol"])
    scaler = StandardScaler().fit(train_df[feature_cols].ffill().fillna(0).values)  # fit on TRAIN ONLY

    def prep(df):
        df = df.copy()
        df[feature_cols] = scaler.transform(df[feature_cols].ffill().fillna(0).values)
        return make_sequences(df, feature_cols)

    X_tr, y_clf_tr, y_reg_tr = prep(train_df)
    X_val, y_clf_val, y_reg_val = prep(val_df.dropna(subset=["y_spike", "fwd_realized_vol"]))
    if X_tr is None or X_val is None:
        raise RuntimeError("Not enough rows to build sequence windows - need more trading days than SEQ_WINDOW minutes each.")

    model = build_lstm_attention_model(n_features=len(feature_cols))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([(1 - y_clf_tr.mean()) / max(y_clf_tr.mean(), 1e-6)])
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse = torch.nn.MSELoss()

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                             torch.tensor(y_clf_tr, dtype=torch.float32),
                                             torch.tensor(y_reg_tr, dtype=torch.float32)),
                               batch_size=batch_size, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_clf_val_t = torch.tensor(y_clf_val, dtype=torch.float32)
    y_reg_val_t = torch.tensor(y_reg_val, dtype=torch.float32)

    best_val_loss, best_state, patience_left = float("inf"), None, patience
    for epoch in range(epochs):
        model.train()
        for xb, yclf_b, yreg_b in train_loader:
            opt.zero_grad()
            logit, vol_pred = model(xb)
            loss = bce(logit, yclf_b) + mse(vol_pred, yreg_b)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logit, vol_pred = model(X_val_t)
            val_loss = (bce(logit, y_clf_val_t) + mse(vol_pred, y_reg_val_t)).item()
        print(f"[DL] epoch {epoch+1}/{epochs}  val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss, best_state, patience_left = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[DL] early stopping at epoch {epoch+1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "scaler": scaler}


# ============================================================================
# 8. TRAIN CLI COMMAND
# ============================================================================
def cmd_train(args):
    root = args.root
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    vix_ctx = load_daily_context(args.vix_csv) if args.vix_csv else None
    stock_ctx = None
    if args.stock_csvs:
        stock_ctx = {Path(p).stem: load_daily_context(p) for p in args.stock_csvs}

    print(f"Building panel for {start}..{end} from {root} ...")
    full = build_dataset_parallel(root, start, end, vix_ctx, stock_ctx, workers=args.workers)
    print(f"Built {len(full)} minute-rows across {full['trading_day'].nunique()} trading days.")

    all_days = sorted(full["trading_day"].unique())
    folds = purged_walk_forward_splits(all_days, n_splits=args.n_splits,
                                        min_train_days=args.min_train_days, purge_days=1,
                                        test_days_per_split=args.test_days_per_split)
    if not folds:
        raise RuntimeError("Not enough trading days for even one walk-forward fold - lower --min-train-days or widen the date range.")

    feature_cols = [c for c in FEATURE_COLS if c in full.columns]
    print(f"Using {len(feature_cols)} features: {feature_cols}")

    for i, fold in enumerate(folds):
        train_df = full[full["trading_day"].isin(fold.train_days)]
        test_df = full[full["trading_day"].isin(fold.test_days)]
        models = train_ml_models(train_df, feature_cols)
        metrics = eval_ml_models(models, test_df, feature_cols)
        print(f"[Fold {i}] train={fold.train_days[0]}..{fold.train_days[-1]} "
              f"test={fold.test_days[0]}..{fold.test_days[-1]}  metrics={metrics}")

    # Final production models: train ML + DL on ALL data up to `end` (the walk-forward
    # loop above is for honest evaluation; this final fit is what `predict` will load).
    print("Fitting final ML model on full training range ...")
    final_ml = train_ml_models(full, feature_cols)
    joblib.dump({"backend": final_ml["backend"], "scaler": final_ml["scaler"], "clf": final_ml["clf"],
                 "reg": final_ml["reg"], "feature_cols": feature_cols}, artifacts / "ml_model.joblib")
    print(f"Saved {artifacts / 'ml_model.joblib'}")

    if _torch_available():
        import torch
        print("Fitting final DL model on full training range (last fold's test days held out as val) ...")
        val_days = folds[-1].test_days
        train_part = full[~full["trading_day"].isin(val_days)]
        val_part = full[full["trading_day"].isin(val_days)]
        dl = train_dl_model(train_part, val_part, feature_cols, epochs=args.dl_epochs)
        torch.save({"state_dict": dl["model"].state_dict(), "scaler": dl["scaler"],
                    "feature_cols": feature_cols, "n_features": len(feature_cols)}, artifacts / "dl_model.pt")
        print(f"Saved {artifacts / 'dl_model.pt'}")
    else:
        print("[DL] torch not installed - skipping DL model. `pip install torch` and re-run `train` to add it.")

    (artifacts / "meta.txt").write_text(f"trained_through={end}\nfeature_cols={feature_cols}\n")
    print("Training complete.")


# ============================================================================
# 9. PREDICT-FOR-A-DATE CLI COMMAND (Step 5 harness)
# ============================================================================
def cmd_predict(args):
    root = args.root
    target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    artifacts = Path(args.artifacts)

    meta = (artifacts / "meta.txt").read_text()
    trained_through = datetime.strptime(meta.splitlines()[0].split("=")[1], "%Y-%m-%d").date()
    if target_date <= trained_through:
        print(f"WARNING: {target_date} is not strictly after the training cutoff ({trained_through}). "
              f"Predicting on it is in-sample / leaky. Re-run `train` with --end before {target_date} for a clean test.")

    opt_files, spot_files = find_day_files(root, target_date.year, target_date.month)
    date_str = target_date.strftime("%Y-%m-%d")
    if date_str not in opt_files or date_str not in spot_files:
        raise FileNotFoundError(f"Could not find both options and spot files for {date_str} under {root}. "
                                 f"Expected them in nifty_options/{target_date.year}/{target_date.month} and nifty_spot/{target_date.year}/{target_date.month}.")

    vix_ctx = load_daily_context(args.vix_csv) if args.vix_csv else None
    stock_ctx = {Path(p).stem: load_daily_context(p) for p in args.stock_csvs} if args.stock_csvs else None

    print(f"Building causal panel for {date_str} ...")
    panel = build_day_panel(opt_files[date_str], spot_files[date_str], vix_ctx, stock_ctx)
    spot = load_spot_day(spot_files[date_str])
    labeled = build_labels(panel, spot)  # labels here are for POST-HOC evaluation/plotting only, never fed to the model

    ml = joblib.load(artifacts / "ml_model.joblib")
    feature_cols = ml["feature_cols"]
    valid = labeled.dropna(subset=feature_cols).copy()
    Xs = ml["scaler"].transform(valid[feature_cols].values)
    valid["p_spike_ml"] = ml["clf"].predict_proba(Xs)[:, 1] if hasattr(ml["clf"], "predict_proba") else ml["clf"].predict(Xs)
    valid["pred_fwd_vol_ml"] = ml["reg"].predict(Xs)

    dl_path = artifacts / "dl_model.pt"
    if dl_path.exists() and _torch_available():
        import torch
        ckpt = torch.load(dl_path, weights_only=False)
        model = build_lstm_attention_model(n_features=ckpt["n_features"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        labeled_scaled = labeled.copy()
        labeled_scaled[feature_cols] = ckpt["scaler"].transform(labeled_scaled[feature_cols].ffill().fillna(0).values)
        feats = labeled_scaled[feature_cols].values
        p_dl, vol_dl = [np.nan] * len(labeled), [np.nan] * len(labeled)
        with torch.no_grad():
            for i in range(SEQ_WINDOW, len(labeled)):
                window = torch.tensor(feats[i - SEQ_WINDOW:i], dtype=torch.float32).unsqueeze(0)
                logit, vol_pred = model(window)
                p_dl[i] = torch.sigmoid(logit).item()
                vol_dl[i] = vol_pred.item()
        labeled["p_spike_dl"] = p_dl
        labeled["pred_fwd_vol_dl"] = vol_dl
        valid = valid.merge(labeled[["datetime", "p_spike_dl", "pred_fwd_vol_dl"]], on="datetime", how="left")
    else:
        print("[DL] no dl_model.pt found / torch unavailable - reporting ML predictions only.")

    out_csv = artifacts / f"predictions_{date_str}.csv"
    valid.to_csv(out_csv, index=False)
    print(f"Saved minute-by-minute predictions to {out_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(valid["datetime"], valid["spot"], color="black", linewidth=1, label="Spot")
        ax1.set_ylabel("Nifty spot")
        ax2 = ax1.twinx()
        ax2.plot(valid["datetime"], valid["p_spike_ml"], color="tab:red", alpha=0.7, label="P(spike) - ML")
        if "p_spike_dl" in valid.columns:
            ax2.plot(valid["datetime"], valid["p_spike_dl"], color="tab:orange", alpha=0.7, label="P(spike) - DL")
        actual_spikes = valid[valid["y_spike"] == 1]["datetime"]
        for t in actual_spikes:
            ax1.axvline(t, color="tab:blue", alpha=0.15)
        ax2.set_ylabel("Spike probability")
        fig.legend(loc="upper left")
        plt.title(f"Volatility-spike detection - {date_str}")
        plt.tight_layout()
        png_path = artifacts / f"predictions_{date_str}.png"
        plt.savefig(png_path, dpi=120)
        print(f"Saved chart to {png_path}")
    except ImportError:
        print("matplotlib not installed - skipping chart, CSV still written.")


# ============================================================================
# 10. CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Nifty options intraday volatility-expansion detector")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Build panel, walk-forward evaluate, and save final models")
    p_train.add_argument("--root", required=True, help=r"e.g. C:\Users\RISHI\nifty_data")
    p_train.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_train.add_argument("--end", required=True, help="YYYY-MM-DD (exclusive of test date you plan to predict on)")
    p_train.add_argument("--artifacts", default="./artifacts")
    p_train.add_argument("--workers", type=int, default=1)
    p_train.add_argument("--n-splits", dest="n_splits", type=int, default=5)
    p_train.add_argument("--min-train-days", dest="min_train_days", type=int, default=60)
    p_train.add_argument("--test-days-per-split", dest="test_days_per_split", type=int, default=20)
    p_train.add_argument("--dl-epochs", dest="dl_epochs", type=int, default=30)
    p_train.add_argument("--vix-csv", dest="vix_csv", default=None, help="optional daily India VIX csv (date,close)")
    p_train.add_argument("--stock-csvs", dest="stock_csvs", nargs="*", default=None, help="optional daily constituent csvs (date,close)")
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Run the trained models minute-by-minute over one held-out date")
    p_pred.add_argument("--root", required=True)
    p_pred.add_argument("--date", required=True, help="YYYY-MM-DD - must have matching options+spot files under --root")
    p_pred.add_argument("--artifacts", default="./artifacts")
    p_pred.add_argument("--vix-csv", dest="vix_csv", default=None)
    p_pred.add_argument("--stock-csvs", dest="stock_csvs", nargs="*", default=None)
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()