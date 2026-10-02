"""
Nifty 30-min ORB 15-min-close breakout  ->  sell 0.4-delta option
=================================================================
(Retest condition REMOVED - this replaces the earlier ORB-retest version.)

STRATEGY
---------------------------------------------------------------
1. 15-min chart. Opening range (ORB) = 09:15-09:45 high/low.
2. Enter as soon as a 15-min candle CLOSES beyond the ORB:
       close above ORB high -> bullish -> SELL a 0.4-delta PE
       close below ORB low  -> bearish -> SELL a 0.4-delta CE
   Entry = OPEN of the first 1-min option bar at/after that candle's close.
3. SL (on the underlying) = the NEARER of
       - the opposite ORB edge, or
       - 100 points from the entry spot (spot = the breakout candle's close).
4. TP = 1.5 x SL distance on the underlying, OR the option premium falling
   90% (buy back at 10% of the premium received), OR 15:15 IST - whichever
   comes first.

ASSUMPTIONS (constants below)
---------------------------------------------------------------
* "short 0.4 delta strike" = WRITING the OTM option closest to 0.40 |delta|.
  No greeks in the data: implied vol is solved per strike from its last
  traded price (Black-Scholes, spot, RISK_FREE=7%), then delta from that IV.
* TP "1.5 x SL" is measured on the UNDERLYING (entry spot -/+ 1.5 x SL distance).
* Exits: the 1-min SPOT bar touching SL/TP (high/low) triggers; premium TP
  uses the 1-min option CLOSE; the fill is the option's close of that minute
  (open for the 15:15 time exit). SL and TP in the same minute -> SL first.
* One trade per day; the breakout candle must close by LAST_ENTRY_TIME.
* 2% adverse slippage on sell and buy-back. No brokerage/STT/taxes.
* P&L is per 1 lot (LOT_SIZE). No margin model.

NO LOOK-AHEAD
---------------------------------------------------------------
ORB is used only from the 09:45 candle on; a candle is acted on only after it
closes; strike/IV/delta use option prints stamped before the signal time.

DATA: 1-min spot + weekly-options CSVs. Set DATA_ROOT (or env NIFTY_DATA_ROOT).
Layout: nifty_spot\\<year>\\<month>\\*.csv and nifty_options\\<year>\\<month>\\*.csv,
or flat nifty_spot*.csv / nifty_options*.csv in DATA_ROOT.
"""

import os
import glob
import gc
import math
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore")

# ----------------------------- PATHS / RANGE -----------------------------
DATA_ROOT = os.environ.get("NIFTY_DATA_ROOT", r"C:\Users\RISHI\nifty_data")
START_YEAR, START_MONTH = 2024, 1
END_YEAR, END_MONTH = 2024, 10

OUT_DIR = "/mnt/user-data/outputs" if os.path.isdir("/mnt/user-data/outputs") else "."
CHART_DIR = os.path.join(OUT_DIR, "orb_breakout_charts")
PLOT_CHARTS = True
PLOT_ALL_DAYS = True          # also chart days with no trade

# ----------------------------- STRATEGY CONFIG ---------------------------
ORB_START = "09:15"
ORB_END = "09:45"             # exclusive: ORB = bars 09:15..09:44
SETUP_TF_MIN = 15

TARGET_DELTA = 0.40
RISK_FREE = 0.07
STRIKE_MIN_PRICE = 0.5        # ignore strikes quoted below this (junk)
MAX_STALE_MIN = 3             # strike's last print must be this recent at signal time
MAX_ENTRY_DELAY_MIN = 2       # first option bar at/after signal end must be within this

SL_MAX_POINTS = 100           # SL = nearer of (opposite ORB edge, this many points)
TP_R = 1.5                    # TP = TP_R x SL distance (underlying)
PREMIUM_TP_PCT = 0.90         # buy back when premium has fallen 90%
EXIT_TIME = "15:15"
LAST_ENTRY_TIME = "15:00"
MAX_TRADES_PER_DAY = 1

LOT_SIZE = 50
SLIPPAGE_PCT = 0.02


# ----------------------------- DATA LOADING ------------------------------
def _month_range(sy, sm, ey, em):
    out, y, m = [], sy, sm
    while (y, m) <= (ey, em):
        out.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _structured(sub, y, m):
    d = os.path.join(DATA_ROOT, sub, str(y), str(m))
    return sorted(glob.glob(os.path.join(d, "*.csv"))) if os.path.isdir(d) else []


def _flat(prefix):
    return sorted(glob.glob(os.path.join(DATA_ROOT, f"{prefix}*.csv")))


def get_batches():
    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(f"DATA_ROOT does not exist: {DATA_ROOT!r} - edit DATA_ROOT "
                                f"or set NIFTY_DATA_ROOT")
    batches = []
    for y, m in _month_range(START_YEAR, START_MONTH, END_YEAR, END_MONTH):
        s, o = _structured("nifty_spot", y, m), _structured("nifty_options", y, m)
        if s and o:
            batches.append({'label': f'{y}-{m:02d}', 'spot': s, 'opt': o})
    if batches:
        return batches
    s, o = _flat("nifty_spot"), _flat("nifty_options")
    if not (s and o):
        raise FileNotFoundError(f"No spot/options CSVs found under {DATA_ROOT}")
    return [{'label': 'all-data (flat)', 'spot': s, 'opt': o}]


_F = {'open': 'float32', 'high': 'float32', 'low': 'float32', 'close': 'float32'}


def _read(files, dtypes):
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, dtype=dtypes))
        except Exception as e:
            print(f"  WARNING: could not load {f}: {e}")
    if not dfs:
        raise ValueError("no files could be loaded")
    return pd.concat(dfs, ignore_index=True)


def load_spot(files):
    df = _read(files, _F)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    return df.sort_values('datetime').set_index('datetime')[['open', 'high', 'low', 'close']]


def load_options(files):
    df = _read(files, dict(_F, oi='float32', volume='int32'))
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    # NIFTY + DDMMMYY + STRIKE + CE/PE   e.g. NIFTY04JAN2418300PE
    df['opt_type'] = df['symbol'].str[-2:]
    df['expiry'] = pd.to_datetime(df['symbol'].str[5:12], format='%d%b%y') + pd.Timedelta(hours=15, minutes=30)
    df['strike'] = df['symbol'].str[12:-2].astype('int32')
    return df.sort_values('datetime').set_index('datetime')[
        ['symbol', 'opt_type', 'strike', 'expiry', 'open', 'high', 'low', 'close', 'oi', 'volume']]


# ----------------------------- BLACK-SCHOLES -----------------------------
def bs_price(S, K, T, r, sig, cp):
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if cp == 'CE':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sig, cp):
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    return norm.cdf(d1) if cp == 'CE' else norm.cdf(d1) - 1.0


def implied_vol(price, S, K, T, r, cp):
    if T <= 0 or price <= 0:
        return np.nan
    intrinsic = max(S - K, 0.0) if cp == 'CE' else max(K - S, 0.0)
    if price <= intrinsic:
        return np.nan
    f = lambda s: bs_price(S, K, T, r, s, cp) - price
    try:
        if f(1e-3) * f(5.0) > 0:
            return np.nan
        return brentq(f, 1e-3, 5.0, xtol=1e-6)
    except Exception:
        return np.nan


# ----------------------------- CANDLES -----------------------------------
def make_candles(spot_day, minutes):
    r = spot_day.resample(f"{minutes}min", label='left', closed='left', origin='start_day').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    r = r.reset_index().rename(columns={'datetime': 'start'})
    r['end'] = r['start'] + pd.Timedelta(minutes=minutes)
    return r


# ----------------------------- SETUP DETECTION ---------------------------
def find_setup(c15, orh, orl, day):
    """First 15-min candle after the ORB that CLOSES beyond it, if it closes
    by LAST_ENTRY_TIME. Returns a dict ('stage' = 'none' | 'signal')."""
    orb_end = pd.Timestamp(f"{day} {ORB_END}")
    last_entry = pd.Timestamp(f"{day} {LAST_ENTRY_TIME}")
    for _, r in c15[c15['start'] >= orb_end].iterrows():
        if r['end'] > last_entry:
            break
        if r['close'] > orh:
            d = 'LONG'
        elif r['close'] < orl:
            d = 'SHORT'
        else:
            continue
        return {'stage': 'signal', 'direction': d, 'bo_start': r['start'], 'bo_end': r['end'],
                'bo_hi': float(r['high']), 'bo_lo': float(r['low']),
                'sig_end': r['end'], 'sig_close': float(r['close'])}
    return {'stage': 'none'}


# ----------------------------- STRIKE SELECTION --------------------------
def pick_delta_strike(day_opts, spot_1min, direction, t_sig):
    """Sell PE when bullish, CE when bearish. Choose the OTM strike whose
    |delta| (from per-strike implied vol) is closest to TARGET_DELTA, using
    only prints stamped before t_sig."""
    cp = 'PE' if direction == 'LONG' else 'CE'
    win = day_opts[(day_opts.index < t_sig) &
                   (day_opts.index >= t_sig - pd.Timedelta(minutes=MAX_STALE_MIN)) &
                   (day_opts['opt_type'] == cp)]
    if win.empty:
        return None
    last = win.groupby('strike').tail(1)
    sp = spot_1min[spot_1min.index < t_sig]
    if sp.empty:
        return None
    S = float(sp['close'].iloc[-1])
    best = None
    for ts, r in last.iterrows():
        K = int(r['strike'])
        if (cp == 'PE' and K >= S) or (cp == 'CE' and K <= S):
            continue                                   # OTM only
        px = float(r['close'])
        if px < STRIKE_MIN_PRICE:
            continue
        T = (r['expiry'] - (ts + pd.Timedelta(minutes=1))).total_seconds() / (365 * 24 * 3600)
        if T <= 0:
            continue
        iv = implied_vol(px, S, K, T, RISK_FREE, cp)
        if np.isnan(iv):
            continue
        dlt = abs(bs_delta(S, K, T, RISK_FREE, iv, cp))
        gap = abs(dlt - TARGET_DELTA)
        if best is None or gap < best['gap']:
            best = {'strike': K, 'opt_type': cp, 'delta': dlt, 'iv': iv, 'gap': gap, 'ref_px': px, 'spot_ref': S}
    return best


# ----------------------------- TRADE SIMULATION --------------------------
def simulate_trade(setup, day, day_opts, spot_1min, orh, orl):
    direction = setup['direction']
    t_sig = setup['sig_end']
    spot_e = setup['sig_close']

    pick = pick_delta_strike(day_opts, spot_1min, direction, t_sig)
    if pick is None:
        return None, 'no valid 0.4-delta strike (no fresh OTM prints / IV failed)'

    leg = day_opts[(day_opts['strike'] == pick['strike']) & (day_opts['opt_type'] == pick['opt_type'])]
    entry_win = leg[leg.index >= t_sig]
    if entry_win.empty:
        return None, 'no option print after signal'
    t_entry = entry_win.index[0]
    if t_entry - t_sig > pd.Timedelta(minutes=MAX_ENTRY_DELAY_MIN):
        return None, 'first option print too long after signal (stale)'
    exit_dt = pd.Timestamp(f"{day} {EXIT_TIME}")
    if t_entry >= exit_dt:
        return None, 'entry at/after time-exit'

    entry_raw = float(entry_win['open'].iloc[0])
    credit = entry_raw * (1 - SLIPPAGE_PCT)                 # sell fills lower
    if credit <= 0:
        return None, 'non-positive entry premium'

    if direction == 'LONG':
        orb_dist = spot_e - orl
        dist = min(orb_dist, SL_MAX_POINTS)
        sl_type = 'orb_low' if orb_dist <= SL_MAX_POINTS else '100pts'
        sl, tp = spot_e - dist, spot_e + TP_R * dist
    else:
        orb_dist = orh - spot_e
        dist = min(orb_dist, SL_MAX_POINTS)
        sl_type = 'orb_high' if orb_dist <= SL_MAX_POINTS else '100pts'
        sl, tp = spot_e + dist, spot_e - TP_R * dist
    if dist <= 0:
        return None, 'SL distance not positive'
    prem_tp_px = (1 - PREMIUM_TP_PCT) * credit

    sp = spot_1min[spot_1min.index >= t_entry]
    opt_close = leg['close'].reindex(sp.index).ffill()
    opt_open = leg['open'].reindex(sp.index)
    if opt_close.isna().all():
        return None, 'no option data after entry'

    exit_t = exit_px = reason = spot_exit = None
    for t, row in sp.iterrows():
        oc = opt_close.get(t, np.nan)
        if t >= exit_dt:
            px = opt_open.get(t, np.nan)
            exit_t, exit_px, reason = t, (px if not np.isnan(px) else oc), 'time_exit_1515'
            spot_exit = float(row['open'])
            break
        if np.isnan(oc):
            continue
        sl_hit = (row['low'] <= sl) if direction == 'LONG' else (row['high'] >= sl)
        tp_hit = (row['high'] >= tp) if direction == 'LONG' else (row['low'] <= tp)
        if sl_hit:
            exit_t, exit_px, reason, spot_exit = t + pd.Timedelta(minutes=1), oc, 'sl_hit', sl
            break
        if tp_hit:
            exit_t, exit_px, reason, spot_exit = t + pd.Timedelta(minutes=1), oc, 'tp_1.5x_sl', tp
            break
        if oc <= prem_tp_px:
            exit_t, exit_px, reason, spot_exit = t + pd.Timedelta(minutes=1), oc, 'tp_90pct_premium', float(row['close'])
            break
    if exit_t is None:
        last_t = sp.index[-1]
        exit_t, exit_px, reason = last_t + pd.Timedelta(minutes=1), float(opt_close.dropna().iloc[-1]), 'eod_data_end'
        spot_exit = float(sp['close'].iloc[-1])

    buyback = exit_px * (1 + SLIPPAGE_PCT)                  # buy-back fills higher
    pnl_pts = credit - buyback
    sgn = 1 if direction == 'LONG' else -1
    trade = {
        'date': day, 'direction': direction, 'sold': f"{pick['strike']}{pick['opt_type']}",
        'strike': pick['strike'], 'opt_type': pick['opt_type'],
        'delta_at_signal': round(pick['delta'], 3), 'iv_at_signal': round(pick['iv'], 3),
        'breakout_candle_start': setup['bo_start'],
        'signal_bar_end': t_sig, 'entry_time': t_entry, 'exit_time': exit_t,
        'spot_entry': spot_e, 'sl_level': sl, 'sl_type': sl_type, 'sl_points': round(dist, 1), 'tp_level': tp, 'spot_exit': spot_exit,
        'premium_sold': round(credit, 2), 'premium_bought': round(buyback, 2),
        'exit_reason': reason, 'pnl_points': round(pnl_pts, 2),
        'pnl_rupee_1lot': round(pnl_pts * LOT_SIZE, 2),
        'spot_R': round(sgn * (spot_exit - spot_e) / dist, 3),
    }
    return trade, None


# ----------------------------- CHARTS ------------------------------------
def plot_day(day, c15, orh, orl, setup, trade, out_path):
    fig, ax = plt.subplots(figsize=(14, 7))
    w = (SETUP_TF_MIN / 1440.0) * 0.7
    for _, r in c15.iterrows():
        x = mdates.date2num(r['start'].to_pydatetime())
        up = r['close'] >= r['open']
        col = '#2e9e5b' if up else '#d64545'
        ax.plot([x + w / 2, x + w / 2], [r['low'], r['high']], color=col, lw=1)
        ax.add_patch(Rectangle((x, min(r['open'], r['close'])), w, max(abs(r['close'] - r['open']), 0.5),
                               facecolor=col, edgecolor=col))
    x0 = mdates.date2num(pd.Timestamp(f"{day} {ORB_START}").to_pydatetime())
    x1 = mdates.date2num(pd.Timestamp(f"{day} {ORB_END}").to_pydatetime())
    xe = mdates.date2num(pd.Timestamp(f"{day} 15:30").to_pydatetime())
    ax.add_patch(Rectangle((x0, orl), x1 - x0, orh - orl, facecolor='#f4d35e', alpha=0.25, edgecolor='none'))
    ax.hlines([orh, orl], x0, xe, colors='#b8860b', linestyles='--', lw=1.2)
    ax.text(xe, orh, f" ORB high {orh:.1f}", va='center', fontsize=8, color='#b8860b')
    ax.text(xe, orl, f" ORB low {orl:.1f}", va='center', fontsize=8, color='#b8860b')

    title = f"{day}  |  15-min chart, 30-min ORB"
    if setup.get('stage') == 'signal':
        bx = mdates.date2num(setup['bo_start'].to_pydatetime())
        ax.axvspan(bx, bx + w, color='#3b6ea5', alpha=0.15)
        ax.text(bx + w / 2, c15['high'].max(), "breakout", ha='center', fontsize=8, color='#3b6ea5')
        sx = mdates.date2num(setup['sig_end'].to_pydatetime())
        ax.plot(sx, setup['sig_close'], marker='v' if setup['direction'] == 'SHORT' else '^',
                color='black', ms=10, zorder=5)
        title += f"  |  breakout {setup['direction']}"
    if trade:
        _lo, _hi = c15['low'].min(), c15['high'].max()
        _pad = (_hi - _lo) * 0.08
        ex = mdates.date2num(trade['exit_time'].to_pydatetime())
        en = mdates.date2num(trade['entry_time'].to_pydatetime())
        for lvl, lab, col in ((trade['sl_level'], 'SL', '#d64545'), (trade['tp_level'], 'TP', '#2e9e5b')):
            if _lo - _pad <= lvl <= _hi + _pad:                  # draw only if on-screen
                ax.hlines(lvl, en, xe, colors=col, linestyles='-.', lw=1)
                ax.text(xe, lvl, f" {lab} {lvl:.1f}", va='center', fontsize=8, color=col)
            else:
                title += f"  |  {lab} {lvl:.1f} off-chart"
        ax.plot(ex, trade['spot_exit'], marker='X', color='crimson', ms=10, zorder=5)
        ax.annotate(f"exit: {trade['exit_reason']}", (ex, trade['spot_exit']), textcoords='offset points',
                    xytext=(-10, -18), fontsize=8, color='crimson')
        title += (f"  |  SELL {trade['sold']} (d={trade['delta_at_signal']}) "
                  f"{trade['premium_sold']:.1f}->{trade['premium_bought']:.1f}  "
                  f"P&L Rs {trade['pnl_rupee_1lot']:,.0f}")
    else:
        title += "  |  no trade"
    ax.set_title(title, fontsize=10)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
    ax.set_xlim(x0 - w, xe + (xe - x0) * 0.12)
    lo, hi = c15['low'].min(), c15['high'].max()
    pad = (hi - lo) * 0.08
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ----------------------------- SUMMARY -----------------------------------
def summarize(trades):
    if trades.empty:
        print("\n--- ORB breakout strategy: no trades ---")
        return
    wins = trades[trades['pnl_rupee_1lot'] > 0]
    print("\n--- ORB breakout (sell 0.4 delta) ---")
    print(f"  Trades: {len(trades)}   Win rate: {len(wins) / len(trades):.1%}")
    print(f"  Total P&L: Rs {trades['pnl_rupee_1lot'].sum():,.0f} per lot   "
          f"Avg/trade: Rs {trades['pnl_rupee_1lot'].mean():,.0f}   "
          f"Best: Rs {trades['pnl_rupee_1lot'].max():,.0f}   Worst: Rs {trades['pnl_rupee_1lot'].min():,.0f}")
    print(f"  Avg spot-R: {trades['spot_R'].mean():.2f}")
    print(f"  Exit reasons: {trades['exit_reason'].value_counts().to_dict()}")


# ----------------------------- MAIN --------------------------------------
def run_backtest():
    batches = get_batches()
    print(f"{len(batches)} batch(es): {[b['label'] for b in batches]}")
    all_trades, day_log = [], []

    for b in batches:
        print(f"\n=== Batch {b['label']} ===")
        spot = load_spot(b['spot'])
        opts = load_options(b['opt'])
        opt_by_day = {d: g for d, g in opts.groupby(opts.index.date)}

        for day, sday in spot.groupby(spot.index.date):
            day_opts = opt_by_day.get(day)
            orb = sday.between_time(ORB_START, "09:44")
            if orb.empty or day_opts is None:
                day_log.append((day, 'no data'))
                continue
            orh, orl = float(orb['high'].max()), float(orb['low'].min())
            c15 = make_candles(sday, SETUP_TF_MIN)
            setup = find_setup(c15, orh, orl, day)

            trade, why = None, None
            if setup['stage'] == 'signal':
                trade, why = simulate_trade(setup, day, day_opts, sday, orh, orl)
                if trade:
                    all_trades.append(trade)
            status = setup['stage'] if not trade else 'traded'
            note = trade['exit_reason'] if trade else (why or "no 15-min close beyond ORB before cutoff")
            day_log.append((day, f"ORB {orl:.1f}-{orh:.1f} | {status} | {note}"))
            print(f"  {day}: ORB {orl:.1f}-{orh:.1f} | stage={setup['stage']}"
                  + (f" dir={setup['direction']}" if 'direction' in setup else "")
                  + (f" | {note}" if note else ""))

            if PLOT_CHARTS and (trade or PLOT_ALL_DAYS):
                plot_day(day, c15, orh, orl, setup, trade, os.path.join(CHART_DIR, f"orb_{day}.png"))
        del spot, opts, opt_by_day
        gc.collect()

    trades = pd.DataFrame(all_trades)
    print(f"\n{'=' * 70}\nRESULTS\n{'=' * 70}")
    summarize(trades)
    if not trades.empty:
        p = os.path.join(OUT_DIR, "trades_orb_breakout.csv")
        trades.to_csv(p, index=False)
        print(f"  Saved to {p}")
    return trades


if __name__ == "__main__":
    run_backtest()