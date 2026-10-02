"""
Nifty Intraday ORB-Retest Strategy (prediction-filtered)
==========================================================
Strategy (as specified):
  1. Wait for the first 30 minutes of the session (09:15-09:45 IST) to form
     the "30-min range" (range_high, range_low).
  2. From 09:45 onward, watch 15-min candles.
     - If a 15-min candle CLOSES BELOW the range low, AND the daily model
       (nifty_predict_report.py's final_forecast(), run with the PRIOR
       trading day as cutoff) says "tomorrow's close likely BELOW today's
       price" -> wait for price to come back UP and TOUCH the range low,
       then SELL there.
     - If a 15-min candle CLOSES ABOVE the range high, AND the model says
       "tomorrow's close likely ABOVE today's price" -> wait for price to
       come back DOWN and TOUCH the range high, then BUY there.
     - Any other combination (breakout direction disagrees with the model,
       or no breakout, or breakout but no retest) -> No Trade that day.
  3. Risk management:
     - SL = the opposite side of the 30-min range, i.e. SL distance =
       range width (range_high - range_low) -- BUT capped at 100 points:
       if the range is wider than 100, the SL distance used is 100, not
       the full range.
     - Target = 1.5x the (possibly capped) SL distance.
     - Exit at whichever of SL / Target / 15:15 forced time-exit comes
       first. If a single candle's High/Low range could contain both the
       SL and the Target (can't be resolved with OHLC-only data), the
       conservative assumption is that SL was hit first.

No look-ahead: the forecast for day D is generated using ONLY data up to
and including the prior trading day (D-1) -- exactly the "tomorrow's
close" forecast nifty_predict_report.py produces when run with
--cutoff-date = D-1.

USAGE
-----
    # Run for a specific target date (fetches/uses that day's intraday data,
    # prints/saves the trade plan and outcome) AND runs a backtest over
    # every date found in the local intraday CSV:
    python3 intraday_orb_strategy.py --target-date 2026-08-17

    # Backtest only, skip the single target-date run:
    python3 intraday_orb_strategy.py --target-date 2026-08-17 --skip-target-run

    # Point at a different intraday data file / disable the live yfinance fallback:
    python3 intraday_orb_strategy.py --target-date 2026-08-17 --intraday-csv my_data.csv --no-live-fetch

DATA
----
Intraday OHLC data is read from --intraday-csv (default set per-script --
see the 1-min and 5-min variants of this file), in the format produced by
nift_1m_data.py / yfinance's `^NSEI` download. If the target date isn't in
that file, the script tries to fetch it live via yfinance (Yahoo's 1m data
only goes back ~7-8 days; 5m/15m go back ~60 days).

Daily OHLC + the four feature sources used by the forecast model are read
via the same defaults as nifty_predict_report.py (Nifty_OHLC_Direction_
Indicators.xlsx, INDIAVIX_OHLC_Direction_Indicators.xlsx,
us_overnight_market_features.csv, Gift_Nifty_50_Futures_Historical_Data.csv).
"""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

from nifty_predict_report import (
    DEFAULT_CONFIG, load_and_merge, engineer_features, final_forecast, backend_name,
)

warnings.filterwarnings('ignore', category=FutureWarning)

IST = 'Asia/Kolkata'
RANGE_MINUTES = 30
CANDLE_MINUTES = 15   # breakout/retest confirmation candle size
SL_CAP = 1000.0         # SL distance (points) is capped at this if the ORB range exceeds it
TARGET_R_MULTIPLE = 10  # profit target = this many times the (possibly capped) SL distance
MARKET_OPEN = '09:15:00'
EXIT_TIME = '15:15:00'
MARKET_CLOSE = '15:30:00'

BAR_MINUTES = 5              # <-- this script is fixed for 5-min intraday bars
DEFAULT_INTRADAY_CSV = 'nifty_5min_60days.csv'


# =============================================================================
# Intraday OHLC data loading (works for any bar size -- 1-min, 5-min, ...)
# =============================================================================
def load_intraday_csv(path):
    """Reads either a plain Datetime,Open,High,Low,Close[,Volume] CSV, or the
    yfinance multi-header export format seen in nifty_5min_60days.csv /
    nifty_1min_30days.csv (Price/Ticker/Datetime header rows)."""
    with open(path) as f:
        head = [f.readline() for _ in range(3)]
    if head[1].startswith('Ticker'):
        df = pd.read_csv(path, skiprows=[1, 2])
    else:
        df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    if df['Datetime'].dt.tz is None:
        df['Datetime'] = df['Datetime'].dt.tz_localize(IST)
    else:
        df['Datetime'] = df['Datetime'].dt.tz_convert(IST)
    keep = [c for c in ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
    df = df[keep].sort_values('Datetime').drop_duplicates('Datetime').reset_index(drop=True)
    for c in ['Open', 'High', 'Low', 'Close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['Open', 'High', 'Low', 'Close']).reset_index(drop=True)


def fetch_live(date_str, interval='5m'):
    """Fallback: fetch a single day's ^NSEI intraday data live via yfinance.
    Yahoo's lookback limits depend on interval: ~7-8 days for 1m, ~60 days for
    2m/5m/15m/30m/60m/90m. Requires internet access + `pip install yfinance`."""
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError(
            "yfinance is not installed, and this date isn't in the local CSV. "
            "Install it with `pip install yfinance` and ensure you have internet access, "
            "or provide a CSV that already includes this date via --intraday-csv.")

    target = pd.Timestamp(date_str)
    start = target.strftime('%Y-%m-%d')
    end = (target + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"[live-fetch] Requesting ^NSEI {interval} data for {start} via yfinance ...")
    raw = yf.download(tickers='^NSEI', start=start, end=end, interval=interval, progress=False)
    if raw is None or raw.empty:
        # Yahoo sometimes only honors a `period=` request, not exact start/end for intraday data
        fallback_period = '8d' if interval == '1m' else '60d'
        raw = yf.download(tickers='^NSEI', period=fallback_period, interval=interval, progress=False)
    if raw is None or raw.empty:
        limit = "7-8 days" if interval == '1m' else "60 days"
        raise RuntimeError(
            f"yfinance returned no {interval} data for {start}. Yahoo only serves {interval} data for "
            f"roughly the trailing {limit} -- if this date is older than that, provide a CSV "
            "with that date's data via --intraday-csv instead.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index().rename(columns={raw.reset_index().columns[0]: 'Datetime'})
    raw['Datetime'] = pd.to_datetime(raw['Datetime'])
    if raw['Datetime'].dt.tz is None:
        raw['Datetime'] = raw['Datetime'].dt.tz_localize(IST)
    else:
        raw['Datetime'] = raw['Datetime'].dt.tz_convert(IST)
    keep = [c for c in ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'] if c in raw.columns]
    df = raw[keep].sort_values('Datetime').reset_index(drop=True)

    df = df[df['Datetime'].dt.date == target.date()].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"yfinance data fetched but contained no bars for {start}.")
    return df


def get_day_bars(all_bars, date_str, csv_path, allow_live_fetch, live_interval='5m'):
    target_date = pd.Timestamp(date_str).date()
    day_df = pd.DataFrame()
    if all_bars is not None and len(all_bars):
        day_df = all_bars[all_bars['Datetime'].dt.date == target_date].reset_index(drop=True)
    if day_df.empty:
        if not allow_live_fetch:
            raise RuntimeError(f"{target_date} not found in {csv_path}, and live fetch is disabled "
                                f"(--no-live-fetch). Provide a CSV that includes this date.")
        day_df = fetch_live(str(target_date), interval=live_interval)
    return day_df.set_index('Datetime').sort_index()


# =============================================================================
# Forecast (delegates entirely to nifty_predict_report.py's final_forecast())
# =============================================================================
def get_daily_forecast(target_date, cfg):
    """Forecast for `target_date`, generated using only data up to the prior
    trading day (cfg['end_date'] is set to the calendar day before target_date;
    load_and_merge() naturally falls back to the most recent trading day at or
    before that, so weekends/holidays are handled automatically)."""
    cutoff = (pd.Timestamp(target_date) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    cfg = dict(cfg)
    cfg['end_date'] = cutoff
    df = load_and_merge(cfg)
    data, feat_cols = engineer_features(df)
    fc = final_forecast(df, data, feat_cols, cfg)

    if fc['Close'] > fc['last_close']:
        direction = 'Up'
        msg = (f"Tomorrow's ({target_date}) closing price likely to be ABOVE today's "
               f"({fc['last_date']}) price of {fc['last_close']:,.1f}, with a high of {fc['High']:,.1f}")
    elif fc['Close'] < fc['last_close']:
        direction = 'Down'
        msg = (f"Tomorrow's ({target_date}) closing price likely to be BELOW today's "
               f"({fc['last_date']}) price of {fc['last_close']:,.1f}, with a low of {fc['Low']:,.1f}")
    else:
        direction = 'Flat'
        msg = f"Tomorrow's ({target_date}) closing price forecast is flat vs. today's ({fc['last_date']})."

    fc['direction'] = direction
    fc['message'] = msg
    return fc


# =============================================================================
# Opening range + breakout/retest mechanics
# =============================================================================
def compute_opening_range(day_bars, session_date, bar_minutes):
    open_ts = pd.Timestamp(f"{session_date} {MARKET_OPEN}", tz=IST)
    range_end_ts = open_ts + pd.Timedelta(minutes=RANGE_MINUTES)
    rng = day_bars[(day_bars.index >= open_ts) & (day_bars.index < range_end_ts)]
    expected_bars = max(1, RANGE_MINUTES // bar_minutes)
    if len(rng) < max(1, int(expected_bars * 0.6)):  # need a reasonably complete first 30 minutes
        return None
    return dict(range_high=rng['High'].max(), range_low=rng['Low'].min(),
                open_ts=open_ts, range_end_ts=range_end_ts)


def get_breakout_candles(day_bars, from_ts, to_ts, open_ts):
    post = day_bars[(day_bars.index >= from_ts) & (day_bars.index <= to_ts)]
    if post.empty:
        return pd.DataFrame()
    candles = post.resample(f'{CANDLE_MINUTES}min', origin=open_ts, label='left', closed='left').agg(
        Open=('Open', 'first'), High=('High', 'max'), Low=('Low', 'min'), Close=('Close', 'last'))
    return candles.dropna()


def find_breakout(candles, range_high, range_low, direction):
    for ts, row in candles.iterrows():
        if direction == 'Down' and row['Close'] < range_low:
            return ts + pd.Timedelta(minutes=CANDLE_MINUTES)  # candle's confirmed-close time
        if direction == 'Up' and row['Close'] > range_high:
            return ts + pd.Timedelta(minutes=CANDLE_MINUTES)
    return None


def find_retest_entry(day_bars, search_from, search_to, level, direction):
    post = day_bars[(day_bars.index >= search_from) & (day_bars.index <= search_to)]
    if direction == 'Down':
        touch = post[post['High'] >= level]
    else:
        touch = post[post['Low'] <= level]
    if touch.empty:
        return None, None
    return touch.index[0], level


def simulate_exit(day_bars, entry_ts, sl_level, target_level, direction, exit_deadline):
    """Walks bar-by-bar after entry, checking SL and target each bar.
    If a single bar's range could have hit BOTH levels (ambiguous with OHLC-only
    data), the conservative assumption is that SL was hit first."""
    post = day_bars[(day_bars.index > entry_ts) & (day_bars.index <= exit_deadline)]
    for ts, row in post.iterrows():
        if direction == 'Down':  # short: SL above (range_high side), target below
            sl_hit = row['High'] >= sl_level
            tgt_hit = row['Low'] <= target_level
        else:  # long: SL below (range_low side), target above
            sl_hit = row['Low'] <= sl_level
            tgt_hit = row['High'] >= target_level
        if sl_hit and tgt_hit:
            return ts, sl_level, 'SL Hit (same-bar w/ target, conservative)'
        if sl_hit:
            return ts, sl_level, 'SL Hit'
        if tgt_hit:
            return ts, target_level, 'Target Hit'

    upto_deadline = day_bars[day_bars.index <= exit_deadline]
    if upto_deadline.empty:
        last_ts, last_row = day_bars.index[-1], day_bars.iloc[-1]
        return last_ts, last_row['Close'], 'Time Exit (EOD, no 15:15 bar)'
    last_ts = upto_deadline.index[-1]
    return last_ts, upto_deadline.loc[last_ts, 'Close'], 'Time Exit (15:15)'


# =============================================================================
# Single-day pipeline
# =============================================================================
def run_one_day(target_date, day_bars, fc, bar_minutes, verbose=True):
    session_date = pd.Timestamp(target_date).date()
    result = dict(Date=session_date, Forecast_Direction=fc['direction'],
                  Forecast_Message=fc['message'], Anchor_Close=round(fc['last_close'], 2),
                  Predicted_Close=round(fc['Close'], 2))

    orb = compute_opening_range(day_bars, session_date, bar_minutes)
    if orb is None:
        result.update(Outcome='No Trade', Reason='Insufficient bars in the first 30 minutes')
        return result
    result.update(Range_High=round(orb['range_high'], 2), Range_Low=round(orb['range_low'], 2))

    if fc['direction'] == 'Flat':
        result.update(Outcome='No Trade', Reason='Model forecast is flat -- no directional signal')
        return result

    exit_deadline = pd.Timestamp(f"{session_date} {EXIT_TIME}", tz=IST)
    market_close = pd.Timestamp(f"{session_date} {MARKET_CLOSE}", tz=IST)
    day_end = min(day_bars.index.max(), market_close) if len(day_bars) else market_close

    candles = get_breakout_candles(day_bars, orb['range_end_ts'], day_end, orb['open_ts'])
    breakout_confirm_ts = find_breakout(candles, orb['range_high'], orb['range_low'], fc['direction'])
    if breakout_confirm_ts is None:
        result.update(Outcome='No Trade',
                       Reason=f"No {CANDLE_MINUTES}-min candle closed "
                              f"{'below range low' if fc['direction']=='Down' else 'above range high'}")
        return result
    result['Breakout_Confirm_Time'] = breakout_confirm_ts

    level = orb['range_low'] if fc['direction'] == 'Down' else orb['range_high']
    entry_ts, entry_price = find_retest_entry(day_bars, breakout_confirm_ts, exit_deadline, level, fc['direction'])
    if entry_ts is None:
        result.update(Outcome='No Trade',
                       Reason=f"Breakout confirmed but price never retested the "
                              f"{'range low' if fc['direction']=='Down' else 'range high'} before {EXIT_TIME}")
        return result

    range_width = orb['range_high'] - orb['range_low']
    sl_distance = min(range_width, SL_CAP)
    target_distance = TARGET_R_MULTIPLE * sl_distance
    if fc['direction'] == 'Down':  # short
        sl_level = entry_price + sl_distance
        target_level = entry_price - target_distance
    else:  # long
        sl_level = entry_price - sl_distance
        target_level = entry_price + target_distance

    side = 'SELL' if fc['direction'] == 'Down' else 'BUY'
    result.update(Side=side, Entry_Time=entry_ts, Entry_Price=round(entry_price, 2),
                  Range_Width=round(range_width, 2), SL_Distance=round(sl_distance, 2),
                  SL=round(sl_level, 2), Target_Distance=round(target_distance, 2),
                  Target=round(target_level, 2))

    exit_ts, exit_price, exit_reason = simulate_exit(day_bars, entry_ts, sl_level, target_level,
                                                       fc['direction'], exit_deadline)
    pnl_points = (entry_price - exit_price) if side == 'SELL' else (exit_price - entry_price)
    result.update(Exit_Time=exit_ts, Exit_Price=round(exit_price, 2), Exit_Reason=exit_reason,
                   PnL_Points=round(pnl_points, 2), PnL_Pct=round(pnl_points / entry_price * 100, 3),
                   Outcome='Win' if pnl_points > 0 else ('Loss' if pnl_points < 0 else 'Breakeven'))

    if verbose:
        print(f"\n[{session_date}] {fc['message']}")
        print(f"  30-min range: {orb['range_low']:,.1f} - {orb['range_high']:,.1f} (width {range_width:,.1f})")
        print(f"  Breakout confirmed at {breakout_confirm_ts.time()} "
              f"({'below range low' if fc['direction']=='Down' else 'above range high'})")
        print(f"  {side} entry {entry_price:,.1f} at {entry_ts.time()}, "
              f"SL {sl_level:,.1f} (dist {sl_distance:,.1f}), Target {target_level:,.1f} (dist {target_distance:,.1f})")
        print(f"  Exit {exit_price:,.1f} at {exit_ts.time()} ({exit_reason}) -> "
              f"P&L {pnl_points:+.1f} pts ({result['PnL_Pct']:+.2f}%)")

    return result


# =============================================================================
# Backtest across every date present in the 1-min CSV
# =============================================================================
def run_backtest(all_bars, cfg, bar_minutes):
    dates = sorted(all_bars['Datetime'].dt.date.unique())
    rows = []
    for d in dates:
        d_str = d.strftime('%Y-%m-%d')
        try:
            fc = get_daily_forecast(d_str, cfg)
        except Exception as e:
            rows.append(dict(Date=d, Outcome='No Trade', Reason=f'Forecast error: {e}'))
            continue
        day_bars = all_bars[all_bars['Datetime'].dt.date == d].set_index('Datetime').sort_index()
        rows.append(run_one_day(d_str, day_bars, fc, bar_minutes, verbose=True))
    return pd.DataFrame(rows)


def summarize_backtest(bt):
    n = len(bt)
    traded = bt[bt['Outcome'].isin(['Win', 'Loss', 'Breakeven'])]
    print(f"\n=== Backtest summary: {n} days, {len(traded)} trades taken ===")
    if len(traded):
        wins = (traded['Outcome'] == 'Win').sum()
        losses = (traded['Outcome'] == 'Loss').sum()
        total_pts = traded['PnL_Points'].sum()
        win_rate = wins / len(traded)
        print(f"  Win rate: {win_rate:.1%}  ({wins}W / {losses}L / {len(traded)-wins-losses}BE)")
        print(f"  Total P&L: {total_pts:+.1f} points | Avg/trade: {traded['PnL_Points'].mean():+.2f} pts")
    no_trade = n - len(traded)
    print(f"  No-trade days: {no_trade}")


# =============================================================================
# XLSX report
# =============================================================================
def write_backtest_xlsx(bt, out_path):
    wb = Workbook()
    ws = wb.active
    ws.title = 'ORB Backtest'

    header_font = Font(name='Arial', bold=True, color='FFFFFF')
    header_fill = PatternFill('solid', fgColor='1F4E78')
    body_font = Font(name='Arial', size=10)
    center = Alignment(horizontal='center', wrap_text=True)
    win_fill = PatternFill('solid', fgColor='C6EFCE')
    loss_fill = PatternFill('solid', fgColor='FFC7CE')
    win_font = Font(name='Arial', size=10, color='006100')
    loss_font = Font(name='Arial', size=10, color='9C0006')

    ws['A1'] = 'Nifty Intraday ORB-Retest Backtest'
    ws['A1'].font = Font(name='Arial', bold=True, size=14)
    n = len(bt)
    traded = bt[bt['Outcome'].isin(['Win', 'Loss', 'Breakeven'])]
    win_rate = (traded['Outcome'] == 'Win').mean() if len(traded) else float('nan')
    total_pts = traded['PnL_Points'].sum() if len(traded) else 0.0
    ws['A2'] = (f"{n} days | {len(traded)} trades taken | win rate "
                f"{win_rate:.1%} | total P&L {total_pts:+.1f} pts" if len(traded)
                else f"{n} days | 0 trades taken")
    ws['A2'].font = Font(name='Arial', italic=True, size=10)
    ws.merge_cells('A1:P1')
    ws.merge_cells('A2:P2')

    header_row = 4
    cols = list(bt.columns)
    for j, col in enumerate(cols, start=1):
        c = ws.cell(row=header_row, column=j, value=col.replace('_', ' '))
        c.font, c.fill, c.alignment = header_font, header_fill, center

    for i, row in bt.iterrows():
        r = header_row + 1 + i
        for j, col in enumerate(cols, start=1):
            val = row[col]
            val = '' if pd.isna(val) else (str(val) if not isinstance(val, (int, float)) else val)
            c = ws.cell(row=r, column=j, value=val)
            c.font, c.alignment = body_font, center
            if col == 'Outcome':
                if val == 'Win':
                    c.fill, c.font = win_fill, win_font
                elif val == 'Loss':
                    c.fill, c.font = loss_fill, loss_font

    for j in range(1, len(cols) + 1):
        ws.column_dimensions[get_column_letter(j)].width = 16
    ws.freeze_panes = ws.cell(row=header_row + 1, column=1)
    wb.save(out_path)


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description=f"Prediction-filtered opening-range breakout+retest intraday strategy for Nifty "
                    f"({BAR_MINUTES}-min bars).")
    p.add_argument('--target-date', '-d', required=True, help="Date to trade today (YYYY-MM-DD).")
    p.add_argument('--intraday-csv', default=DEFAULT_INTRADAY_CSV,
                    help=f"Local intraday OHLC CSV, {BAR_MINUTES}-min bars (yfinance ^NSEI export format). "
                         f"Default: {DEFAULT_INTRADAY_CSV}")
    p.add_argument('--no-live-fetch', action='store_true',
                    help="Disable the yfinance live-fetch fallback when --target-date isn't in the CSV.")
    p.add_argument('--skip-target-run', action='store_true', help="Skip the single target-date run.")
    p.add_argument('--skip-backtest', action='store_true', help="Skip the backtest over the CSV's dates.")
    p.add_argument('--nifty-xlsx', default=DEFAULT_CONFIG['nifty_xlsx'])
    p.add_argument('--vix-xlsx', default=DEFAULT_CONFIG['vix_xlsx'])
    p.add_argument('--us-csv', default=DEFAULT_CONFIG['us_csv'])
    p.add_argument('--gift-csv', default=DEFAULT_CONFIG['gift_csv'])
    return p.parse_args()


def main():
    args = parse_args()
    live_interval = f"{BAR_MINUTES}m"

    cfg = dict(DEFAULT_CONFIG)
    cfg.update(nifty_xlsx=args.nifty_xlsx, vix_xlsx=args.vix_xlsx, us_csv=args.us_csv, gift_csv=args.gift_csv)
    print(f"[gbm backend] {backend_name()}")

    try:
        all_bars = load_intraday_csv(args.intraday_csv)
        print(f"[intraday data] {args.intraday_csv} ({BAR_MINUTES}-min bars): "
              f"{all_bars['Datetime'].dt.date.nunique()} days, "
              f"{all_bars['Datetime'].min()} -> {all_bars['Datetime'].max()}")
    except FileNotFoundError:
        all_bars = None
        print(f"[intraday data] {args.intraday_csv} not found -- will rely on live fetch for the target date only.")

    if not args.skip_target_run:
        print(f"\n=== Target date: {args.target_date} ===")
        fc = get_daily_forecast(args.target_date, cfg)
        print(fc['message'])
        try:
            day_bars = get_day_bars(all_bars, args.target_date, args.intraday_csv,
                                     allow_live_fetch=not args.no_live_fetch, live_interval=live_interval)
            result = run_one_day(args.target_date, day_bars, fc, BAR_MINUTES, verbose=True)
            out_csv = f"ORB_Trade_{args.target_date}_{BAR_MINUTES}min.csv"
            pd.DataFrame([result]).to_csv(out_csv, index=False)
            print(f"[saved] {out_csv}")
        except RuntimeError as e:
            print(f"[warning] Could not get intraday data for {args.target_date}: {e}")

    if not args.skip_backtest:
        if all_bars is None or all_bars.empty:
            print("\n[backtest] No local intraday data available to backtest -- skipping.")
        else:
            print(f"\n=== Backtest over all {all_bars['Datetime'].dt.date.nunique()} days in "
                  f"{args.intraday_csv} ===")
            bt = run_backtest(all_bars, cfg, BAR_MINUTES)
            summarize_backtest(bt)
            out_csv = f"ORB_Backtest_Report_{BAR_MINUTES}min.csv"
            bt.to_csv(out_csv, index=False)
            print(f"\n[saved] {out_csv}")

#%run intraday_orb_strategy_5min.py --target-date 2026-08-15 
if __name__ == '__main__':
    main()
