"""
download_all_data.py
=====================
One consolidated script to (re)download every dataset that nifty_forecast.py
consumes, all driven by a single point-in-time cutoff argument.

How many datasets does nifty_forecast.py use?  FOUR.
------------------------------------------------------
Confirmed straight from nifty_forecast.py's DEFAULT_CONFIG:

  1. Nifty_OHLC_Direction_Indicators.xlsx     -- NIFTY 50 (^NSEI) daily OHLC + indicators
                                                  (design copied from data_download.py)
  2. INDIAVIX_OHLC_Direction_Indicators.xlsx  -- India VIX (^INDIAVIX) daily OHLC + indicators
                                                  (design copied from data_download.py -- same
                                                  code, this is literally what data_download.py
                                                  already produces when TICKER="^INDIAVIX")
  3. us_overnight_market_features.csv         -- SP500/NASDAQ/Crude/DXY/US_VIX overnight returns,
                                                  1-day shifted (design copied from the bottom
                                                  half of data_download.py)
  4. Gift_Nifty_50_Futures_Historical_Data.csv -- GIFT Nifty futures daily OHLC (source: investing.com,
                                                  best-effort scrape here, see download_gift_nifty())

  NOTE: nifty_1min_30days.csv is NOT one of the four -- grep nifty_forecast.py and it
  never appears. It looks like a leftover/legacy file from a different workflow, so this
  script does not touch it. Say the word if you actually want it refreshed too.

  NOTE: the NIFTY 50 09:15-09:45 IST opening-session workbook
  (nifty_opening_session_60d.xlsx, and the per-entity <STEM>_opening_session_60d.xlsx
  files below) has been removed from this script -- download_opening_session() and
  the CLI's --skip-opening-session flag are gone. Say the word if you want it back.

Universe extension (NSE/BSE indices + global macro + NIFTY 50 stocks)
------------------------------------------------------------------------
On top of the original 4 NIFTY 50 files, this script ALSO downloads, for every
entity listed in entities_universe.py (major NSE indices, the SENSEX, a handful
of non-NSE/BSE global-macro entities that are well known to heavily move NIFTY 50
-- USD/INR, Dow Jones, Nikkei 225, Hang Seng, crude oil, US 10y yield -- and all
50 NIFTY 50 constituent stocks), the SAME file type NIFTY 50 gets -- using the
exact same function (download_daily_ohlc_indicators), just parameterized by
ticker/label:

  <STEM>_OHLC_Direction_Indicators.xlsx   -- daily OHLC + indicators

where <STEM> is entities_universe.file_stem(label), e.g. "NIFTY BANK" ->
"NIFTY_BANK". See --skip-universe/--skip-indices/--skip-stocks/--only below
to control this.

NOTE: NIFTY AUTO/FMCG/METAL/REALTY/ENERGY/MEDIA/PSU BANK, NIFTY NEXT 50,
NIFTY FIN SERVICE, and BSE 100/200/500/MIDCAP/SMALLCAP were removed from
entities_universe.py -- none of them had usable yfinance data as of the
2026-07-31 verification pass, so they were dropped rather than silently
downloading stale/missing data. See entities_universe.py's docstring for
detail.

Point-in-time cutoff
---------------------
Every downloader is handed the SAME cutoff timestamp (assumed IST) and trims its
output so nothing after that instant leaks in -- each source has different session
timing, so "as of 09:45 IST" means something different for each one:

  - NIFTY / India VIX daily candles: the trading day's candle isn't final until the
    15:30 IST close. A 09:45 cutoff on day D means the LAST usable daily row is D-1
    (or D itself only if the cutoff time is >= 15:30).
  - US overnight features: the US cash session runs ~19:00 IST to ~01:30 IST
    (next day). By 09:45 IST on day D that session (dated D-1 in US terms) is long
    closed and safe to use; D's own US session hasn't even started yet.
  - GIFT Nifty futures: trades nearly round-the-clock overlapping both the US and
    Indian sessions, so investing.com's own daily "close" for day D isn't posted
    until D's full session wraps late that evening. A 09:45 cutoff on day D means
    the last safe daily row is still D-1.

Usage
-----
    python3 download_all_data.py "2026-08-17 09:45"
    python3 download_all_data.py "2026-08-17 09:45 AM"
    python3 download_all_data.py 2026-08-17                 # time defaults to 09:45 IST
    python3 download_all_data.py "2026-08-17 09:45" --skip-gift

Requires: yfinance, pandas, numpy, openpyxl, requests
"""
import argparse
import sys
from datetime import time as dtime, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("Missing dependency. Run: pip install yfinance pandas numpy openpyxl requests")

from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import LineChart, Reference

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except ImportError:  # pragma: no cover
    import pytz
    IST = pytz.timezone("Asia/Kolkata")

from entities_universe import (
    all_index_entities, all_stock_entities, file_stem,
)

MARKET_CLOSE = dtime(15, 30)     # NSE cash-session close
US_SESSION_START_IST = dtime(18, 0)  # rough US cash-open in IST, used only to decide
                                       # whether "today's" US session has even begun


# =============================================================================
# Output filenames -- match nifty_forecast.py's DEFAULT_CONFIG exactly
# =============================================================================
NIFTY_XLSX = "Nifty_OHLC_Direction_Indicators.xlsx"
VIX_XLSX = "INDIAVIX_OHLC_Direction_Indicators.xlsx"
US_CSV = "us_overnight_market_features.csv"
GIFT_CSV = "Gift_Nifty_50_Futures_Historical_Data.csv"


# =============================================================================
# Cutoff helpers
# =============================================================================
def parse_cutoff(cutoff_str):
    """Parses a flexible date/datetime string into an IST-aware Timestamp.
    Accepts '2026-08-17', '2026-08-17 09:45', '2026-08-17 9:45 AM', and (since we
    always assume IST anyway) an explicit '... IST' suffix like '2026-08-17 9:45 am IST'
    -- pandas doesn't recognize "IST" as a timezone name on its own (it's ambiguous
    with Israel Standard Time), so that suffix is stripped before parsing rather than
    passed through. A date with no time defaults to 09:45 IST (the original
    default cutoff time this script was built around)."""
    cleaned = cutoff_str.strip()
    if cleaned.upper().endswith("IST"):
        cleaned = cleaned[:-3].strip()

    ts = pd.Timestamp(cleaned)
    if ts.time() == dtime(0, 0) and " " not in cleaned and ":" not in cleaned:
        ts = ts.replace(hour=9, minute=45)  # default cutoff time
    if ts.tzinfo is None:
        ts = ts.tz_localize(IST)
    else:
        ts = ts.tz_convert(IST)
    return ts


def last_complete_daily_date(cutoff_ts):
    """Last NSE trading date whose 15:30 IST close has already happened as of cutoff_ts."""
    d = cutoff_ts.date()
    if cutoff_ts.time() >= MARKET_CLOSE:
        return d
    return d - timedelta(days=1)


def last_complete_us_date(cutoff_ts):
    """Last US calendar trading date whose session has already closed as of cutoff_ts
    (IST). The US session for IST-date D runs roughly 19:00 D -> 01:30 (D+1), so unless
    the cutoff is very late at night (after that close), the most recent SAFE row is D-1."""
    d = cutoff_ts.date()
    if cutoff_ts.time() < US_SESSION_START_IST:
        return d - timedelta(days=1)
    return d - timedelta(days=1)  # today's US session hasn't closed yet either way


def last_complete_gift_date(cutoff_ts):
    """GIFT Nifty runs nearly round-the-clock; investing.com only finalizes a day's
    daily row after that day's full session wraps in the evening. Same conservative
    D-1 boundary as US data for a morning cutoff."""
    return last_complete_us_date(cutoff_ts)


# =============================================================================
# Shared Excel styling helpers (same design as data_download.py)
# =============================================================================
def style_header(ws, ncols):
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF")
    for col in range(1, ncols + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
    ws.freeze_panes = "A2"


def autofit_columns(ws, df):
    # NOTE: on some pandas builds (PyArrow-backed string dtype), .astype(str) on a
    # column containing NaN can leave those cells as an actual float NaN instead of
    # the string "nan", which then breaks .map(len). Using .apply(lambda x: len(str(x)))
    # forces a real Python str() per element (indicator columns like SMA20/SMA50
    # legitimately have NaN for their first N warm-up rows, so this has to be handled).
    for i, col in enumerate(df.columns, start=1):
        max_len = max(df[col].apply(lambda x: len(str(x))).max(), len(str(col))) + 3
        ws.column_dimensions[get_column_letter(i)].width = min(max_len, 22)


def color_direction_column(ws, df, col_name):
    if col_name not in df.columns:
        return
    col_idx = df.columns.get_loc(col_name) + 1
    green = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    red = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    yellow = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    for row in range(2, len(df) + 2):
        cell = ws.cell(row=row, column=col_idx)
        if cell.value == "Bullish":
            cell.fill = green
        elif cell.value == "Bearish":
            cell.fill = red
        elif cell.value in ("Neutral", "No Trend"):
            cell.fill = yellow


# =============================================================================
# Indicator calculations (identical to data_download.py)
# =============================================================================
def add_sma_crossover(df):
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA_Direction"] = np.where(df["SMA20"] > df["SMA50"], "Bullish",
                             np.where(df["SMA20"] < df["SMA50"], "Bearish", "Neutral"))
    return df


def add_ema_crossover(df):
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_Direction"] = np.where(df["EMA20"] > df["EMA50"], "Bullish",
                             np.where(df["EMA20"] < df["EMA50"], "Bearish", "Neutral"))
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df["MACD_Direction"] = np.where(df["MACD"] > df["MACD_Signal"], "Bullish",
                              np.where(df["MACD"] < df["MACD_Signal"], "Bearish", "Neutral"))
    return df


def add_rsi(df, period=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI14"] = 100 - (100 / (1 + rs))
    df["RSI_Direction"] = np.where(df["RSI14"] > 55, "Bullish",
                             np.where(df["RSI14"] < 45, "Bearish", "Neutral"))
    return df


def add_adx(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[(plus_dm < 0) | (plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < 0) | (minus_dm < plus_dm)] = 0

    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    df["Plus_DI"] = plus_di
    df["Minus_DI"] = minus_di
    df["ADX14"] = adx
    df["ADX_Direction"] = np.where(
        df["ADX14"] < 20, "No Trend",
        np.where(df["Plus_DI"] > df["Minus_DI"], "Bullish", "Bearish")
    )
    return df


def build_summary(df):
    latest = df.iloc[-1]
    rows = [
        ("SMA 20/50 Crossover", latest["SMA_Direction"], f"SMA20={latest['SMA20']:.2f}, SMA50={latest['SMA50']:.2f}"),
        ("EMA 20/50 Crossover", latest["EMA_Direction"], f"EMA20={latest['EMA20']:.2f}, EMA50={latest['EMA50']:.2f}"),
        ("MACD (12,26,9)", latest["MACD_Direction"], f"MACD={latest['MACD']:.2f}, Signal={latest['MACD_Signal']:.2f}"),
        ("RSI (14)", latest["RSI_Direction"], f"RSI={latest['RSI14']:.2f}"),
        ("ADX (14) + DI", latest["ADX_Direction"], f"ADX={latest['ADX14']:.2f}, +DI={latest['Plus_DI']:.2f}, -DI={latest['Minus_DI']:.2f}"),
    ]
    bullish = sum(1 for r in rows if r[1] == "Bullish")
    bearish = sum(1 for r in rows if r[1] == "Bearish")
    overall = "Bullish" if bullish > bearish else ("Bearish" if bearish > bullish else "Mixed/Neutral")

    summary = pd.DataFrame(rows, columns=["Indicator", "Signal", "Details"])
    summary.loc[len(summary)] = ["OVERALL CONSENSUS", overall, f"{bullish} Bullish / {bearish} Bearish out of 5"]
    return summary


def write_ohlc_indicator_workbook(ohlc_df, indicators_df, summary_df, filename, chart_title):
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        ohlc_df.to_excel(writer, sheet_name="OHLC Data", index=False)
        indicators_df.to_excel(writer, sheet_name="Indicators", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        ws1 = writer.sheets["OHLC Data"]
        style_header(ws1, len(ohlc_df.columns))
        autofit_columns(ws1, ohlc_df)

        ws2 = writer.sheets["Indicators"]
        style_header(ws2, len(indicators_df.columns))
        autofit_columns(ws2, indicators_df)
        for col_name in ["SMA_Direction", "EMA_Direction", "MACD_Direction", "RSI_Direction", "ADX_Direction"]:
            color_direction_column(ws2, indicators_df, col_name)

        ws3 = writer.sheets["Summary"]
        style_header(ws3, len(summary_df.columns))
        autofit_columns(ws3, summary_df)
        color_direction_column(ws3, summary_df, "Signal")

        chart = LineChart()
        chart.title = chart_title
        chart.y_axis.title = "Price"
        chart.x_axis.title = "Date"
        close_col_idx = ohlc_df.columns.get_loc("Close") + 1
        data_ref = Reference(ws1, min_col=close_col_idx, min_row=1, max_row=len(ohlc_df) + 1)
        chart.add_data(data_ref, titles_from_data=True)
        chart.width = 24
        chart.height = 10
        ws1.add_chart(chart, "H2")

    print(f"  [saved] {filename}  ({len(ohlc_df)} daily rows, {ohlc_df['Date'].iloc[0]} -> {ohlc_df['Date'].iloc[-1]})")


# =============================================================================
# 1 & 2. NIFTY / India VIX daily OHLC + indicators
# =============================================================================
def download_daily_ohlc_indicators(ticker, out_file, cutoff_ts, label, period="10y"):
    print(f"[{label}] Downloading {period} of daily data for {ticker} ...")
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        print(f"  [FAILED] No data downloaded for {ticker}. Check ticker symbol or your internet connection.")
        return False

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    boundary = last_complete_daily_date(cutoff_ts)
    before = len(df)
    df = df[df["Date"] <= boundary].reset_index(drop=True)
    print(f"  [cutoff] Keeping daily rows <= {boundary} (dropped {before - len(df)} row(s) not yet closed as of {cutoff_ts})")

    if df.empty:
        print(f"  [FAILED] No data left for {ticker} after applying the cutoff.")
        return False

    indicators_df = df.copy()
    indicators_df = add_sma_crossover(indicators_df)
    indicators_df = add_ema_crossover(indicators_df)
    indicators_df = add_macd(indicators_df)
    indicators_df = add_rsi(indicators_df)
    indicators_df = add_adx(indicators_df)

    num_cols = indicators_df.select_dtypes(include=[np.number]).columns
    indicators_df[num_cols] = indicators_df[num_cols].round(2)

    summary_df = build_summary(indicators_df)
    write_ohlc_indicator_workbook(df, indicators_df, summary_df, out_file,
                                   chart_title=f"{label} Close Price")
    return True


# =============================================================================
# 3. US overnight market features
# =============================================================================
def download_us_overnight_features(cutoff_ts, out_file=US_CSV, period="10y"):
    print("[US overnight] Downloading SP500/NASDAQ/Crude/DXY/US_VIX ...")
    tickers = {
        "SP500": "^GSPC",
        "NASDAQ": "^NDX",
        "CRUDE": "BZ=F",
        "DXY": "DX-Y.NYB",
        "US_VIX": "^VIX",
    }
    raw = yf.download(list(tickers.values()), period=period, interval="1d", progress=False)
    if raw.empty:
        print("  [FAILED] No US data downloaded.")
        return False

    us_data = raw["Close"]
    us_data.columns = [k for k in tickers.keys()]
    us_data.index = pd.to_datetime(us_data.index).date
    us_data = pd.DataFrame(us_data).ffill()

    boundary = last_complete_us_date(cutoff_ts)
    before = len(us_data)
    us_data = us_data[us_data.index <= boundary]
    print(f"  [cutoff] Keeping US daily closes <= {boundary} "
          f"(dropped {before - len(us_data)} row(s) not yet closed as of {cutoff_ts})")

    if us_data.empty:
        print("  [FAILED] No US data left after applying the cutoff.")
        return False

    us_features = pd.DataFrame(index=us_data.index)
    us_features["SP500_Return"] = np.log(us_data["SP500"] / us_data["SP500"].shift(1))
    us_features["NASDAQ_Return"] = np.log(us_data["NASDAQ"] / us_data["NASDAQ"].shift(1))
    us_features["Crude_Change"] = np.log(us_data["CRUDE"] / us_data["CRUDE"].shift(1))
    us_features["DXY_Change"] = np.log(us_data["DXY"] / us_data["DXY"].shift(1))
    us_features["US_VIX_Change"] = np.log(us_data["US_VIX"] / us_data["US_VIX"].shift(1))

    # Shift by 1 day so Day t (India) uses Day t-1 US close -- prevents lookahead bias,
    # same as data_download.py.
    us_features_shifted = us_features.shift(1)
    us_features_shifted.to_csv(out_file, index_label="Date")
    print(f"  [saved] {out_file}  ({len(us_features_shifted)} rows)")
    return True


# =============================================================================
# 4. GIFT Nifty futures (investing.com) -- best-effort, likely needs manual fallback
# =============================================================================
def download_gift_nifty(cutoff_ts, out_file=GIFT_CSV):
    """Attempts to pull GIFT Nifty futures daily history from investing.com's
    historical-data API. investing.com actively blocks non-browser traffic
    (Cloudflare + session-token requirements), so this is a best-effort attempt
    only -- if it fails, download it manually from investing.com:
        https://www.investing.com/indices/cnx-nifty-futures-historical-data
    and save it as Gift_Nifty_50_Futures_Historical_Data.csv in the same format
    as the existing file (Date, Price, Open, High, Low, Vol., Change %)."""
    print("[GIFT Nifty] Attempting investing.com scrape (often blocked -- see fallback note) ...")
    try:
        import requests
    except ImportError:
        print("  [FAILED] 'requests' not installed. pip install requests, or download manually from investing.com.")
        return False

    boundary = last_complete_gift_date(cutoff_ts)
    end_str = boundary.strftime("%m/%d/%Y")
    start_str = (boundary - timedelta(days=730)).strftime("%m/%d/%Y")

    # investing.com's GIFT Nifty / SGX Nifty futures page ID changes over time and
    # requires a live browser session (cookies + CSRF header) to hit the historical-data
    # endpoint reliably -- this is a genuine best-effort attempt, expected to fail
    # in a headless/server context.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": "https://www.investing.com/indices/cnx-nifty-futures-historical-data",
    }
    payload = {
        "curr_id": "1177390",   # GIFT Nifty / SGX Nifty futures instrument id on investing.com
                                 # (subject to change -- verify against the page URL if this fails)
        "st_date": start_str,
        "end_date": end_str,
        "interval_sec": "Daily",
        "sort_col": "date",
        "sort_ord": "DESC",
        "action": "historical_data",
    }
    try:
        resp = requests.post("https://www.investing.com/instruments/HistoricalDataAjax",
                              data=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        if not tables:
            raise ValueError("No table found in response")
        gift = tables[0]
        gift.to_csv(out_file, index=False)
        print(f"  [saved] {out_file}  ({len(gift)} rows)")
        return True
    except Exception as e:
        print(f"  [FAILED] investing.com scrape didn't work ({type(e).__name__}: {e}).")
        print(f"  [ACTION NEEDED] Download GIFT Nifty history manually from investing.com and save as:")
        print(f"                  {out_file}")
        print(f"                  (Date, Price, Open, High, Low, Vol., Change % -- keep rows <= {boundary})")
        return False


# =============================================================================
# 5. Universe -- every major NSE/BSE index + every NIFTY 50 stock
# =============================================================================
def download_entity(label, ticker, cutoff_ts):
    """Downloads the one file an entity needs (daily OHLC+indicators workbook),
    using exactly the same function/design used for NIFTY 50 and India VIX
    above -- just parameterized by ticker/label."""
    stem = file_stem(label)
    ohlc_file = f"{stem}_OHLC_Direction_Indicators.xlsx"

    print(f"--- {label} ({ticker}) ---")
    ok_ohlc = download_daily_ohlc_indicators(ticker, ohlc_file, cutoff_ts, label=label)
    print()
    return {ohlc_file: ok_ohlc}


def download_universe(cutoff_ts, skip_indices=False, skip_stocks=False, only=None):
    """Loops over every NSE/BSE index (excluding NIFTY 50 itself, which is handled
    by its own original filename above) and every NIFTY 50 constituent stock,
    downloading the one OHLC+indicators file each needs. `only`, if given, is an
    iterable of labels to restrict the run to (case-sensitive, must match
    entities_universe labels exactly) -- handy for retrying just the entities
    that failed.

    A failure on any single entity (missing ticker, no data, yfinance hiccup)
    is caught by download_daily_ohlc_indicators itself (it prints [FAILED] and
    returns False) -- it does NOT stop the loop, so one bad ticker never blocks
    the rest of the universe."""
    results = {}

    if not skip_indices:
        print("\n=== NSE + BSE indices ===\n")
        for label, ticker in all_index_entities().items():
            if only and label not in only:
                continue
            try:
                results.update(download_entity(label, ticker, cutoff_ts))
            except Exception as e:
                print(f"  [FAILED] {label} ({ticker}): {type(e).__name__}: {e}\n")

    if not skip_stocks:
        print("\n=== NIFTY 50 constituent stocks ===\n")
        for label, ticker in all_stock_entities().items():
            if only and label not in only:
                continue
            try:
                results.update(download_entity(label, ticker, cutoff_ts))
            except Exception as e:
                print(f"  [FAILED] {label} ({ticker}): {type(e).__name__}: {e}\n")

    return results


# =============================================================================
# CLI entry point
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Download all datasets nifty_forecast.py needs (NIFTY 50 + shared "
                     "macro data, plus every major NSE/BSE index and NIFTY 50 stock), "
                     "trimmed to what was actually observable as of a given IST cutoff timestamp.")
    p.add_argument("cutoff", help="Cutoff date/time, IST. e.g. '2026-08-17 09:45' or "
                                   "'2026-08-17 9:45 AM' or just '2026-08-17' (defaults to 09:45).")
    p.add_argument("--skip-nifty", action="store_true")
    p.add_argument("--skip-vix", action="store_true")
    p.add_argument("--skip-us", action="store_true")
    p.add_argument("--skip-gift", action="store_true")
    p.add_argument("--skip-universe", action="store_true",
                    help="Skip ALL of the NSE/BSE indices and NIFTY 50 stocks below -- "
                         "just refresh the original 5 NIFTY 50 files.")
    p.add_argument("--skip-indices", action="store_true",
                    help="Skip the NSE/BSE indices (still downloads NIFTY 50 stocks).")
    p.add_argument("--skip-stocks", action="store_true",
                    help="Skip the NIFTY 50 stocks (still downloads NSE/BSE indices).")
    p.add_argument("--only", default=None,
                    help="Comma-separated list of entity labels to restrict the universe "
                         "download to, e.g. --only 'NIFTY BANK,Reliance Industries'.")
    return p.parse_args()


def main(cutoff=None, skip_nifty=False, skip_vix=False, skip_us=False,
         skip_gift=False, skip_universe=False,
         skip_indices=False, skip_stocks=False, only=None):
    if cutoff is None:
        args = parse_args()
        cutoff = args.cutoff
        skip_nifty, skip_vix = args.skip_nifty, args.skip_vix
        skip_us, skip_gift = args.skip_us, args.skip_gift
        skip_universe = args.skip_universe
        skip_indices, skip_stocks = args.skip_indices, args.skip_stocks
        only = [s.strip() for s in args.only.split(",")] if args.only else None

    cutoff_ts = parse_cutoff(cutoff)
    print(f"=== Downloading all datasets as of cutoff: {cutoff_ts} ===\n")

    results = {}

    if not skip_nifty:
        results[NIFTY_XLSX] = download_daily_ohlc_indicators(
            "^NSEI", NIFTY_XLSX, cutoff_ts, label="NIFTY 50")
        print()

    if not skip_vix:
        results[VIX_XLSX] = download_daily_ohlc_indicators(
            "^INDIAVIX", VIX_XLSX, cutoff_ts, label="India VIX")
        print()

    if not skip_us:
        results[US_CSV] = download_us_overnight_features(cutoff_ts, US_CSV)
        print()

    if not skip_gift:
        results[GIFT_CSV] = download_gift_nifty(cutoff_ts, GIFT_CSV)
        print()

    if not skip_universe:
        results.update(download_universe(cutoff_ts, skip_indices=skip_indices,
                                          skip_stocks=skip_stocks, only=only))

    # India VIX is a required input to nifty_forecast.py's feature set (VIX_Close,
    # VIX_change, VIX_level_norm) -- not optional the way most universe entities
    # are. If it wasn't successfully downloaded above (either --skip-vix was
    # passed, or the normal attempt failed), force one more attempt here so a
    # forgotten/failed flag doesn't silently leave the forecaster without it.
    if not results.get(VIX_XLSX):
        reason = "was skipped (--skip-vix)" if skip_vix else "failed above"
        print(f"[fallback] India VIX {reason} but is required by nifty_forecast.py -- retrying now ...")
        results[VIX_XLSX] = download_daily_ohlc_indicators(
            "^INDIAVIX", VIX_XLSX, cutoff_ts, label="India VIX")
        print()

    print("=== Summary ===")
    for fname, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {fname}")
    if not results.get(GIFT_CSV, True):
        print(f"\n  GIFT Nifty needs a manual download from investing.com -> save as {GIFT_CSV}")

    return results


if __name__ == "__main__":
    # NOTE: this now also refreshes every NSE/BSE index + NIFTY 50 stock in
    # entities_universe.py (in addition to the original 5 NIFTY 50 files).
    # Pass skip_universe=True (or --skip-universe on the CLI) to get the old,
    # NIFTY-50-only behavior back.
    main('2026-09-03')
    