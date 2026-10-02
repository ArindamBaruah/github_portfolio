import sys
import numpy as np
import pandas as pd
 
try:
    import yfinance as yf
except ImportError:
    sys.exit("Missing dependency. Run: pip install yfinance pandas numpy openpyxl")
 
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.chart import LineChart, Reference
 
TICKER = "^INDIAVIX"          # NIFTY 50 index on Yahoo Finance
PERIOD = "5y"              # last 1 year
OUTPUT_FILE = "INDIAVIX_OHLC_Direction_Indicators.xlsx"
 
 
# --------------------------------------------------------------------------
# 1. DOWNLOAD DATA
# --------------------------------------------------------------------------
def download_ohlc(ticker: str, period: str) -> pd.DataFrame:
    print(f"Downloading {period} of daily data for {ticker} ...")
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        sys.exit("No data downloaded. Check ticker symbol or your internet connection.")
 
    # yfinance sometimes returns a MultiIndex column structure -> flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
 
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    print(f"Downloaded {len(df)} daily candles: {df['Date'].iloc[0]} -> {df['Date'].iloc[-1]}")
    return df
 
 
# --------------------------------------------------------------------------
# 2. INDICATOR CALCULATIONS
# --------------------------------------------------------------------------
def add_sma_crossover(df: pd.DataFrame) -> pd.DataFrame:
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA_Direction"] = np.where(df["SMA20"] > df["SMA50"], "Bullish",
                             np.where(df["SMA20"] < df["SMA50"], "Bearish", "Neutral"))
    return df
 
 
def add_ema_crossover(df: pd.DataFrame) -> pd.DataFrame:
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_Direction"] = np.where(df["EMA20"] > df["EMA50"], "Bullish",
                             np.where(df["EMA20"] < df["EMA50"], "Bearish", "Neutral"))
    return df
 
 
def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df["MACD_Direction"] = np.where(df["MACD"] > df["MACD_Signal"], "Bullish",
                              np.where(df["MACD"] < df["MACD_Signal"], "Bearish", "Neutral"))
    return df
 
 
def add_rsi(df: pd.DataFrame, period=14) -> pd.DataFrame:
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
 
 
def add_adx(df: pd.DataFrame, period=14) -> pd.DataFrame:
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
    # Direction only counted when trend is strong enough (ADX > 20)
    df["ADX_Direction"] = np.where(
        df["ADX14"] < 20, "No Trend",
        np.where(df["Plus_DI"] > df["Minus_DI"], "Bullish", "Bearish")
    )
    return df
 
 
def build_summary(df: pd.DataFrame) -> pd.DataFrame:
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
 
 
# --------------------------------------------------------------------------
# 3. EXCEL WRITER WITH FORMATTING
# --------------------------------------------------------------------------
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
    for i, col in enumerate(df.columns, start=1):
        max_len = max(df[col].astype(str).map(len).max(), len(str(col))) + 3
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
 
 
def write_excel(ohlc_df, indicators_df, summary_df, filename):
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        ohlc_df.to_excel(writer, sheet_name="OHLC Data", index=False)
        indicators_df.to_excel(writer, sheet_name="Indicators", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
 
        wb = writer.book
 
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
 
        # Add a Close price line chart on the OHLC sheet
        chart = LineChart()
        chart.title = "INDIAVIX Close Price (1 Year)"
        chart.y_axis.title = "Price"
        chart.x_axis.title = "Date"
        close_col_idx = ohlc_df.columns.get_loc("Close") + 1
        data_ref = Reference(ws1, min_col=close_col_idx, min_row=1, max_row=len(ohlc_df) + 1)
        chart.add_data(data_ref, titles_from_data=True)
        chart.width = 24
        chart.height = 10
        ws1.add_chart(chart, f"H2")
 
    print(f"Saved workbook: {filename}")
 
 
# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    ohlc_df = download_ohlc(TICKER, PERIOD)
 
    indicators_df = ohlc_df.copy()
    indicators_df = add_sma_crossover(indicators_df)
    indicators_df = add_ema_crossover(indicators_df)
    indicators_df = add_macd(indicators_df)
    indicators_df = add_rsi(indicators_df)
    indicators_df = add_adx(indicators_df)
 
    # round numeric columns for readability
    num_cols = indicators_df.select_dtypes(include=[np.number]).columns
    indicators_df[num_cols] = indicators_df[num_cols].round(2)
 
    summary_df = build_summary(indicators_df)
 
    write_excel(ohlc_df, indicators_df, summary_df, OUTPUT_FILE)
    
    # 1. Fetch US Indices & Overnight Drivers
    tickers = {
        'SP500': '^GSPC',
        'NASDAQ': '^NDX',
        'CRUDE': 'BZ=F',
        'DXY': 'DX-Y.NYB',
        'US_VIX': '^VIX'
    }

    # Download past 5 years of daily data
    us_data = yf.download(list(tickers.values()), period="5y", interval="1d")['Close']
    us_data.columns = [k for k in tickers.keys()]

    # Forward fill missing dates (handles non-overlapping holidays between US and India)
    us_data = us_data.ffill()

    # 2. Engineer Overnight Return Features
    us_features = pd.DataFrame(index=us_data.index)

    # US Index intraday/overnight log returns
    us_features['SP500_Return'] = np.log(us_data['SP500'] / us_data['SP500'].shift(1))
    us_features['NASDAQ_Return'] = np.log(us_data['NASDAQ'] / us_data['NASDAQ'].shift(1))
    us_features['Crude_Change'] = np.log(us_data['CRUDE'] / us_data['CRUDE'].shift(1))
    us_features['DXY_Change'] = np.log(us_data['DXY'] / us_data['DXY'].shift(1))
    us_features['US_VIX_Change'] = np.log(us_data['US_VIX'] / us_data['US_VIX'].shift(1))

    # 3. Merging with Nifty Data
    # Shift by 1 day so Day t uses Day t-1 US close to prevent lookahead bias
    us_features_shifted = us_features.shift(1)

    # Export to CSV
    us_features_shifted.to_csv("us_overnight_market_features.csv", index_label="Date")
 
 
if __name__ == "__main__":
    main()
    
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd
import yfinance as yf

# 1. Fetch 5-minute data for NIFTY 50 (^NSEI) for the last 60 days
print("Downloading 60-day 5-minute intraday data for ^NSEI...")
df = yf.download(
    tickers="^NSEI", period="60d", interval="5m", progress=False
)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Convert timezone to IST (Asia/Kolkata)
if df.index.tz is None:
    df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
else:
    df.index = df.index.tz_convert("Asia/Kolkata")

# Filter for full market session hours (09:15 AM to 03:30 PM IST)
df = df.between_time("09:15", "15:30").dropna(subset=["Open", "High", "Low", "Close"])

# Reset index so Datetime becomes a regular column
df = df.reset_index().rename(columns={df.columns[0]: "Datetime"})

# 2. Prepare full-day 5-minute records
records = []
for _, row in df.iterrows():
    dt = row["Datetime"]
    records.append({
        "Datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "Date": dt.strftime("%Y-%m-%d"),
        "Time": dt.strftime("%H:%M"),
        "Open": round(float(row["Open"]), 2),
        "High": round(float(row["High"]), 2),
        "Low": round(float(row["Low"]), 2),
        "Close": round(float(row["Close"]), 2),
        "Volume": int(row["Volume"]) if "Volume" in row and pd.notna(row["Volume"]) else 0,
    })

# Save clean CSV copy (useful for strategy backtesting)
csv_filename = "nifty_5min_60days.csv"
pd.DataFrame(records)[["Datetime", "Open", "High", "Low", "Close", "Volume"]].to_csv(csv_filename, index=False)
print(f"CSV saved as: {csv_filename}")

# 3. Build Excel Workbook with openpyxl styling
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "NIFTY_5Min_60Days"
ws.views.sheetView[0].showGridLines = True

# Title banner
ws.merge_cells("A1:H1")
ws["A1"] = "NIFTY 50 — Full Day 5-Minute Intraday Data (60 Days)"
ws["A1"].font = Font(name="Arial", size=14, bold=True, color="FFFFFF")
ws["A1"].fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
ws.row_dimensions[1].height = 35

# Table Headers
headers = ["Datetime", "Date", "Time", "Open", "High", "Low", "Close", "Volume"]
header_row = 3

for col_idx, text in enumerate(headers, 1):
    cell = ws.cell(row=header_row, column=col_idx, value=text)
    cell.font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
    cell.fill = PatternFill(start_color="2F5597", end_color="2F5597", fill_type="solid")
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

ws.row_dimensions[header_row].height = 25

# Populate Data Rows
start_row = 4
thin_border = Border(
    left=Side(style="thin", color="D9D9D9"),
    right=Side(style="thin", color="D9D9D9"),
    top=Side(style="thin", color="D9D9D9"),
    bottom=Side(style="thin", color="D9D9D9"),
)

for r_idx, rec in enumerate(records, start=start_row):
    row_vals = list(rec.values())

    for c_idx, val in enumerate(row_vals, 1):
        cell = ws.cell(row=r_idx, column=c_idx, value=val)
        cell.font = Font(name="Arial", size=10)
        cell.border = thin_border

        if c_idx in [1, 2, 3]:  # Datetime, Date, Time
            cell.alignment = Alignment(horizontal="center", vertical="center")
        elif c_idx in [4, 5, 6, 7]:  # OHLC
            cell.number_format = "#,##0.00"
            cell.alignment = Alignment(horizontal="right", vertical="center")
        elif c_idx == 8:  # Volume
            cell.number_format = "#,##0"
            cell.alignment = Alignment(horizontal="right", vertical="center")

# Auto-adjust column widths
for col in ws.columns:
    max_len = max(len(str(cell.value or '')) for cell in col)
    col_letter = get_column_letter(col[0].column)
    ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

# Freeze panes below header
ws.freeze_panes = ws.cell(row=header_row + 1, column=1)

# Save Excel file
excel_filename = "nifty_5min_full_day_60d.xlsx"
wb.save(excel_filename)
print(f"Excel report saved successfully as: {excel_filename}")