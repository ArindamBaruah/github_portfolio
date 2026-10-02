"""
entities_universe.py
=====================
Single source of truth for every entity that download_all_data.py downloads
and nifty_forecast.py forecasts, beyond the original NIFTY 50 -only design.

Both scripts import from here so the ticker list and the filename convention
can never drift apart between "what got downloaded" and "what the forecaster
looks for".

Groups
------
  NIFTY_50_LABEL / NIFTY_50_TICKER  -- the original entity. Kept on its
      original, unprefixed filename (Nifty_OHLC_Direction_Indicators.xlsx)
      for backward compatibility -- it is NOT re-downloaded under a new
      name by the universe loop below.

  NSE_INDICES     -- major NSE benchmark + sectoral indices (NIFTY BANK, IT,
                      PHARMA, NEXT 50, MIDCAP 50, FIN SERVICE, NIFTY 500 ...)
  BSE_INDICES     -- major BSE indices (SENSEX, BSE 100/200/500, MIDCAP, SMALLCAP)
  GLOBAL_MACRO_INDICATORS -- non-NSE/BSE tickers with reliable yfinance history
                      that are well known to heavily move NIFTY 50 (USD/INR,
                      other Asian benchmarks that open before India, crude oil,
                      US long-term yields). See that dict's own comment for why
                      each one is included.
  NIFTY50_STOCKS  -- the 50 current NIFTY 50 constituents

  Removed: NIFTY AUTO/FMCG/METAL/REALTY/ENERGY/MEDIA/PSU BANK (^CNXAUTO,
  ^CNXFMCG, ^CNXMETAL, ^CNXREALTY, ^CNXENERGY, ^CNXMEDIA, ^CNXPSUBANK) used
  to be listed under NSE_INDICES, but yfinance's historical-data feed for
  these legacy tickers stopped returning current daily bars (each stalled
  around the same date regardless of cutoff), so they were dropped rather
  than silently feeding stale rows into the model. If yfinance's coverage
  for these ever catches back up, they can be re-added here.

  Also removed, for the same reason (no usable data as of the 2026-07-31
  verification pass -- these never produced a file at all, unlike the
  legacy ^CNX* tickers above which at least stalled on old data):
  NIFTY NEXT 50 (^NIFTYJR) and NIFTY FIN SERVICE (^CNXFINANCE) from
  NSE_INDICES, and BSE 100/200/500/MIDCAP/SMALLCAP (^BSE100, ^BSE200,
  ^BSE500, ^BSEMID, ^BSESML) from BSE_INDICES -- SENSEX (^BSESN) is the
  only BSE index left, since it's the only one yfinance actually served.

Filename convention
--------------------
file_stem(label) turns a human label into the stem used for that entity's
file, e.g. "NIFTY BANK" -> "NIFTY_BANK":
    NIFTY_BANK_OHLC_Direction_Indicators.xlsx
    NIFTY_BANK_trained_model.pkl
    NIFTY_BANK_LiveForecast_<date>.csv

IMPORTANT -- things that WILL need periodic upkeep
---------------------------------------------------
1. NSE reconstitutes the NIFTY 50 index every semi-annual review (cut-off
   dates Jan 31 / Jul 31), so NIFTY50_STOCKS below is a snapshot and can
   drift -- swap tickers in/out here if a constituent changes.
2. Not every NSE/BSE index below is guaranteed to have reliable daily AND
   5-minute intraday coverage on yfinance/Yahoo Finance -- sectoral/niche
   indices in particular are best-effort, exactly like the GIFT Nifty
   investing.com scrape already in download_all_data.py: if yfinance has
   no data (or only stale data) for a ticker, that one entity's download
   prints [FAILED] and the pipeline moves on to the next entity rather
   than crashing. If a ticker below turns out to be stale/wrong, just
   correct or remove it here (see the removed-tickers note above).
"""
import re

# ---------------------------------------------------------------------------
# The original entity (unchanged filenames, handled by the pre-existing code
# path in both scripts -- listed here only so the forecast script's "run
# every entity" loop can include it).
# ---------------------------------------------------------------------------
NIFTY_50_LABEL = "NIFTY 50"
NIFTY_50_TICKER = "^NSEI"

# ---------------------------------------------------------------------------
# Major NSE indices (label -> yfinance ticker)
# ---------------------------------------------------------------------------
NSE_INDICES = {
    "NIFTY BANK":        "^NSEBANK",
    "NIFTY MIDCAP 50":   "^NSEMDCP50",
    "NIFTY IT":          "^CNXIT",
    "NIFTY PHARMA":      "^CNXPHARMA",
    "NIFTY 500":         "^CRSLDX",
}

# ---------------------------------------------------------------------------
# Major BSE indices (label -> yfinance ticker)
# ---------------------------------------------------------------------------
BSE_INDICES = {
    "SENSEX":        "^BSESN",
}

# ---------------------------------------------------------------------------
# Global macro entities (label -> yfinance ticker)
# -----------------------------------------------------------------------
# Not NSE/BSE instruments, but each one is a well-known, heavily-cited driver
# of NIFTY 50's own next-session moves, and each has a reliable, actively
# updated yfinance history (unlike the legacy ^CNX* sectoral tickers removed
# above), so each gets forecast as its own entity just like every index/stock:
#   USD/INR       -- INR=X    -- FII/FPI flows into Indian equities are highly
#                                 sensitive to rupee strength/weakness.
#   Dow Jones     -- ^DJI     -- overnight US sentiment, alongside the S&P 500/
#                                 NASDAQ already folded in as NIFTY's own
#                                 pre-open features.
#   Nikkei 225    -- ^N225    -- first major Asian market to open each day;
#                                 sets the regional risk tone before NSE opens.
#   Hang Seng     -- ^HSI     -- ditto, Hong Kong/China-linked regional risk
#                                 appetite, also pre-opens NSE.
#   Crude Oil     -- CL=F     -- India is a large net oil importer, so crude
#                                 moves feed straight through to the currency,
#                                 inflation expectations, and equity sentiment.
#   US 10Y Yield  -- ^TNX     -- global risk-free rate; drives FII allocation
#                                 into/out of emerging markets like India.
# ---------------------------------------------------------------------------
GLOBAL_MACRO_INDICATORS = {
    "USD/INR":          "INR=X",
    "Dow Jones":        "^DJI",
    "Nikkei 225":       "^N225",
    "Hang Seng":        "^HSI",
    "Crude Oil (WTI)":  "CL=F",
    "US 10Y Yield":     "^TNX",
}

# ---------------------------------------------------------------------------
# NIFTY 50 constituent stocks (label -> yfinance ticker, .NS = NSE)
# Snapshot -- see the "upkeep" note in the module docstring.
# ---------------------------------------------------------------------------
NIFTY50_STOCKS = {
    "Reliance Industries":        "RELIANCE.NS",
    "Tata Consultancy Services":  "TCS.NS",
    "HDFC Bank":                  "HDFCBANK.NS",
    "Bharti Airtel":              "BHARTIARTL.NS",
    "ICICI Bank":                 "ICICIBANK.NS",
    "State Bank of India":        "SBIN.NS",
    "Infosys":                    "INFY.NS",
    "Larsen & Toubro":            "LT.NS",
    "Hindustan Unilever":         "HINDUNILVR.NS",
    "ITC":                        "ITC.NS",
    "Bajaj Finance":              "BAJFINANCE.NS",
    "HCL Technologies":           "HCLTECH.NS",
    "Sun Pharmaceutical":         "SUNPHARMA.NS",
    "Maruti Suzuki":              "MARUTI.NS",
    "Mahindra & Mahindra":        "M&M.NS",
    "Kotak Mahindra Bank":        "KOTAKBANK.NS",
    "Axis Bank":                  "AXISBANK.NS",
    "UltraTech Cement":           "ULTRACEMCO.NS",
    "NTPC":                       "NTPC.NS",
    "Titan Company":              "TITAN.NS",
    "Bajaj Finserv":              "BAJAJFINSV.NS",
    "Adani Enterprises":          "ADANIENT.NS",
    "Power Grid Corporation":     "POWERGRID.NS",
    "Oil & Natural Gas Corp":     "ONGC.NS",
    "Nestle India":               "NESTLEIND.NS",
    "Wipro":                      "WIPRO.NS",
    "JSW Steel":                  "JSWSTEEL.NS",
    "Tata Steel":                 "TATASTEEL.NS",
    "Asian Paints":               "ASIANPAINT.NS",
    "Coal India":                 "COALINDIA.NS",
    "Bajaj Auto":                 "BAJAJ-AUTO.NS",
    "Grasim Industries":          "GRASIM.NS",
    "Tech Mahindra":              "TECHM.NS",
    "HDFC Life Insurance":        "HDFCLIFE.NS",
    "Adani Ports & SEZ":          "ADANIPORTS.NS",
    "SBI Life Insurance":         "SBILIFE.NS",
    "Cipla":                      "CIPLA.NS",
    "Dr. Reddy's Laboratories":   "DRREDDY.NS",
    "Eicher Motors":              "EICHERMOT.NS",
    "Apollo Hospitals":           "APOLLOHOSP.NS",
    "Bharat Electronics":         "BEL.NS",
    "Tata Consumer Products":     "TATACONSUM.NS",
    "Hindalco Industries":        "HINDALCO.NS",
    "Trent":                      "TRENT.NS",
    "Shriram Finance":            "SHRIRAMFIN.NS",
    "Jio Financial Services":     "JIOFIN.NS",
    "Eternal (Zomato)":           "ETERNAL.NS",
    "Tata Motors (Passenger Vehicles)": "TMPV.NS", #post-demerger, depending on data provider
    "IndiGo":                     "INDIGO.NS",
    "Max Healthcare":             "MAXHEALTH.NS",
}


def file_stem(label):
    """'NIFTY BANK' -> 'NIFTY_BANK'; 'Dr. Reddy's Laboratories' -> 'DR_REDDYS_LABORATORIES'."""
    stem = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").upper()
    return stem


def all_index_entities():
    """{label: ticker} for every NSE + BSE index plus the global-macro entities,
    EXCLUDING NIFTY 50 itself (which keeps its original, separately-managed
    filenames)."""
    merged = {}
    merged.update(NSE_INDICES)
    merged.update(BSE_INDICES)
    merged.update(GLOBAL_MACRO_INDICATORS)
    return merged


def all_stock_entities():
    """{label: ticker} for every NIFTY 50 constituent stock."""
    return dict(NIFTY50_STOCKS)


def all_forecast_entities():
    """{label: ticker} for every entity nifty_forecast.py should forecast,
    INCLUDING NIFTY 50 itself (mapped to its original filenames)."""
    merged = {NIFTY_50_LABEL: NIFTY_50_TICKER}
    merged.update(NSE_INDICES)
    merged.update(BSE_INDICES)
    merged.update(GLOBAL_MACRO_INDICATORS)
    merged.update(NIFTY50_STOCKS)
    return merged