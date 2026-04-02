"""
prepare.py - Scorpion AutoResearch V1 data preparation.
IMMUTABLE. Agent never touches this file.

Three data sources:
  1. Polygon API     - all price data (OHLCV, intraday bars)
  2. yfinance        - market cap, float, shares, sector, news
  3. SEC EDGAR       - filings (8-K, S-3, 424B) - free public API

No cache. Every run fetches fresh. No pickle files.
No lookahead bias. Features only use data available at entry time.
No synthetic data. Missing = log and skip.
"""

import os, json, time, math, signal
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import pytz
from datetime import datetime, timedelta
from pathlib import Path

# ══════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════

POLYGON_KEY    = os.environ["POLYGON_API_KEY"]
POLYGON_BASE   = "https://api.polygon.io"
EDGAR_BASE     = "https://efts.sec.gov/LATEST/search-index"
ET             = pytz.timezone("America/New_York")
RAW_DIR        = Path("./raw_cache")
LOG_FILE       = RAW_DIR / "validation_log.txt"

# Universe filters
MIN_GAP_PCT    = 0.20
MIN_PRICE      = 1.00
MAX_PRICE      = 50.00
MIN_VOLUME     = 500_000
MIN_DOLLAR_VOL = 500_000
MAX_MKTCAP     = 1_000_000_000

# Win/loss simulation
WIN_TARGET     = 0.15
STOP_LOSS      = 0.10
COMMISSION     = 0.01
MAX_ENTRY_MIN  = 270   # no entries after 2pm ET

# Score gates - ALL must pass or score = 0.0
MIN_WR         = 0.65
MIN_AVG_RET    = 0.04
MIN_N_MONTH    = 15
MIN_N_SCORE    = 30
OOS_DECAY_MAX  = 0.12

# Bucketing
MIN_BUCKET_N   = 30

# Data windows
TRAIN_MONTHS   = 12
VAL_MONTHS     = 3
BUFFER_DAYS    = 30


# ══════════════════════════════════════════════════════════
# SECTION 1 - HELPERS
# ══════════════════════════════════════════════════════════

def _log(msg):
    RAW_DIR.mkdir(exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().isoformat()} {msg}\n")


def _polygon(endpoint, params, retries=3):
    """Polygon API call with retry. Returns {} on failure."""
    params = {**params, "apiKey": POLYGON_KEY}
    for i in range(retries):
        try:
            r = requests.get(f"{POLYGON_BASE}{endpoint}",
                             params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(60); continue
            if r.status_code >= 500:
                time.sleep(5 * (i + 1)); continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            _log(f"Polygon error {endpoint}: {e}")
            time.sleep(5)
    return {}


def _trading_days_before(date_str, n):
    """Return list of n weekday dates before date_str."""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    days = []
    while len(days) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            days.append(d.strftime("%Y-%m-%d"))
    return days


def _yf_info(ticker):
    """
    Fetch fundamentals from yfinance.
    Returns dict with marketCap, floatShares, sharesOutstanding,
    sector, industry. Returns {} on failure.
    """
    try:
        info = yf.Ticker(ticker, session=None).fast_info
        return {
            "market_cap":         getattr(info, "market_cap", 0) or 0,
            "float_shares":       getattr(info, "float_shares", 0) or 0,
            "shares_outstanding": getattr(info, "shares", 0) or 0,
            "sector":             "",
            "industry":           "",
        }
    except Exception as e:
        _log(f"yfinance error {ticker}: {e}")
        return {}


def _yf_news(ticker, date_str):
    """
    Fetch recent news headlines from yfinance.
    Returns list of title strings published before market open on date_str.
    Note: yfinance news is recent-only, not deeply historical.
    For older dates this may return empty - that is correct behavior,
    log as NOT_FOUND and continue.
    """
    try:
        news_items = yf.Ticker(ticker).news or []
        gap_ts = datetime.strptime(date_str, "%Y-%m-%d").timestamp() + 9.5 * 3600
        headlines = []
        for item in news_items:
            pub = item.get("providerPublishTime", 0)
            if pub and pub < gap_ts:
                title = item.get("title", "")
                if title:
                    headlines.append(title)
        return headlines
    except Exception as e:
        _log(f"yfinance news error {ticker} {date_str}: {e}")
        return []


def _edgar_filings(ticker, date_str):
    """
    Fetch SEC filings from EDGAR free public API.
    Returns list of form_type strings filed in the 5 trading days
    before and including date_str.

    Uses EDGAR full-text search endpoint - no API key needed.
    Rate limit: ~10 requests/second. Add sleep between calls.

    SEC EDGAR filing date alignment:
      Only includes filings where filing_date <= gap_date.
      This ensures we only use information available at market open.
    """
    prev5 = _trading_days_before(date_str, 5)[-1]
    try:
        resp = requests.get(
            EDGAR_BASE,
            params={
                "q":          f'"{ticker}"',
                "forms":      "8-K,S-3,S-1,424B3,424B4,SC 13D,SC 13G",
                "dateRange":  "custom",
                "startdt":    prev5,
                "enddt":      date_str,
            },
            timeout=15,
            headers={"User-Agent": "Scorpion Research admin@scorpion.ai"}
        )
        resp.raise_for_status()
        data  = resp.json()
        hits  = data.get("hits", {}).get("hits", [])
        forms = []
        for hit in hits:
            src = hit.get("_source", {})
            ft  = src.get("form_type", "")
            fd  = src.get("file_date", "")
            # Only include if filed on or before gap date
            if ft and fd and fd <= date_str:
                forms.append(ft.upper())
        return forms
    except Exception as e:
        _log(f"EDGAR error {ticker} {date_str}: {e}")
        return []


def _avg_volume(ticker, date_str, days=20):
    """20-day avg daily volume before date_str using Polygon."""
    prev = _trading_days_before(date_str, days + 5)[days]
    data = _polygon(
        f"/v2/aggs/ticker/{ticker}/range/1/day/{prev}/{date_str}",
        {"adjusted": "false", "sort": "asc", "limit": 50}
    )
    results = data.get("results", [])
    if not results:
        return 0
    return float(np.mean([r["v"] for r in results[-days:]]))


# ══════════════════════════════════════════════════════════
# SECTION 2 - UNIVERSE FETCH (Polygon)
# ══════════════════════════════════════════════════════════

def fetch_universe(start_date, end_date):
    """
    Walk every trading day. Fetch grouped daily bars from Polygon.
    Filter to gap >= 20%, price $1-$50, volume >= 500k.
    Returns DataFrame: ticker, date, open, high, low, close, volume,
      prev_close, gap_pct.
    NO CACHE. Always fresh.
    """
    RAW_DIR.mkdir(exist_ok=True)
    rows = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end     = datetime.strptime(end_date,   "%Y-%m-%d")

    while current <= end:
        if current.weekday() >= 5:
            current += timedelta(days=1)
            continue

        date_str = current.strftime("%Y-%m-%d")
        data     = _polygon(
            f"/v2/aggs/grouped/locale/us/market/stocks/{date_str}",
            {"adjusted": "false", "include_otc": "false"}
        )
        today_results = data.get("results", [])
        if not today_results:
            _log(f"Empty grouped daily: {date_str}")
            current += timedelta(days=1)
            time.sleep(0.12)
            continue

        # Prev close map
        prev_day  = _trading_days_before(date_str, 1)[0]
        prev_data = _polygon(
            f"/v2/aggs/grouped/locale/us/market/stocks/{prev_day}",
            {"adjusted": "false"}
        )
        prev_map = {r["T"]: r["c"] for r in prev_data.get("results", [])}
        time.sleep(0.12)

        for r in today_results:
            ticker = r.get("T", "")
            if not ticker:
                continue
            o, h, l, c, v = r.get("o",0), r.get("h",0), r.get("l",0), r.get("c",0), r.get("v",0)
            prev_close = prev_map.get(ticker, 0)
            if not prev_close or prev_close <= 0:
                continue
            gap_pct = (o - prev_close) / prev_close
            if gap_pct < MIN_GAP_PCT: continue
            if o < MIN_PRICE or o > MAX_PRICE: continue
            if v < MIN_VOLUME: continue
            if o * v < MIN_DOLLAR_VOL: continue
            rows.append({
                "ticker": ticker, "date": date_str,
                "open": o, "high": h, "low": l, "close": c,
                "volume": v, "prev_close": prev_close, "gap_pct": gap_pct,
            })

        current += timedelta(days=1)
        time.sleep(0.12)

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    _log(f"fetch_universe: {len(df)} events {start_date}->{end_date}")
    return df


# ══════════════════════════════════════════════════════════
# SECTION 3 - FUNDAMENTALS (yfinance - free)
# ══════════════════════════════════════════════════════════

def enrich_fundamentals(df):
    """
    For each unique ticker, fetch market_cap, float_shares,
    shares_outstanding, sector, industry via yfinance.
    Filters out tickers where market_cap > MAX_MKTCAP.
    Sleep 0.5s between yfinance calls to avoid rate limiting.
    """
    df = df.copy()
    df["market_cap"]          = 0.0
    df["float_shares"]        = 0.0
    df["shares_outstanding"]  = 0.0
    df["sector"]              = ""
    df["industry"]            = ""

    for ticker in df["ticker"].unique():
        info = _yf_info(ticker)
        if not info:
            continue
        mask = df["ticker"] == ticker
        df.loc[mask, "market_cap"]         = info["market_cap"]
        df.loc[mask, "float_shares"]       = info["float_shares"]
        df.loc[mask, "shares_outstanding"] = info["shares_outstanding"]
        df.loc[mask, "sector"]             = info["sector"]
        df.loc[mask, "industry"]           = info["industry"]
        time.sleep(0.5)   # yfinance rate limit

    df = df[df["market_cap"] > 0].copy()
    df = df[df["market_cap"] <= MAX_MKTCAP].copy()
    return df


# ══════════════════════════════════════════════════════════
# SECTION 4 - NEWS + SEC FILINGS
# ══════════════════════════════════════════════════════════

def enrich_news_and_filings(df):
    """
    For each (ticker, date):
    - news_headlines: yfinance news titles published before 9:30am ET on gap day
    - filing_types: SEC EDGAR forms filed in 5 trading days before gap day
    - has_8k, has_s3, has_424b: derived booleans

    SEC filing date rule: only filings where filing_date <= gap_date.
    This ensures no lookahead - we only use what was public at open.
    Sleep 0.5s between yfinance, 0.2s between EDGAR calls.
    """
    df = df.copy()
    df["news_headlines"] = [[] for _ in range(len(df))]
    df["filing_types"]   = [[] for _ in range(len(df))]
    df["has_8k"]         = False
    df["has_s3"]         = False
    df["has_424b"]       = False

    for idx, row in df.iterrows():
        ticker   = row["ticker"]
        date_str = row["date"]

        # News via yfinance (free)
        headlines = _yf_news(ticker, date_str)
        df.at[idx, "news_headlines"] = headlines
        time.sleep(0.5)

        # SEC filings via EDGAR (free)
        forms = _edgar_filings(ticker, date_str)
        df.at[idx, "filing_types"] = forms
        df.at[idx, "has_8k"]  = any("8-K"  in f for f in forms)
        df.at[idx, "has_s3"]  = any("S-3"  in f or "S-1" in f for f in forms)
        df.at[idx, "has_424b"]= any("424B" in f for f in forms)
        time.sleep(0.2)

    return df


# ══════════════════════════════════════════════════════════
# SECTION 5 - CATALYST CLASSIFICATION
# ══════════════════════════════════════════════════════════

def classify_catalyst(headlines, form_types):
    """
    Returns one of exactly 7 strings.
    NOT_FOUND = no news found AND no relevant filing found.
    Priority: EARNINGS > FDA > ACQUISITION > SPAC_UPLIST > DILUTION > FLUFF_PR > NOT_FOUND
    """
    text = " ".join(headlines).lower()
    ft   = " ".join(form_types).upper()

    # DILUTION - SEC filing is definitive
    if any(x in ft for x in ["424B", "S-3", "S-1"]):
        return "DILUTION"

    earn_kw = ["earn","eps","revenue","profit","loss","guidance",
               "results","quarter","annual","beat","miss","raised"]
    if any(kw in text for kw in earn_kw):
        return "EARNINGS"

    fda_kw  = ["fda","approval","approved","pdufa","clinical","trial",
               "phase","drug","biologic","nda","bla"]
    if any(kw in text for kw in fda_kw):
        return "FDA"

    acq_kw  = ["acqui","merger","buyout","takeover","definitive agreement"]
    if any(kw in text for kw in acq_kw):
        return "ACQUISITION"

    spac_kw = ["spac","uplisting","reverse merger","business combination",
               "nasdaq listing","nyse listing"]
    if any(kw in text for kw in spac_kw):
        return "SPAC_UPLIST"

    fluff_kw= ["partner","collaborat"," ai ","blockchain","nft","celebrity",
               "influencer","letter of intent","mou ","memorandum","pilot"]
    if any(kw in text for kw in fluff_kw):
        return "FLUFF_PR"

    if headlines:
        return "FLUFF_PR"   # has news but nothing classified

    return "NOT_FOUND"      # zero news AND zero relevant filings


# ══════════════════════════════════════════════════════════
# SECTION 6 - INTRADAY BARS (Polygon, 3 timeframes, no cache)
# ══════════════════════════════════════════════════════════

def fetch_bars(df):
    """
    Fetch 5m, 15m, 30m bars from Polygon for each (ticker, date).
    Market hours 09:30-15:59:59 ET only.
    NO CACHE. Always fresh.
    Returns: {(ticker, date): {"5m": [...], "15m": [...], "30m": [...]}}
    Each bar: {t, o, h, l, c, v, t_min}
    t_min = minutes from 9:30am open (9:30 = 0, 10:00 = 30, etc.)
    """
    result = {}

    for _, row in df.iterrows():
        ticker   = row["ticker"]
        date_str = row["date"]
        key      = (ticker, date_str)
        result[key] = {"5m": [], "15m": [], "30m": []}

        for tf in ["5", "15", "30"]:
            data = _polygon(
                f"/v2/aggs/ticker/{ticker}/range/{tf}/minute/{date_str}/{date_str}",
                {"adjusted": "false", "sort": "asc", "limit": 1000}
            )
            valid = []
            for b in data.get("results", []):
                t_et = datetime.fromtimestamp(b["t"] / 1000, tz=ET)
                # Market hours filter
                total_min = t_et.hour * 60 + t_et.minute
                if total_min < 9 * 60 + 30: continue
                if total_min >= 16 * 60:     continue
                if b.get("h", 0) < b.get("l", 0): continue
                if b.get("v", -1) < 0:             continue
                t_min = total_min - (9 * 60 + 30)
                valid.append({
                    "t": b["t"], "o": b["o"], "h": b["h"],
                    "l": b["l"], "c": b["c"], "v": b["v"],
                    "t_min": t_min
                })
            result[key][f"{tf}m"] = valid
            time.sleep(0.12)

    return result


# ══════════════════════════════════════════════════════════
# SECTION 7 - FEATURE COMPUTATION (lookahead-safe)
# ══════════════════════════════════════════════════════════
#
# NAMING RULE:
#   f{T}_name = feature computed using ONLY bars up to T minutes from open.
#   T=0  = 9:30am open (gap_pct, fundamentals, filings, news available)
#   T=30 = data through 10:00am
#   T=60 = data through 10:30am
#   T=90 = data through 11:00am
#   T=120= data through 11:30am
#   T=150= data through 12:00pm
#   Features with no T prefix = available at 9:30am open only.

def _running_vwap(bars, up_to_min):
    cum_pv, cum_v = 0.0, 0.0
    for b in bars:
        if b["t_min"] > up_to_min: break
        mid = (b["h"] + b["l"] + b["c"]) / 3
        cum_pv += mid * b["v"]
        cum_v  += b["v"]
    return cum_pv / cum_v if cum_v > 0 else 0.0


def _mini_rsi(closes, period=14):
    if len(closes) < period + 1: return 50.0
    d  = np.diff(closes)
    ag = np.mean(np.where(d > 0, d, 0)[-period:])
    al = np.mean(np.where(d < 0, -d, 0)[-period:])
    return 100 - (100 / (1 + ag / al)) if al > 0 else 100.0


def _bars_up_to(bars, up_to_min):
    return [b for b in bars if b["t_min"] <= up_to_min]


def compute_all_features(df, bars_dict):
    """
    Compute all features. Lookahead-safe - see naming convention.
    Wrap each row in try/except. Log errors, skip row, never crash.
    """
    df = df.copy()

    # Pre-compute avg volumes (one call per unique ticker, before row loop)
    avg_vol_cache = {}
    for ticker in df["ticker"].unique():
        first_date = df[df["ticker"] == ticker]["date"].min()
        avg_vol_cache[ticker] = _avg_volume(ticker, first_date, 20)

    # All feature column names initialized to NaN
    feature_names = [
        "catalyst_type","news_count","has_8k","has_s3","has_424b",
        "rvol_full_day",
        # T=5
        "f5_first_bar_dir","f5_first_bar_range","f5_open_return",
        # T=30
        "f30_rvol_vs_first5m","f30_vol_pct_30m","f30_vwap_dist",
        "f30_price_vs_open","f30_high_vs_open","f30_or_range",
        "f30_bars_green","f30_vol_slope",
        # T=60
        "f60_rvol_vs_first5m","f60_rvol_vs_first15m","f60_rvol_vs_first30m",
        "f60_rvol_vs_first15m_15m","f60_vol_pct_60m","f60_vwap_dist",
        "f60_vwap_dist_15m","f60_price_vs_open","f60_vol_decay_last3",
        "f60_vol_slope_last3","f60_lower_high","f60_rsi14","f60_or_60m",
        # T=90
        "f90_rvol_vs_first5m","f90_rvol_vs_first30m","f90_rvol_vs_first30m_30m",
        "f90_vwap_dist","f90_vwap_dist_15m","f90_vol_decay_last5",
        "f90_vol_slope_last5","f90_lower_high","f90_lower_high_lower_vol",
        "f90_rsi14","f90_consec_red","f90_third_30m_vs_first",
        # T=120
        "f120_rvol_vs_first5m","f120_vwap_dist","f120_vwap_dist_15m",
        "f120_vol_decay_last5","f120_lower_high","f120_rsi14",
        # T=150
        "f150_rvol_vs_first5m","f150_vwap_dist","f150_vol_decay","f150_rsi14",
        # Opening ranges
        "or_5m","or_15m","or_30m",
    ]
    df["catalyst_type"] = ""
    for col in feature_names:
        if col != "catalyst_type":
            df[col] = np.nan

    for idx, row in df.iterrows():
        try:
            ticker   = row["ticker"]
            date_str = row["date"]
            key      = (ticker, date_str)
            b5       = bars_dict.get(key, {}).get("5m",  [])
            b15      = bars_dict.get(key, {}).get("15m", [])
            b30      = bars_dict.get(key, {}).get("30m", [])
            if not b5:
                _log(f"No 5m bars: {ticker} {date_str}")
                continue

            op  = row["open"]
            avg = avg_vol_cache.get(ticker, 0)

            # -- Available at 9:30am ----------------------------------
            cat = classify_catalyst(row["news_headlines"], row["filing_types"])
            df.at[idx, "catalyst_type"] = cat
            df.at[idx, "news_count"]    = float(len(row["news_headlines"]))
            df.at[idx, "has_8k"]        = float(row["has_8k"])
            df.at[idx, "has_s3"]        = float(row["has_s3"])
            df.at[idx, "has_424b"]      = float(row["has_424b"])
            day_vol                     = sum(b["v"] for b in b5) or 1
            df.at[idx, "rvol_full_day"] = row["volume"] / avg if avg > 0 else 0

            # -- T=5 --------------------------------------------------
            if b5:
                b0 = b5[0]
                df.at[idx, "f5_first_bar_dir"]   = 1.0 if b0["c"] > b0["o"] else -1.0
                df.at[idx, "f5_first_bar_range"]  = (b0["h"] - b0["l"]) / op if op > 0 else 0
                df.at[idx, "f5_open_return"]      = (b0["c"] - op) / op if op > 0 else 0

            # Opening ranges
            for T, col in [(5,"or_5m"),(15,"or_15m"),(30,"or_30m")]:
                s = _bars_up_to(b5, T)
                if s:
                    df.at[idx, col] = (max(b["h"] for b in s) - min(b["l"] for b in s)) / op if op > 0 else 0

            # -- T=30 -------------------------------------------------
            s30 = _bars_up_to(b5, 30)
            if s30:
                v30  = [b["v"] for b in s30]
                fv   = v30[0] if v30 else 1
                vw30 = _running_vwap(b5, 30)
                p30  = s30[-1]["c"]
                h30  = max(b["h"] for b in s30)
                df.at[idx, "f30_rvol_vs_first5m"] = (np.mean(v30[1:]) / fv) if len(v30) > 1 and fv > 0 else 1.0
                df.at[idx, "f30_vol_pct_30m"]     = sum(v30) / day_vol
                df.at[idx, "f30_vwap_dist"]       = (p30 - vw30) / vw30 if vw30 > 0 else 0
                df.at[idx, "f30_price_vs_open"]   = (p30 - op) / op if op > 0 else 0
                df.at[idx, "f30_high_vs_open"]    = (h30 - op) / op if op > 0 else 0
                df.at[idx, "f30_or_range"]        = (h30 - min(b["l"] for b in s30)) / op if op > 0 else 0
                df.at[idx, "f30_bars_green"]      = float(sum(1 for b in s30 if b["c"] > b["o"]))
                if len(v30) >= 2:
                    sl = np.polyfit(range(len(v30)), v30, 1)[0]
                    df.at[idx, "f30_vol_slope"]   = sl / (np.mean(v30) or 1)

            # -- T=60 -------------------------------------------------
            s60_5  = _bars_up_to(b5,  60)
            s60_15 = _bars_up_to(b15, 60)
            if s60_5 and s30:
                v60  = [b["v"] for b in s60_5]
                fv5  = v60[0]  if v60  else 1
                v15  = sum(b["v"] for b in _bars_up_to(b5, 15)) or 1
                v30s = sum(b["v"] for b in s30) or 1
                vw60 = _running_vwap(b5, 60)
                p60  = s60_5[-1]["c"]
                c60  = [b["c"] for b in s60_5]
                l3v  = v60[-3:] if len(v60) >= 3 else v60

                df.at[idx, "f60_rvol_vs_first5m"]  = (np.mean(v60[1:])  / fv5)          if len(v60) > 1 and fv5 > 0 else 1.0
                df.at[idx, "f60_rvol_vs_first15m"] = (np.mean(v60[3:])  / (v15/3))       if len(v60) > 3 and v15  > 0 else 1.0
                df.at[idx, "f60_rvol_vs_first30m"] = (np.mean(v60[6:])  / (v30s/6))      if len(v60) > 6 and v30s > 0 else 1.0
                df.at[idx, "f60_vol_pct_60m"]      = sum(v60) / day_vol
                df.at[idx, "f60_vwap_dist"]        = (p60 - vw60) / vw60 if vw60 > 0 else 0
                df.at[idx, "f60_price_vs_open"]    = (p60 - op) / op if op > 0 else 0
                df.at[idx, "f60_rsi14"]            = _mini_rsi(c60)
                df.at[idx, "f60_or_60m"]           = (max(b["h"] for b in s60_5) - min(b["l"] for b in s60_5)) / op if op > 0 else 0
                if len(l3v) == 3:
                    df.at[idx, "f60_vol_decay_last3"] = 1.0 if l3v[0] > l3v[1] > l3v[2] else 0.0
                if len(v60) >= 3:
                    sl = np.polyfit(range(len(v60[-3:])), v60[-3:], 1)[0]
                    df.at[idx, "f60_vol_slope_last3"] = sl / (np.mean(v60[-3:]) or 1)
                h60 = [b["h"] for b in s60_5]
                df.at[idx, "f60_lower_high"] = 1.0 if len(h60) >= 2 and h60[-1] < max(h60[:-1]) else 0.0

            if s60_15:
                v15b = [b["v"] for b in s60_15]
                fv15 = v15b[0] if v15b else 1
                vw60_15 = _running_vwap(b15, 60)
                p60_15  = s60_15[-1]["c"]
                df.at[idx, "f60_rvol_vs_first15m_15m"] = (np.mean(v15b[1:]) / fv15) if len(v15b) > 1 and fv15 > 0 else 1.0
                df.at[idx, "f60_vwap_dist_15m"]        = (p60_15 - vw60_15) / vw60_15 if vw60_15 > 0 else 0

            # -- T=90 -------------------------------------------------
            s90_5  = _bars_up_to(b5,  90)
            s90_15 = _bars_up_to(b15, 90)
            s90_30 = _bars_up_to(b30, 90)
            if s90_5 and s30:
                v90    = [b["v"] for b in s90_5]
                c90    = [b["c"] for b in s90_5]
                h90    = [b["h"] for b in s90_5]
                vw90   = _running_vwap(b5, 90)
                p90    = s90_5[-1]["c"]
                v30avg = (sum(b["v"] for b in s30) / 6) or 1

                df.at[idx, "f90_rvol_vs_first5m"]  = (np.mean(v90[1:]) / v90[0]) if v90[0] > 0 and len(v90) > 1 else 1.0
                df.at[idx, "f90_rvol_vs_first30m"] = (np.mean(v90[6:]) / v30avg) if len(v90) > 6 else 1.0
                df.at[idx, "f90_vwap_dist"]        = (p90 - vw90) / vw90 if vw90 > 0 else 0
                df.at[idx, "f90_rsi14"]            = _mini_rsi(c90)
                df.at[idx, "f90_consec_red"]       = float(sum(1 for b in s90_5[-5:] if b["c"] < b["o"]))

                if len(v90) >= 5:
                    l5v = v90[-5:]
                    sl  = np.polyfit(range(5), l5v, 1)[0]
                    df.at[idx, "f90_vol_slope_last5"] = sl / (np.mean(l5v) or 1)
                    df.at[idx, "f90_vol_decay_last5"] = 1.0 if all(l5v[i] > l5v[i+1] for i in range(4)) else 0.0

                if len(h90) >= 6:
                    mid = len(h90) // 2
                    fh  = max(h90[:mid])
                    sh  = max(h90[mid:])
                    lh  = 1.0 if sh < fh else 0.0
                    df.at[idx, "f90_lower_high"] = lh
                    if lh == 1.0:
                        fv_ = np.mean(v90[:mid]) or 1
                        sv_ = np.mean(v90[mid:])
                        df.at[idx, "f90_lower_high_lower_vol"] = 1.0 if sv_ < fv_ else 0.0
                    else:
                        df.at[idx, "f90_lower_high_lower_vol"] = 0.0

            if s90_15:
                vw90_15 = _running_vwap(b15, 90)
                p90_15  = s90_15[-1]["c"]
                df.at[idx, "f90_vwap_dist_15m"] = (p90_15 - vw90_15) / vw90_15 if vw90_15 > 0 else 0

            if s90_30 and len(s90_30) >= 2:
                v30_90 = [b["v"] for b in s90_30]
                fv30_  = v30_90[0] if v30_90 else 1
                df.at[idx, "f90_rvol_vs_first30m_30m"] = (np.mean(v30_90[1:]) / fv30_) if len(v30_90) > 1 and fv30_ > 0 else 1.0
                df.at[idx, "f90_third_30m_vs_first"]    = (v30_90[2] / fv30_) if len(v30_90) >= 3 and fv30_ > 0 else 1.0

            # -- T=120 ------------------------------------------------
            s120_5  = _bars_up_to(b5,  120)
            s120_15 = _bars_up_to(b15, 120)
            if s120_5 and s30:
                v120 = [b["v"] for b in s120_5]
                c120 = [b["c"] for b in s120_5]
                h120 = [b["h"] for b in s120_5]
                vw120 = _running_vwap(b5, 120)
                p120  = s120_5[-1]["c"]
                df.at[idx, "f120_rvol_vs_first5m"] = (np.mean(v120[1:]) / v120[0]) if v120[0] > 0 and len(v120) > 1 else 1.0
                df.at[idx, "f120_vwap_dist"]       = (p120 - vw120) / vw120 if vw120 > 0 else 0
                df.at[idx, "f120_rsi14"]           = _mini_rsi(c120)
                if len(v120) >= 5:
                    l5 = v120[-5:]
                    df.at[idx, "f120_vol_decay_last5"] = 1.0 if all(l5[i] > l5[i+1] for i in range(4)) else 0.0
                if len(h120) >= 8:
                    mid = len(h120) // 2
                    df.at[idx, "f120_lower_high"] = 1.0 if max(h120[mid:]) < max(h120[:mid]) else 0.0

            if s120_15:
                vw120_15 = _running_vwap(b15, 120)
                p120_15  = s120_15[-1]["c"]
                df.at[idx, "f120_vwap_dist_15m"] = (p120_15 - vw120_15) / vw120_15 if vw120_15 > 0 else 0

            # -- T=150 ------------------------------------------------
            s150_5 = _bars_up_to(b5, 150)
            if s150_5 and s30:
                v150 = [b["v"] for b in s150_5]
                c150 = [b["c"] for b in s150_5]
                vw150 = _running_vwap(b5, 150)
                p150  = s150_5[-1]["c"]
                df.at[idx, "f150_rvol_vs_first5m"] = (np.mean(v150[1:]) / v150[0]) if v150[0] > 0 and len(v150) > 1 else 1.0
                df.at[idx, "f150_vwap_dist"]       = (p150 - vw150) / vw150 if vw150 > 0 else 0
                df.at[idx, "f150_rsi14"]           = _mini_rsi(c150)
                if len(v150) >= 3:
                    lv = v150[-3:]
                    df.at[idx, "f150_vol_decay"] = 1.0 if lv[0] > lv[1] > lv[2] else 0.0

        except Exception as e:
            _log(f"Feature error {row.get('ticker','')} {row.get('date','')}: {e}")
            continue

    return df


# ══════════════════════════════════════════════════════════
# SECTION 8 - BUCKET ASSIGNMENT (9:30am data only)
# ══════════════════════════════════════════════════════════

def assign_buckets(df):
    df = df.copy()

    def _mktcap(mc):
        if mc < 50_000_000:  return "nano"
        if mc < 300_000_000: return "micro"
        return "small"

    def _float(sh):
        if sh < 5_000_000:  return "ultra"
        if sh < 20_000_000: return "low"
        if sh < 50_000_000: return "medium"
        return "high"

    def _gap(g):
        if g < 0.40: return "gap_20_40"
        if g < 0.75: return "gap_40_75"
        return "gap_75plus"

    def _rvol(r):
        if r < 2:  return "rvol_low"
        if r < 5:  return "rvol_2_5"
        if r < 10: return "rvol_5_10"
        return "rvol_10x"

    # Use float_shares for float bucket, shares_outstanding as fallback
    float_col = df["float_shares"].where(df["float_shares"] > 0, df["shares_outstanding"])

    df["bucket_mktcap"]   = df["market_cap"].apply(_mktcap)
    df["bucket_float"]    = float_col.apply(_float)
    df["bucket_gap"]      = df["gap_pct"].apply(_gap)
    df["bucket_catalyst"] = df["catalyst_type"]
    df["bucket_rvol"]     = df["rvol_full_day"].apply(_rvol)
    df["bucket_key"]      = (
        df["bucket_mktcap"]   + "\u00b7" + df["bucket_float"]    + "\u00b7" +
        df["bucket_gap"]      + "\u00b7" + df["bucket_catalyst"]  + "\u00b7" +
        df["bucket_rvol"]
    )
    return df


# ══════════════════════════════════════════════════════════
# SECTION 9 - BUCKET STATS
# ══════════════════════════════════════════════════════════

def compute_bucket_stats(df):
    stats = {}
    for key, grp in df.groupby("bucket_key"):
        n = len(grp)
        if n < MIN_BUCKET_N:
            continue
        stats[key] = {
            "n":        n,
            "base_wr":  float((grp["close"] < grp["open"]).mean()),
            "avg_fade": float(((grp["close"] - grp["open"]) / grp["open"]).mean()),
            "valid":    True
        }
    Path("buckets.json").write_text(json.dumps(stats, indent=2))
    rows = sorted(stats.items(), key=lambda x: x[1]["base_wr"], reverse=True)
    print(f"\n{'Bucket':<55} {'N':>5} {'BaseWR':>8} {'AvgFade':>9}")
    print("-" * 82)
    for k, v in rows:
        print(f"{k:<55} {v['n']:>5} {v['base_wr']:>8.1%} {v['avg_fade']:>9.1%}")
    return stats


def get_valid_buckets():
    if not Path("buckets.json").exists():
        return []
    stats = json.loads(Path("buckets.json").read_text())
    valid = [(k, v["base_wr"]) for k, v in stats.items() if v.get("valid") and v["n"] >= MIN_BUCKET_N]
    return [k for k, _ in sorted(valid, key=lambda x: x[1], reverse=True)]


# ══════════════════════════════════════════════════════════
# SECTION 10 - SCORE FUNCTION
# ══════════════════════════════════════════════════════════

def score(trades_df, bars_dict):
    """
    Simulate W/L on 15m bars after entry candle.
    1% commission deducted every trade.
    Hard gates: WR<65%, avg_ret<4%, N<30, N/month<15 -> score=0.0
    """
    if trades_df is None or len(trades_df) == 0:
        return {"score":0.0,"wr":0,"avg_ret_net":0,"n":0,"n_per_month":0,
                "win_count":0,"loss_count":0,"partial_count":0}

    returns, outcomes = [], []
    for _, row in trades_df.iterrows():
        key   = (row["ticker"], row["date"])
        b15   = bars_dict.get(key, {}).get("15m", [])
        ep    = row["entry_price"]
        et    = row.get("entry_t_min", 0)
        if not b15 or ep <= 0 or et > MAX_ENTRY_MIN:
            continue

        win_l  = ep * (1 - WIN_TARGET)
        stop_l = ep * (1 + STOP_LOSS)
        outcome, ret = "partial", None

        for b in b15:
            if b["t_min"] <= et:
                continue
            if b["l"] <= win_l:   # check low first
                outcome, ret = "win",  WIN_TARGET  - COMMISSION; break
            if b["h"] >= stop_l:
                outcome, ret = "loss", -STOP_LOSS  - COMMISSION; break

        if ret is None:
            fc  = b15[-1]["c"]
            ret = (ep - fc) / ep - COMMISSION

        returns.append(ret)
        outcomes.append(outcome)

    n = len(returns)
    if n == 0:
        return {"score":0.0,"wr":0,"avg_ret_net":0,"n":0,"n_per_month":0,
                "win_count":0,"loss_count":0,"partial_count":0}

    wr      = sum(1 for o in outcomes if o == "win") / n
    avg_ret = float(np.mean(returns))
    dates   = pd.to_datetime(trades_df["date"].unique())
    span    = (dates.max() - dates.min()).days
    npm     = n / max(span / 30.0, 1.0)

    base = {"wr":round(wr,4),"avg_ret_net":round(avg_ret,4),"n":n,
            "n_per_month":round(npm,2),
            "win_count":outcomes.count("win"),
            "loss_count":outcomes.count("loss"),
            "partial_count":outcomes.count("partial")}

    # Hard gates - ALL must pass
    if n   < MIN_N_SCORE:  return {"score":0.0, **base}
    if wr  < MIN_WR:       return {"score":0.0, **base}
    if avg_ret < MIN_AVG_RET: return {"score":0.0, **base}
    if npm < MIN_N_MONTH:  return {"score":0.0, **base}

    sc = ((wr - MIN_WR) / (1.0 - MIN_WR)) * min(avg_ret / MIN_AVG_RET, 3.0) * min(npm / 30.0, 2.0) * min(1.0, n / 100.0)
    return {"score":round(float(sc),6), **base}


# ══════════════════════════════════════════════════════════
# SECTION 11 - load_all_data() - NO CACHE, ALWAYS FRESH
# ══════════════════════════════════════════════════════════

def load_all_data():
    """
    Fetches everything fresh from Polygon + yfinance + EDGAR.
    No cache files. No pickle files.
    Returns (df_train, df_val, bars_train, bars_val)
    """
    today       = datetime.today()
    val_end     = today - timedelta(days=BUFFER_DAYS)
    val_start   = val_end - timedelta(days=VAL_MONTHS * 30)
    train_end   = val_start
    train_start = train_end - timedelta(days=TRAIN_MONTHS * 30)

    ts = train_start.strftime("%Y-%m-%d")
    ve = val_end.strftime("%Y-%m-%d")
    vs = val_start.strftime("%Y-%m-%d")

    print(f"[prepare] Fetching universe {ts} -> {ve} (NO CACHE)")
    df = fetch_universe(ts, ve)
    if df.empty:
        raise RuntimeError("FATAL: fetch_universe returned empty DataFrame")

    print(f"[prepare] Enriching fundamentals via yfinance ({df.ticker.nunique()} tickers)")
    df = enrich_fundamentals(df)

    print(f"[prepare] Fetching news (yfinance) + filings (EDGAR)")
    df = enrich_news_and_filings(df)

    print(f"[prepare] Fetching intraday bars (Polygon, 5m/15m/30m, NO CACHE)")
    bars = fetch_bars(df)

    print(f"[prepare] Computing features")
    df = compute_all_features(df, bars)
    df = assign_buckets(df)

    df_train = df[df["date"] <  vs].copy()
    df_val   = df[df["date"] >= vs].copy()
    bars_train = {k: v for k, v in bars.items() if k[1] <  vs}
    bars_val   = {k: v for k, v in bars.items() if k[1] >= vs}

    compute_bucket_stats(df_train)

    # Assertions
    assert len(df_train) > 50,      f"FATAL: only {len(df_train)} train events"
    assert "bucket_key" in df_train.columns
    assert df_train["bucket_key"].notna().all()
    assert df_train["gap_pct"].min() >= MIN_GAP_PCT * 0.99
    assert not df_train[["ticker","date"]].duplicated().any()

    valid = get_valid_buckets()
    print(f"\n[prepare] DONE - train={len(df_train)} val={len(df_val)} valid_buckets={len(valid)}")
    print("[prepare] ALL ASSERTIONS PASSED")
    return df_train, df_val, bars_train, bars_val
