"""
train.py - Scorpion AutoResearch V1 strategy definition.
THIS IS THE ONLY FILE the autoresearch agent rewrites each experiment.

The agent writes THREE functions. Each experiment it replaces all three
from scratch - this is the Karpathy loop. The agent has full freedom
to compute any indicator from first principles using raw bars.
"""

import pandas as pd
import numpy as np
import math


def screen_at_930(df: pd.DataFrame) -> pd.Series:
    """
    Runs at 9:30am. Uses ONLY data available at market open.
    This is the pre-market universe screen - the bucket filter.

    Available columns (all known at 9:30am open):
      gap_pct, open, volume, market_cap, float_shares,
      shares_outstanding, sector, industry,
      catalyst_type, news_count, has_8k, has_s3, has_424b,
      rvol_full_day,
      bucket_mktcap, bucket_float, bucket_gap,
      bucket_catalyst, bucket_rvol, bucket_key,
      f5_first_bar_dir, f5_first_bar_range, f5_open_return,
      or_5m, or_15m, or_30m

    Use 2 to 8 features. Returns boolean pd.Series.
    True = this stock passes the 9:30am screen.
    """
    try:
        # BASELINE - agent replaces entirely:
        return df["gap_pct"] >= 0.20
    except Exception:
        return pd.Series(True, index=df.index)


def compute_extra_features(df: pd.DataFrame, bars_dict: dict) -> pd.DataFrame:
    """
    OPTIONAL. Agent may compute any additional features from raw bars.
    This is where agent has full freedom from first principles.

    bars_dict[(ticker, date)] = {
        "5m":  [{t, o, h, l, c, v, t_min}, ...],
        "15m": [{t, o, h, l, c, v, t_min}, ...],
        "30m": [{t, o, h, l, c, v, t_min}, ...]
    }
    t_min = minutes from 9:30am open (9:30=0, 10:00=30, 10:30=60 etc)

    Agent can invent anything:
      - Custom volume ratios not in prepare.py
      - Bar sequence pattern counters
      - Statistical moments across bar windows
      - Crossover signals between timeframes
      - Fibonacci-style retracement levels
      - Any mathematical combination of price and volume

    CRITICAL: only use bars where t_min <= your entry_time_key.
    Do not use bars from after the entry time (lookahead).

    Import ONLY: pandas, numpy, math, collections, itertools, statistics
    Never call external APIs or import other modules.
    Wrap each row in try/except. Never crash.
    Return df unchanged if nothing to add.
    """
    return df


def get_entry(df: pd.DataFrame) -> pd.Series:
    """
    Intraday entry signal. Runs at the entry_time_key time.
    Uses only features available at or before entry_time_key.

    ENTRY TIME KEYS and their valid features:

    t30 = enter at 10:00am, use f30_* or earlier:
      f30_rvol_vs_first5m, f30_vol_pct_30m, f30_vwap_dist,
      f30_price_vs_open, f30_high_vs_open, f30_or_range,
      f30_bars_green, f30_vol_slope,
      f30_first_30m_bar_direction, f30_first_30m_bar_vol

    t60 = enter at 10:30am, use f60_* or earlier:
      f60_rvol_vs_first5m, f60_rvol_vs_first15m, f60_rvol_vs_first30m,
      f60_rvol_vs_first15m_15m, f60_vol_pct_60m,
      f60_vwap_dist, f60_vwap_dist_15m,
      f60_price_vs_open, f60_vol_decay_last3, f60_vol_slope_last3,
      f60_lower_high, f60_rsi14, f60_or_60m

    t90 = enter at 11:00am, use f90_* or earlier:
      f90_rvol_vs_first5m, f90_rvol_vs_first30m,
      f90_rvol_vs_first30m_30m, f90_vwap_dist, f90_vwap_dist_15m,
      f90_vol_decay_last5, f90_vol_slope_last5,
      f90_lower_high, f90_lower_high_lower_vol,
      f90_rsi14, f90_consec_red, f90_third_30m_vs_first

    t120 = enter at 11:30am, use f120_* or earlier:
      f120_rvol_vs_first5m, f120_vwap_dist, f120_vwap_dist_15m,
      f120_vol_decay_last5, f120_lower_high, f120_rsi14

    t150 = enter at 12:00pm, use f150_* or earlier:
      f150_rvol_vs_first5m, f150_vwap_dist, f150_vol_decay, f150_rsi14

    Plus any columns added by compute_extra_features().
    Plus any f30_* or earlier features when entry_time_key is t60+.

    Use 2 to 9 features. Returns boolean pd.Series.
    True = enter short on this event.
    Entry price = close of the 5m bar at entry_time_key.
    """
    try:
        # BASELINE - agent replaces entirely:
        return (
            (df["f60_rvol_vs_first5m"] < 0.5) &
            (df["f60_vol_decay_last3"] == 1.0)
        )
    except Exception:
        return pd.Series(False, index=df.index)


HYPOTHESIS_META = {
    "screening": {
        "features":    ["gap_pct"],
        "description": "gap >= 20%"
    },
    "entry": {
        "features":    ["f60_rvol_vs_first5m", "f60_vol_decay_last3"],
        "description": "RVOL dying vs first 5m AND volume decaying last 3 candles at 10:30am"
    },
    "entry_time_key": "t60",
    "total_features": 3,
    "mechanism":      "trivial baseline - autoresearch replaces this",
    "hypothesis_id":  "BASELINE_00",
    "timestamp":      "manual"
}
