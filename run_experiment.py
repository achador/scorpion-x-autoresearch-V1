"""
run_experiment.py - Atomic unit of the Karpathy loop.
IMMUTABLE. Agent never modifies this.

Input: a strategy written in train.py + a bucket to test on.
Output: a single float printed to stdout (the score).
The orchestrator reads this float and decides keep or discard.
"""

import os, sys, time, signal, importlib, json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, ".")
import prepare


def main():
    t0     = time.time()
    BUCKET = os.environ.get("BUCKET_KEY", "")
    if not BUCKET:
        print("0.0"); return

    # -- 1. Fresh data every single run ------------------------------
    # No cache. Always live from Polygon + yfinance + EDGAR.
    try:
        df_train, df_val, bars_train, bars_val = prepare.load_all_data()
    except Exception as e:
        sys.stderr.write(f"[run] FATAL: load_all_data failed: {e}\n")
        print("0.0"); return

    # -- 2. Filter both splits to this bucket -------------------------
    dbt = df_train[df_train["bucket_key"] == BUCKET].copy()
    dbv = df_val[df_val["bucket_key"]     == BUCKET].copy()

    if len(dbt) < 10:
        sys.stderr.write(f"[run] bucket '{BUCKET}' has only {len(dbt)} train rows\n")
        print("0.0"); return

    # -- 3. Import train.py (always reload latest agent version) ------
    try:
        if "train" in sys.modules:
            importlib.reload(sys.modules["train"])
        else:
            import train
        from train import screen_at_930, compute_extra_features, get_entry, HYPOTHESIS_META
    except SyntaxError as e:
        sys.stderr.write(f"[run] SyntaxError in train.py: {e}\n")
        print("0.0"); return
    except Exception as e:
        sys.stderr.write(f"[run] Import error: {e}\n")
        print("0.0"); return

    # -- 4. Validate feature count (max 17 total) ---------------------
    total_feats = HYPOTHESIS_META.get("total_features", 0)
    if total_feats > 17:
        sys.stderr.write(f"[run] REJECTED: {total_feats} features > 17 cap\n")
        print("0.0"); return
    if total_feats < 2:
        sys.stderr.write(f"[run] REJECTED: only {total_feats} features\n")
        print("0.0"); return

    entry_time_key = HYPOTHESIS_META.get("entry_time_key", "t60")
    entry_min_map  = {"t30": 30, "t60": 60, "t90": 90, "t120": 120, "t150": 150}
    entry_min      = entry_min_map.get(entry_time_key, 60)

    # -- 5. Build trades for one split --------------------------------
    def build_trades(df_bucket, bars):
        # Extra features (agent's custom compute)
        try:
            df_ext = compute_extra_features(df_bucket.copy(), bars)
        except Exception as e:
            sys.stderr.write(f"[run] compute_extra_features error: {e}\n")
            df_ext = df_bucket.copy()

        # 9:30am screen
        try:
            screen_mask = screen_at_930(df_ext)
            df_screened = df_ext[screen_mask].copy()
        except Exception as e:
            sys.stderr.write(f"[run] screen_at_930 error: {e}\n")
            return pd.DataFrame()

        if len(df_screened) == 0:
            return pd.DataFrame()

        # Entry signal
        try:
            entry_mask  = get_entry(df_screened)
            df_entries  = df_screened[entry_mask].copy()
        except Exception as e:
            sys.stderr.write(f"[run] get_entry error: {e}\n")
            return pd.DataFrame()

        if len(df_entries) == 0:
            return pd.DataFrame()

        # Determine entry price: close of 5m bar at entry_min
        trade_rows = []
        for _, row in df_entries.iterrows():
            key = (row["ticker"], row["date"])
            b5  = bars.get(key, {}).get("5m", [])
            if not b5:
                continue

            # Find 5m bar at or just before entry_min
            entry_bar = None
            for b in b5:
                if b["t_min"] <= entry_min:
                    entry_bar = b
                else:
                    break

            if entry_bar is None:
                continue

            ep  = entry_bar["c"]
            etm = entry_bar["t_min"]

            if ep <= 0 or etm > prepare.MAX_ENTRY_MIN:
                continue

            trade_rows.append({
                "ticker":      row["ticker"],
                "date":        row["date"],
                "entry_price": ep,
                "entry_t_min": etm
            })

        return pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame()

    # -- 6. Timeout wrapper (90 seconds max per experiment) -----------
    def _timeout(sig, frame):
        sys.stderr.write("[run] TIMEOUT after 90s\n")
        print("0.0")
        sys.exit(0)

    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(90)

    trades_train = build_trades(dbt, bars_train)
    trades_val   = build_trades(dbv, bars_val)

    signal.alarm(0)

    # -- 7. Score both splits -----------------------------------------
    res_train = prepare.score(trades_train, bars_train)
    res_val   = prepare.score(trades_val,   bars_val)

    # OOS stability penalty
    wr_decay = abs(res_train["wr"] - res_val["wr"])
    oos_pen  = 0.5 if wr_decay > prepare.OOS_DECAY_MAX else 1.0
    final    = res_train["score"] * oos_pen

    # -- 8. Write row to results.tsv ----------------------------------
    screen_feats = "|".join(HYPOTHESIS_META.get("screening", {}).get("features", []))
    entry_feats  = "|".join(HYPOTHESIS_META.get("entry",    {}).get("features", []))
    screen_desc  = HYPOTHESIS_META.get("screening", {}).get("description", "")
    entry_desc   = HYPOTHESIS_META.get("entry",     {}).get("description", "")

    row = {
        "hypothesis_id":     HYPOTHESIS_META.get("hypothesis_id", "?"),
        "bucket_key":        BUCKET,
        "entry_time_key":    entry_time_key,
        "total_features":    total_feats,
        "score":             round(final, 6),
        "wr_train":          round(res_train["wr"], 4),
        "avg_ret_train":     round(res_train["avg_ret_net"], 4),
        "n_train":           res_train["n"],
        "n_per_month_train": round(res_train["n_per_month"], 2),
        "wr_val":            round(res_val["wr"], 4),
        "avg_ret_val":       round(res_val["avg_ret_net"], 4),
        "n_val":             res_val["n"],
        "oos_penalty":       oos_pen,
        "screen_vars":       screen_feats,
        "entry_vars":        entry_feats,
        "screen_desc":       screen_desc,
        "entry_desc":        entry_desc,
        "mechanism":         HYPOTHESIS_META.get("mechanism", ""),
        "elapsed_sec":       round(time.time() - t0, 1),
        "timestamp":         datetime.now().isoformat()
    }

    results_path = Path("results.tsv")
    write_header = not results_path.exists() or results_path.stat().st_size == 0
    with open(results_path, "a") as f:
        if write_header:
            f.write("\t".join(row.keys()) + "\n")
        f.write("\t".join(str(v) for v in row.values()) + "\n")

    # -- 9. Output score to stdout (orchestrator reads ONLY this) -----
    print(f"{final:.6f}")

    sys.stderr.write(
        f"[run] {row['hypothesis_id']} [{BUCKET[:35]}] "
        f"score={final:.4f} wr={res_train['wr']:.1%} "
        f"ret={res_train['avg_ret_net']:.1%} N={res_train['n']} "
        f"N/mo={res_train['n_per_month']:.1f} "
        f"wr_val={res_val['wr']:.1%} oos={oos_pen} "
        f"t={entry_time_key} [{row['elapsed_sec']}s]\n"
    )


if __name__ == "__main__":
    main()
