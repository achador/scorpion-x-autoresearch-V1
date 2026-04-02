"""
orchestrator.py - The Karpathy autoresearch loop.
IMMUTABLE. Agent never modifies this.

Runs autonomously. Never stops until the budget is exhausted.
Never asks for permission. Reads results, generates a hypothesis
via Claude API, writes train.py, runs run_experiment.py as a subprocess,
reads the score, git commits if improved else git resets, repeats.
"""

import os, sys, re, json, time, subprocess
import pandas as pd
import requests
import anthropic
from pathlib import Path
from datetime import datetime

ANTHROPIC_KEY   = os.environ["ANTHROPIC_API_KEY"]
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
BUDGET          = int(os.environ.get("EXPERIMENT_BUDGET", "100"))
TARGET_BUCKET   = os.environ.get("TARGET_BUCKET", "")
MODEL           = "claude-sonnet-4-5"
MAX_TOKENS      = 1400

client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)


# -- HELPERS ----------------------------------------------------------

def load_results():
    p = Path("results.tsv")
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, sep="\t")
    except Exception:
        return pd.DataFrame()


def load_bucket_results(bucket):
    df = load_results()
    if df.empty or "bucket_key" not in df.columns:
        return pd.DataFrame()
    return df[df["bucket_key"] == bucket].copy()


def git_commit(msg):
    subprocess.run(["git", "add", "train.py", "results.tsv"],
                   capture_output=True)
    subprocess.run(["git", "commit", "-m", msg],
                   capture_output=True)


def git_reset():
    subprocess.run(["git", "checkout", "--", "train.py"],
                   capture_output=True)


def git_push():
    try:
        subprocess.run(["git", "push"],
                       capture_output=True, timeout=60)
    except Exception as e:
        sys.stderr.write(f"[orch] git push failed: {e}\n")


def run_experiment(bucket):
    """Run run_experiment.py as subprocess. Read score from stdout."""
    env  = {**os.environ, "BUCKET_KEY": bucket}
    try:
        proc = subprocess.run(
            [sys.executable, "run_experiment.py"],
            capture_output=True, text=True,
            timeout=600, env=env
        )
        score_str = proc.stdout.strip()
        if proc.stderr:
            sys.stderr.write(proc.stderr[-500:])
        return float(score_str) if score_str else 0.0
    except subprocess.TimeoutExpired:
        sys.stderr.write("[orch] Experiment timed out\n")
        return 0.0
    except Exception as e:
        sys.stderr.write(f"[orch] Subprocess error: {e}\n")
        return 0.0


def format_prior(prior, n=20):
    if prior.empty:
        return "No prior experiments for this bucket yet."
    top = prior.nlargest(min(n, len(prior)), "score")
    lines = []
    for _, r in top.iterrows():
        lines.append(
            f"  [{r.get('hypothesis_id','?')}] "
            f"score={r.get('score',0):.4f} "
            f"wr={r.get('wr_train',0):.1%} "
            f"ret={r.get('avg_ret_train',0):.1%} "
            f"N/mo={r.get('n_per_month_train',0):.1f} "
            f"t={r.get('entry_time_key','?')} "
            f"feats={r.get('total_features',0)}"
        )
        lines.append(f"    screen: {r.get('screen_desc','')}")
        lines.append(f"    entry:  {r.get('entry_desc','')}")
    return "\n".join(lines)


def list_tested(prior):
    if prior.empty or "entry_desc" not in prior.columns:
        return "(none yet)"
    descs = prior["entry_desc"].dropna().unique()[:40]
    return "\n".join(f"  - {d}" for d in descs)


# -- HYPOTHESIS GENERATION --------------------------------------------

def generate(bucket, prior, n):
    """Call Claude API to generate one new train.py hypothesis."""
    program   = Path("program.md").read_text()
    prior_txt = format_prior(prior, n=20)
    tested    = list_tested(prior)

    prompt = f"""{program}

CURRENT BUCKET: {bucket}

PRIOR RESULTS FOR THIS BUCKET ({len(prior)} experiments so far):
{prior_txt}

ENTRY DESCRIPTIONS ALREADY TESTED (do not repeat any of these):
{tested}

Generate experiment #{n:04d}.
Explore entry_time_keys and feature combinations not yet tried.
Respond in EXACT format below - no preamble, no markdown:

SCREEN_CODE:
    [python body of screen_at_930(df), must end with: return mask]

EXTRA_FEATURES_CODE:
    [python body of compute_extra_features(df, bars_dict), must end with: return df]
    [if not needed, write just: return df]

ENTRY_CODE:
    [python body of get_entry(df), must end with: return mask]

ENTRY_TIME_KEY: [one of: t30 | t60 | t90 | t120 | t150]
SCREEN_FEATURES: feat1,feat2,feat3
ENTRY_FEATURES: feat1,feat2,feat3
TOTAL_FEATURES: [integer - screen count + entry count combined, max 17]
SCREEN_DESC: [one sentence describing the 9:30am screen]
ENTRY_DESC: [one sentence describing the intraday entry signal]
MECHANISM: [one sentence - WHY does this predict a fade in bucket: {bucket}?]
HYPOTHESIS_ID: H_{n:04d}
"""

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.content[0].text
    except Exception as e:
        sys.stderr.write(f"[orch] Claude API error: {e}\n")
        return None

    def ext(tag, multi=False):
        if multi:
            m = re.search(rf"{tag}:\s*\n(.*?)(?=\n[A-Z_]+:|$)", text, re.DOTALL)
            return m.group(1).strip() if m else ""
        m = re.search(rf"{tag}:\s*(.+)", text)
        return m.group(1).strip() if m else ""

    screen_code  = ext("SCREEN_CODE",        multi=True)
    extra_code   = ext("EXTRA_FEATURES_CODE", multi=True) or "    return df"
    entry_code   = ext("ENTRY_CODE",          multi=True)
    entry_t      = ext("ENTRY_TIME_KEY")
    screen_feats = ext("SCREEN_FEATURES")
    entry_feats  = ext("ENTRY_FEATURES")
    total_str    = ext("TOTAL_FEATURES")
    screen_desc  = ext("SCREEN_DESC")
    entry_desc   = ext("ENTRY_DESC")
    mechanism    = ext("MECHANISM")
    hyp_id       = ext("HYPOTHESIS_ID") or f"H_{n:04d}"

    if not screen_code or not entry_code:
        sys.stderr.write(f"[orch] Parse failed for H_{n:04d}\n")
        return None

    if entry_t not in ("t30","t60","t90","t120","t150"):
        entry_t = "t60"

    try:
        total_int = int(total_str)
    except Exception:
        total_int = len(screen_feats.split(",")) + len(entry_feats.split(","))

    return {
        "screen_code":  screen_code,
        "extra_code":   extra_code,
        "entry_code":   entry_code,
        "entry_t":      entry_t,
        "screen_feats": screen_feats,
        "entry_feats":  entry_feats,
        "total":        total_int,
        "screen_desc":  screen_desc,
        "entry_desc":   entry_desc,
        "mechanism":    mechanism,
        "hyp_id":       hyp_id,
    }


# -- WRITE train.py ---------------------------------------------------

def write_train(h):
    def indent(code, spaces=8):
        pad = " " * spaces
        return "\n".join(pad + line for line in code.splitlines())

    content = f'''"""
train.py - {h["hyp_id"]} - {datetime.now().isoformat()}
entry_time_key={h["entry_t"]} total_features={h["total"]}
AUTOGENERATED BY SCORPION AUTORESEARCH - DO NOT EDIT MANUALLY
"""
import pandas as pd
import numpy as np
import math


def screen_at_930(df: pd.DataFrame) -> pd.Series:
    """Screen: {h["screen_desc"]}"""
    try:
{indent(h["screen_code"])}
    except Exception:
        return pd.Series(True, index=df.index)


def compute_extra_features(df: pd.DataFrame, bars_dict: dict) -> pd.DataFrame:
    try:
{indent(h["extra_code"])}
    except Exception:
        return df


def get_entry(df: pd.DataFrame) -> pd.Series:
    """Entry: {h["entry_desc"]}"""
    try:
{indent(h["entry_code"])}
    except Exception:
        return pd.Series(False, index=df.index)


HYPOTHESIS_META = {{
    "screening": {{
        "features":    {json.dumps([f.strip() for f in h["screen_feats"].split(",") if f.strip()])},
        "description": "{h["screen_desc"].replace(chr(34), chr(39))}"
    }},
    "entry": {{
        "features":    {json.dumps([f.strip() for f in h["entry_feats"].split(",") if f.strip()])},
        "description": "{h["entry_desc"].replace(chr(34), chr(39))}"
    }},
    "entry_time_key": "{h["entry_t"]}",
    "total_features": {h["total"]},
    "mechanism":      "{h["mechanism"].replace(chr(34), chr(39))}",
    "hypothesis_id":  "{h["hyp_id"]}",
    "timestamp":      "{datetime.now().isoformat()}"
}}
'''
    Path("train.py").write_text(content)


# -- DISCORD REPORT ---------------------------------------------------

def send_discord(bucket, prior):
    if not DISCORD_WEBHOOK:
        return
    top5 = prior.nlargest(5, "score") if not prior.empty else pd.DataFrame()
    def lvl(n):
        if n < 30:  return "L1-SKIP"
        if n < 100: return "L2-10%"
        if n < 500: return "L3-25%"
        return "L4-FULL"
    lines = [
        f"**SCORPION {datetime.now():%Y-%m-%d}**",
        f"Bucket: `{bucket}`",
        f"Experiments: `{len(prior)}`\n"
    ]
    for _, r in top5.iterrows():
        lines.append(
            f"**[{lvl(r.get('n_train',0))}]** "
            f"score=`{r['score']:.4f}` "
            f"WR=`{r['wr_train']:.1%}` "
            f"ret=`{r['avg_ret_train']:.1%}` "
            f"N/mo=`{r['n_per_month_train']:.1f}` "
            f"t=`{r.get('entry_time_key','?')}`\n"
            f"_{r.get('screen_desc','')}_ -> _{r.get('entry_desc','')}_"
        )
    try:
        requests.post(
            DISCORD_WEBHOOK,
            json={"content": "\n".join(lines)[:2000]},
            timeout=10
        )
    except Exception as e:
        sys.stderr.write(f"[orch] Discord error: {e}\n")


# -- MAIN LOOP --------------------------------------------------------

def main():
    # Pick bucket
    bucket = TARGET_BUCKET
    if not bucket:
        if not Path("buckets.json").exists():
            sys.stderr.write("[orch] No buckets.json. Run prepare.load_all_data() first.\n")
            sys.exit(1)
        stats  = json.loads(Path("buckets.json").read_text())
        valid  = sorted(
            [k for k, v in stats.items() if v.get("valid")],
            key=lambda k: stats[k]["base_wr"], reverse=True
        )
        if not valid:
            sys.stderr.write("[orch] No valid buckets found.\n")
            sys.exit(1)
        # Rotate: pick valid bucket with fewest prior experiments
        all_res = load_results()
        if all_res.empty or "bucket_key" not in all_res.columns:
            bucket = valid[0]
        else:
            counts = all_res.groupby("bucket_key").size()
            bucket = min(valid, key=lambda b: counts.get(b, 0))

    print(f"\n[orch] ══════════════════════════════════════════")
    print(f"[orch] SCORPION AUTORESEARCH - Karpathy Loop")
    print(f"[orch] Bucket:  {bucket}")
    print(f"[orch] Budget:  {BUDGET} experiments")
    print(f"[orch] Model:   {MODEL}")
    print(f"[orch] ══════════════════════════════════════════\n")

    prior      = load_bucket_results(bucket)
    best_score = prior["score"].max() if not prior.empty and "score" in prior.columns else 0.0
    print(f"[orch] Prior experiments: {len(prior)} | Best score: {best_score:.4f}\n")

    for i in range(1, BUDGET + 1):
        print(f"[orch] -- Experiment {i}/{BUDGET} ---------------------------------")

        # GENERATE
        hyp = generate(bucket, prior, i)
        if hyp is None:
            sys.stderr.write("[orch] Generation failed, sleeping 10s...\n")
            time.sleep(10)
            continue

        print(f"[orch] {hyp['hyp_id']} | t={hyp['entry_t']} | feats={hyp['total']}")
        print(f"[orch] screen: {hyp['screen_desc']}")
        print(f"[orch] entry:  {hyp['entry_desc']}")

        # WRITE
        write_train(hyp)

        # RUN
        score = run_experiment(bucket)

        # EVALUATE - git commit or reset
        if score > best_score and score > 0:
            best_score = score
            git_commit(f"{hyp['hyp_id']}[{bucket[:25]}] score={score:.4f}")
            print(f"[orch] NEW BEST {score:.4f} - committed to git")
        else:
            git_reset()
            print(f"[orch] discarded {score:.4f} (best={best_score:.4f})")

        # UPDATE prior for next generation context
        prior = load_bucket_results(bucket)

        # Brief pause - respect Claude API rate limits
        time.sleep(3)

    # DONE
    print(f"\n[orch] == RUN COMPLETE ==")
    print(f"[orch] {BUDGET} experiments | Best: {best_score:.4f}")
    send_discord(bucket, prior)
    git_push()


if __name__ == "__main__":
    main()
