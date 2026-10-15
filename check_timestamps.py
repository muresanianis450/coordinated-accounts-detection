# check_timestamps.py
# ---------------------------------------------------------------------------
# GOAL: check whether the timing features (gap_median_s and friends) measure
# real posting behavior or how the timestamps were COLLECTED.
# For every campaign it compares information-operation (IO) accounts with
# control accounts on:
#   - timestamp resolution (do posts have exact seconds, or only minutes?)
#   - timestamps at exactly 00:00:00 (a sign of date-only data)
#   - identical consecutive timestamps (zero gaps)
#   - how long a period the first CAP_POSTS posts cover
#   - the typical posting date
# It uses the same account selection as build_features.py (MIN_POSTS, CAP_POSTS),
# so it describes exactly the posts the features were computed on.
# ---------------------------------------------------------------------------

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import *                       # DATA_ROOT, CAMPAIGNS, MIN_POSTS, CAP_POSTS

SHOW_PLOTS = True                          # True: open the figure at the end
RESULTS = Path(f"results_min{MIN_POSTS}_cap{CAP_POSTS}")   # same folder naming as the other scripts
RESULTS.mkdir(exist_ok=True)


def load_posts(name):
    """Read only the three columns we need from all parquet parts of a campaign."""
    files = sorted((DATA_ROOT / name).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_ROOT / name}")
    return pd.concat((pd.read_parquet(f, columns=["accountid", "post_time", "is_control"])
                      for f in files), ignore_index=True)


def capped_posts(df):
    """Reproduce the selection of build_features.py:
    keep accounts with at least MIN_POSTS posts, then each account's first CAP_POSTS posts."""
    n = df.groupby("accountid").post_time.transform("size")   # posts per account, on every row
    df = df[n >= MIN_POSTS]
    return (df.sort_values("post_time")
              .groupby("accountid").head(CAP_POSTS))


def summarize(g):
    """Timestamp statistics for the posts of ONE class in ONE campaign."""
    g = g.sort_values(["accountid", "post_time"]).copy()
    # Does the timestamp have a non-zero seconds part? If most posts have seconds == 0,
    # the time was probably recorded at minute resolution. By chance only ~1.7% (1/60) would.
    g["sec0"] = g.post_time.dt.second == 0
    # Exactly midnight: the time of day is 00:00:00 (typical when only the date is known)
    g["midnight"] = g.post_time == g.post_time.dt.normalize()
    # Gap to the previous post of the SAME account, in seconds (NaN for the first post)
    g["gap"] = g.groupby("accountid").post_time.diff().dt.total_seconds()

    per_acc = g.groupby("accountid").agg(
        first=("post_time", "min"), last=("post_time", "max"),
        sec0=("sec0", "mean"), median_gap=("gap", "median"))
    per_acc["window_days"] = (per_acc["last"] - per_acc["first"]).dt.total_seconds() / 86400

    gaps = g.gap.dropna()
    return dict(
        n_accounts=len(per_acc),
        n_posts=len(g),
        pct_sec0=100 * g.sec0.mean(),                       # % of posts with seconds == 0 (chance: 1.7)
        pct_midnight=100 * g.midnight.mean(),               # % of posts at exactly 00:00:00
        pct_minres_accounts=100 * (per_acc.sec0 >= 0.9).mean(),   # % of accounts with >= 90% sec0 posts
        pct_zero_gap=100 * (gaps == 0).mean(),              # % of gaps equal to 0 s (same timestamp)
        median_gap_s=per_acc.median_gap.median(),           # typical account's median gap
        median_window_days=per_acc.window_days.median(),    # time covered by the first CAP_POSTS posts
        median_date=g.post_time.median().date(),            # typical posting date
    )


rows = []
for name in CAMPAIGNS:
    capped = capped_posts(load_posts(name))
    # is_control == True -> control account, False -> information-operation account
    for is_control, g in capped.groupby("is_control"):
        rows.append(dict(campaign=name, group="control" if is_control else "IO", **summarize(g)))
res = pd.DataFrame(rows)

# ---- Print and save --------------------------------------------------------
pd.set_option("display.width", 220)                 # keep the wide table on one line
print(f"Settings: MIN_POSTS={MIN_POSTS}, CAP_POSTS={CAP_POSTS}")
print("Chance level for pct_sec0 is about 1.7 (1 second out of 60).")
print(res.round(1).to_string(index=False))
res.to_csv(RESULTS / "timestamp_check.csv", index=False)

# ---- Figure: three panels, IO vs control per campaign -----------------------
def grouped_bar(ax, col, title, log=False):
    """Bars: x = campaign, two bars per campaign (IO in red, control in blue)."""
    piv = res.pivot(index="campaign", columns="group", values=col).reindex(CAMPAIGNS)
    x, w = np.arange(len(piv)), 0.38
    ax.bar(x - w / 2, piv["IO"], w, label="IO", color="#d62728")
    ax.bar(x + w / 2, piv["control"], w, label="control", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=45, ha="right")
    ax.set_title(title)
    if log:
        ax.set_yscale("log")


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
grouped_bar(axes[0], "pct_sec0", "% of posts with seconds = 0")
axes[0].axhline(100 / 60, ls="--", color="black", lw=1)       # chance level
axes[0].legend()
grouped_bar(axes[1], "pct_midnight", "% of posts at exactly 00:00:00")
grouped_bar(axes[2], "median_window_days",
            f"Median period covered by the first {CAP_POSTS} posts (days, log)", log=True)
fig.tight_layout()
fig.savefig(RESULTS / "fig_timestamps.png", dpi=150)

print(f"\nDone. Saved in {RESULTS}/")
if SHOW_PLOTS:
    plt.show()