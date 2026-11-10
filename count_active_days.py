# count_active_days.py
# ---------------------------------------------------------------------------
# GOAL: find out how many IO accounts could be compared with control accounts
# on equal terms.
#
# Control accounts were collected one day at a time (up to 100 posts per day),
# so 30 control posts usually fall inside one day. To treat IO accounts the
# same way, we need IO accounts that have at least one calendar day with
# MIN_POSTS or more posts. This script only COUNTS them; it changes nothing.
#
# Output: results_<RUN_TAG>/active_day_counts.csv  (one row per campaign)
# Run it from the project folder, like the other scripts.
# ---------------------------------------------------------------------------

from pathlib import Path

import pandas as pd

from config import DATA_ROOT, CAMPAIGNS, MIN_POSTS, RUN_TAG

# We only need three columns, so loading is light even for big campaigns.
COLS = ["accountid", "post_time", "is_control"]

rows = []
for name in CAMPAIGNS:
    files = sorted((DATA_ROOT / name).glob("*.parquet"))
    if not files:
        print(f"{name}: no parquet files found, skipped")
        continue
    df = pd.concat((pd.read_parquet(f, columns=COLS) for f in files), ignore_index=True)

    # The calendar day (UTC) of every post
    df["day"] = pd.to_datetime(df["post_time"], utc=True).dt.floor("D")

    # Number of posts of each account on each day
    per_day = df.groupby(["accountid", "day"]).size().rename("n").reset_index()
    # Label of each account (True = control, False = IO)
    is_control = df.groupby("accountid").is_control.first()

    # For each account: its busiest day, and how many days reach MIN_POSTS
    acc = per_day.groupby("accountid").n.agg(
        busiest_day="max",
        days_with_min_posts=lambda s: (s >= MIN_POSTS).sum(),
    )
    acc["is_control"] = is_control
    acc["total_posts"] = per_day.groupby("accountid").n.sum()

    io = acc[~acc.is_control]
    ctrl = acc[acc.is_control]
    rows.append({
        "campaign": name,
        # IO accounts
        "io_all": len(io),
        f"io_total_ge_{MIN_POSTS}": int((io.total_posts >= MIN_POSTS).sum()),
        f"io_one_day_ge_{MIN_POSTS}": int((io.busiest_day >= MIN_POSTS).sum()),
        "io_median_qualifying_days": float(io.days_with_min_posts[io.days_with_min_posts > 0].median())
        if (io.days_with_min_posts > 0).any() else 0.0,
        # Control accounts, same counts for comparison
        "ctrl_all": len(ctrl),
        f"ctrl_total_ge_{MIN_POSTS}": int((ctrl.total_posts >= MIN_POSTS).sum()),
        f"ctrl_one_day_ge_{MIN_POSTS}": int((ctrl.busiest_day >= MIN_POSTS).sum()),
    })
    print(rows[-1])

out = pd.DataFrame(rows)
total = out.drop(columns=["campaign", "io_median_qualifying_days"]).sum()
total["campaign"] = "TOTAL"
out = pd.concat([out, total.to_frame().T], ignore_index=True)

results = Path(f"results_{RUN_TAG}")
results.mkdir(exist_ok=True)
out.to_csv(results / "active_day_counts.csv", index=False)
print()
print(out.to_string(index=False))
print(f"\nSaved to {results / 'active_day_counts.csv'}")
