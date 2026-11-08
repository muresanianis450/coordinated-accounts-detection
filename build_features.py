# ---------------------------------------------------------------------------
# GOAL: turn the raw post-level data (one row per post) into an account-level
# table (one row per account) with numeric features and a label.
# We classify ACCOUNTS, not posts, so every feature describes one account.
#
# Output: one file per campaign, data/processed/<campaign>.parquet
# ---------------------------------------------------------------------------


import re                                           # regular expressions, used to clean the text
import numpy as np                                  # numerical functions (log2, nan)
import pandas as pd                                 # tables (DataFrame)

from config import *

# The only columns we load from parquet files, Loading fewer columns is
# faster and uses less memory. Everything else in the files is ignored.

COLS = ["accountid", "post_text", "post_time", "in_reply_to_postid", "is_repost",
        "hashtags", "urls", "account_mentions", "follower_count",
        "following_count", "account_creation_date", "is_control"]


# Two compiled patterns used to clean text before looking for duplicates:
#   URL_RE matches "http://..." or "https://..." up to the next whitespace
#   MENT_RE matches "@username"

URL_RE, MENT_RE = re.compile(r"https?://\S+"), re.compile(r"@\w+")

def norm(s):
    """Normalize a post-text so that near-identical posts become identical.
    Steps: lowercase -> remove URL -> remove @mentions -> strip spaces.
    Example: 'Vote NOW! https://x.co/abc @bob' -> 'vote now!'
    Two posts that differ only in the link or the mentioned user then count as duplicates,
    which is typical for coordinated accounts."""
    return MENT_RE.sub("", URL_RE.sub("",s.lower())).strip()

def n_items(x):
    """Length of a list-like cell (hashtags, urls, mentions).
    The cell can be missing (None); then the count is 0."""
    return 0 if x is None else len(x)

def load_campaign(name):
    """Read all parquet part files of one campaign and stack them into one table."""
    # Find every *.parquet file in the campaign folder (Bangladesh_part_1, _part_2, ...)
    files = sorted((DATA_ROOT / name).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {DATA_ROOT / name}")

    # Read each file (only the COLS columns) and concatenate them vertically
    # An account's posts can be spread over several parts, so we must combine
    # all parts BEFORE computing any per-account feature.
    return pd.concat((pd.read_parquet(f, columns=COLS) for f in files),
                     ignore_index=True)


def meta_features(df):
    """One row per account with 'metadata' features.
    These are risky: they may depend on HOW the data was collected, not on
    the behavior (e.g., IO accounts were collected with their whole history, so
    their post-count is huge.) We keep them in a separate group so evaluate.py
    can test models with and without them."""
    # groupby("accountid") splits the table into one group per account;
    # agg(...) computes one summary value per group for each name output.
    m = df.groupby("accountid").agg(
        is_control=("is_control","first"),                              # label (same for all posts of an account)
        n_posts_total=("post_time","size"),                             # how many posts the account has
        first=("post_time", "min"), last=("post_time", "max"),          # time of its first post
        followers=("follower_count","median"),                          # median, since the value changes over time
        following=("following_count","median"),
        created=("account_creation_date","first"))                      # account creation date

    # Length of the observed activity period, in days
    m["span_days"] = (m["last"]  - m["first"]).dt.days
    # Account age at its last observed post, in days
    m["account_age_days"] = (m["last"] - pd.to_datetime(m["created"])).dt.days

    # first/last/created were only needed to compute the two columns above
    return m.drop(columns=["first","last","created"])


def behavior_row(g):
    """Compute the behavioral features for ONE account
    g = a table with only that account's posts (already capped to CAP_POSTS)
    Returns a dictionary {feature name: value}."""

    # Sort chronologically; many features depend on the order of posts
    g = g.sort_values("post_time")
    t = g.post_time

    # Time gaps between consecutive posts, in seconds.
    # diff() gives the difference to the previous post; the first row has no
    # previous post (NaN), so dropna() removes it.

    gaps = t.diff().dt.total_seconds().dropna()
    mu,sd = gaps.mean(), gaps.std()         # mean and standard deviation of the gaps

    # Distribution of posts over the 24 hours of the day as proportions.
    # Example: p = {14: 0.5, 15:.05} means half the posts at 14h, half at 15h
    p = t.dt.hour.value_counts(normalize=True)

    # Text handling: missing text becomes an empty string
    text = g.post_text.fillna("")
    clean = text.map(norm)              # normalized version used for duplicate detection
    clean = clean[clean.str.len() > 0]  # drop posts that are empty after cleaning


    return {
        # ---------------------- TIMING features -----------------------------
        # Shanon entropy of the hour-of-day distribution, in bits.
        # Low = posts concentrated in few hours (scheduled or scripted)
        # high (max log2(24) = 4.58) = spread over the whole day
        "hour_entropy": -(p * np.log2(p)).sum(),
        # Typical (median) waiting time between posts, in seconds
        "gap_median_s": gaps.median(),
        # Coefficient of variation = std / mean of the gaps.
        # Near 0 = very regular posting; large = irregular.
        "gap_cv":  sd / mu if mu > 0 else np.nan,
        # Burstiness B = (sd-mu) / (sd + mu), between -1 and 1.
        "burstiness": (sd - mu) / (sd + mu) if (sd + mu) > 0 else np.nan,
        # Fraction of gaps shorter than 60 seconds (rapid-fire posting)
        "frac_gap_lt_60s": (gaps < 60).mean(),

        # ---------------------- CONTENT features -----------------------------
        # Share of posts whose normalized text already appeared earlier
        # 0 = every post is unique, close to 1 = the account repeats itself.
        # nunique() = number of distinct texts, len(clean) = number of post.
        "dup_text_share": 1 - clean.nunique() / len(clean) if len(clean) > 0 else np.nan,
        # Average post-length in characters
        "text_len_mean": text.str.len().mean(),
        # Average number of hashtags / URLs / mentions per post
        "hashtag_rate": g.hashtags.map(n_items).mean(),
        "url_rate": g.urls.map(n_items).mean(),
        "mention_rate": g.account_mentions.map(n_items).mean(),

        # ---------------------- ACTIVITY features -----------------------------
        # Fraction of posts that are reposts (retweets). True counts as 1, False as 0.
        "repost_share": g.is_repost.mean(),
        # Fraction of posts that are replies (the "reply to" field is filled in)
        "reply_share": g.in_reply_to_postid.notna().mean(),

        # Diagnostic only: observation window covered by the capped posts.
        "window_days": (t.max() - t.min()).total_seconds() / 86400,
    }

def window_table(capped):
    """One row per account with its observation window and class label."""
    # TODO Plot bins to see how many IOs and controlled are there
    W = capped.groupby("accountid").agg(
        first=("post_time", "min"),
        last=("post_time", "max"),
        is_control=("is_control", "first"),
    )
    W["window_days"] = (
    (W["last"] - W["first"]).dt.total_seconds() / 86400 # days
    )
    W["bin"] = pd.cut(
        W["window_days"],
        bins=WINDOWS_EDGES_DAYS,
        include_lowest=True
    )
    return W

def match_windows(capped):
    """Keep accounts in window bins containing enough examples of both classes. """
    W = window_table(capped)

    counts = (
        W.groupby(["bin","is_control"],observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[False,True],fill_value=0)
        )

    counts.columns = ["IO","control"]

    ok = counts[
        (counts.IO >= MIN_PER_CLASS_PER_BIN) &
        (counts.control >= MIN_PER_CLASS_PER_BIN)
    ]

    print("Acounts per window bin (days), before matching")
    print(counts.rename_axis("bin").to_string().replace("\n","\n   "))

    keep_acc = W.index[
        W["bin"].astype(str).isin(ok.index.astype(str))
    ]

    return capped[capped.accountid.isin(keep_acc)]


def count_classes(capped):
    """Return the number of IO accounts and control accounts."""
    lab = capped.groupby("accountid").is_control.first()

    return int((~lab).sum()), int(lab.sum())

def build(name):
    """Full pipeline for one campaign: load -> meta features -> behavior features -> save."""
    df = load_campaign(name)

    # Give IO and control accounts the same timestamp resolution.
    if FLOOR_TO_MINUTE:
        df["post_time"] = df["post_time"].dt.floor("min")

    acc = meta_features(df)     #one row per account (computed on ALL posts, uncapped)

    # Sanity check: count accounts per class and compare with Table 1 of the paper.
    # is_control == True means control account; ~ means "not", so ~is_control = IO account
    n_io, n_ctrl = (~acc.is_control).sum(), acc.is_control.sum()
    print(f"{name}: IO={n_io} control={n_ctrl} expected={EXPECTED.get(name)}")

    # Keep only accounts with at least MIN_POSTS posts
    keep = acc.index[acc.n_posts_total >= MIN_POSTS]

    # For the behavioral features: select the posts of the kept accounts,
    # sort by time, and take only the first CAP_POSTS posts per account.
    # head(N) after groupby return the first N rows of each group.
    capped = (df[df.accountid.isin(keep)].sort_values("post_time")
              .groupby("accountid").head(CAP_POSTS))

    print(
        f"  after MIN_POSTS: "
        f"IO={count_classes(capped)[0]} "
        f"control={count_classes(capped)[1]}"
    )

    # Keep only accounts observed over comparable periods.
    if MATCH_WINDOWS:
        capped = match_windows(capped)

        print(
            f"  after window matching: "
            f"IO={count_classes(capped)[0]} "
            f"control={count_classes(capped)[1]}"
        )

    if capped.empty:
        print("  no accounts left, campaign skipped")
        return

    # Compute behavior_row for every account and collect the results.
    # capped.groupby("accountid") yields (account id, that account's posts).
    rows = {a: behavior_row(g) for a, g in capped.groupby("accountid")}
    # Convert {account: {feature: value}} into a table with accounts as rows
    feats = pd.DataFrame.from_dict(rows, orient = "index")

    # Join meta features and behavior features on the account id.
    # how="inner" keeps only accounts present in both (i.e. the kept ones).
    out = acc.join(feats, how="inner")

    # Label for the model: 1 = information-operation accounts, 0 = control
    out["label"] = (~out.is_control).astype(int)
    #Remember which campaign each row came from (needed for leave-one-campaign-out)
    out["campaign"] = name
    out = out.drop(columns=["is_control"])      # redundant now that we have "label"

    # A campaign without IO accounts cannot be used as a test set.
    if out.label.sum() == 0:
        print("  no IO accounts left, campaign skipped")
        return

    # Save the account-level table for evaluate.py
    out.to_parquet(OUT / f"{name}.parquet")
    print(f" kept {len(out)} accounts, positives = {out.label.sum()}")

    # TODO save the csv for each of the country so I can see the data
    # How each
    out.to_csv(OUT / f"{name}.csv")

if __name__ == "__main__":
    print(f"Run tag: {RUN_TAG} -> saving in {OUT}")

    for c in CAMPAIGNS:
        build(c)

