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
        "burstiness": (sd - mu) / (sd + mu) if sd > 0 else np.nan,
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
    }

def build(name):
    """Full pipeline for one campaign: load -> meta features -> behavior features -> save."""
    df = load_campaign(name)
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

    # Save the account-level table for evaluate.py
    out.to_parquet(OUT / f"{name}.parquet")
    print(f" kept {len(out)} accounts, positives = {out.label.sum()}")


if __name__ == "__main__" :
    for c in CAMPAIGNS:
        build(c)
