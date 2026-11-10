"""
Goal: for every feature, show how it is distributed for IO accounts vs control accounts
so we can judge whether the feature is actually useful

What it produces (all inside results_<RUN_TAG>/feature_report/):
1. feature_all.csv  -> one row per account, all campaigns together
2. per_campaign/<campaign.csv>  -> the same table, split by campaign
3. feture_summary.csv  -> per feature and campaign: medians, means,
                        single-feature AUC, KS statistic, missing share
4. plots/<feature>.png     -> one figure per feature, one panel per campaign
                            + one panel with all campaigns pooled
5. auc_heatmap.png         -> features x campaigns, colored by AUC

HOW TO READ AUC (IO is the positive class):
0.5     -> the feature does not separate IO from control at all
>0.5    -> IO accounts have HIGHER values than controls
<0.5    -> IO accounts have LOWER values than controls
The further from 0.5, the stronger the separation (direction does not matter for the strenght,
                    only for consistency across campaigns).


"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

import config

matplotlib.use('Agg') # draw to files, no window needed

# -------------------------------------
# SETTINGS
# -------------------------------------

PROCESSED_DIR = os.path.join("data",f"processed_{config.RUN_TAG}")
OUT_DIR = os.path.join(f"results_{config.RUN_TAG}","feature_report")

LABEL_COL = "label"     # column holding the class
IO_VALUE = 1            # value of LABEL_COL meaning "IO account"
CAMPAIGN_COL = "campaign"

# Feature groups (from build.features.py)
FEATURE_GROUPS = {
    "timing":   ["hour_entropy", "gap_median_s", "burstiness", "frac_gap_lt_60s"],
    "content":  ["dup_text_share", "text_len_mean", "hashtag_rate", "url_rate", "mention_rate"],
    "activity": ["repost_share", "reply_share"],
    "diagnostic": ["window_days"],
    "meta":     ["n_posts_total", "followers", "following", "span_days", "account_age_days"],
}


# Features with a very long right tail are drawn on log10(1 + x) so the histogram is readable
# this only affects the PLOTS, not the CSVs or the AUC.
LOG_FEATURES = {
    "gap_median_s", "text_len_mean", "n_posts_total", "followers", "following",
    "span_days", "account_age_days", "window_days"
}
MIN_IO_FOR_RELIABLE = 20 # campaigns with fewer IO accounts get a "*" in the heatmap


# -------------------------------------
# STEP 1: load the processed one-row-per-account table
# -------------------------------------

def load_table():
    """Read every parquet (or csv) file in PROCESSED_DIR and stack them."""
    files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.parquet")))
    if files:
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {PROCESSED_DIR}")
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Loaded {len(df)} accounts from {len(files)} file(s) in {PROCESSED_DIR}")
    print("Columns:",list(df.columns))
    return df



# -------------------------------------
# STEP 2: statistics for one feature on one subset of accounts
# -------------------------------------
def describe(feature, group, name, sub):
    """Return one summary row (a dict) for this feature on this subset."""
    io = sub.loc[sub["is_io"], feature].dropna()
    ctrl = sub.loc[~sub["is_io"], feature].dropna()
    row = {
        "feature": feature, "group": group, "campaign": name,
        "n_io": len(io), "n_ctrl": len(ctrl),
        "median_io": io.median() if len(io) else np.nan,
        "median_ctrl": ctrl.median() if len(ctrl) else np.nan,
        "mean_io": io.mean() if len(io) else np.nan,
        "mean_ctrl": ctrl.mean() if len(ctrl) else np.nan,
        "frac_nan": sub[feature].isna().mean(),
        "auc_io_higher": np.nan, "ks": np.nan
    }

    # AUC and KS need both classes present
    if len(io) >= 2 and len(ctrl) >= 2:
        values = np.concatenate([io.values,ctrl.values])
        labels = np.concatenate([np.ones(len(io)), np.zeros(len(ctrl))])
        row["auc_io_higher"] = roc_auc_score(labels,values)
        row["ks"] = ks_2samp(io.values, ctrl.values).statistic

    return row

# -------------------------------------
# STEP 3: one figure per feature, one panel per campaign (+ pooled)
# -------------------------------------
def plot_feature(feature, df, campaigns, summary, out_path):
    """One figure per feature: a grid of panels, one per campaign + one pooled."""
    panels = campaigns + ["ALL (pooled)"]
    ncols = 6                                   # panels per row
    nrows = -(-len(panels) // ncols)            # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows),
                             squeeze=False)
    axes = axes.ravel()                         # flat list of axes, easier to loop over
    log = feature in LOG_FEATURES

    for ax, name in zip(axes, panels):
        # Select the accounts of this panel (all accounts for the pooled panel)
        sub = df if name.startswith("ALL") else df[df[CAMPAIGN_COL] == name]
        io = sub.loc[sub["is_io"], feature].dropna().astype(float)
        ctrl = sub.loc[~sub["is_io"], feature].dropna().astype(float)

        # Only the log features are transformed; the others stay as they are
        if log:
            io = np.log10(1 + io.clip(lower=0))
            ctrl = np.log10(1 + ctrl.clip(lower=0))

        # This check now runs for EVERY feature (it was inside "if log" before)
        both = pd.concat([io, ctrl])
        if len(io) == 0 or len(ctrl) == 0 or both.nunique() < 2:
            ax.text(0.5, 0.5, "no data or constant", ha="center", va="center")
            ax.set_title(f"{name}\nIO={len(io)} ctrl={len(ctrl)}", fontsize=9)
            continue

        # Same bins for both classes, clipped to the 1st-99th percentile
        lo, hi = both.quantile(0.01), both.quantile(0.99)
        if lo == hi:
            lo, hi = both.min(), both.max()
        bins = np.linspace(lo, hi, 30)

        # density=True: each class sums to 1, so classes of very different size
        # can be compared by shape
        ax.hist(ctrl.clip(lo, hi), bins=bins, density=True, alpha=0.5, label="control")
        ax.hist(io.clip(lo, hi), bins=bins, density=True, alpha=0.5, label="IO")

        auc = summary[(summary.feature == feature) &
                      (summary.campaign == name)]["auc_io_higher"]
        auc_txt = f"AUC={auc.iloc[0]:.2f}" if len(auc) and pd.notna(auc.iloc[0]) else "AUC=n/a"
        ax.set_title(f"{name}\nIO={len(io)} ctrl={len(ctrl)}  {auc_txt}", fontsize=9)
        ax.set_xlabel(f"log10(1+{feature})" if log else feature, fontsize=8)
        ax.set_yticks([])

    # Hide the unused axes in the last row
    for ax in axes[len(panels):]:
        ax.axis("off")

    # Legend and saving happen ONCE, after all panels are drawn
    handles, labels = [], []
    for ax in axes[:len(panels)]:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)

    fig.suptitle(feature, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])      # leave room for the title
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

# ----------------------------------------------------------------------------
# STEP 4: heatmap of AUC (features x campaigns)
# ----------------------------------------------------------------------------
def plot_heatmap(summary, features, campaigns, out_path):
    cols = campaigns + ["ALL (pooled)"]
    auc = summary.pivot(index="feature", columns="campaign", values="auc_io_higher")
    auc = auc.reindex(index=features, columns=cols)
    n_io = summary.pivot(index="feature", columns="campaign", values="n_io")
    n_io = n_io.reindex(index=features, columns=cols)

    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 3, 0.45 * len(features) + 2))
    im = ax.imshow(auc.values.astype(float), cmap="coolwarm", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)

    # write the number in each cell; "*" = fewer than MIN_IO_FOR_RELIABLE IO accounts
    for i in range(len(features)):
        for j in range(len(cols)):
            v = auc.values[i, j]
            if pd.notna(v):
                star = "*" if n_io.values[i, j] < MIN_IO_FOR_RELIABLE else ""
                ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="AUC (red: IO higher, blue: IO lower, white: no signal)")
    ax.set_title(f"Single-feature AUC (* = fewer than {MIN_IO_FOR_RELIABLE} IO accounts)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    df = load_table()

    # boolean helper column: True = IO account
    df["is_io"] = df[LABEL_COL] == IO_VALUE
    print("Label counts:\n", df[LABEL_COL].value_counts())

    # keep only the features that really exist in the table
    groups = {g: [f for f in fs if f in df.columns] for g, fs in FEATURE_GROUPS.items()}
    missing = [f for fs in FEATURE_GROUPS.values() for f in fs if f not in df.columns]
    if missing:
        print("WARNING, columns not found (skipped):", missing)
    features = [f for fs in groups.values() for f in fs]
    group_of = {f: g for g, fs in groups.items() for f in fs}

    campaigns = sorted(df[CAMPAIGN_COL].unique())

    os.makedirs(os.path.join(OUT_DIR, "per_campaign"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "plots"), exist_ok=True)

    # --- 1) CSV files -------------------------------------------------------
    df.drop(columns=["is_io"]).to_csv(os.path.join(OUT_DIR, "features_all.csv"), index=False)
    for c in campaigns:
        df[df[CAMPAIGN_COL] == c].drop(columns=["is_io"]).to_csv(
            os.path.join(OUT_DIR, "per_campaign", f"{c}.csv"), index=False)

    # --- 2) summary table ---------------------------------------------------
    rows = []
    for f in features:
        for c in campaigns:
            rows.append(describe(f, group_of[f], c, df[df[CAMPAIGN_COL] == c]))
        rows.append(describe(f, group_of[f], "ALL (pooled)", df))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "feature_summary.csv"), index=False)

    # --- 3) one figure per feature -----------------------------------------
    for f in features:
        plot_feature(f, df, campaigns, summary, os.path.join(OUT_DIR, "plots", f"{f}.png"))
        print("plotted", f)

    # --- 4) heatmap ---------------------------------------------------------
    plot_heatmap(summary, features, campaigns, os.path.join(OUT_DIR, "auc_heatmap.png"))

    # --- 5) quick console view: AUC table -----------------------------------
    table = summary.pivot(index="feature", columns="campaign", values="auc_io_higher")
    print("\nSingle-feature AUC (IO = positive class):")
    print(table.reindex(features).round(2).to_string())
    print(f"\nDone. Everything is in {OUT_DIR}")


if __name__ == "__main__":
    main()

