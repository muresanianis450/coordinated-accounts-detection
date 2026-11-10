"""
Goal: test whether single-feature AUC survives when the observation window is
held (roughly) fixed. This is a DIAGNOSTIC, it does not change the data.

Idea: inside one campaign, group accounts into window-length bins. In a bin all
accounts have a similar window, so a window-driven feature (hour_entropy,
gap_median_s) can no longer separate the classes through the window alone.
If its AUC stays far from 0.5 inside bins, it carries something else.
If it falls to ~0.5, it was mostly the window.

Output: results_<RUN_TAG>/window_bin_test/cells.csv and summary.csv
"""

import os
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config

PROCESSED_DIR = os.path.join("data", f"processed_{config.RUN_TAG}")
OUT_DIR = os.path.join(f"results_{config.RUN_TAG}", "window_bin_test")

MIN_PER_CLASS = 20    # a cell needs at least this many IO AND control accounts

# Fine bins in days; narrow bins keep the window almost fixed inside a cell.
EDGES = [0, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 15, 60, 365, np.inf]

# window_days is included on purpose: it is the control row. Inside narrow bins
# its AUC should drop towards 0.5; if not, the bins are too wide.
FEATURES = ["window_days", "hour_entropy", "gap_median_s",
            "frac_gap_lt_60s", "dup_text_share", "text_len_mean",
            "hashtag_rate", "url_rate", "mention_rate",
            "repost_share", "reply_share"]


def main():
    # Load all processed account tables and stack them
    files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Assign every account to a window bin
    df["bin"] = pd.cut(df["window_days"], bins=EDGES, include_lowest=True)

    rows = []
    # One cell = one campaign + one bin
    for (camp, b), cell in df.groupby(["campaign", "bin"], observed=True):
        n_io = int((cell.label == 1).sum())
        n_ctrl = int((cell.label == 0).sum())
        if n_io < MIN_PER_CLASS or n_ctrl < MIN_PER_CLASS:
            continue                      # too small, skip this cell
        for f in FEATURES:
            sub = cell[[f, "label"]].dropna()
            # after dropping missing values both classes must still be big enough
            if (sub.label == 1).sum() < MIN_PER_CLASS or (sub.label == 0).sum() < MIN_PER_CLASS:
                continue
            rows.append({"campaign": camp, "bin": str(b), "feature": f,
                         "n_io": n_io, "n_ctrl": n_ctrl,
                         "auc": roc_auc_score(sub.label, sub[f])})

    cells = pd.DataFrame(rows)
    cells.to_csv(os.path.join(OUT_DIR, "cells.csv"), index=False)
    if cells.empty:
        print("No cell has enough accounts of both classes.")
        return

    # Summarize per feature over all cells.
    # weight = number of IO accounts, so tiny cells count less.
    def summarize(g):
        w = g.n_io
        return pd.Series({
            "n_cells": len(g),
            "n_campaigns": g.campaign.nunique(),
            "median_auc": g.auc.median(),
            "weighted_auc": np.average(g.auc, weights=w),
            "share_above_0.5": (g.auc > 0.5).mean(),
        })

    summary = cells.groupby("feature").apply(summarize).reindex(FEATURES)
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"))

    print(f"{cells[['campaign','bin']].drop_duplicates().shape[0]} cells")
    print(cells.groupby("campaign").bin.nunique().rename("cells per campaign").to_string())
    print()
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()