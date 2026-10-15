# -----------------------------------------
# GOAL: make the first leave-one-campaign_out (LOCO) results trustworthy.
# Three analyses:
#       A. LEAKAGE DIAGNOSIS: which metadata feature lets a model "cheat"?
#       B. UNCERTAINTY: bootstrap confidence intervals for every PR_AUC score.
#       C. RANDOM SPLIT vs LOCO: how much does a normal random split inflate scores?
#
#       Run after buld_features.py. Results are printed and saved in results/
# -----------------------------------------


from pathlib import Path

import matplotlib.pyplot as plt     #plotting
from matplotlib.lines import Line2D   # used to build custom legends
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import *

N_BOOT = 500                            # bootstrap repetition (more = smoother, slower)
LEAK_MODEL = "rf"                       # model used for the leakage diagnosis (part A)
RESULTS = f"results_{RUN_TAG}"
Path(RESULTS).mkdir(exist_ok=True)
SHOW_PLOTS = True

# Load the account-level tables made by build_features.py
# reset_index(drop=True) replaces the account-id index by 0...n-1, because
# cross_val_prediction below works with row positions.
data = pd.concat(pd.read_parquet(p) for p in OUT.glob("*.parquet")).reset_index(drop=True)
CAMPAIGNS_PRESENT = sorted(data.campaign.unique())

# Same feature groups as in evaluate.py
BEHAVIOR = ["hour_entropy", "gap_median_s", "gap_cv", "burstiness", "frac_gap_lt_60s",
            "dup_text_share", "text_len_mean", "hashtag_rate", "url_rate",
            "mention_rate", "repost_share", "reply_share"]
META = ["n_posts_total", "followers", "following", "span_days", "account_age_days"]
GROUPS = {"behavior": BEHAVIOR, "meta_only": META, "behavior+meta": BEHAVIOR + META}
MODEL_NAMES = ["logreg", "rf", "hgb"]     # the dummy baseline is not needed here


def models():
    """Fresh untrained models (same settings as in evaluate.py)"""
    imp = lambda: SimpleImputer(strategy="median")  # fills NaN with the column median
    return {
        "logreg": make_pipeline(imp(),StandardScaler(),
                                LogisticRegression(class_weight="balanced",max_iter=2000)),
        "rf": make_pipeline(imp(), RandomForestClassifier(n_estimators= 300, class_weight="balanced", random_state=SEED, n_jobs=-1)),
        "hgb": make_pipeline(
    imp(),
    HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=SEED
    )
)
    }

def loco_predict(cols,mname):
    """Leave-one-campaign-out for ONE feature set and ONE model.
    For each campaign: train on all the others, predict the held-out one
    Returns {campaign: (true labels, predicted scored)}"""
    out = {}
    for c in CAMPAIGNS_PRESENT:
        train,test = data[data.campaign != c], data[data.campaign == c]
        model = models()[mname]             # new model for every fold
        model.fit(train[cols],train.label)
        out[c] = (test.label.values, model.predict_proba(test[cols])[:, 1])
    return out



# -----------------------------------------
# PART A: LEAKAGE DIAGNOSIS
# If a single metadata feature scores well on its own, the label is carried
# by how the data was collected, not by the behavior.
# -----------------------------------------
print("\n=== A. Leakage diagnosis (PR-AUC, model = %s) ===" % LEAK_MODEL)

# Positive rate per campaign = what the dummy baseline scores
pos_rate = data.groupby("campaign").label.mean().round(3)
print("Dummy baseline (positive rage):", pos_rate.to_dict())


LEAK_SETS = {f"{f} only": [f] for f in META}            # each metadata feature alone
LEAK_SETS["all meta (reference)"] = META
LEAK_SETS["meta without n_posts_total"] = [f for f in META if f != "n_posts_total"]
LEAK_SETS["behavior (reference)"] = BEHAVIOR
LEAK_SETS["behavior + meta without n_posts_total"] = (
    BEHAVIOR + [f for f in META if f != "n_posts_total"])

rows=[]
for name, cols in LEAK_SETS.items():
    for c, (y,s) in loco_predict(cols, LEAK_MODEL).items():
        rows.append(dict(features=name, test=c, pr_auc=average_precision_score(y,s)))

leak = (pd.DataFrame(rows)
        .pivot(index="features",columns="test",values="pr_auc")
        .reindex(list(LEAK_SETS)))                          # keep our row order

print(leak.round(3).to_string())
leak.round(3).to_csv(f"{RESULTS}/leakage_diagnosis.csv")




# -----------------------------------------
# PART B: BOOTSTRAP CONFIDENCE INTERVALS
# Idea: resample the TEST accounts with replacement many times and recompute
# PR-AUC each time. The 2.5% and 97.5% percentiles of those values give a 95%
# interval. Limitations: this captures uncertainty from WHICH test accounts
# we have, not uncertainty from the training data
# -----------------------------------------
print("\n=== B. LOCO PR-AUC with 95% bootstrap confidence intervals ===")
rng = np.random.default_rng(SEED)           # reproducible resampling


def bootstrap_ci(y,s):
    """95% bootstrap interval of PR-AUC over the test accounts."""
    vals=[]
    for _ in range(N_BOOT):
        idx = rng.integers(0,len(y),len(y))     # draw len(y) accounts with replacement
        if y[idx].sum() == 0:                   # PR-AUC is undefined without positives
            continue
        vals.append(average_precision_score(y[idx],s[idx]))
    return np.percentile(vals,[2.5,97.5])


rows = []
for gname, cols in GROUPS.items():
    for mname in MODEL_NAMES:
        for c, (y,s) in loco_predict(cols,mname).items():
            lo, hi = bootstrap_ci(y,s)
            rows.append(dict(test = c, features = gname, model = mname, n_pos=int(y.sum()),
                        pr_auc=average_precision_score(y,s), ci_low=lo, ci_high =hi,
                        dummy=y.mean()))
loco = pd.DataFrame(rows).round(3)
print(loco.to_string(index=False))
loco.to_csv(f"{RESULTS}/loco_with_ci.csv", index = False)

# -----------------------------------------
# PART C: RANDOM SPLIT vs LEAVE-ONE-CAMPAIGN-OUT
# Random split: accounts of the SAME campaign appear in train and test, so the
# model can learn campaign-specific patterns. We evaluate on the same accounts
# per campaign as in LOCO, so the two numbers are directly comparable.
# -----------------------------------------
print("\n=== C. Random 5-fold split vs LOCO (PR-AUC per campaign) ===")

# Stratify on campaign + label, so every fold has accounts from every campaign
# and from both classes.
strata = data.campaign + "_" + data.label.astype(str)
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
             .split(data,strata))

rows = []
for gname, cols in GROUPS.items():
    for mname in MODEL_NAMES:
        # Out-of-fold scores: every account is scored by a model that did not
        # see it during training.
        oof = cross_val_predict(models()[mname], data[cols],data.label,
                                cv=folds,method="predict_proba")[:, 1]
        for c in CAMPAIGNS_PRESENT:
            mask = (data.campaign == c).values
            rows.append(dict(test= c, features=gname, model=mname,
                             random_split=average_precision_score(data.label[mask],oof[mask])))
rand = pd.DataFrame(rows)


# Put the random-split score next to the LOCO score
cmp = loco.merge(rand.round(3), on=["test", "features", "model"])
cmp["inflation"] = (cmp.random_split - cmp.pr_auc).round(3)       # random minus LOCO
print(cmp[["test", "features", "model", "pr_auc", "random_split", "inflation"]]
      .rename(columns={"pr_auc": "loco"}).to_string(index=False))
cmp.to_csv(f"{RESULTS}/random_vs_loco.csv", index=False)


# PART D: FIGURES
# Each figure is saved as a PNG in results/ (150 dpi, good enough for a thesis
# draft). They use the tables computed above: leak, loco and cmp.
# ===========================================================================
# One fixed color per feature group, used in figures 2 and 3
GROUP_COLORS = {"behavior": "#1f77b4", "meta_only": "#d62728", "behavior+meta": "#7f7f7f"}

# ---- Figure 1: leakage heatmap ---------------------------------------------
# Rows = feature sets, columns = held-out campaigns, color = PR-AUC.
# A bright row for a single metadata feature means that feature leaks the label.
fig, ax = plt.subplots(figsize=(2 + 1.3 * len(leak.columns), 0.55 * len(leak) + 1.8))
im = ax.imshow(leak.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(leak.columns)))
ax.set_xticklabels(leak.columns, rotation=45, ha="right")
ax.set_yticks(range(len(leak.index)))
ax.set_yticklabels(leak.index)
if len(leak.columns) <= 8:  # write the numbers in the cells (few campaigns only)
    for i in range(leak.shape[0]):
        for j in range(leak.shape[1]):
            v = leak.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if v < 0.6 else "black")  # white text on dark cells
fig.colorbar(im, ax=ax, label="PR-AUC")
ax.set_title("Leakage diagnosis: PR-AUC by feature set\n"
             "(dummy baseline is about 0.02 for every campaign)")
fig.tight_layout()
fig.savefig(f"{RESULTS}/fig1_leakage.png", dpi=150)

# ---- Figure 2: PR-AUC with 95% bootstrap intervals --------------------------
# One panel per held-out campaign. Each row is one (feature group, model).
# The dot is the PR-AUC, the bar is the interval, the dashed line is the dummy
# baseline. A wide bar means the score is uncertain.
labels = [(g, m) for g in GROUPS for m in MODEL_NAMES]
ncols = min(3, len(CAMPAIGNS_PRESENT))
nrows = int(np.ceil(len(CAMPAIGNS_PRESENT) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows),
                         sharex=True, squeeze=False)
for ax, c in zip(axes.flat, CAMPAIGNS_PRESENT):
    sub = loco[loco.test == c].set_index(["features", "model"])
    for k, (g, m) in enumerate(labels):
        r = sub.loc[(g, m)]
        # Error bar lengths to the left and right of the dot. max(..., 0) avoids a
        # negative length when the bootstrap interval does not contain the point value.
        err = [[max(r.pr_auc - r.ci_low, 0)], [max(r.ci_high - r.pr_auc, 0)]]
        ax.errorbar(r.pr_auc, k, xerr=err, fmt="o", color=GROUP_COLORS[g], capsize=3)
    ax.axvline(sub.dummy.iloc[0], ls="--", color="black", lw=1)  # dummy baseline
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"{g} / {m}" for g, m in labels], fontsize=8)
    ax.invert_yaxis()  # first row at the top
    ax.set_xlim(0, 1.02)
    ax.set_title(f"{c} ({int(sub.n_pos.iloc[0])} IO accounts)")
    ax.set_xlabel("PR-AUC (dashed line = dummy)")
for ax in axes.flat[len(CAMPAIGNS_PRESENT):]:  # hide unused panels
    ax.axis("off")
fig.tight_layout()
fig.savefig(f"{RESULTS}/fig2_loco_ci.png", dpi=150)

# ---- Figure 3: random split vs leave-one-campaign-out ------------------------
# Each point is one (campaign, feature group, model). Points ABOVE the dashed
# diagonal scored higher under the random split than under LOCO, i.e. the
# random split inflated the score.
MARKERS = {"logreg": "o", "rf": "s", "hgb": "^"}
fig, ax = plt.subplots(figsize=(6.5, 6.5))
for (g, m), sub in cmp.groupby(["features", "model"]):
    ax.scatter(sub.pr_auc, sub.random_split, color=GROUP_COLORS[g],
               marker=MARKERS[m], s=55, alpha=0.8)
ax.plot([0, 1], [0, 1], "k--", lw=1)  # y = x line
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)
ax.set_xlabel("PR-AUC, leave-one-campaign-out")
ax.set_ylabel("PR-AUC, random 5-fold split")
ax.set_title("Does a random split inflate the score?\n(above the dashed line = yes)")
# Two legends: colors = feature group, shapes = model
handles = [Line2D([], [], marker="o", ls="", color=col, label=g) for g, col in GROUP_COLORS.items()]
handles += [Line2D([], [], marker=mk, ls="", color="black", label=m) for m, mk in MARKERS.items()]
ax.legend(handles=handles, loc="lower right")
fig.tight_layout()
fig.savefig(f"{RESULTS}/fig3_random_vs_loco.png", dpi=150)

print("\nDone. Tables and figures saved in the results/ folder.")
if SHOW_PLOTS:
    plt.show()  # opens all three figures; the script ends when you close them