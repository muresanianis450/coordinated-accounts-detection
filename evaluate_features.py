# evaluate_features.py
# ---------------------------------------------------------------------------
# GOAL: find out WHICH behavioral features carry the signal.
#   A. FEATURE-GROUP ABLATION: remove or keep whole groups (timing, content,
#      activity) and see how the leave-one-campaign-out (LOCO) score changes.
#      Includes a test without the two "posting speed" features, to check that
#      the signal is not just "IO accounts post faster".
#   B. PERMUTATION IMPORTANCE: shuffle one feature at a time in the held-out
#      campaign and measure how much the score drops.
#
# Run after build_features.py. Results go in a folder named after the settings
# in config.py (e.g. results_min30_cap30_1/), so different settings never
# overwrite each other.
# ---------------------------------------------------------------------------

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import *                       # OUT, SEED, MIN_POSTS, CAP_POSTS

N_BOOT = 500                               # bootstrap repetitions for the intervals
N_REPEATS = 20                             # shuffles per feature in permutation importance
IMPORTANCE_MODEL = "rf"                    # model used for permutation importance
SHOW_PLOTS = True                       # True: open the figures at the end

# Output folder named after all experiment settings.
RESULTS = Path(f"results_{RUN_TAG}")
RESULTS.mkdir(exist_ok=True)
print(f"Run tag: {RUN_TAG} -> reading {OUT}/, saving in {RESULTS}/")

# Load the account-level tables made by build_features.py
data = pd.concat(pd.read_parquet(p) for p in OUT.glob("*.parquet")).reset_index(drop=True)
CAMPAIGNS_PRESENT = sorted(data.campaign.unique())

# ---- Behavioral features, split into the three groups used in build_features.py
TIMING = ["hour_entropy", "gap_median_s", "gap_cv", "burstiness", "frac_gap_lt_60s"]
CONTENT = ["dup_text_share", "text_len_mean", "hashtag_rate", "url_rate", "mention_rate"]
ACTIVITY = ["repost_share", "reply_share"]
BEHAVIOR = TIMING + CONTENT + ACTIVITY

# "Speed" features: they partly measure how fast an account posts, and IO
# accounts post faster. If the signal survives without them, it is not just speed.
SPEED = ["gap_median_s", "frac_gap_lt_60s"]


def drop(cols, remove):
    """Return cols without the features listed in remove."""
    return [c for c in cols if c not in remove]


# The feature sets to compare (the first one is the reference)
SETS = {
    "behavior (all)": BEHAVIOR,
    "timing only": TIMING,
    "content only": CONTENT,
    "activity only": ACTIVITY,
    "behavior without timing": CONTENT + ACTIVITY,
    "behavior without content": TIMING + ACTIVITY,
    "behavior without activity": TIMING + CONTENT,
    "behavior without speed features": drop(BEHAVIOR, SPEED),
    "timing without speed features": drop(TIMING, SPEED),

    # Diagnostic experiment: test window length alone.
    "window length only (probe)": ["window_days"],
}

MODEL_NAMES = ["logreg", "rf", "hgb"]


def models():
    """Fresh untrained models (same settings as in evaluate.py)."""
    imp = lambda: SimpleImputer(strategy="median")     # fills NaN with the column median
    return {
        "logreg": make_pipeline(imp(), StandardScaler(),
                                LogisticRegression(class_weight="balanced", max_iter=2000)),
        "rf": make_pipeline(imp(), RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)),
        # Gradient boosting handles NaN by itself, so no imputer and no pipeline here
        "hgb": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    }


# ===========================================================================
# PART A: FEATURE-GROUP ABLATION
# For every feature set: leave-one-campaign-out with all three models.
# Because model rankings are unstable with so few positives, the headline
# number is the PR-AUC AVERAGED over the three models.
# ===========================================================================
print("\n=== A. Feature-group ablation (mean PR-AUC over logreg, rf, hgb) ===")
rng = np.random.default_rng(SEED)                      # reproducible bootstrap


def loco_scores(cols):
    """LOCO for one feature set and all models.
    Returns {campaign: (true labels, {model name: predicted scores})}."""
    out = {}
    for c in CAMPAIGNS_PRESENT:
        train, test = data[data.campaign != c], data[data.campaign == c]
        scores = {}
        for mname in MODEL_NAMES:
            model = models()[mname]                    # new model for every fold
            model.fit(train[cols], train.label)
            scores[mname] = model.predict_proba(test[cols])[:, 1]
        out[c] = (test.label.values, scores)
    return out


def mean_ap(y, scores):
    """PR-AUC averaged over the models."""
    return np.mean([average_precision_score(y, s) for s in scores.values()])


def bootstrap_ci(y, scores):
    """95% bootstrap interval of the model-averaged PR-AUC. The SAME resampled
    accounts are used for all models in each repetition."""
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))          # draw accounts with replacement
        if y[idx].sum() == 0:                          # PR-AUC is undefined without positives
            continue
        vals.append(np.mean([average_precision_score(y[idx], s[idx]) for s in scores.values()]))
    return np.percentile(vals, [2.5, 97.5])


rows = []
for name, cols in SETS.items():
    for c, (y, scores) in loco_scores(cols).items():
        lo, hi = bootstrap_ci(y, scores)
        row = dict(features=name, test=c, n_features=len(cols), n_pos=int(y.sum()),
                   pr_auc=mean_ap(y, scores), ci_low=lo, ci_high=hi, dummy=y.mean())
        for mname, s in scores.items():                # also keep each model's own score
            row[f"pr_auc_{mname}"] = average_precision_score(y, s)
        rows.append(row)
abl = pd.DataFrame(rows).round(3)
print("Dummy baseline:", abl.groupby("test").dummy.first().to_dict())
print(abl.pivot(index="features", columns="test", values="pr_auc")
         .reindex(list(SETS)).to_string())
abl.to_csv(RESULTS / "feature_ablation.csv", index=False)


# ===========================================================================
# PART B: PERMUTATION IMPORTANCE
# Train on the other campaigns, then on the held-out campaign shuffle one
# feature's values (breaking its link to the label) and measure how much the
# PR-AUC drops. A big drop = the model relies on that feature.
# Caveats: correlated features share their importance (each looks less
# important than it is), and with few positives the numbers are noisy.
# A value near 0 or below 0 means the feature does not help on that campaign.
# ===========================================================================
print("\n=== B. Permutation importance (drop in PR-AUC, model = %s) ===" % IMPORTANCE_MODEL)

rows = []
for c in CAMPAIGNS_PRESENT:
    train, test = data[data.campaign != c], data[data.campaign == c]
    model = models()[IMPORTANCE_MODEL]
    model.fit(train[BEHAVIOR], train.label)
    r = permutation_importance(model, test[BEHAVIOR], test.label,
                               scoring="average_precision",
                               n_repeats=N_REPEATS, random_state=SEED)
    for f, m in zip(BEHAVIOR, r.importances_mean):
        rows.append(dict(feature=f, test=c, importance=m))
imp = pd.DataFrame(rows)
piv = imp.pivot(index="feature", columns="test", values="importance")
piv["mean"] = piv.mean(axis=1)                         # average over campaigns
piv = piv.sort_values("mean", ascending=False)         # most important first
print(piv.round(3).to_string())
piv.round(4).to_csv(RESULTS / "permutation_importance.csv")


# ===========================================================================
# PART C: FIGURES
# ===========================================================================
# ---- Figure 1: ablation with 95% bootstrap intervals -------------------------
# One panel per held-out campaign. Each row is a feature set. The dot is the
# model-averaged PR-AUC, the bar is the interval, the dashed line is the dummy.
ncols = min(3, len(CAMPAIGNS_PRESENT))
nrows = int(np.ceil(len(CAMPAIGNS_PRESENT) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.0 * nrows),
                         sharex=True, squeeze=False)
names = list(SETS)
for ax, c in zip(axes.flat, CAMPAIGNS_PRESENT):
    sub = abl[abl.test == c].set_index("features")
    for k, name in enumerate(names):
        r = sub.loc[name]
        # max(..., 0) avoids a negative bar length if the interval does not contain the point value
        err = [[max(r.pr_auc - r.ci_low, 0)], [max(r.ci_high - r.pr_auc, 0)]]
        # The reference set (all behavior) is drawn in a different color
        ax.errorbar(r.pr_auc, k, xerr=err, fmt="o", capsize=3,
                    color="#d62728" if k == 0 else "#1f77b4")
    ax.axvline(sub.dummy.iloc[0], ls="--", color="black", lw=1)    # dummy baseline
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()                                              # first row at the top
    ax.set_xlim(0, 1.02)
    ax.set_title(f"{c} ({int(sub.n_pos.iloc[0])} IO accounts)")
    ax.set_xlabel("mean PR-AUC over 3 models (dashed = dummy)")
for ax in axes.flat[len(CAMPAIGNS_PRESENT):]:                      # hide unused panels
    ax.axis("off")
fig.tight_layout()
fig.savefig(RESULTS / "fig_ablation.png", dpi=150)

# ---- Figure 2: permutation importance heatmap -------------------------------
# Rows = features (most important first), columns = held-out campaigns.
# Red = shuffling the feature lowers the score (the model uses it).
# Blue = shuffling does not hurt or even helps (the feature adds noise there).
heat = piv.drop(columns="mean")
vmax = max(np.abs(heat.values).max(), 1e-6)            # symmetric color scale around 0
fig, ax = plt.subplots(figsize=(2 + 1.3 * heat.shape[1], 0.45 * heat.shape[0] + 1.8))
im = ax.imshow(heat.values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(heat.shape[1]))
ax.set_xticklabels(heat.columns, rotation=45, ha="right")
ax.set_yticks(range(heat.shape[0]))
ax.set_yticklabels(heat.index)
if heat.shape[1] <= 8:                                 # write numbers in the cells (few campaigns only)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, label="drop in PR-AUC when shuffled")
ax.set_title(f"Permutation importance ({IMPORTANCE_MODEL}, behavior features)")
fig.tight_layout()
fig.savefig(RESULTS / "fig_permutation_importance.png", dpi=150)

print(f"\nDone. Tables and figures saved in {RESULTS}/")
if SHOW_PLOTS:
    plt.show()          # opens the figures; the script ends when you close them