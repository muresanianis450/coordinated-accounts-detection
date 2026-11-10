# evaluate_features.py
# ---------------------------------------------------------------------------
# GOAL: test whether behavioral features detect IO accounts on campaigns the
# model has never seen (leave-one-campaign-out, LOCO), and compare every
# feature set with a probe that sees ONLY the observation window length.
#
# Headline numbers per held-out campaign:
#   pr_auc = average precision (mean over 3 models)
#   lift   = pr_auc / dummy, where dummy = share of IO accounts in that campaign
# Lift makes campaigns with different class ratios comparable (roughly).
# If a feature set does not beat the window probe, it has not shown behavior
# detection, because the window is a collection artifact.
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
from config import *                       # OUT, SEED, RUN_TAG

# ---------------------------- SETTINGS -------------------------------------
WEIGHTING = "equal"          # "equal": every training campaign counts the same
                             # "none":  every account counts the same (old behavior)
N_BOOT = 200                 # bootstrap repetitions for the intervals
MIN_IO_RELIABLE = 20         # campaigns with fewer IO accounts are flagged "small"
RUN_IMPORTANCE = False       # permutation importance is slow, switch on later
N_REPEATS = 5                # shuffles per feature in permutation importance
SHOW_PLOTS = False           # True: open the figures at the end

RESULTS = Path(f"results_{RUN_TAG}")
RESULTS.mkdir(exist_ok=True)
print(f"Run tag: {RUN_TAG} -> reading {OUT}/, saving in {RESULTS}/ (weighting={WEIGHTING})")

# Load the account-level tables made by build_features.py
data = pd.concat(pd.read_parquet(p) for p in OUT.glob("*.parquet")).reset_index(drop=True)
CAMPAIGNS_PRESENT = sorted(data.campaign.unique())
print(f"{len(CAMPAIGNS_PRESENT)} campaigns, {len(data)} accounts, {int(data.label.sum())} IO")

# ---------------------------- FEATURE SETS ---------------------------------
# Main set: features whose tie to the window is moderate or unreliable.
MAIN = ["burstiness", "text_len_mean", "url_rate",
        "dup_text_share", "repost_share", "reply_share"]
# Shortcut group: most tied to the window across campaigns (correlation analysis).
SHORTCUT = ["hour_entropy", "gap_median_s", "frac_gap_lt_60s", "mention_rate"]
# Not used anywhere: hashtag_rate (controls were selected through hashtags),
# gap_cv (identical information to burstiness).

SETS = {
    "main": MAIN,
    "main + shortcut group": MAIN + SHORTCUT,
    "shortcut group only": SHORTCUT,
    "burstiness only": ["burstiness"],
    # Reference row: the window length alone, a collection artifact.
    "window length only (probe)": ["window_days"],
    "window + main": ["window_days"] + MAIN,
    "window + main + shortcut": ["window_days"] + MAIN + SHORTCUT,
}
PROBE = "window length only (probe)"

# Fail early if a column is missing (typo in a feature name)
needed = {c for cols in SETS.values() for c in cols} | {"label", "campaign"}
missing = needed - set(data.columns)
assert not missing, f"Columns missing in the processed table: {missing}"

MODEL_NAMES = ["logreg", "rf", "hgb"]


def models():
    """Fresh untrained models."""
    imp = lambda: SimpleImputer(strategy="median")     # fills NaN with the column median
    return {
        "logreg": make_pipeline(imp(), StandardScaler(),
                                LogisticRegression(class_weight="balanced", max_iter=2000)),
        "rf": make_pipeline(imp(), RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3, class_weight="balanced",
            random_state=SEED, n_jobs=-1)),
        # Gradient boosting handles NaN by itself
        "hgb": HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
    }


def campaign_weights(train):
    """Equal total weight for every training campaign.
    Each account gets 1 / (accounts in its campaign), rescaled to mean 1."""
    n = train.groupby("campaign").campaign.transform("size")
    w = 1.0 / n
    return (w / w.mean()).values


def fit_model(model, X, y, w):
    """Fit with or without sample weights. A Pipeline needs the weight argument
    addressed to its LAST step (e.g. 'randomforestclassifier__sample_weight')."""
    if WEIGHTING != "equal":
        model.fit(X, y)
    elif hasattr(model, "steps"):                      # a Pipeline
        model.fit(X, y, **{model.steps[-1][0] + "__sample_weight": w})
    else:                                              # HistGradientBoosting
        model.fit(X, y, sample_weight=w)


# ===========================================================================
# PART A: LOCO for every feature set
# ===========================================================================
rng = np.random.default_rng(SEED)                      # reproducible bootstrap


def loco_scores(cols):
    """Returns {campaign: (true labels, {model name: predicted scores})}."""
    out = {}
    for c in CAMPAIGNS_PRESENT:
        train, test = data[data.campaign != c], data[data.campaign == c]
        w = campaign_weights(train)
        scores = {}
        for mname in MODEL_NAMES:
            model = models()[mname]                    # new model for every fold
            fit_model(model, train[cols], train.label, w)
            scores[mname] = model.predict_proba(test[cols])[:, 1]
        out[c] = (test.label.values, scores)
    return out


def mean_ap(y, scores):
    """PR-AUC averaged over the models."""
    return np.mean([average_precision_score(y, s) for s in scores.values()])


def bootstrap_ci(y, scores):
    """95% bootstrap interval of the model-averaged PR-AUC (test-sample
    uncertainty only; it ignores variability of the training data)."""
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))          # draw accounts with replacement
        if y[idx].sum() == 0:                          # PR-AUC undefined without positives
            continue
        vals.append(np.mean([average_precision_score(y[idx], s[idx]) for s in scores.values()]))
    return np.percentile(vals, [2.5, 97.5])


print("\n=== A. LOCO results ===")
rows = []
for name, cols in SETS.items():
    print(f"  running: {name}")
    for c, (y, scores) in loco_scores(cols).items():
        if y.sum() == 0:
            continue                                   # no IO account, PR-AUC undefined
        lo, hi = bootstrap_ci(y, scores)
        dummy = y.mean()
        pr = mean_ap(y, scores)
        rows.append(dict(features=name, test=c, n_features=len(cols),
                         n_pos=int(y.sum()), dummy=dummy,
                         pr_auc=pr, lift=pr / dummy,
                         lift_low=lo / dummy, lift_high=hi / dummy,
                         small=int(y.sum()) < MIN_IO_RELIABLE))
res = pd.DataFrame(rows)
res.round(4).to_csv(RESULTS / f"model_results_{WEIGHTING}.csv", index=False)

# --- per-campaign lift table (rows = feature sets, columns = campaigns) -----
lift = res.pivot(index="features", columns="test", values="lift").reindex(list(SETS))
small = res.groupby("test").small.first()
lift.columns = [c + ("*" if small[c] else "") for c in lift.columns]   # * = fewer than 20 IO
print("\nLift over dummy per held-out campaign (* = fewer than 20 IO accounts):")
print(lift.round(2).to_string())

# --- macro-averages ---------------------------------------------------------
# Every campaign counts once, whatever its size. Shown for all campaigns and for
# the reliable ones only (>= 20 IO accounts).
rel = res[~res.small]
macro = pd.DataFrame({
    "macro_lift_all": res.groupby("features").lift.mean(),
    "macro_lift_reliable": rel.groupby("features").lift.mean(),
    "median_lift_reliable": rel.groupby("features").lift.median(),
    "macro_pr_auc_reliable": rel.groupby("features").pr_auc.mean(),
}).reindex(list(SETS))

# --- how often does a set beat the window probe? (reliable campaigns) -------
lift_rel = rel.pivot(index="features", columns="test", values="lift")
macro["beats_probe"] = [
    f"{int((lift_rel.loc[s] > lift_rel.loc[PROBE]).sum())}/{lift_rel.shape[1]}"
    for s in SETS
]
print(f"\nMacro-averages ({len(lift_rel.columns)} reliable campaigns):")
print(macro.round(2).to_string())
macro.round(3).to_csv(RESULTS / f"macro_{WEIGHTING}.csv")
print("\nReminder: Egypt has more IO than control accounts (dummy about 0.56),"
      " so its lift is capped near 1.8 and not comparable with the others.")


# ===========================================================================
# PART B: PERMUTATION IMPORTANCE (optional, slow)
# Model: random forest on main + shortcut group, reliable campaigns only.
# A big drop in PR-AUC when a feature is shuffled = the model relies on it.
# Correlated features share their importance, so read it as a rough guide.
# ===========================================================================
if RUN_IMPORTANCE:
    print("\n=== B. Permutation importance (rf) ===")
    FEATS = MAIN + SHORTCUT
    reliable = [c for c in CAMPAIGNS_PRESENT if not small[c]]
    rows = []
    for c in reliable:
        train, test = data[data.campaign != c], data[data.campaign == c]
        model = models()["rf"]
        fit_model(model, train[FEATS], train.label, campaign_weights(train))
        r = permutation_importance(model, test[FEATS], test.label,
                                   scoring="average_precision",
                                   n_repeats=N_REPEATS, random_state=SEED)
        for f, m in zip(FEATS, r.importances_mean):
            rows.append(dict(feature=f, test=c, importance=m))
    piv = pd.DataFrame(rows).pivot(index="feature", columns="test", values="importance")
    piv["mean"] = piv.mean(axis=1)
    piv = piv.sort_values("mean", ascending=False)
    print(piv.round(3).to_string())
    piv.round(4).to_csv(RESULTS / "permutation_importance.csv")


# ===========================================================================
# PART C: FIGURE, lift per campaign with 95% intervals
# One panel per held-out campaign. Dot = lift, bar = interval, dashed line =
# 1 (the dummy). The first row (main) is red.
# ===========================================================================
ncols = 5
nrows = int(np.ceil(len(CAMPAIGNS_PRESENT) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows),
                         sharex=False, squeeze=False)
names = list(SETS)
for ax, c in zip(axes.flat, CAMPAIGNS_PRESENT):
    sub = res[res.test == c].set_index("features")
    if sub.empty:
        ax.axis("off")
        continue
    for k, name in enumerate(names):
        r = sub.loc[name]
        err = [[max(r.lift - r.lift_low, 0)], [max(r.lift_high - r.lift, 0)]]
        ax.errorbar(r.lift, k, xerr=err, fmt="o", capsize=3,
                    color="#d62728" if k == 0 else "#1f77b4")
    ax.axvline(1, ls="--", color="black", lw=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names if c == CAMPAIGNS_PRESENT[0] or ax in axes[:, 0] else [], fontsize=7)
    ax.invert_yaxis()
    ax.set_title(f"{c} ({int(sub.n_pos.iloc[0])} IO)", fontsize=9)
for ax in axes.flat[len(CAMPAIGNS_PRESENT):]:
    ax.axis("off")
fig.supxlabel("lift = PR-AUC / dummy (dashed line = dummy)")
fig.tight_layout()
fig.savefig(RESULTS / f"fig_lift_{WEIGHTING}.png", dpi=130)

print(f"\nDone. Tables and figures saved in {RESULTS}/")
if SHOW_PLOTS:
    plt.show()