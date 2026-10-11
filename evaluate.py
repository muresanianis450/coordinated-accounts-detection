# ------------------------------------------------------------------
# GOAL: measure how well a classifier trained on some campaigns detects
# information-operation (IO) accounts in a campaign it has NEVER seen.
# This is "leave-one-campaign-out" (LOCO) cross-validation:
#   for each campaign C: train on all other campaigns, test on C.
# It is stricter than a random split, because a random split would put
# accounts of the same campaign in both train and test, which inflates scores.
# ------------------------------------------------------------------

import pandas as pd
from sklearn.dummy import DummyClassifier           # "no skill" baseline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer            # fills missing values
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline          # chains steps (impute -> scale -> model)
from sklearn.preprocessing import StandardScaler    # rescales features to mean 0, std 1
from config import *                                # OUT, SEED

# Load every processed campaign made by build_features.py and stack them
# into one table (one row per account, with "campaign" and "label" columns).

data = pd.concat(pd.read_parquet(p) for p in OUT.glob("*.parquet"))


# ----------- Feature groups ----------------------------------------
# Behavior features: describe WHAT the account does (timing, content, activity)
BEHAVIOR = ["hour_entropy","gap_median_s","gap_cv","burstiness","frac_gap_lt_60s",
            "dup_text_share","text_len_mean","hashtag_rate","url_rate",
            "mention_rate","repost_share","reply_share"]

# Metadata features: may reflect HOW the data was collected (leakage risk)
META = ["n_posts_total","followers","following","span_days","account_age_days"]

# The three experiments (ablation):
#   behavior -> the real goal
#   meta_only -> if this scores high, collection artifacts carry the label
#   behavior + meta -> everything together
GROUPS = {"behavior" : BEHAVIOR, "meta_only" : META,"behavior+meta": BEHAVIOR + META}

def models():
    """Creates fresh (untrained) models. Called again for every fold, so no
    model keeps what it learned in a previous fold"""
    # lambda = a tiny function; each call gives a new imputer.
    # Median imputation replaces missing features values (NaN) by the column median,
    # needed because e.g. gap_cv is NaN for some accounts and sklearn models
    # cannot handle NaN (except HistGradientBoosting, which handles it itself).
    imp = lambda: SimpleImputer(strategy="median")
    return {
        # Always model. Needs scaled features, hence StandardScaler
        # It tells you what "knowing nothing" scores.
        "dummy": DummyClassifier(strategy="prior"),
        # Linear model. Needs scaled features, hence StandardScaler.
        # class_weight="balance" gives the rare IO class more weight, so the
        # model does not just predict "control" for everyone.
        "logreg": make_pipeline(imp(), StandardScaler(),
                                LogisticRegression(class_weight="balanced", max_iter=2000)),
        # Ensemble of 300 decision trees; n_jobs=-1 uses all CPU cores.
        "rf": make_pipeline(imp(),RandomForestClassifier(
            n_estimators=300, class_weight="balanced",random_state=SEED,n_jobs=-1)),
        # Gradient boosting (scikit-learn's fast version); handles Nan natively.
        "hgb": HistGradientBoostingClassifier(class_weight="balanced",random_state=SEED),
    }

rows = [] # one dictionary of results per (campaign, features, model) combination

# Loop 1: which campaign is held out as the TEST set
for held_out in data.campaign.unique():
    train = data[data.campaign != held_out]    # all other campaigns
    test = data[data.campaign == held_out]     # the unseen campaigns

    # Loop 2: which feature group to use
    for gname, cols in GROUPS.items():
        # Loop 3: which model
        for mname, model in models().items():
            # Learn from the training campaigns (feature X, labels y)
            model.fit(train[cols], train.label)
            # Score for each test account: predicted probability of being IO.
            # predict_proba returns [P(control), P(IO); [:, 1] keeps P(IO).
            s = model.predict_proba(test[cols])[:, 1]
            rows.append(dict(
                test=held_out, features=gname, model=mname,
                n_pos=test.label.sum(),                     # IO accounts is the test campaign
                n=len(test),                                 # all accounts in the test campaign
                # PR-AUC (average precision): area under the precision-recall curve.
                # Best metric for imbalance data; compare with the dummy value.
                pr_auc=average_precision_score(test.label, s),
                # ROC_AUC: chance that a random IO account is ranked above a random
                # control (0.5 = random guessing, 1.0 = perfect)
                roc_auc=roc_auc_score(test.label, s)))


# Collect all results in a table, round to 3 decimals, print and save as CSV
res = pd.DataFrame(rows).round(3)
print(res.to_string(index=False))
res.to_csv("results_loco.csv", index=False)
