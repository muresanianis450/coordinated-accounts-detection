# Context update: 11 campaigns (13 Oct 2026)

How to use: replace the same-numbered sections in `context` / `project_context.md` with the sections below. Header line: "Last updated: 13 October 2026 (11 campaigns, floor-only run `min30_cap30_floor_11c`)".

Everything numeric below was computed from the printed outputs of `build_features.py`, `evaluate.py`, `evaluate_extra.py` and `evaluate_features.py` (run `min30_cap30_floor_11c`). Single runs, no matching, no `check_timestamps.py` yet.

---

## 5 (replace the "Available locally" bullet). Data available locally: 11 of 26 campaigns

Parquet files in `...\Theses\Data\Extracted\<Campaign>\*.parquet`. Counts before and after `MIN_POSTS = 30`:

| Campaign | IO before | Control before | IO kept | Control kept | Positive rate (dummy PR-AUC) |
|---|---|---|---|---|---|
| Bangladesh | 11 | 929 | 8 | 347 | 0.023 |
| Armenia | 31 | 1,767 | 19 | 721 | 0.026 |
| Catalonia | 76 | 2,607 | 30 | 1,164 | 0.025 |
| China_1 | 699 | 42,120 | 440 | 18,991 | 0.023 |
| China_2 | 191 | 35,806 | 166 | 17,444 | 0.009 |
| Cuba | 503 | 30,099 | 437 | 12,908 | 0.033 |
| Ecuador | 787 | 21,137 (expected 21,138) | 416 | 7,915 | 0.050 |
| Egypt | 240 | 370 | 214 | 168 | **0.560** |
| Iran_1 | 660 | 5,015 | 540 | 2,136 | **0.202** |
| Iran_2 | 389 | 6,615 | 286 | 3,169 | 0.083 |
| Iran_3 | 210 | 2,240 | 192 | 944 | **0.169** |
| **Total** | **3,797** | **148,706** | **2,748 (72%)** | **65,907 (44%)** | **0.040 overall** |

Total accounts kept: 68,655.

Notes:
- `MIN_POSTS = 30` removes 28% of IO accounts but 56% of controls.
- Ecuador control count is 21,137, one fewer than Table 1 (21,138). Minor; check why (one account missing an id or lost in loading) and mention it.
- Egypt has a very different design (370 controls, 56% IO after filtering). Its PR-AUC cannot be compared with other campaigns; read ROC-AUC or "skill over dummy" (section 12).
- Iran_1 and Iran_3 also have high positive rates (0.20, 0.17).
- Bangladesh (8 IO) is too small to be a test set; keep it for training and report it only as a footnote.
- Remaining 15 campaigns: the coordinator would have to request them (Zenodo access policy).

---

## 11 (replace). Code status and reference: what each file calculates

Environment: Windows, PyCharm, Python 3.12, venv at `C:\Users\mures\PycharmProjects\PythonProject\Theses\.venv`. Packages: pandas, pyarrow, scikit-learn, matplotlib. All scripts live in the project root and read their settings from `config.py`. Current run tag: `min30_cap30_floor_11c`.

### 11.1 Files

| File | What it calculates | Output |
|---|---|---|
| `config.py` | Nothing. Holds paths, `CAMPAIGNS`, `EXPECTED` (IO, control account counts from Table 1 of the paper), `SEED=42`, `MIN_POSTS`, `CAP_POSTS`, `FLOOR_TO_MINUTE`, `MATCH_WINDOWS`, `WINDOWS_EDGES_DAYS`, `MIN_PER_CLASS_PER_BIN`, `RUN_TAG`, `OUT` | none |
| `build_features.py` | Turns raw posts into **one row per account**. Checks account counts against `EXPECTED`; drops accounts with fewer than `MIN_POSTS` posts; computes behavior features from each account's first `CAP_POSTS` posts; computes metadata columns and the diagnostic `window_days`; attaches label (1 = IO, 0 = control) and campaign. Optional minute-flooring and window matching | `data/processed_<RUN_TAG>/` |
| `evaluate.py` | Basic leave-one-campaign-out (LOCO) for three feature sets (behavior, meta_only, behavior+meta) and four models (dummy, logreg, rf, hgb). Reports PR-AUC and ROC-AUC per test campaign | `results_loco.csv` |
| `evaluate_extra.py` | **A** leakage diagnosis: single metadata features and combinations (rf). **B** 95% bootstrap intervals (500 resamples of test accounts). **C** random 5-fold split vs LOCO, per campaign, with the difference ("inflation"). **D** figures | `results_<RUN_TAG>/` |
| `evaluate_features.py` | **A** feature-group ablation (behavior, timing, content, activity, without timing/content/activity/speed, window-length probe), mean PR-AUC over logreg, rf, hgb. **B** permutation importance (rf, behavior features) | `results_<RUN_TAG>/` |
| `check_timestamps.py` | Raw-timestamp diagnostic: seconds-resolution, exact midnight, zero gaps, window length of the first posts, median date, by class. **Not yet run on the 11 campaigns** | `results_min{MIN}_cap{CAP}/` |

Run order: edit `config.py`, run `build_features.py`, then the evaluation scripts. Rebuild after every settings change. The last part of `RUN_TAG` (`_11c`) is the number of campaigns, so runs never overwrite each other.

### 11.2 Feature groups
- **Timing:** `hour_entropy`, `gap_median_s`, `gap_cv`, `burstiness`, `frac_gap_lt_60s`.
- **Content:** `dup_text_share` (exact duplicates only), `text_len_mean`, `hashtag_rate`, `url_rate`, `mention_rate`.
- **Activity:** `repost_share`, `reply_share`.
- **Speed features:** `gap_median_s`, `frac_gap_lt_60s` (subset of timing).
- **Meta (leakage risk, kept out of behavior models):** `n_posts_total`, `followers`, `following`, `span_days`, `account_age_days`.
- **`window_days`:** length in days of the window covered by the first `CAP_POSTS` posts. Diagnostic probe only, never a model feature in the behavior set.

### 11.3 Models and how they behave
All use `class_weight="balanced"`, no hyperparameter tuning, seed 42. Headline numbers are the mean over logreg, rf and hgb, because the ranking of single models is unstable with few positives.

| Model | What it is | Behavior observed here |
|---|---|---|
| **Dummy** | Ranks accounts at random; PR-AUC equals the positive rate, ROC-AUC 0.5 | The floor. Differs a lot by campaign (0.009 China_2 to 0.56 Egypt), so PR-AUC must be read against it |
| **Logistic regression** | Linear model of the features, one weight per feature | Weakest on behavior in some campaigns (Cuba 0.070, Armenia 0.105) because timing effects are non-linear. Hardly benefits from a random split (inflation below 0.05 everywhere), so it cannot memorize campaign-specific patterns. Useful as a sanity check |
| **Random forest (rf)** | Many decision trees on random subsets, averaged | Best or near-best in LOCO. Under a random split it gains a lot in the large campaigns (inflation +0.32 China_1, +0.25 Cuba, +0.34 Iran_3), i.e. it memorizes campaign-specific structure |
| **Histogram gradient boosting (hgb)** | Trees added one at a time, each fixing the previous errors | Similar to rf in LOCO and similar inflation (+0.27 China_1, +0.29 Iran_3). Gets the highest scores on metadata (0.96 to 0.99 in several campaigns) |

Exact settings (confirmed from the code): **logreg** = median imputer, then `StandardScaler`, then `LogisticRegression(class_weight="balanced", max_iter=2000)`; **rf** = median imputer, then `RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1)`; **hgb** = `HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)` with no imputer (it handles missing values itself). The dummy model exists only in `evaluate.py`, which I have not seen.

### 11.4 Functions: what each one does exactly

Written from the code you pasted (`build_features.py`, `evaluate_extra.py`, `evaluate_features.py`; the second paste of `evaluate_features.py` is identical to the first). **Still missing:** `evaluate.py`, `check_timestamps.py`, `config.py`. Their sections below stay as stubs until the code is pasted.

#### `build_features.py` (raw posts to one row per account)

Module level: `COLS` = the 12 parquet columns that are loaded (all others are ignored to save memory). `URL_RE` matches `https?://\S+`, `MENT_RE` matches `@\w+`.

| Function | Input to output | What it does exactly |
|---|---|---|
| `norm(s)` | text to text | Lowercases, removes URLs, removes `@mentions`, strips outer spaces. Used only for `dup_text_share`. Punctuation and hashtags are kept, so "vote now!" and "vote now" count as different texts |
| `n_items(x)` | list-like cell to int | Length of a hashtags/urls/mentions cell; 0 if the cell is `None`. Does not handle `NaN` floats (works on your data, would fail on a different export) |
| `load_campaign(name)` | campaign name to DataFrame | Finds all `*.parquet` files in `DATA_ROOT/<name>`, reads only `COLS`, stacks them. Raises `FileNotFoundError` if none. All parts are combined **before** any feature is computed, because one account's posts can be spread over several part files |
| `meta_features(df)` | all posts to one row per account | Groups by `accountid`. `is_control` (first value), `n_posts_total` (number of posts, **uncapped**), first and last post time, `followers` and `following` (median over the account's posts), `created` (first value). Then `span_days = (last - first).dt.days` (whole days, truncated) and `account_age_days = (last - created).dt.days`. Helper columns dropped. These are the leakage-risk columns |
| `behavior_row(g)` | one account's capped posts to dict | Sorts by time, takes the 29 gaps between consecutive posts (when `CAP_POSTS = 30`), and returns the features below |
| `window_table(capped)` | capped posts to one row per account | First time, last time, class, `window_days = (last - first)` in days, and `bin = pd.cut(window_days, WINDOWS_EDGES_DAYS, include_lowest=True)`. An account whose window falls outside all edges gets a `NaN` bin |
| `match_windows(capped)` | capped posts to capped posts | Counts IO and control accounts per window bin, keeps bins with at least `MIN_PER_CLASS_PER_BIN` of **each** class, prints the table, returns only the posts of accounts in kept bins. Guarantees overlap only, not equal class ratios. Accounts with a `NaN` bin are dropped silently |
| `count_classes(capped)` | capped posts to (n_IO, n_control) | Used only for the printed counts |
| `build(name)` | campaign name to parquet file | The pipeline, in this order: (1) load; (2) if `FLOOR_TO_MINUTE`, round every timestamp down to the minute; (3) `meta_features` on all posts; (4) print IO/control counts next to `EXPECTED`; (5) keep accounts with `n_posts_total >= MIN_POSTS`; (6) take each kept account's first `CAP_POSTS` posts by time; (7) print counts "after MIN_POSTS"; (8) if `MATCH_WINDOWS`, apply `match_windows` and print counts; (9) `behavior_row` for every account; (10) inner-join meta and behavior tables; (11) `label = 1` for IO (`~is_control`), add `campaign`, drop `is_control`; (12) skip the campaign if it has no IO accounts; (13) save `OUT/<name>.parquet` and print the number of accounts and positives |
| `__main__` | none | Prints the run tag and output folder, then calls `build` for every name in `CAMPAIGNS` |

Features returned by `behavior_row` (exact definitions):

| Feature | Definition |
|---|---|
| `hour_entropy` | Shannon entropy in bits of the share of posts per hour of day, `-sum(p * log2 p)`; maximum 4.585 (= log2 24). Hour is taken from the timestamp as stored (time zone not verified) |
| `gap_median_s` | Median of the gaps in seconds. With minute flooring it is a multiple of 60 |
| `gap_cv` | `std / mean` of the gaps; `NaN` if the mean is 0 (for example all posts in the same minute) |
| `burstiness` | `(std - mean) / (std + mean)`; `NaN` if the denominator is 0 |
| `frac_gap_lt_60s` | Share of gaps shorter than 60 s. After minute flooring this equals the share of consecutive posts in the same minute |
| `dup_text_share` | `1 - (distinct normalized texts) / (non-empty normalized texts)` = share of posts that repeat an earlier text; exact match only |
| `text_len_mean` | Mean number of characters of the raw text (missing text counts as length 0) |
| `hashtag_rate`, `url_rate`, `mention_rate` | Mean number of hashtags / URLs / mentions per post |
| `repost_share` | Mean of `is_repost` |
| `reply_share` | Share of posts whose `in_reply_to_postid` is filled |
| `window_days` | `(max - min)` of the capped post times, in days. **Diagnostic only**, never in `BEHAVIOR` |

#### `evaluate_extra.py` (leakage, intervals, random split)

Setup: `N_BOOT = 500`, `LEAK_MODEL = "rf"`, results folder `results_<RUN_TAG>`. Loads all processed parquet files into one table `data` (index reset, because `cross_val_predict` uses row positions). Defines `BEHAVIOR`, `META`, `GROUPS` (behavior, meta_only, behavior+meta) and `MODEL_NAMES = [logreg, rf, hgb]`.

| Function / block | What it does exactly |
|---|---|
| `models()` | Returns a fresh dictionary of untrained models (settings in 11.3). Called again for every fit so nothing leaks between folds |
| `loco_predict(cols, mname)` | For each campaign: train on all other campaigns, test on the held-out one; fit; take the predicted probability of the IO class. Returns `{campaign: (true labels, scores)}` |
| Part A | Prints the positive rate per campaign (= dummy). For each feature set in `LEAK_SETS` (each metadata column alone, all meta, meta without `n_posts_total`, behavior, behavior + meta without `n_posts_total`) runs `loco_predict` with rf and records PR-AUC. Pivot table saved as `leakage_diagnosis.csv` |
| `bootstrap_ci(y, s)` | Draws 500 resamples of the test accounts with replacement, recomputes PR-AUC, skips resamples without positives, returns the 2.5% and 97.5% percentiles. Covers test-sample uncertainty only. Uses one global random generator (seed 42), so the exact interval values depend on call order |
| Part B | For 3 feature groups x 3 models x campaigns: PR-AUC, interval, number of positives, dummy (`y.mean()`). Saved as `loco_with_ci.csv` |
| Part C | Stratifies on `campaign + "_" + label` and makes a 5-fold shuffled split over **all campaigns pooled**, so every fold has every campaign and both classes. `cross_val_predict` gives out-of-fold scores (each account scored by a model that never saw it). PR-AUC is computed per campaign on the same accounts as in LOCO. `inflation = random_split - LOCO`. Saved as `random_vs_loco.csv` |
| Part D | `fig1_leakage.png` (heatmap, feature set x campaign; numbers in cells only when 8 campaigns or fewer, so none with 11), `fig2_loco_ci.png` (one panel per campaign, dot = PR-AUC, bar = interval, dashed = dummy), `fig3_random_vs_loco.png` (scatter, points above the diagonal = random split scored higher). Opens the figures if `SHOW_PLOTS` |

#### `evaluate_features.py` (which features carry the signal)

Setup: `N_BOOT = 500`, `N_REPEATS = 20` (shuffles per feature), `IMPORTANCE_MODEL = "rf"`. Defines `TIMING`, `CONTENT`, `ACTIVITY`, `BEHAVIOR = TIMING + CONTENT + ACTIVITY`, `SPEED = [gap_median_s, frac_gap_lt_60s]`, and `SETS` (10 feature sets, listed in 12.1, including the probe `["window_days"]`).

| Function / block | What it does exactly |
|---|---|
| `drop(cols, remove)` | Returns `cols` without the names in `remove`; used to build the "without speed features" sets |
| `models()` | Same three models as in `evaluate_extra.py` |
| `loco_scores(cols)` | LOCO for one feature set with **all three models**; returns `{campaign: (true labels, {model: scores})}` |
| `mean_ap(y, scores)` | PR-AUC averaged over the three models (the headline number) |
| `bootstrap_ci(y, scores)` | 500 resamples of the test accounts; in each resample the **same** accounts are used for all three models, and the model-averaged PR-AUC is recorded; returns the 2.5% and 97.5% percentiles |
| Part A | For each set and campaign: `pr_auc` (model average), interval, dummy, number of features, number of positives, and each model's own score (`pr_auc_logreg`, `pr_auc_rf`, `pr_auc_hgb`). Printed pivot and `feature_ablation.csv`. The per-model columns are what you need to show how the models differ |
| Part B | For each held-out campaign: train rf on the other campaigns using `BEHAVIOR`, then `permutation_importance` on the held-out campaign (scoring = average precision, 20 shuffles). The importance is the mean drop in PR-AUC; averaged over campaigns and sorted. Saved as `permutation_importance.csv`. Caveat: correlated features split their importance |
| Part C | `fig_ablation.png` (panels per campaign, reference set in red) and `fig_permutation_importance.png` (coolwarm heatmap, numbers in cells only for 8 campaigns or fewer) |

#### Stubs (code not yet seen)
- `evaluate.py`: LOCO over three feature sets (behavior, meta_only, behavior+meta) and four models (dummy, logreg, rf, hgb); prints PR-AUC and ROC-AUC per test campaign (output format known from your run); writes `results_loco.csv`. Functions: to be filled.
- `check_timestamps.py`: raw timestamp diagnostic (resolution, midnight, zero gaps, window length, median date). Functions: to be filled.
- `config.py`: constants only, listed in 11.1.

### 11.5 Known small issues and code observations

Already fixed in the pasted `evaluate_extra.py`: `hgb` is now a plain `HistGradientBoostingClassifier`, and the Figure 1 title no longer hard-codes the dummy value. Remove these two from the to-do list.

Still open:
1. **Typos:** "positive rage" (print in part A), "buld_features.py" and "results/" in the header comment of `evaluate_extra.py`, "Acounts" in `match_windows`.
2. **Unstable sort at the cap.** `sort_values("post_time")` is not stable by default. After minute flooring many posts tie, so which tied posts fall inside the first 30 can vary. Use `sort_values("post_time", kind="stable")` in `build` and in `behavior_row`. Low impact, but it affects reproducibility.
3. **`NaN` window bins.** Accounts whose window exceeds the last edge in `WINDOWS_EDGES_DAYS` are silently dropped by `match_windows`. Check that the last edge is very large (or `np.inf`) in `config.py`.
4. **Figure cell numbers.** With 11 campaigns the heatmaps show no numbers (threshold is 8 campaigns). Raising the threshold to 12 and using font size 7 would show them.
5. **`frac_gap_lt_60s` after flooring** is exactly "share of same-minute pairs"; describe it that way in the thesis.
6. **Time zone** of `hour_entropy` is not verified (stored timestamps, probably UTC).
7. **Bootstrap generator** is shared across calls, so intervals shift slightly if the code order changes. Fine, but do not compare intervals digit by digit across versions.
8. `config.py`: check there is no stray `import glob`.
9. Git: commit status unknown; do not commit data or processed tables.

---

## 12 (replace). Results with 11 campaigns (floor only, MIN 30, CAP 30, LOCO)

### 12.1 Feature-group ablation, mean PR-AUC over logreg, rf, hgb
| Feature set | Armenia | Bangl.* | Catalonia | China_1 | China_2 | Cuba | Ecuador | Egypt | Iran_1 | Iran_2 | Iran_3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dummy | 0.026 | 0.023 | 0.025 | 0.023 | 0.009 | 0.033 | 0.050 | 0.560 | 0.202 | 0.083 | 0.169 |
| Behavior (all) | 0.297 | 0.777 | 0.906 | 0.405 | 0.517 | 0.160 | 0.592 | 0.922 | 0.791 | 0.529 | 0.489 |
| Timing only | 0.128 | 0.503 | 0.911 | 0.389 | 0.455 | 0.098 | 0.475 | 0.889 | 0.706 | 0.476 | 0.513 |
| Content only | 0.142 | 0.435 | 0.050 | 0.064 | 0.047 | 0.046 | 0.094 | 0.697 | 0.425 | 0.186 | 0.211 |
| Activity only | 0.040 | 0.023 | 0.020 | 0.020 | 0.012 | 0.043 | 0.059 | 0.641 | 0.236 | 0.105 | 0.157 |
| Behavior without speed | 0.212 | 0.517 | 0.156 | 0.089 | 0.166 | 0.095 | 0.226 | 0.870 | 0.605 | 0.297 | 0.354 |
| **Window length only (probe)** | 0.371 | 0.675 | 0.480 | 0.492 | 0.369 | 0.348 | 0.591 | 0.966 | 0.898 | 0.633 | 0.688 |

\*Bangladesh has 8 IO accounts: ignore.

### 12.2 "Skill over dummy" (my rescaling, not a standard metric)
skill = (PR-AUC − dummy) / (1 − dummy). It is 0 for the dummy and 1 for perfect, so campaigns with different positive rates become roughly comparable. Excluding Bangladesh (10 campaigns):

| Feature set | Mean skill | Range |
|---|---|---|
| Behavior (all) | 0.52 | 0.13 (Cuba) to 0.90 (Catalonia) |
| Behavior without speed features | 0.25 | 0.06 (Cuba) to 0.70 (Egypt) |
| Window length only (probe) | 0.56 | 0.33 (Cuba) to 0.92 (Egypt) |

Per campaign (behavior / without speed / probe): Armenia 0.28 / 0.19 / 0.35; Catalonia 0.90 / 0.13 / 0.47; China_1 0.39 / 0.07 / 0.48; China_2 0.51 / 0.16 / 0.36; Cuba 0.13 / 0.06 / 0.33; Ecuador 0.57 / 0.19 / 0.57; Egypt 0.82 / 0.70 / 0.92; Iran_1 0.74 / 0.51 / 0.87; Iran_2 0.49 / 0.23 / 0.60; Iran_3 0.39 / 0.22 / 0.62.

### 12.3 Metadata leakage (rf, PR-AUC)
- All five metadata columns: 0.67 (China_2) to 0.999 (Iran_3); above 0.89 in 7 of 11 campaigns.
- `n_posts_total` alone: 0.51 to 0.93 in 8 campaigns (e.g. Cuba 0.697 vs dummy 0.033, Iran_3 0.918 vs 0.169). `span_days` alone: Egypt 0.998, Iran_1 0.810, Bangladesh 0.821.
- `followers`, `following`, `account_age_days` alone: close to the dummy in most campaigns (exceptions: Egypt, Iran_1 followers 0.239 vs 0.202, Ecuador followers 0.144 vs 0.050).
- Metadata without `n_posts_total` is still 0.51 to 0.99, so `span_days` carries a lot, and combinations carry more.
- Behavior + metadata without `n_posts_total`: 0.64 to 1.00.

### 12.4 Random 5-fold split vs LOCO (behavior only)
| Model | Inflation (random − LOCO) |
|---|---|
| Logistic regression | below 0.05 in every campaign (largest China_1 +0.049) |
| Random forest | large in China_1 +0.32, Iran_3 +0.34, Cuba +0.25, Iran_2 +0.24, Ecuador +0.20; small elsewhere (Armenia, Bangladesh, Catalonia about 0; Egypt +0.03; Iran_1 +0.05; China_2 +0.06) |
| Hist. gradient boosting | same pattern, slightly smaller (China_1 +0.27, Iran_3 +0.29) |

### 12.5 Permutation importance (rf, behavior, mean over 11 campaigns)
`gap_median_s` 0.38, `hour_entropy` 0.11, `url_rate` 0.09, `hashtag_rate` 0.06, `repost_share` 0.04, `frac_gap_lt_60s` 0.03, others near 0. `dup_text_share` is about 0 (exact duplicates only).

### 12.6 Bootstrap intervals (95%, LOCO, test accounts only)
Narrow in large campaigns (e.g. China_1 rf 0.413 [0.372, 0.464]; Iran_1 rf 0.842 [0.818, 0.866]). Wide in the small ones (Armenia rf 0.340 [0.168, 0.604]; Bangladesh n_pos = 8).

---

## 13 (replace). Next steps
1. **Run `check_timestamps.py` on all 11 campaigns** and paste the summary table: do minute resolution (IO) and short control windows hold everywhere? If yes, it is a dataset-level finding for the coordinator.
2. **Stricter window matching** (fixed control-to-IO ratio per window bin; Bangladesh train-only). Success criterion: the probe falls close to the dummy. This is the decisive experiment: it shows whether any behavioral signal remains once the window is neutralized.
3. **Profile-only metadata test:** `followers`, `following`, `account_age_days` without `n_posts_total` and `span_days`, to separate profile leakage from volume/window leakage.
4. **Hashtag test:** content features without `hashtag_rate` (controls were selected through hashtags).
5. **Report skill over dummy and macro-average** (excluding Bangladesh; show Egypt separately); try equal campaign weights in training.
6. **Content/structure features that do not depend on timing:** near-duplicate text (TF-IDF cosine), repost/reply structure, then graph features (co-repost network from `reposted_accountid`, shared hashtags).
7. Fix issues in 11.5; commit code and result CSVs (not data).
8. **Coordinator meeting before 30.10.2026** (see 14.5 for the talking points).
9. **Lab1:** MSC and ACM CCS classification, Author Guidelines, reviewer form, ask the lab teacher what "conspectare" requires.
10. **Reading (by 27.11):** Seckin et al. in full, plus work on cross-campaign generalization and collection artifacts.
11. **Romanian case study:** check data volume (YouTube, Telegram, Bluesky); weak labels, aggregates only.

---

## 14.2 (add to threats to validity)
- **Egypt design:** very different class ratio (56% IO); scores not comparable with other campaigns.
- **Positive rates vary** from 0.9% to 56%; always report dummy and skill over dummy.
- **Matching not applied** in the 11-campaign run; all conclusions here are for the floor-only setting.
- **Rescaled "skill" is my own convenience metric**; define it in the thesis or use lift plus ROC-AUC.

## 14.3 (replace finding 1; add findings 11 to 15)
1. **(Revised.)** Metadata separates IO from controls far better than behavior. Single columns differ: `n_posts_total` and `span_days` are strong alone in most campaigns, `followers`, `following`, `account_age_days` are mostly weak alone. The earlier statement "not mainly post count" holds only for the combination result (metadata without `n_posts_total` is still high).
11. **Window length alone matches or beats all behavior features** in 7 of 10 campaigns with enough IO accounts (Armenia, China_1, Cuba, Egypt, Iran_1, Iran_2, Iran_3), ties in Ecuador, loses in Catalonia and China_2. Mean skill: probe 0.56 vs behavior 0.52.
12. **Removing speed features cuts mean skill from 0.52 to 0.25**; in Catalonia (0.90 to 0.13), China_1 (0.39 to 0.07) and China_2 (0.51 to 0.16) most of the signal disappears. Some signal remains in Egypt, Iran_1 and the other Iran campaigns, but those also have high positive rates and windows that separate easily.
13. **Random-split inflation is a tree-model effect:** logistic regression does not inflate, rf and hgb inflate by 0.2 to 0.34 in the large campaigns. Interpretation (provisional): trees exploit campaign-specific structure when the campaign is in training.
14. **More training campaigns helped Armenia** (behavior 0.07 with 4 campaigns, 0.30 with 11), but with the shortcut still present, this cannot yet be called real signal.
15. **Content-only and activity-only signals are weak** outside Egypt, Iran and Bangladesh (e.g. content-only China_1 0.064, Cuba 0.046).

## 14.5 Talking points for the coordinator
1. What I have: 11 of 26 campaigns, 68,655 accounts, 2,748 IO accounts after the 30-post filter.
2. The pipeline works and the evaluation is leave-one-campaign-out.
3. Headline: apparent detection comes largely from collection design (metadata, observation window, posting speed), not from behavior alone.
4. Proposed thesis: audit of this effect plus detectors based on less collection-dependent signals (content, repost/reply structure, graph).
5. Questions: (a) can you request the other 15 campaigns; (b) is audit-plus-detector acceptable for a bachelor's thesis; (c) how were control timelines collected and why do IO timestamps have minute resolution; (d) reuse of lab work, ethics for user-level data, template; (e) may processed account-level tables be shared.

## 15 (add rows)
| Date | RUN_TAG | Campaigns | Script | Key result | Note |
|---|---|---|---|---|---|
| 13 Oct 2026 | min30_cap30_floor_11c | 11 (added China_2, Cuba, Ecuador, Egypt, Iran_1, Iran_2, Iran_3) | build_features, evaluate, evaluate_extra, evaluate_features | Behavior mean skill 0.52, without speed 0.25, window probe 0.56 (10 campaigns, Bangladesh excluded); rf/hgb random-split inflation up to +0.34, logreg below 0.05 | Floor only, no matching, `check_timestamps.py` not run; Ecuador control count 21,137 vs 21,138 expected |

---

## 16 (new section). Where we are: the problem, the coordinator agenda, the adjustments

### 16.1 The problem in one paragraph
- IO and control accounts were **collected differently** (paper's limitation, confirmed in our data): IO timelines are complete and span days to years, control timelines are cropped at 100 posts and cover hours, IO timestamps have minute resolution, controls were selected through hashtags.
- A classifier can therefore score well by learning **how an account was collected** (observation window, post volume, posting speed, metadata) instead of **how it behaves**.
- Evidence on 11 campaigns (mean skill over dummy, 10 campaigns): window length alone 0.56, all behavior features 0.52, behavior without speed features 0.25. All five metadata columns reach PR-AUC above 0.89 in 7 of 11 campaigns. Random forest and gradient boosting gain up to +0.34 PR-AUC under a random split, logistic regression almost nothing.
- Consequence: a high score on this dataset is **not** evidence of detection skill. The current numbers cannot support the claim "behavioral features detect coordinated accounts".
- What is still unknown: whether any real behavioral signal remains once the observation window is neutralized. Window length also reflects posting rate, so the two cannot be fully separated; only a stricter control of the window (16.3, items 1 and 2) can narrow this down.
- This is not a "too few campaigns" problem: all campaigns share the same collection design, so more campaigns add statistical power but do not remove the artifacts.

### 16.2 What to discuss with the coordinator
- **What I did:** 11 of 26 campaigns, 68,655 accounts (2,748 IO) after a 30-post filter; account-level features; leave-one-campaign-out evaluation; leakage, window-probe and random-split diagnostics.
- **What I found:** the artifact result above, in two sentences, with one table (behavior vs window-only vs without speed features).
- **Decision 1, thesis shape:** is an **audit plus detectors on artifact-robust signals** (content, repost/reply structure, graph) acceptable, or must the thesis be a detector with a positive result? What does the coordinator expect for the grade (process vs result)?
- **Decision 2, data:** can the coordinator request the remaining 15 campaigns? (More campaigns help LOCO and statistical power, not the artifacts.)
- **Decision 3, collection design:** does the coordinator know how control timelines were collected and why IO timestamps are at minute resolution? Is it worth asking the dataset authors (Seckin et al.) for the collection details, or for uncropped control timelines?
- **Decision 4, scope:** drop the Hawkes extension (timestamps and cropped control windows make it unreliable on this data); keep TGN optional; keep the Romanian case study only if data volume allows.
- **Administrative:** reuse of lab work in the thesis, ethics rules for user-level data, whether the faculty requires its own template, whether processed account-level tables may be shared (license CC BY-NC-ND 4.0), and whether the lab teacher and coordinator are the same person.
- **Timeline:** topic by 30.10.2026; papers read and plan by 27.11.2026.

### 16.3 What to adjust and add (experiments, in priority order)
1. **Stricter window matching:** in every window bin keep the same control-to-IO ratio (subsample controls, or IO when controls are few). Success criterion: the window-only probe falls close to the dummy in every test campaign with enough accounts. Bangladesh becomes train-only.
2. **Re-crop IO timelines to mimic the control collection** (my suggestion, untested): for each IO account use a window of the same length as a typical control window (for example a random 100-post or short time window) instead of its first 30 posts over months. Needs enough IO posts in the window; report how many accounts survive.
3. **`check_timestamps.py` on all 11 campaigns:** do minute resolution (IO) and short control windows hold everywhere? Paste the summary table.
4. **Profile-only metadata test:** `followers`, `following`, `account_age_days` without `n_posts_total` and `span_days`.
5. **Hashtag test:** content features without `hashtag_rate`.
6. **Reporting:** use skill over dummy (or lift plus ROC-AUC) and a macro-average over campaigns; show Egypt separately (56% IO); exclude Bangladesh from test results; try equal campaign weights in training.
7. **Artifact-robust features:** near-duplicate text (TF-IDF cosine), repost/reply structure, hashtag and URL co-use, then graph features (co-repost network from `reposted_accountid`, shared-hashtag network). Re-run the same window probe on them.
8. **Code fixes:** stable sort (`kind="stable"`) after minute flooring, check the last edge of `WINDOWS_EDGES_DAYS`, fix typos, raise the heatmap number threshold (see 11.5). Paste `evaluate.py`, `check_timestamps.py`, `config.py` so section 11.4 can be completed.

### 16.4 What to adjust in the thesis text and claims
- Replace "detects bots / manipulators" with "separates IO-archive accounts from hashtag-selected controls"; use "coordinated-looking" for any Romanian case-study labels.
- Present the artifact audit (leakage, window probe, random vs LOCO) as a **result**, with the caveats in 14.2.
- State in the abstract and discussion that scores on this dataset are upper bounds contaminated by collection design.
- Keep the Hawkes claim at "statistical association" and list it as future work unless the data supports it.
- Define "skill over dummy" in the methods, or use lift and ROC-AUC instead.
- Update the old one-page proposal and the LaTeX/bib to the toned-down scope; verify every bibliography entry.