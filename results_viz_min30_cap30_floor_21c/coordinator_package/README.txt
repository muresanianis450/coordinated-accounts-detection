FEATURE PACKAGE - 21 campaigns, RUN_TAG viz_min30_cap30_floor_21c
131,640 accounts (7,013 IO, 124,627 control), accounts with at least 30 posts,
features computed on each account's first 30 posts, timestamps floored to the minute.

FILES
00_feature_overview.csv   one row per feature: verdict, reason, and the numbers behind it
00_feature_overview.png   all 18 features in one figure
features/NN_<feature>/
   <feature>_distribution.png   histograms IO vs control, one panel per campaign + pooled
   <feature>_performance.png    AUC per campaign: raw, and with window length held fixed
   <feature>_by_campaign.csv    per campaign: counts, medians, means, AUC, KS, correlation with window
   <feature>_window_bins.csv    AUC inside each (campaign, window-length bin) cell
model_level/                    leave-one-campaign-out results for feature SETS (lift over dummy)

HOW TO READ
AUC (IO = positive class): 0.5 no separation, above 0.5 IO has higher values, below 0.5 IO lower.
"Reliable" campaigns have at least 20 IO accounts (17 of 21). Armenia, Bangladesh, Russia_3, Russia_4 are noisy.
"Window held fixed": accounts are grouped by the time span of their first 30 posts (window_days,
11 bins); AUC is computed inside each bin that has at least 20 accounts of each class, then
averaged with IO-count weights. Only 9 campaigns have such bins (33 cells).
Short windows (<= 1 day): 13 cells, 6 campaigns. Long windows (> 1 day): 20 cells, 7 campaigns.

WHY THE WINDOW MATTERS (Seckin et al. 2025, control data collection and limitations)
IO accounts have complete timelines. Control accounts were found through the campaign hashtags
and only their posts on the matching dates were collected (up to 100 per day). So 30 control
posts usually fall within one day (median window 0.27 days) while 30 IO posts span weeks
(median 27 days). A feature that mainly measures this span is not behavior.

LIMITS OF THIS EVIDENCE
- Single-feature AUCs, not model results. No confidence intervals here.
- The window bins are coarse: window_days itself still shows AUC 0.39 inside bins (ideal 0.5).
- In long-window bins the controls are an unusual subgroup (about 2% of controls, sampled on
  several dates), and their gaps jump between sampled days. Timing features in long bins are
  therefore still affected by collection design; short-window cells are the cleaner comparison.
- Verdicts are a proposal. The decisive test is a leave-one-feature-out ablation (not yet run).
