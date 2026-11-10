import pandas as pd
s = pd.read_csv("results_viz_min30_cap30_floor_21c/feature_report/feature_summary.csv")
s = s[s.campaign != "ALL (pooled)"]
p = s.pivot(index="campaign", columns="feature", values="auc_io_higher")
n = s[s.feature == "window_days"].set_index("campaign")["n_io"]

variants = {
    "all": [],
    "no Iran_3": ["Iran_3"],
    "no small (<20 IO)": list(n[n < 20].index),
    "neither": list(n[n < 20].index) + ["Iran_3"],
}
out = pd.DataFrame({
    k: p.drop(index=drop).corr(method="spearman")["window_days"]
    for k, drop in variants.items()
})
print(out.drop(index="window_days").sort_values("all").round(2).to_string())