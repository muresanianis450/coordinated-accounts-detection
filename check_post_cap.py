# check_post_cap.py
# Purpose: test whether control accounts are capped at about 100 posts (paper's limitation),
# which would make n_posts_total a pure collection artifact.
import pandas as pd
from config import *

# Load the account-level tables made by build_features.py (one row per account)
data = pd.concat(pd.read_parquet(p) for p in OUT.glob("*.parquet"))
ctrl = data[data.label == 0]   # control accounts
io = data[data.label == 1]     # IO accounts

summary = pd.DataFrame({
    "control_max_posts": ctrl.groupby("campaign").n_posts_total.max(),
    "control_share_over_100": ctrl.groupby("campaign").n_posts_total.apply(lambda s: (s > 100).mean()),
    "io_share_over_100": io.groupby("campaign").n_posts_total.apply(lambda s: (s > 100).mean()),
})
print(summary.round(3))