# inspect_campaigns.py
# Purpose: for every extracted campaign folder, report file sizes, number of posts,
# and number of IO / control accounts, so we can compare with Table 1 of the paper
# BEFORE touching config.py or rebuilding features.
# Version 2: a corrupt or unfinished parquet file no longer stops the script;
# it is listed and skipped, and the campaign is flagged as INCOMPLETE.
import glob
import os

import pyarrow.parquet as pq

# Folder that contains one sub-folder per campaign (change if yours differs)
ROOT = r"C:\Users\mures\OneDrive\Desktop\Theses\Data\Extracted"

for campaign in sorted(os.listdir(ROOT)):
    folder = os.path.join(ROOT, campaign)
    if not os.path.isdir(folder):
        continue  # skip stray files

    files = sorted(glob.glob(os.path.join(folder, "**", "*.parquet"), recursive=True))
    if not files:
        print(f"{campaign}: NO parquet files found (empty folder or unfinished extraction?)")
        continue

    n_posts = 0
    size_mb = 0.0
    accounts = set()
    bad_files = []

    for f in files:
        try:
            # Row count comes from the parquet metadata: nothing is loaded into memory
            n_posts += pq.ParquetFile(f).metadata.num_rows
            # Read only two light columns to count accounts per class
            tbl = pq.read_table(f, columns=["accountid", "is_control"]).to_pandas()
            accounts.update(zip(tbl["accountid"], tbl["is_control"]))
            size_mb += os.path.getsize(f) / 1e6
        except Exception as e:  # corrupt / incomplete file
            bad_files.append((os.path.basename(f), os.path.getsize(f), type(e).__name__))

    n_control = sum(1 for _, is_ctrl in accounts if is_ctrl)
    n_io = len(accounts) - n_control
    flag = "  <-- INCOMPLETE" if bad_files else ""
    print(f"{campaign}: {len(files)} file(s), {size_mb:,.0f} MB, "
          f"{n_posts:,} posts, IO accounts {n_io}, control accounts {n_control}{flag}")
    for name, size, err in bad_files[:5]:
        print(f"    bad file: {name} ({size:,} bytes, {err})")
    if len(bad_files) > 5:
        print(f"    ... and {len(bad_files) - 5} more bad files")