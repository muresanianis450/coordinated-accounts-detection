# check_cuba_merge.py
# Purpose: test whether Cuba_1 and Cuba_2 are two pieces of ONE campaign, by comparing
# the union of their accounts with the paper's Table 1 values (503 IO / 30,099 control).
import glob
import os

import pyarrow.parquet as pq

ROOT = r"C:\Users\mures\OneDrive\Desktop\Theses\Data\Extracted"


def load_accounts(name):
    """Return the set of (accountid, is_control) pairs found in one campaign folder."""
    files = sorted(glob.glob(os.path.join(ROOT, name, "**", "*.parquet"), recursive=True))
    accounts = set()
    for f in files:
        t = pq.read_table(f, columns=["accountid", "is_control"]).to_pandas()
        accounts.update(zip(t["accountid"], t["is_control"]))
    return accounts


a = load_accounts("Cuba_1")
b = load_accounts("Cuba_2")
union = a | b
overlap = a & b

n_control = sum(1 for _, is_ctrl in union if is_ctrl)
n_io = len(union) - n_control
print(f"Accounts in both folders: {len(overlap)}")
print(f"Union: IO {n_io}, control {n_control}   (paper: IO 503, control 30,099)")