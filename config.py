from pathlib import Path

DATA_ROOT = Path(r"C:\Users\mures\OneDrive\Desktop\Theses\Data\Extracted")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

CAMPAIGNS = ["Bangladesh", "Armenia", "Catalonia"] # folder names
SEED = 42


# --- Volume control -------------------------------------------------------
# Problem: campaign (IO) accounts post thousands of times, controls only ~15-40.
# If a model can see "how many posts", it can separate the classes without
# learning any real behavior. Two protections:
#   MIN_POSTS: accounts with fewer posts than this are dropped, because
#              behavioral features (gaps, entropy) are meaningless on 1-2 posts.
#   CAP_POSTS: for the behavioral features we only use each account's first
#              CAP_POSTS posts, so every account is described by a similar
#              amount of data.
MIN_POSTS = 10
CAP_POSTS = 50


# Sanity check against Table 1 of the paper: (IO accounts, control accounts).
# build_features.py prints these next to the counts it finds in your files.
# If they differ, something is wrong with the loading.
EXPECTED = {"Bangladesh": (11,929), "Armenia":(31,1767), "Catalonia": (76,2607)}
