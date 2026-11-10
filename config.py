from pathlib import Path

DATA_ROOT = Path(r"C:\Users\mures\OneDrive\Desktop\Theses\Data\Extracted")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

CAMPAIGNS = ["Bangladesh", "Armenia", "Catalonia", "China_1", "China_2","Cuba", "Ecuador",
             "Egypt","Iran_1","Iran_2","Iran_3","Iran_4","Iran_5","Iran_6","Qatar",
             "Russia_1","Russia_2","Russia_3","Russia_4","Russia_5",
             "Spain"] # folder names
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
MIN_POSTS = CAP_POSTS = 30

# --- Collection-bias control -------------------------------
# Two known differences between IO and control data
#
# 1. Timestamp resolution. IO timestamps are at minute resolution
#   (seconds are always 0), control timestamps have seconds.
#   FLOOR_TO_MINUTE rounds EVERY timestamp down to the minute,
#   so both classes have the same resolution.
FLOOR_TO_MINUTE = True


#
# 2. Observation window. IO timelines are complete, control timelines are
#    cropped at 100 posts, so the first CAP_POSTS posts of a control account
#    cover hours while an IO account's cover days or months. MATCH_WINDOWS
#    keeps only accounts whose window lengths fall in a bin (see WINDOW_EDGE_DAYS)
#    that contain accounts of BOTH classes, so IO and control accounts are
#    compared at similar window lengths. This removes many accounts.
MATCH_WINDOWS = False

# Bin edge for the window length, in days (from 0 to up infinity).
# Fixed edges, so every campaign is binned the same way.
WINDOWS_EDGES_DAYS = [0,0.05,0.1,0.25,0.5,1,2,5,15,60,365, float("inf")]
# A bin is kept only if it has at least this many accounts of EACH class.
MIN_PER_CLASS_PER_BIN = 1

# --- Run tag ----------------------------------------------------
# A short name describing the settings above. The processed data and the results
# of each setting combination go to their own folders, so runs never overwrite
# each other and the evaluation script always read the data built with the
# CURRENT settings. (ALWAYS run build_features.py after changing settings.)

# The tag describes only the settings that change the data. The "viz_" prefix
# keeps these files separate from the earlier evaluation runs.
RUN_TAG = (f"viz_min{MIN_POSTS}_cap{CAP_POSTS}"
           + ("_floor" if FLOOR_TO_MINUTE else "")
           + f"_{len(CAMPAIGNS)}c")


# Where the processed tables (one row per account) will be saved.
# This is a relative path, so it is created inside your PyCharm project folder.
OUT = Path(f"data/processed_{RUN_TAG}")
# Create the folder if it does not exist yet. parents=True also creates "data"
# if needed; exist_ok=True means "no error if it is already there".
OUT.mkdir(parents=True, exist_ok=True)


# Sanity check against Table 1 of the paper: (IO accounts, control accounts).
# build_features.py prints these next to the counts it finds in your files.
# If they differ, something is wrong with the loading.
EXPECTED = {
    "Bangladesh": (11, 929),
    "Armenia": (31, 1767),
    "Catalonia": (76, 2607),
    "China_1": (699, 42120),
    "China_2": (191, 35806),
    "Cuba": (503, 30099),
    "Ecuador": (787, 21138),
    "Egypt": (240, 370)   ,
    "Iran_1": (660,5015),
    "Iran_2": (389,6615),
    "Iran_3": (210,2240),
    "Iran_4": (2519,7612),
    "Iran_5": (104,1247),
    "Iran_6": (209,16885),
    "Russia_1": (3293,31303),
    "Russia_2": (361,2175),
    "Russia_3": (5,780),
    "Russia_4": (24,36533),
    "Russia_5": (51,13077),
    "Spain": (216,1681),
    "Qatar": (29,19481)
}
