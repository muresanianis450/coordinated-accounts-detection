import glob
import pandas as pd

FOLDER = r"C:\Users\mures\OneDrive\Desktop\Theses\Data\Extracted\Bangladesh"
files = sorted(glob.glob(FOLDER+ r"\*.parquet"))
print(len(files), "part files")

cols = ["accountid","post_time","post_language","is_repost","follower_count","account_creation_date","is_control"]

df = pd.concat((pd.read_parquet(f, columns=cols) for f in files), ignore_index=True)

print("posts: ",len(df), "| accounts: ", df.accountid.nunique())
print("time range: ", df.post_time.min(),"->",df.post_time.max())


# 1. class balance, by posts and by accounts
print(df.is_control.value_counts())
print(df.groupby("is_control").accountid.nunique())

# 2. posts per account, per class
print(df.groupby(["is_control","accountid"]).size()
      .groupby("is_control").describe())

# 3. does any account have both labels?

lab = df.groupby("accountid").is_control.nunique()
print("accounts with mixed labels: ", (lab > 1).sum)

# 4. do campaign and control cover the same time period?
print(df.groupby('is_control').post_time.agg(["min","median","max"]))

# 5. languages and missing values
print(df.post_language.value_counts().head(10))
print(df.isna().mean())
