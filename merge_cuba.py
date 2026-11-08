# merge_cuba.py
# Purpose: combine the two halves of the Cuba campaign (Cuba_1, Cuba_2) into one folder
# "Cuba", so the pipeline treats it as a single campaign. Originals are NOT modified.
# File names are prefixed with the source folder name so nothing can be overwritten.
import glob
import os
import shutil

ROOT = r"C:\Users\mures\OneDrive\Desktop\Theses\Data\Extracted"
DEST = os.path.join(ROOT, "Cuba")
os.makedirs(DEST, exist_ok=True)

n_copied = 0
for part in ["Cuba_1", "Cuba_2"]:
    files = sorted(glob.glob(os.path.join(ROOT, part, "**", "*.parquet"), recursive=True))
    for f in files:
        target = os.path.join(DEST, f"{part}__{os.path.basename(f)}")
        shutil.copy2(f, target)  # copy, keep the original
        n_copied += 1

print(f"Copied {n_copied} files into {DEST}")
print("Expected: 100 (Cuba_1) + 24 (Cuba_2) = 124 files")