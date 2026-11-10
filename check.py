import os
import config

folder = f"results_{config.RUN_TAG}"
for root, dirs, files in os.walk(folder):
    for f in files:
        path = os.path.join(root, f)
        print(path, os.path.getsize(path), "bytes")