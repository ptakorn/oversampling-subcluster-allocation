import sys, time, warnings, pandas as pd
import bench                      # updates caos.CLF with LR etc.
import per_cluster as pc
from keel import load_keel
warnings.filterwarnings("ignore")
names = open("names_keel.txt").read().split(","); rows = []; t0 = time.time()
for i, n in enumerate(names):
    X, y = load_keel(n); pc.run("KEEL_" + n, X, y, ["kNN", "DT", "SVM", "LR"], 2, rows)
    if i % 5 == 4: pd.DataFrame(rows).to_csv("pc_keel_rows.csv", index=False); print(f"[{time.time()-t0:.0f}s] {i+1}/{len(names)}", flush=True)
pd.DataFrame(rows).to_csv("pc_keel_rows.csv", index=False); print("ALLDONE")
