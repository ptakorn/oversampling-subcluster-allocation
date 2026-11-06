import glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon
d = pd.concat([pd.read_csv(f, keep_default_na=False) for f in sorted(glob.glob("ks_out/*.csv"))], ignore_index=True)
for c in ("gmean", "ap", "f1", "hpn"): d[c] = pd.to_numeric(d[c], errors="coerce")
print("datasets:", d.dataset.nunique())
rng = np.random.RandomState(0)
def ds_mean(df, key, col="gmean"): return df.groupby(["dataset", key])[col].mean().unstack()
def boot(x, B=4000):
    b = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(B)]; return np.percentile(b, [2.5, 97.5])
# ---- A. k sensitivity
a = d[d.method.isin(["uniform", "difficulty", "sparsity1"])].copy(); a["cfg"] = a.method + "|" + a.ktag
for col in ("gmean", "ap"):
    T = ds_mean(a, "cfg", col); print(f"\n[{col}] mean over datasets by rule x k-tag (def = sqrt(n_min))")
    print(pd.DataFrame({r: {t: T[f"{r}|{t}"].mean() for t in ["k2", "k3", "half", "def", "double", "quarter"]} for r in ["uniform", "difficulty", "sparsity1"]}).round(4).T.to_string())
print("\nMedian k by tag:", a.groupby("ktag").k.median().to_dict())
T = ds_mean(a, "cfg")
print("\n[gmean] difficulty - uniform and sparsity1 - uniform, paired by dataset, at each k (bootstrap CI, Wilcoxon p)")
for r in ["difficulty", "sparsity1"]:
    for t in ["k2", "k3", "half", "def", "double", "quarter"]:
        x = (T[f"{r}|{t}"] - T[f"uniform|{t}"]).dropna().values
        try: p = wilcoxon(x).pvalue
        except Exception: p = np.nan
        lo, hi = boot(x); print(f"{r:10s} {t:8s} diff={x.mean():+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}")
print("\n[gmean] uniform at each k minus uniform at default k (does k itself matter?)")
for t in ["k2", "k3", "half", "double", "quarter"]:
    x = (T[f"uniform|{t}"] - T["uniform|def"]).dropna().values; lo, hi = boot(x); print(f"uniform {t:8s} {x.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")
# ---- B. K-means SMOTE tuning
b = d[d.method.isin(["KMS-tuned", "KMS-oracle", "KMS-cfg"])].copy()
b["m"] = np.where(b.method == "KMS-cfg", b.ktag, b.method)
T = ds_mean(b, "m", "gmean"); print("\n[gmean] K-means SMOTE configs (dataset means):"); print(T.mean().round(4).to_string())
fb = pd.to_numeric(b[b.method == "KMS-cfg"].fallback, errors="coerce").groupby(b[b.method == "KMS-cfg"].ktag).mean()
print("\nfallback rate by config:"); print(fb.round(3).to_string())
fb = pd.to_numeric(b[b.method == "KMS-tuned"].fallback, errors="coerce"); print("fallback rate of inner-CV-selected config:", round(fb.mean(), 4))
# ---- C. compare with main-run references on the same folds (first 2 repeats, kNN/DT/LR)
r = pd.concat([pd.read_csv(f, keep_default_na=False) for f in sorted(glob.glob("r_final/ff_*.csv"))], ignore_index=True)
r["fold"] = pd.to_numeric(r.fold); r = r[(r.fold < 10) & r.clf.isin(["kNN", "DT", "LR"]) & r.method.isin(["SMOTE", "Alloc-uniform", "KMeansSMOTE"])]
r["gmean"] = pd.to_numeric(r.gmean); R = r.groupby(["dataset", "method"]).gmean.mean().unstack()
K = d[d.method.isin(["KMS-tuned", "KMS-oracle"])].groupby(["dataset", "method"]).gmean.mean().unstack()
U = d[(d.method == "uniform") & (d.ktag == "def")].groupby("dataset").gmean.mean().rename("uniform_here")
M = R.join(K, how="inner").join(U, how="inner").dropna(); print(f"\n[gmean] reference comparison on {len(M)} datasets"); print(M.mean().round(4).to_string())
for a_, b_ in [("KMS-tuned", "SMOTE"), ("KMS-oracle", "SMOTE"), ("KMS-tuned", "KMeansSMOTE"), ("KMS-tuned", "Alloc-uniform"), ("KMS-oracle", "Alloc-uniform")]:
    x = (M[a_] - M[b_]).values; lo, hi = boot(x); print(f"{a_} - {b_}: {x.mean():+.4f} [{lo:+.4f},{hi:+.4f}] wins={int((x>0).sum())}/{len(x)} p={wilcoxon(x).pvalue:.4f}")
print("sanity: uniform here vs main run uniform diff:", round((M.uniform_here - M['Alloc-uniform']).abs().max(), 6))
