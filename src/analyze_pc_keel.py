import pandas as pd, numpy as np
from scipy.stats import wilcoxon
d = pd.read_csv("pc_keel_rows.csv"); d = d[d.k >= 3]
def terc(s):
    s = s.fillna(s.median()); p = s.rank(method="first") / len(s); return np.where(p <= 1/3, "low", np.where(p > 2/3, "high", "mid"))
b = d[d.method == "SMOTE"][["dataset", "fold", "clf", "cluster", "n_c", "sparse", "diff"]].copy()
for c, nm in [("n_c", "size"), ("sparse", "sparsity"), ("diff", "difficulty")]:
    b[nm] = b.groupby(["dataset", "fold", "clf"])[c].transform(lambda s: pd.Series(terc(s), index=s.index))
d = d.merge(b[["dataset", "fold", "clf", "cluster", "size", "sparsity", "difficulty"]], on=["dataset", "fold", "clf", "cluster"])
def rec(dim, tier, a, bm):
    g = d[(d[dim] == tier) & d.method.isin([a, bm])].groupby(["dataset", "method"]).agg(h=("hits", "sum"), n=("nte", "sum")).reset_index()
    g["r"] = g.h / g.n.replace(0, np.nan); P = g.pivot(index="dataset", columns="method", values="r").dropna(); return (P[a] - P[bm]).values
def holm(p):
    p = np.array(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj
rng = np.random.RandomState(0)
def ci(x): return np.percentile([x[rng.randint(0, len(x), len(x))].mean() for _ in range(4000)], [2.5, 97.5])
H = [("H1 uniform-SMOTE, smallest clusters (expect >0)", "size", "low", "Alloc-uniform", "SMOTE", "greater"),
     ("H2 uniform-SMOTE, largest clusters (expect <0)", "size", "high", "Alloc-uniform", "SMOTE", "less"),
     ("H3 uniform-difficulty, easiest clusters (expect >0)", "difficulty", "low", "Alloc-uniform", "Alloc-difficulty", "greater"),
     ("H4 uniform-SMOTE, sparsest clusters (expect >0)", "sparsity", "high", "Alloc-uniform", "SMOTE", "greater")]
res = []
for name, dim, tier, a, bm, alt in H:
    x = rec(dim, tier, a, bm); p2 = wilcoxon(x).pvalue if np.any(x != 0) else 1.0; lo, hi = ci(x)
    res.append((name, len(x), x.mean(), lo, hi, int((x > 0).sum()), p2))
r = pd.DataFrame(res, columns=["hypothesis", "n_datasets", "mean_diff", "ci_lo", "ci_hi", "n_pos", "p_two_sided"]); r["p_holm"] = holm(r.p_two_sided.values)
print("=== PRE-SPECIFIED CONFIRMATORY TESTS (KEEL, 59 datasets; Holm over 4) ==="); print(r.round(4).to_string(index=False))
print("\n=== EXPLORATORY: all terciles x contrasts (unadjusted p) ===")
for dim in ["size", "sparsity", "difficulty"]:
    for tier in ["low", "mid", "high"]:
        out = []
        for a, bm in [("Alloc-uniform", "SMOTE"), ("Alloc-uniform", "Alloc-difficulty"), ("Alloc-difficulty", "SMOTE")]:
            x = rec(dim, tier, a, bm); out.append(f"{a[6:]}-{bm[:6]} {x.mean():+.3f} (p={wilcoxon(x).pvalue:.3f})")
        print(f"{dim:>10}/{tier:<4}: " + " | ".join(out))
g = d.groupby("method").agg(spp=("synth_per_pt", "mean"))
for dim in ["size", "sparsity", "difficulty"]:
    print(f"\nsynthetic per original point by {dim} tercile:"); print(d.groupby([dim, "method"]).synth_per_pt.mean().unstack().loc[["low", "mid", "high"]].round(2).to_string())
