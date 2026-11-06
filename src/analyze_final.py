"""Final KEEL analysis: 19+ methods, G-mean / AUC-PR / F1 / HPN, Friedman + Holm + bootstrap CI, fair threshold comparison, fallback rate."""
import glob, sys, numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
kw = dict(keep_default_na=False, na_values=[""])
files = sorted(glob.glob("r_final/*.csv")); d = pd.concat([pd.read_csv(f, **kw) for f in files], ignore_index=True)
d["method"] = d.method.replace("", "None"); d = d.drop_duplicates(["dataset", "fold", "clf", "method"])
d = d[~((d.clf == "RF") & (d.fold >= 10))]                     # RF main run has 2 repeats (folds 0-9)
KEY = ["None", "ClassWeight", "ThreshTune", "SMOTE+Thresh", "Uniform+Thresh", "SMOTE", "Borderline", "SVMSMOTE", "ADASYN", "SMOTE-ENN", "SMOTE-Tomek", "KMeansSMOTE",
       "CBO-dup", "Alloc-uniform", "Alloc-difficulty", "Alloc-sparsity-1", "Alloc-sparsity-d", "CAOS-a0.5", "U-greedy", "SMOTE-rho-G", "V3-both-G"]
fb = d[d.method == "KMeansSMOTE"]; d = d[d.method.isin(KEY)]
rng = np.random.RandomState(0)
def holm(p):
    p = np.array(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1, run)
    return adj
def boot(x, B=4000):
    x = np.asarray(x); return np.percentile([x[rng.randint(0, len(x), len(x))].mean() for _ in range(B)], [2.5, 97.5])
def analyse(sub, label, mets, refs):
    print(f"\n{'#'*6} {label}: {sub.dataset.nunique()} datasets, clf {sorted(sub.clf.unique())}, folds/dataset~{int(sub.groupby('dataset').fold.nunique().median())} {'#'*6}")
    for met in mets:
        if met not in sub or sub[met].isna().all(): continue
        P = sub.groupby(["dataset", "method"])[met].mean().unstack(); cols = [c for c in KEY if c in P.columns]; P = P[cols].dropna()
        rk = P.rank(axis=1, ascending=False).mean().sort_values()
        try: fr = friedmanchisquare(*[P[c] for c in cols]); frs = f"Friedman chi2={fr.statistic:.1f}, p={fr.pvalue:.2g}"
        except Exception: frs = "Friedman n/a"
        print(f"\n[{met}] n={len(P)} datasets, {len(cols)} methods, {frs}")
        print(pd.DataFrame({"mean": P.mean().round(4), "avg_rank": rk.reindex(P.columns).round(2)}).loc[rk.index].T.to_string())
        for ref in refs:
            if ref not in P: continue
            rows = []
            for c in cols:
                if c == ref: continue
                x = (P[c] - P[ref]).values
                try: p = wilcoxon(x).pvalue if np.any(x != 0) else 1.0
                except Exception: p = 1.0
                lo, hi = boot(x); rows.append((c, x.mean(), lo, hi, int((x > 0).sum()), p))
            r = pd.DataFrame(rows, columns=["method", "diff", "ci_lo", "ci_hi", "wins", "p"]); r["p_holm"] = holm(r.p.values)
            print(f"  -- minus {ref} (bootstrap 95% CI; Holm over {len(r)}) --"); print(r.sort_values("diff", ascending=False).round(4).to_string(index=False))
if __name__ == "__main__":
    fast = d[d.clf.isin(["kNN", "DT", "SVM", "LR"])]
    analyse(fast, "FAST CLASSIFIERS (kNN,DT,SVM,LR)", ["gmean", "ap", "f1", "hpn"], ["Alloc-uniform", "SMOTE"])
    analyse(d[d.clf == "RF"], "RANDOM FOREST", ["gmean", "ap", "f1", "hpn"], ["Alloc-uniform", "SMOTE"])
    for cn in ["kNN", "DT", "SVM", "LR"]: analyse(d[d.clf == cn], f"CLASSIFIER {cn}", ["gmean", "ap"], ["Alloc-uniform"])
    print("\n##### K-means SMOTE fallback rate (share of fold-classifier runs that fell back to plain SMOTE) #####")
    if "fallback" in fb: 
        x = pd.to_numeric(fb.fallback, errors="coerce"); print("overall:", round(x.mean(), 4)); print(fb.assign(fallback=x).groupby("clf").fallback.mean().round(3).to_dict())
        g = fb.assign(fallback=x).groupby("dataset").fallback.mean(); print("datasets with fallback>50%:", int((g > .5).sum()), "of", len(g))
    print("\n##### Allocation statistics (mean over runs) #####")
    print(d.groupby("method")[["Hn", "maxshare"]].apply(lambda g: g.apply(pd.to_numeric, errors="coerce").mean()).dropna().round(3).to_string())
