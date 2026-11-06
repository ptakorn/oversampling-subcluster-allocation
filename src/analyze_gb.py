import glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
d = pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]) for f in sorted(glob.glob("gb_out/*.csv"))]); rng = np.random.RandomState(2)
def boot(x, B=4000): x = np.asarray(x); return np.percentile([x[rng.randint(0, len(x), len(x))].mean() for _ in range(B)], [2.5, 97.5])
out = []
def pr(*a): s = " ".join(str(x) for x in a); print(s); out.append(s)
pr("datasets:", d.dataset.nunique(), "folds/dataset:", d.groupby("dataset").fold.nunique().median())
M = {m: d.groupby(["dataset", "method"])[m].mean().unstack() for m in ("gmean", "ap", "f1", "hpn")}
order = ["None", "ClassWeight", "ThreshTune", "SMOTE", "Alloc-uniform", "Alloc-size", "Alloc-difficulty", "SMOTE-ENN", "SMOTE+Thresh", "Uniform+Thresh"]
T = pd.DataFrame({m: M[m][order].mean() for m in M}).round(4); T["rank_g"] = M["gmean"][order].rank(axis=1, ascending=False).mean().round(2); pr(T.to_string())
fr = friedmanchisquare(*[M["gmean"][c] for c in order]); pr("Friedman G-mean chi2=%.1f p=%.2g" % (fr.statistic, fr.pvalue))
for met in ("gmean", "ap"):
    for a, b in (("SMOTE", "Alloc-uniform"), ("Alloc-size", "Alloc-uniform"), ("Alloc-difficulty", "Alloc-uniform"), ("SMOTE-ENN", "Alloc-uniform"), ("ThreshTune", "Alloc-uniform"),
                 ("SMOTE+Thresh", "Uniform+Thresh"), ("SMOTE+Thresh", "ThreshTune"), ("Uniform+Thresh", "ThreshTune"), ("SMOTE", "None"), ("ClassWeight", "None")):
        z = (M[met][a] - M[met][b]).dropna(); p = wilcoxon(z).pvalue if np.any(z != 0) else 1; lo, hi = boot(z)
        pr(f"{met} {a}-{b}: {z.mean():+.4f} [{lo:+.4f},{hi:+.4f}] W{int((z>0).sum())}/L{int((z<0).sum())} p={p:.3g}")
open("../analysis/analysis_gb.txt", "w").write("\n".join(out))
