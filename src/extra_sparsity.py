"""Sparsity-proportional allocation (K-means SMOTE style weights, restricted to minority clusters): same folds/seeds as bench.py."""
import argparse, time, warnings, numpy as np, pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import bench
from bench import CLF
from caos import metrics, ClusterSMOTE, to_counts, entropy
from keel import load_keel
warnings.filterwarnings("ignore")
def weights(Xmin, lab, k, expo):
    sp = np.full(k, np.nan)
    for c in range(k):
        P = Xmin[lab == c]
        if len(P) >= 2: sp[c] = np.log(pdist(P).mean() + 1e-12) * expo - np.log(len(P))     # log sparsity = expo*log(meanDist) - log(n_c)
    sp[np.isnan(sp)] = np.nanmedian(sp) if np.isfinite(sp).any() else 0.0
    w = np.exp(sp - sp.max()); return w / w.sum()
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--names", required=True); ap.add_argument("--clfs", default="kNN,DT,SVM,LR")
    ap.add_argument("--repeats", type=int, default=2); ap.add_argument("--out", required=True); a = ap.parse_args(); rows = []
    for dn in a.names.split(","):
        X, y = load_keel(dn)
        for f, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats, random_state=42).split(X, y)):
            sc = StandardScaler().fit(X[tr]); Xtr, Xte, ytr, yte = sc.transform(X[tr]), sc.transform(X[te]), y[tr], y[te]
            Xmin = Xtr[ytr == 1]; n_min = len(Xmin); S = int((ytr == 0).sum() - n_min)
            k = int(max(2, min(round(np.sqrt(n_min)), n_min // 2))); km = KMeans(k, n_init=5, random_state=f).fit(Xmin); lab, cent = km.labels_, km.cluster_centers_
            gen = ClusterSMOTE(Xmin, lab, k)
            for name, expo in [("Alloc-sparsity-d", Xtr.shape[1]), ("Alloc-sparsity-1", 1)]:
                p = weights(Xmin, lab, k, expo); p = p * (gen.sizes > 0); p /= p.sum(); Xs = gen.generate(to_counts(p, S), f)
                q = np.bincount(np.argmin(((Xs[:, None] - cent[None]) ** 2).sum(-1), 1), minlength=k); q = q / q.sum()
                for cn in a.clfs.split(","):
                    r = metrics(yte, CLF[cn]().fit(np.vstack([Xtr, Xs]), np.r_[ytr, np.ones(len(Xs), int)]).predict(Xte))
                    rows.append(dict(dataset=dn, fold=f, clf=cn, method=name, Hn=entropy(q) / np.log(k), maxshare=q.max(), **{m: r[m] for m in ("hpn", "f1", "gmean", "prec", "rec", "ap")}))
        pd.DataFrame(rows).to_csv(a.out, index=False)
