"""Per-minority-cluster analysis: which clusters benefit from uniform allocation vs SMOTE / difficulty allocation?"""
import numpy as np, pandas as pd, time, itertools, warnings
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from caos import load_datasets, ClusterSMOTE, to_counts, init_alloc, CLF
from synth import make
warnings.filterwarnings("ignore")

def run(name, X, y, clfs, reps, rows):
    for f, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=42).split(X, y)):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte, ytr, yte = sc.transform(X[tr]), sc.transform(X[te]), y[tr], y[te]
        Xmin = Xtr[ytr == 1]; n_min = len(Xmin); S = int((ytr == 0).sum() - n_min)
        k = int(max(2, min(round(np.sqrt(n_min)), n_min // 2)))
        km = KMeans(k, n_init=5, random_state=f).fit(Xmin); lab, cent = km.labels_, km.cluster_centers_
        # cluster descriptors from TRAIN data
        nn_maj = NearestNeighbors(n_neighbors=6).fit(Xtr).kneighbors(Xmin, return_distance=False)[:, 1:]
        frac_maj = (ytr[nn_maj] == 0).mean(1)
        desc = []
        for c in range(k):
            P = Xmin[lab == c]; m = len(P)
            if m >= 3:
                dd = NearestNeighbors(n_neighbors=min(4, m)).fit(P).kneighbors(P)[0][:, 1:].mean()
            else: dd = np.nan
            desc.append((m, dd, frac_maj[lab == c].mean()))
        desc = np.array(desc, float)
        tm = Xte[yte == 1]; tcl = np.argmin(((tm[:, None] - cent[None]) ** 2).sum(-1), 1)     # test minority -> nearest centroid
        gen = ClusterSMOTE(Xmin, lab, k)
        syn = {"SMOTE": SMOTE(random_state=f, k_neighbors=min(5, n_min - 1)).fit_resample(Xtr, ytr)[0][len(Xtr):],
               "Alloc-uniform": gen.generate(to_counts(init_alloc("uniform", Xtr, ytr, Xmin, lab, k), S), f),
               "Alloc-difficulty": gen.generate(to_counts(init_alloc("difficulty", Xtr, ytr, Xmin, lab, k), S), f)}
        for cn in clfs:
            for mth, Xs in syn.items():
                Xa = np.vstack([Xtr, Xs]); ya = np.r_[ytr, np.ones(len(Xs), int)]
                pred = CLF[cn]().fit(Xa, ya).predict(Xte); pm = pred[yte == 1]
                a = np.bincount(np.argmin(((Xs[:, None] - cent[None]) ** 2).sum(-1), 1), minlength=k); share = a / a.sum()
                spec = ((pred == 0) & (yte == 0)).sum() / (yte == 0).sum()
                for c in range(k):
                    rows.append(dict(dataset=name, fold=f, clf=cn, method=mth, cluster=c, k=k, n_c=desc[c, 0], sparse=desc[c, 1], diff=desc[c, 2],
                                     nte=int((tcl == c).sum()), hits=int(pm[tcl == c].sum()), share=share[c], synth_per_pt=a[c] / desc[c, 0], spec=spec))

if __name__ == "__main__":
    t0 = time.time(); rows = []
    cfgs = list(itertools.product([2.0, 3.5, 5.0, 8.0], [3, 6], [50, 100], [0, 1]))
    for ci, (R, m, nm, s) in enumerate(cfgs):
        X, y = make(R, m, nm, seed=1000 * ci + s); run(f"S_R{R}_m{m}_n{nm}_s{s}", X, y, ["kNN", "DT", "SVM"], 2, rows)
        if ci % 8 == 7: print(f"[{time.time()-t0:.0f}s] synth {ci+1}/32", flush=True); pd.DataFrame(rows).to_csv("pc_rows.csv", index=False)
    for dn, (X, y, frac) in load_datasets().items():
        if frac < 1: X, _, y, _ = train_test_split(X, y, train_size=frac, stratify=y, random_state=0)
        run("REAL_" + dn, X, y, ["kNN", "DT"] if dn in ("Mammography", "Phoneme", "Digits(8)") else ["kNN", "DT", "SVM"], 3, rows)
        print(f"[{time.time()-t0:.0f}s] {dn}", flush=True); pd.DataFrame(rows).to_csv("pc_rows.csv", index=False)
    print("done")
