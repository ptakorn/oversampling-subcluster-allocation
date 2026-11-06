"""Controlled synthetic study: minority = m Gaussian clusters (unequal sizes, heterogeneous distance to majority)."""
import numpy as np, pandas as pd, time, itertools, warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from caos import run_fold
from caos2 import run_fold2
warnings.filterwarnings("ignore")

def make(R, m, n_min, seed, n_maj=800, d=8, sig=0.6):
    rng = np.random.RandomState(seed)
    Xmaj = rng.randn(n_maj, d)
    w = np.array([0.5 ** i for i in range(m)]); w /= w.sum()             # unequal cluster sizes
    sizes = np.maximum(4, np.round(w * n_min).astype(int))
    Xs = []
    for c in range(m):
        dirv = rng.randn(d); dirv /= np.linalg.norm(dirv)
        Rc = R * rng.uniform(0.6, 1.4)                                    # heterogeneous distance to majority
        Xs.append(Rc * dirv + sig * rng.randn(sizes[c], d))
    X = np.vstack([Xmaj] + Xs); y = np.r_[np.zeros(n_maj, int), np.ones(len(X) - n_maj, int)]
    return X, y

if __name__ == "__main__":
    t0 = time.time(); rows = []
    cfgs = list(itertools.product([2.0, 3.5, 5.0, 8.0], [3, 6], [50, 100], [0, 1]))
    for ci, (R, m, nm, s) in enumerate(cfgs):
        X, y = make(R, m, nm, seed=1000 * ci + s); name = f"R{R}_m{m}_n{nm}_s{s}"
        print(f"[{time.time()-t0:5.0f}s] {name} n={len(y)} min={int(y.sum())}", flush=True)
        for f, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42).split(X, y)):
            Xt = StandardScaler().fit_transform(X[tr]); Xm = Xt[y[tr] == 1]; n = len(Xm)
            k = int(max(2, min(round(np.sqrt(n)), n // 2))); sil = silhouette_score(Xm, KMeans(k, n_init=5, random_state=f).fit_predict(Xm))
            for cn in ["kNN", "DT", "SVM"]:
                res = run_fold(X[tr], y[tr], X[te], y[te], cn, seed=f, alphas=[0.5], metric="gmean")
                res2 = run_fold2(X[tr], y[tr], X[te], y[te], cn, seed=f)
                for mth, r in {**res, **res2}.items():
                    rows.append(dict(dataset=name, R=R, m=m, n_min=nm, fold=f, clf=cn, method=mth, sil=sil, **r))
        pd.DataFrame(rows).to_csv("r_synth.csv", index=False)
    print("done", time.time() - t0)
