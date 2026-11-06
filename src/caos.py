"""Concentration-aware cluster oversampling (CAOS) - faithful implementation of Algorithm 1
plus benchmark harness. Run: python caos.py --out results.csv
"""
import argparse, time, warnings, sys, os
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE
warnings.filterwarnings("ignore")
DATA = os.environ.get("UCI_DATA", "data_uci/")   # only used by the legacy UCI loader (not part of the reported results)

# ---------------------------------------------------------------- datasets
def _csv(name, header=None, **kw):
    return pd.read_csv(DATA + name, header=header, **kw)

def load_datasets():
    from sklearn.datasets import load_breast_cancer, load_wine, load_digits
    D = {}
    def put(name, X, y, frac=1.0):
        D[name] = (np.asarray(X, float), np.asarray(y, int), frac)
    d = _csv("pima-indians-diabetes.csv"); put("Pima", d.iloc[:, :-1], d.iloc[:, -1] == 1)
    d = _csv("haberman.csv"); put("Haberman", d.iloc[:, :-1], d.iloc[:, -1] == 2)
    d = _csv("ionosphere.csv"); put("Ionosphere", d.iloc[:, :-1], d.iloc[:, -1] == "b")
    d = _csv("sonar.csv"); put("Sonar", d.iloc[:, :-1], d.iloc[:, -1] == "R")
    d = _csv("german.csv", sep=r"\s+|,", engine="python") if False else None
    d = _csv("glass.csv"); put("Glass(cls6)", d.iloc[:, :-1], d.iloc[:, -1] == 6)
    d = _csv("ecoli.csv"); put("Ecoli(om)", d.iloc[:, :-1], d.iloc[:, -1] == "om")
    d = _csv("mammography.csv"); put("Mammography", d.iloc[:, :-1], d.iloc[:, -1].astype(str).str.contains("1") & ~d.iloc[:, -1].astype(str).str.contains("-1"), 0.25)
    d = _csv("oil-spill.csv"); put("OilSpill", d.iloc[:, :-1], d.iloc[:, -1] == 1)
    d = _csv("phoneme.csv"); put("Phoneme", d.iloc[:, :-1], d.iloc[:, -1] == 1, 0.25)
    X, y = load_breast_cancer(return_X_y=True); put("BreastWDBC", X, y == 0)
    X, y = load_wine(return_X_y=True); put("Wine(cls1)", X, y == 1)
    X, y = load_digits(return_X_y=True); put("Digits(8)", X, y == 8)
    return D

# ---------------------------------------------------------------- metrics
def metrics(yt, yp):
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    sd = lambda a, b: a / b if b > 0 else 0.0
    prec, rec = sd(tp, tp + fp), sd(tp, tp + fn)
    npv, spec = sd(tn, tn + fn), sd(tn, tn + fp)
    f1 = sd(2 * prec * rec, prec + rec)
    hpn = sd(2 * prec * npv, prec + npv)
    return dict(hpn=hpn, f1=f1, gmean=np.sqrt(rec * spec), prec=prec, rec=rec)

def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

# ---------------------------------------------------------------- within-cluster SMOTE
class ClusterSMOTE:
    def __init__(self, Xmin, labels, k, knn=5):
        self.k = k; self.pts = []; self.nn = []
        for c in range(k):
            P = Xmin[labels == c]; self.pts.append(P)
            m = len(P)
            if m >= 2:
                kk = min(knn, m - 1)
                idx = NearestNeighbors(n_neighbors=kk + 1).fit(P).kneighbors(P, return_distance=False)[:, 1:]
            else:
                idx = None
            self.nn.append(idx)
        self.sizes = np.array([len(p) for p in self.pts])

    def generate(self, counts, seed):
        rng = np.random.RandomState(seed); out = []
        for c, s in enumerate(counts):
            s = int(s)
            if s <= 0 or self.sizes[c] == 0: continue
            P = self.pts[c]; base = rng.randint(0, len(P), s)
            if self.nn[c] is None:                      # singleton cluster: jittered duplicate
                out.append(P[base] + rng.normal(0, 1e-3, (s, P.shape[1]))); continue
            nb = self.nn[c][base, rng.randint(0, self.nn[c].shape[1], s)]
            gap = rng.rand(s, 1)
            out.append(P[base] + gap * (P[nb] - P[base]))
        return np.vstack(out) if out else np.empty((0, self.pts[0].shape[1]))

def to_counts(p, S):
    p = np.asarray(p, float); p = p / p.sum()
    raw = p * S; c = np.floor(raw).astype(int)
    rem = S - c.sum()
    if rem > 0: c[np.argsort(-(raw - c))[:rem]] += 1
    return c

# ---------------------------------------------------------------- allocation init
def init_alloc(mode, Xtr, ytr, Xmin, labels, k):
    sizes = np.bincount(labels, minlength=k).astype(float)
    if mode == "uniform": w = np.ones(k)
    elif mode == "size": w = sizes
    else:  # difficulty: fraction of majority points among 5-NN of each minority point
        nn = NearestNeighbors(n_neighbors=6).fit(Xtr).kneighbors(Xmin, return_distance=False)[:, 1:]
        frac = (ytr[nn] == 0).mean(1)
        w = np.array([frac[labels == c].mean() if (labels == c).any() else 0 for c in range(k)]) + 0.05
        w = w ** 2          # sharpen so initial allocation is realistically concentrated
    return w / w.sum()

# ---------------------------------------------------------------- Algorithm 1
def caos_allocate(p0, Xin, yin, Xmin_in, lab_in, Xval, yval, make_clf, k, alpha, metric,
                  T=10, M=4, beta=0.05, seed=0, log=None):
    """Greedy pairwise allocation search. Works on proportions p (sum 1); S_in = n_maj-n_min of inner train."""
    gen = ClusterSMOTE(Xmin_in, lab_in, k)
    S = max(1, int((yin == 0).sum() - (yin == 1).sum()))
    delta = max(1, int(beta * S))

    def J_of(p):
        counts = to_counts(p, S)
        Xs = gen.generate(counts, seed)             # fixed seed => common random numbers
        Xa = np.vstack([Xin, Xs]); ya = np.r_[yin, np.ones(len(Xs), int)]
        yp = make_clf().fit(Xa, ya).predict(Xval)
        P = metrics(yval, yp)[metric]
        return P + alpha * entropy(counts / counts.sum()), P

    counts = to_counts(p0, S); J, P = J_of(p0); rng = np.random.RandomState(seed)
    hist = [(J, P)]
    for t in range(T):
        improved = False
        cnt = counts.copy()
        pairs = [(int(np.argmin(cnt)), int(np.argmax(cnt)))]       # lowest <- highest
        for _ in range(M - 1):
            i, j = rng.choice(k, 2, replace=False); pairs.append((int(i), int(j)))
        for i, j in pairs:
            if counts[j] < delta or i == j: continue
            cand = counts.copy(); cand[i] += delta; cand[j] -= delta
            Jc, Pc = J_of(cand / cand.sum())
            if Jc > J + 1e-12:
                counts, J, P, improved = cand, Jc, Pc, True
        hist.append((J, P))
        if not improved: break
    return counts / counts.sum(), hist

# ---------------------------------------------------------------- classifiers
CLF = {
    "kNN": lambda: KNeighborsClassifier(5),
    "DT": lambda: DecisionTreeClassifier(random_state=0),
    "SVM": lambda: SVC(kernel="rbf", C=1.0, gamma="scale"),
    "RF": lambda: RandomForestClassifier(100, random_state=0, n_jobs=1),
}

# ---------------------------------------------------------------- one fold
def run_fold(Xtr, ytr, Xte, yte, clf_name, seed, alphas, metric="hpn"):
    make_clf = CLF[clf_name]
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    Xmin = Xtr[ytr == 1]; n_min = len(Xmin); S = int((ytr == 0).sum() - n_min)
    k = int(max(2, min(round(np.sqrt(n_min)), n_min // 2)))
    km = KMeans(k, n_init=5, random_state=seed).fit(Xmin); lab = km.labels_; cent = km.cluster_centers_
    out = {}

    def finish(name, Xs, extra=None):
        Xa = np.vstack([Xtr, Xs]) if len(Xs) else Xtr
        ya = np.r_[ytr, np.ones(len(Xs), int)]
        r = metrics(yte, make_clf().fit(Xa, ya).predict(Xte))
        if len(Xs) > 0:   # allocation metrics under the common cluster partition
            a = np.bincount(np.argmin(((Xs[:, None, :] - cent[None]) ** 2).sum(-1), 1), minlength=k)
            p = a / a.sum(); r.update(H=entropy(p), Hn=entropy(p) / np.log(k), maxshare=p.max())
        r["k"] = k
        if extra: r.update(extra)
        out[name] = r

    finish("None", np.empty((0, Xtr.shape[1])))
    for nm, smp in [("SMOTE", SMOTE(random_state=seed)), ("ADASYN", ADASYN(random_state=seed)),
                    ("Borderline", BorderlineSMOTE(random_state=seed)),
                    ("KMeansSMOTE", KMeansSMOTE(random_state=seed, k_neighbors=min(5, max(1, n_min // k - 1)),
                                                cluster_balance_threshold=0.1, kmeans_estimator=min(10, max(2, n_min // 3))))]:
        fb = 0
        try:
            Xr, _ = smp.fit_resample(Xtr, ytr); Xs = Xr[len(Xtr):]
            if len(Xs) == 0: raise ValueError
        except Exception:
            fb = 1
            Xr, _ = SMOTE(random_state=seed, k_neighbors=min(5, n_min - 1)).fit_resample(Xtr, ytr); Xs = Xr[len(Xtr):]
        finish(nm, Xs, extra=dict(fallback=fb))

    gen = ClusterSMOTE(Xmin, lab, k)
    for mode in ["uniform", "size", "difficulty"]:
        p0 = init_alloc(mode, Xtr, ytr, Xmin, lab, k)
        finish(f"Alloc-{mode}", gen.generate(to_counts(p0, S), seed))

    # inner split for optimisation (no test leakage)
    try:
        itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.3, stratify=ytr, random_state=seed)
    except ValueError:
        itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.3, random_state=seed)
    Xin, yin, Xva, yva = Xtr[itr], ytr[itr], Xtr[iva], ytr[iva]
    mask = yin == 1
    # cluster labels for inner-train minority points: reuse the global partition (no re-clustering)
    lab_in = lab[np.searchsorted(np.where(ytr == 1)[0], itr[mask])] if False else None
    min_idx = np.where(ytr == 1)[0]; pos = {g: i for i, g in enumerate(min_idx)}
    lab_in = np.array([lab[pos[g]] for g in itr[mask]])
    Xmin_in = Xin[mask]
    p0 = init_alloc("difficulty", Xin, yin, Xmin_in, lab_in, k)
    for a in alphas:
        p_opt, hist = caos_allocate(p0, Xin, yin, Xmin_in, lab_in, Xva, yva, make_clf, k, a, metric, seed=seed)
        finish(f"CAOS-a{a}", gen.generate(to_counts(p_opt, S), seed), extra=dict(iters=len(hist) - 1, Jgain=hist[-1][0] - hist[0][0]))
    return out

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results.csv"); ap.add_argument("--clfs", default="kNN,DT,SVM,RF")
    ap.add_argument("--datasets", default=""); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2); ap.add_argument("--alphas", default="0,0.5,1.0")
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]
    D = load_datasets()
    names = a.datasets.split(",") if a.datasets else list(D)
    rows = []; t0 = time.time()
    for dn in names:
        X, y, frac = D[dn]
        if frac < 1:
            X, _, y, _ = train_test_split(X, y, train_size=frac, stratify=y, random_state=0)
        print(f"[{time.time()-t0:6.0f}s] {dn} n={len(y)} d={X.shape[1]} min={int(y.sum())} IR={((y==0).sum()/y.sum()):.1f}", flush=True)
        rskf = RepeatedStratifiedKFold(n_splits=a.folds, n_repeats=a.repeats, random_state=42)
        for f, (tr, te) in enumerate(rskf.split(X, y)):
            for cn in a.clfs.split(","):
                res = run_fold(X[tr], y[tr], X[te], y[te], cn, seed=f, alphas=alphas)
                for m, r in res.items():
                    rows.append(dict(dataset=dn, fold=f, clf=cn, method=m, **r))
            pd.DataFrame(rows).to_csv(a.out, index=False)
    print("done", time.time() - t0)
