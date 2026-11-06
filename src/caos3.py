"""CAOS-U: start from uniform allocation, move away only when inner-CV validation says it helps.
Variants: greedy (accept any gain), margin (gain > 0.02), pen (paper objective P + alpha*H, alpha=0.5).
rho fixed at 1.0 to isolate the allocation effect. Same outer folds/seeds as caos.py / caos2.py / synth.py.
"""
import argparse, time, warnings
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from caos import load_datasets, metrics, entropy, ClusterSMOTE, to_counts, CLF
warnings.filterwarnings("ignore")

VARIANTS = {"U-greedy": dict(kind="greedy"), "U-margin": dict(kind="margin", margin=0.02), "U-pen": dict(kind="pen", alpha=0.5)}

def search(Xtr, ytr, lab, k, make_clf, seed, var, T=6, M=3, beta=0.10, cache=None, folds=None):
    pos = {g: i for i, g in enumerate(np.where(ytr == 1)[0])}
    if folds is None:
        folds = []
        for itr, iva in StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr):
            Xin, yin = Xtr[itr], ytr[itr]; mk = yin == 1
            lab_in = np.array([lab[pos[g]] for g in itr[mk]])
            folds.append((Xin, yin, iva, ClusterSMOTE(Xin[mk], lab_in, k), int((yin == 0).sum() - mk.sum())))
    sizes = np.bincount(lab, minlength=k)

    def P_of(p):
        key = tuple(np.round(p, 6))
        if key in cache: return cache[key]
        gs = []
        for Xin, yin, iva, gen, S in folds:
            Xs = gen.generate(to_counts(p, S), seed)
            Xa = np.vstack([Xin, Xs]) if len(Xs) else Xin; ya = np.r_[yin, np.ones(len(Xs), int)]
            gs.append(metrics(ytr[iva], make_clf().fit(Xa, ya).predict(Xtr[iva]))["gmean"])
        cache[key] = float(np.mean(gs)); return cache[key]

    p = (sizes > 0).astype(float); p /= p.sum()                    # uniform over non-empty clusters
    P = P_of(p); rng = np.random.RandomState(seed); moved = 0
    def J(Pv, pv): return Pv + var.get("alpha", 0.0) * entropy(pv)
    Jcur = J(P, p)
    for _ in range(T):
        improved = False
        nz = np.where(sizes > 0)[0]
        pairs = [(int(nz[np.argmin(p[nz])]), int(nz[np.argmax(p[nz])]))]
        for _ in range(M - 1):
            i, j = rng.choice(nz, 2, replace=False); pairs.append((int(i), int(j)))
        for i, j in pairs:
            if i == j or p[j] < beta - 1e-12: continue
            c = p.copy(); c[i] += beta; c[j] -= beta; Pc = P_of(c)
            if var["kind"] == "greedy": ok = Pc > P + 1e-12
            elif var["kind"] == "margin": ok = Pc > P + var["margin"]
            else: ok = J(Pc, c) > Jcur + 1e-12
            if ok: p, P, Jcur, improved, moved = c, Pc, J(Pc, c), True, moved + 1
        if not improved: break
    return p, moved, folds

def run_fold3(Xtr, ytr, Xte, yte, clf_name, seed):
    make_clf = CLF[clf_name]; sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    Xmin = Xtr[ytr == 1]; n_min = len(Xmin); S = int((ytr == 0).sum() - n_min)
    k = int(max(2, min(round(np.sqrt(n_min)), n_min // 2)))
    km = KMeans(k, n_init=5, random_state=seed).fit(Xmin); lab, cent = km.labels_, km.cluster_centers_
    gen = ClusterSMOTE(Xmin, lab, k); out = {}; cache = {}; folds = None
    for name, var in VARIANTS.items():
        p, moved, folds = search(Xtr, ytr, lab, k, make_clf, seed, var, cache=cache, folds=folds)
        Xs = gen.generate(to_counts(p, S), seed)
        r = metrics(yte, make_clf().fit(np.vstack([Xtr, Xs]), np.r_[ytr, np.ones(len(Xs), int)]).predict(Xte))
        a = np.bincount(np.argmin(((Xs[:, None, :] - cent[None]) ** 2).sum(-1), 1), minlength=k); q = a / a.sum()
        r.update(H=entropy(q), Hn=entropy(q) / np.log(k), maxshare=q.max(), k=k, moves=moved); out[name] = r
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="synth"); ap.add_argument("--out", default="r3.csv")
    a = ap.parse_args(); t0 = time.time(); rows = []
    if a.mode == "synth":
        import itertools; from synth import make
        from sklearn.metrics import silhouette_score
        cfgs = list(itertools.product([2.0, 3.5, 5.0, 8.0], [3, 6], [50, 100], [0, 1])); jobs = []
        for ci, (R, m, nm, s) in enumerate(cfgs):
            X, y = make(R, m, nm, seed=1000 * ci + s); jobs.append((f"R{R}_m{m}_n{nm}_s{s}", X, y, ["kNN", "DT", "SVM"], 2))
    else:
        D = load_datasets(); jobs = []
        for dn, (X, y, frac) in D.items():
            if frac < 1: X, _, y, _ = train_test_split(X, y, train_size=frac, stratify=y, random_state=0)
            jobs.append((dn, X, y, ["kNN", "DT"] if dn in ("Mammography", "Phoneme", "Digits(8)") else ["kNN", "DT", "SVM"], 5))
    for name, X, y, clfs, reps in jobs:
        print(f"[{time.time()-t0:5.0f}s] {name}", flush=True)
        for f, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=42).split(X, y)):
            for cn in clfs:
                for mth, r in run_fold3(X[tr], y[tr], X[te], y[te], cn, seed=f).items():
                    rows.append(dict(dataset=name, fold=f, clf=cn, method=mth, **r))
        pd.DataFrame(rows).to_csv(a.out, index=False)
    print("done", time.time() - t0)
