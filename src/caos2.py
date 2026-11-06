"""CAOS v2 - fixes: (1) tune synthetic budget rho, (2) lambda-interpolation instead of entropy penalty,
(3) inner 3-fold CV with G-mean/HPN objective. Same outer folds/seeds as caos.py.
Run: python caos2.py --out r2.csv --datasets ... --clfs kNN,DT,SVM
"""
import argparse, time, warnings
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from caos import (load_datasets, metrics, entropy, ClusterSMOTE, to_counts, init_alloc, CLF)
warnings.filterwarnings("ignore")

RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
LAMS = [1.0, 0.5, 0.0]          # 1 = uniform across clusters, 0 = difficulty-proportional (concentrated)

def _p(p_diff, lam, sizes):
    p = (1 - lam) * p_diff + lam * np.ones_like(p_diff) / len(p_diff)
    p = p * (sizes > 0); return p / p.sum()          # never allocate to empty clusters

def smote_rho(Xin, yin, rho, seed):
    n_min, n_maj = int((yin == 1).sum()), int((yin == 0).sum()); S = int(round(rho * (n_maj - n_min)))
    if S <= 0 or n_min < 2: return np.empty((0, Xin.shape[1]))
    ratio = min(1.0, (n_min + S) / n_maj)
    Xr, _ = SMOTE(sampling_strategy=ratio, k_neighbors=min(5, n_min - 1), random_state=seed).fit_resample(Xin, yin)
    return Xr[len(Xin):]

def inner_tables(Xtr, ytr, lab, k, make_clf, seed):
    """Return dict: ('cfg',rho,lam)->[(gmean,hpn)...] and ('smote',rho)->[...] over 3 inner folds."""
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    pos = {g: i for i, g in enumerate(np.where(ytr == 1)[0])}
    T = {}
    for itr, iva in skf.split(Xtr, ytr):
        Xin, yin = Xtr[itr], ytr[itr]; mask = yin == 1
        Xmin_in = Xin[mask]; lab_in = np.array([lab[pos[g]] for g in itr[mask]])
        gen = ClusterSMOTE(Xmin_in, lab_in, k)
        S_full = int((yin == 0).sum() - mask.sum())
        p_diff = init_alloc("difficulty", Xin, yin, Xmin_in, lab_in, k)

        def ev(Xs):
            Xa = np.vstack([Xin, Xs]) if len(Xs) else Xin
            ya = np.r_[yin, np.ones(len(Xs), int)]
            m = metrics(ytr[iva], make_clf().fit(Xa, ya).predict(Xtr[iva]))
            return (m["gmean"], m["hpn"])
        base = None
        for rho in RHOS:
            if rho == 0.0:
                base = ev(np.empty((0, Xtr.shape[1])))
                for lam in LAMS: T.setdefault(("cfg", 0.0, lam), []).append(base)
                T.setdefault(("smote", 0.0), []).append(base); continue
            S = int(round(rho * S_full))
            for lam in LAMS:
                Xs = gen.generate(to_counts(_p(p_diff, lam, gen.sizes), S), seed)
                T.setdefault(("cfg", rho, lam), []).append(ev(Xs))
            T.setdefault(("smote", rho), []).append(ev(smote_rho(Xin, yin, rho, seed)))
    return {key: np.mean(v, axis=0) for key, v in T.items()}       # -> (gmean, hpn)

def run_fold2(Xtr, ytr, Xte, yte, clf_name, seed):
    make_clf = CLF[clf_name]
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    Xmin = Xtr[ytr == 1]; n_min = len(Xmin); S_full = int((ytr == 0).sum() - n_min)
    k = int(max(2, min(round(np.sqrt(n_min)), n_min // 2)))
    km = KMeans(k, n_init=5, random_state=seed).fit(Xmin); lab, cent = km.labels_, km.cluster_centers_
    gen = ClusterSMOTE(Xmin, lab, k); p_diff = init_alloc("difficulty", Xtr, ytr, Xmin, lab, k)
    tab = inner_tables(Xtr, ytr, lab, k, make_clf, seed); out = {}

    def finish(name, Xs, extra):
        Xa = np.vstack([Xtr, Xs]) if len(Xs) else Xtr; ya = np.r_[ytr, np.ones(len(Xs), int)]
        r = metrics(yte, make_clf().fit(Xa, ya).predict(Xte))
        if len(Xs) > 0:
            a = np.bincount(np.argmin(((Xs[:, None, :] - cent[None]) ** 2).sum(-1), 1), minlength=k)
            p = a / a.sum(); r.update(H=entropy(p), Hn=entropy(p) / np.log(k), maxshare=p.max())
        r.update(k=k, **extra); out[name] = r

    def pick(cands, obj):                                  # cands ordered; argmax picks first max
        return max(cands, key=lambda c: tab[c][obj])

    cfgs = [("cfg", r, l) for r in RHOS for l in LAMS]
    variants = {
        "V1-rho(unif)-G": ([("cfg", r, 1.0) for r in RHOS], 0),
        "V2-lam(rho=1)-G": ([("cfg", 1.0, l) for l in LAMS], 0),
        "V3-both-G": (cfgs, 0),
        "V3-both-HPN": (cfgs, 1),
        "SMOTE-rho-G": ([("smote", r) for r in RHOS], 0),
        "SMOTE-rho-HPN": ([("smote", r) for r in RHOS], 1),
    }
    for name, (cands, obj) in variants.items():
        best = pick(cands, obj)
        if best[0] == "smote":
            rho = best[1]; Xs = smote_rho(Xtr, ytr, rho, seed) if rho > 0 else np.empty((0, Xtr.shape[1])); lam = np.nan
        else:
            _, rho, lam = best; S = int(round(rho * S_full))
            Xs = gen.generate(to_counts(_p(p_diff, lam, gen.sizes), S), seed) if S > 0 else np.empty((0, Xtr.shape[1]))
        finish(name, Xs, dict(rho=rho, lam=lam))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="r2.csv"); ap.add_argument("--clfs", default="kNN,DT,SVM")
    ap.add_argument("--datasets", default=""); ap.add_argument("--folds", type=int, default=5); ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args(); D = load_datasets(); names = a.datasets.split(",") if a.datasets else list(D)
    rows = []; t0 = time.time()
    for dn in names:
        X, y, frac = D[dn]
        if frac < 1: X, _, y, _ = train_test_split(X, y, train_size=frac, stratify=y, random_state=0)
        print(f"[{time.time()-t0:6.0f}s] {dn}", flush=True)
        for f, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=a.folds, n_repeats=a.repeats, random_state=42).split(X, y)):
            for cn in a.clfs.split(","):
                for m, r in run_fold2(X[tr], y[tr], X[te], y[te], cn, seed=f).items():
                    rows.append(dict(dataset=dn, fold=f, clf=cn, method=m, **r))
            if f % 5 == 4: pd.DataFrame(rows).to_csv(a.out, index=False)
        pd.DataFrame(rows).to_csv(a.out, index=False)
    print("done", time.time() - t0)
