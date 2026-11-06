"""Gradient-boosting check (HistGradientBoosting, default hyperparameters). One dataset per call; resumable.
usage: python gb.py NAME [repeats] -> gb_out/NAME.csv"""
import sys, os, time, warnings, numpy as np, pandas as pd
import apmetrics
import caos
from caos import metrics, ClusterSMOTE, to_counts, init_alloc
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from keel import load_keel
warnings.filterwarnings("ignore")
mk = lambda cw=None: HGB(max_iter=100, early_stopping=False, random_state=0, class_weight=cw)

def fitpred(Xa, ya, Xte, yte, cw=None):
    c = mk(cw).fit(Xa, ya); apmetrics.set_scores(c.predict_proba(Xte)[:, 1]); return metrics(yte, c.predict(Xte))

def alloc_resample(rule):
    def f(Xa, ya, seed):
        Xm = Xa[ya == 1]; nm = len(Xm); k = int(max(2, min(round(np.sqrt(nm)), nm // 2)))
        lab = KMeans(k, n_init=5, random_state=seed).fit_predict(Xm); gen = ClusterSMOTE(Xm, lab, k)
        S = int((ya == 0).sum() - nm); p = init_alloc(rule, Xa, ya, Xm, lab, k) * (gen.sizes > 0); p /= p.sum()
        Xs = gen.generate(to_counts(p, S), seed); return np.vstack([Xa, Xs]), np.r_[ya, np.ones(len(Xs), int)]
    return f
def smote(Xa, ya, seed): return SMOTE(random_state=seed, k_neighbors=max(1, min(5, int(ya.sum()) - 1))).fit_resample(Xa, ya)
def smoteenn(Xa, ya, seed): return SMOTEENN(smote=SMOTE(random_state=seed, k_neighbors=max(1, min(5, int(ya.sum()) - 1))), random_state=seed).fit_resample(Xa, ya)

def thr_fold(Xtr, ytr, Xte, yte, seed, resample=None):
    skf = StratifiedKFold(3, shuffle=True, random_state=seed); ss, yy = [], []
    for itr, iva in skf.split(Xtr, ytr):
        Xa, ya = (Xtr[itr], ytr[itr]) if resample is None else resample(Xtr[itr], ytr[itr], seed)
        ss.append(mk().fit(Xa, ya).predict_proba(Xtr[iva])[:, 1]); yy.append(ytr[iva])
    s_, y_ = np.concatenate(ss), np.concatenate(yy); grid = np.unique(np.quantile(s_, np.linspace(0.05, 0.99, 60)))
    best = max(grid, key=lambda t: caos.metrics(y_, (s_ >= t).astype(int))["gmean"])
    Xa, ya = (Xtr, ytr) if resample is None else resample(Xtr, ytr, seed)
    sc = mk().fit(Xa, ya).predict_proba(Xte)[:, 1]; apmetrics.set_scores(sc); return metrics(yte, (sc >= best).astype(int))

def main(dn, reps):
    out = f"gb_out/{dn}.csv"
    if os.path.exists(out): return
    X, y = load_keel(dn); rows = []; t0 = time.time()
    for f, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=42).split(X, y)):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte, ytr, yte = sc.transform(X[tr]), sc.transform(X[te]), y[tr], y[te]
        res = {}
        def ev(name, fn):
            try: res[name] = fn()
            except Exception as e: print("ERR", dn, name, type(e).__name__, str(e)[:80], file=sys.stderr)
        ev("None", lambda: fitpred(Xtr, ytr, Xte, yte))
        ev("ClassWeight", lambda: fitpred(Xtr, ytr, Xte, yte, "balanced"))
        for name, rs in (("SMOTE", smote), ("Alloc-uniform", alloc_resample("uniform")), ("Alloc-size", alloc_resample("size")),
                         ("Alloc-difficulty", alloc_resample("difficulty")), ("SMOTE-ENN", smoteenn)):
            ev(name, lambda rs=rs: fitpred(*rs(Xtr, ytr, f), Xte, yte))
        ev("ThreshTune", lambda: thr_fold(Xtr, ytr, Xte, yte, f))
        ev("SMOTE+Thresh", lambda: thr_fold(Xtr, ytr, Xte, yte, f, smote))
        ev("Uniform+Thresh", lambda: thr_fold(Xtr, ytr, Xte, yte, f, alloc_resample("uniform")))
        for m, r in res.items(): rows.append(dict(dataset=dn, fold=f, clf="GB", method=m, **{k: r[k] for k in ("gmean", "ap", "f1", "hpn", "rec", "prec")}))
    os.makedirs("gb_out", exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False); print("DONE", dn, round(time.time() - t0), flush=True)
if __name__ == "__main__": main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
