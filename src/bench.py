"""Unified benchmark on KEEL imbalanced datasets: our methods + literature baselines + class weight + threshold tuning."""
import argparse, time, warnings, sys
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
import apmetrics            # patch caos.metrics (adds AP) before the other modules bind it
import caos, caos2, caos3
from caos import metrics
from keel import load_keel
warnings.filterwarnings("ignore")
# make every module use the same classifier set
CLF = {"kNN": lambda: KNeighborsClassifier(5), "DT": lambda: DecisionTreeClassifier(random_state=0),
       "SVM": lambda: SVC(kernel="rbf", C=1.0, gamma="scale"), "LR": lambda: LogisticRegression(max_iter=500),
       "RF": lambda: RandomForestClassifier(50, random_state=0, n_jobs=1)}
CLFW = {"DT": lambda: DecisionTreeClassifier(random_state=0, class_weight="balanced"), "SVM": lambda: SVC(kernel="rbf", gamma="scale", class_weight="balanced"),
        "LR": lambda: LogisticRegression(max_iter=500, class_weight="balanced"), "RF": lambda: RandomForestClassifier(50, random_state=0, n_jobs=1, class_weight="balanced")}
for m in (caos, caos2, caos3): m.CLF.update(CLF)

def score(clf, X):
    return clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") and not isinstance(clf, SVC) else clf.decision_function(X)

def run_extra(Xtr, ytr, Xte, yte, cn, seed):
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    n_min = int(ytr.sum()); n_maj = len(ytr) - n_min; out = {}
    def ev(name, f):
        try: out[name] = f()
        except Exception as e: pass
    def fitpred(Xa, ya, mk=CLF[cn]): return metrics(yte, mk().fit(Xa, ya).predict(Xte))
    ev("ClassWeight", lambda: fitpred(Xtr, ytr, CLFW[cn]) if cn in CLFW else (_ for _ in ()).throw(ValueError))
    kk = min(5, n_min - 1)
    ev("SVMSMOTE", lambda: fitpred(*SVMSMOTE(random_state=seed, k_neighbors=kk, m_neighbors=min(10, n_min)).fit_resample(Xtr, ytr)))
    ev("SMOTE-ENN", lambda: fitpred(*SMOTEENN(smote=SMOTE(random_state=seed, k_neighbors=kk), random_state=seed).fit_resample(Xtr, ytr)))
    ev("SMOTE-Tomek", lambda: fitpred(*SMOTETomek(smote=SMOTE(random_state=seed, k_neighbors=kk), random_state=seed).fit_resample(Xtr, ytr)))
    def cbo():                       # simplified cluster-based oversampling: duplicate within minority clusters to equal size
        k = int(max(2, min(round(np.sqrt(n_min)), n_min // 2))); Xm = Xtr[ytr == 1]
        lab = KMeans(k, n_init=5, random_state=seed).fit_predict(Xm); rng = np.random.RandomState(seed); tgt = n_maj // k; add = []
        for c in range(k):
            P = Xm[lab == c]
            if len(P) and tgt > len(P): add.append(P[rng.randint(0, len(P), tgt - len(P))])
        Xs = np.vstack(add); return fitpred(np.vstack([Xtr, Xs]), np.r_[ytr, np.ones(len(Xs), int)])
    ev("CBO-dup", cbo)
    def thr():                       # threshold moving: tune cut-off on inner CV scores, maximise G-mean
        mk = CLF[cn]; skf = StratifiedKFold(3, shuffle=True, random_state=seed)
        method = "predict_proba" if hasattr(mk(), "predict_proba") and cn != "SVM" else "decision_function"
        s = cross_val_predict(mk(), Xtr, ytr, cv=skf, method=method); s = s[:, 1] if s.ndim == 2 else s
        grid = np.unique(np.quantile(s, np.linspace(0.05, 0.99, 60)))
        best = max(grid, key=lambda t: metrics(ytr, (s >= t).astype(int))["gmean"])
        clf = mk().fit(Xtr, ytr); sc = score(clf, Xte); apmetrics.set_scores(sc); return metrics(yte, (sc >= best).astype(int))
    ev("ThreshTune", thr)
    from caos import ClusterSMOTE, to_counts
    def resamp_smote(Xa, ya):
        nm = int(ya.sum()); return SMOTE(random_state=seed, k_neighbors=max(1, min(5, nm - 1))).fit_resample(Xa, ya)
    def resamp_uniform(Xa, ya):
        Xm = Xa[ya == 1]; nm = len(Xm); k = int(max(2, min(round(np.sqrt(nm)), nm // 2)))
        lab = KMeans(k, n_init=5, random_state=seed).fit_predict(Xm); gen = ClusterSMOTE(Xm, lab, k)
        S = int((ya == 0).sum() - nm); p = (gen.sizes > 0).astype(float); p /= p.sum()
        Xs = gen.generate(to_counts(p, S), seed); return np.vstack([Xa, Xs]), np.r_[ya, np.ones(len(Xs), int)]
    def thr_after(resample):                # resample, then tune the threshold on inner-CV scores of the RESAMPLED model
        mk = CLF[cn]; skf = StratifiedKFold(3, shuffle=True, random_state=seed); ss, yy = [], []
        for itr, iva in skf.split(Xtr, ytr):
            Xa, ya = resample(Xtr[itr], ytr[itr]); clf = mk().fit(Xa, ya); ss.append(score(clf, Xtr[iva])); yy.append(ytr[iva])
        s_, y_ = np.concatenate(ss), np.concatenate(yy); grid = np.unique(np.quantile(s_, np.linspace(0.05, 0.99, 60)))
        best = max(grid, key=lambda t: metrics(y_, (s_ >= t).astype(int))["gmean"])
        Xa, ya = resample(Xtr, ytr); clf = mk().fit(Xa, ya); sc = score(clf, Xte); apmetrics.set_scores(sc)
        return metrics(yte, (sc >= best).astype(int))
    ev("SMOTE+Thresh", lambda: thr_after(resamp_smote))
    ev("Uniform+Thresh", lambda: thr_after(resamp_uniform))
    return out

def run_all(Xtr, ytr, Xte, yte, cn, seed):
    res = {}
    for fn in (lambda: caos.run_fold(Xtr, ytr, Xte, yte, cn, seed, alphas=[0.5], metric="gmean"),
               lambda: caos2.run_fold2(Xtr, ytr, Xte, yte, cn, seed),
               lambda: caos3.run_fold3(Xtr, ytr, Xte, yte, cn, seed),
               lambda: run_extra(Xtr, ytr, Xte, yte, cn, seed)):
        try: res.update(fn())
        except Exception as e: print("ERR", type(e).__name__, str(e)[:80], file=sys.stderr)
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--names", required=True); ap.add_argument("--clfs", default="kNN,DT,SVM,LR")
    ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--out", required=True); a = ap.parse_args()
    t0 = time.time(); rows = []
    for dn in a.names.split(","):
        X, y = load_keel(dn)
        for f, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats, random_state=42).split(X, y)):
            for cn in a.clfs.split(","):
                for m, r in run_all(X[tr], y[tr], X[te], y[te], cn, f).items():
                    rows.append(dict(dataset=dn, fold=f, clf=cn, method=m, **{k: v for k, v in r.items() if k in ("hpn", "f1", "gmean", "prec", "rec", "ap", "Hn", "maxshare", "rho", "lam", "moves", "iters", "fallback")}))
        pd.DataFrame(rows).to_csv(a.out, index=False); print(f"[{time.time()-t0:5.0f}s] {dn} n={len(y)} min={int(y.sum())}", flush=True)
    print("done")
