"""k-sensitivity of allocation rules + tuned K-means SMOTE. One dataset per call; resumable.
usage: python ksens.py NAME [repeats]  -> ks_out/NAME.csv"""
import sys, os, time, warnings, numpy as np, pandas as pd
import apmetrics                                  # must precede bench/caos metrics use
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, KMeansSMOTE
import bench
from bench import CLF
from caos import metrics, ClusterSMOTE, to_counts, init_alloc, entropy
from extra_sparsity import weights
from keel import load_keel
warnings.filterwarnings("ignore")
CLFS = ["kNN", "DT", "LR"]
KMS_GRID = [dict(cbt=c, ke=e, kn=kn) for c in (0.05, 0.1, 0.3) for e in (2, 5, 10, 20) for kn in (5,)] + \
           [dict(cbt=None, ke=None, kn=5)]            # last = imblearn defaults (auto threshold, auto clusters)

def kms_sample(Xtr, ytr, cfg, seed):
    n_min = int((ytr == 1).sum())
    try:
        s = KMeansSMOTE(random_state=seed, k_neighbors=min(cfg["kn"], max(1, n_min - 1)),
                        cluster_balance_threshold=("auto" if cfg["cbt"] is None else cfg["cbt"]),
                        **({} if cfg["ke"] is None else dict(kmeans_estimator=cfg["ke"])))
        Xr, _ = s.fit_resample(Xtr, ytr); Xs = Xr[len(Xtr):]
        if len(Xs) == 0: raise ValueError
        return Xs, 0
    except Exception:
        Xr, _ = SMOTE(random_state=seed, k_neighbors=min(5, n_min - 1)).fit_resample(Xtr, ytr); return Xr[len(Xtr):], 1

def fit_eval(Xtr, ytr, Xs, Xte, yte, cn):
    Xa = np.vstack([Xtr, Xs]) if len(Xs) else Xtr; ya = np.r_[ytr, np.ones(len(Xs), int)]
    return metrics(yte, CLF[cn]().fit(Xa, ya).predict(Xte))

def main(dn, reps):
    out = f"ks_out/{dn}.csv"
    if os.path.exists(out): return
    X, y = load_keel(dn); rows = []; t0 = time.time()
    for f, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=42).split(X, y)):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte, ytr, yte = sc.transform(X[tr]), sc.transform(X[te]), y[tr], y[te]
        Xmin = Xtr[ytr == 1]; n_min = len(Xmin); S = int((ytr == 0).sum() - n_min); sq = np.sqrt(n_min)
        tags = {"k2": 2, "k3": 3, "half": round(sq / 2), "def": round(sq), "double": round(2 * sq), "quarter": round(n_min / 4)}
        tags = {t: int(max(2, min(v, n_min // 2, 60))) for t, v in tags.items()}
        cache = {}
        for kk in sorted(set(tags.values())):
            km = KMeans(kk, n_init=5, random_state=f).fit(Xmin); lab, cent = km.labels_, km.cluster_centers_
            gen = ClusterSMOTE(Xmin, lab, kk); res = {}
            for rule in ("uniform", "difficulty", "sparsity1"):
                if rule == "sparsity1": p = weights(Xmin, lab, kk, 1)
                else: p = init_alloc(rule, Xtr, ytr, Xmin, lab, kk)
                p = p * (gen.sizes > 0); p /= p.sum(); res[rule] = gen.generate(to_counts(p, S), f)
            cache[kk] = res
        for tag, kk in tags.items():
            for rule, Xs in cache[kk].items():
                for cn in CLFS:
                    r = fit_eval(Xtr, ytr, Xs, Xte, yte, cn)
                    rows.append(dict(dataset=dn, fold=f, clf=cn, method=f"{rule}", ktag=tag, k=kk, **{m: r[m] for m in ("gmean", "ap", "f1", "hpn")}))
        # K-means SMOTE: every config on test (oracle bound) + inner-CV selected config
        inner = list(StratifiedKFold(3, shuffle=True, random_state=f).split(Xtr, ytr))
        for cn in CLFS:
            test_g, inner_g, fbs = [], [], []
            for ci, cfg in enumerate(KMS_GRID):
                Xs, fb = kms_sample(Xtr, ytr, cfg, f); r = fit_eval(Xtr, ytr, Xs, Xte, yte, cn); test_g.append(r)
                fbs.append(fb); g = []
                for itr, iva in inner:
                    Xi, yi = Xtr[itr], ytr[itr]; Xsi, _ = kms_sample(Xi, yi, cfg, f); g.append(fit_eval(Xi, yi, Xsi, Xtr[iva], ytr[iva], cn)["gmean"])
                inner_g.append(np.mean(g))
                rows.append(dict(dataset=dn, fold=f, clf=cn, method="KMS-cfg", ktag=f"cfg{ci}", k=-1, fallback=fb, **{m: r[m] for m in ("gmean", "ap", "f1", "hpn")}))
            b = int(np.argmax(inner_g)); r = test_g[b]
            rows.append(dict(dataset=dn, fold=f, clf=cn, method="KMS-tuned", ktag=f"cfg{b}", k=-1, fallback=fbs[b], **{m: r[m] for m in ("gmean", "ap", "f1", "hpn")}))
            ob = int(np.argmax([t["gmean"] for t in test_g])); r = test_g[ob]
            rows.append(dict(dataset=dn, fold=f, clf=cn, method="KMS-oracle", ktag=f"cfg{ob}", k=-1, fallback=fbs[ob], **{m: r[m] for m in ("gmean", "ap", "f1", "hpn")}))
    os.makedirs("ks_out", exist_ok=True); pd.DataFrame(rows).to_csv(out, index=False); print("DONE", dn, round(time.time() - t0), flush=True)

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
