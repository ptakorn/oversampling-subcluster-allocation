"""Extra analyses (A): size-proportional allocation, moderators, family-robustness, rho/lambda distribution, figures."""
import glob, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr
kw = dict(keep_default_na=False, na_values=[""])
names = [n for n in open("names_keel.txt").read().replace("\n", ",").split(",") if n]
d = pd.concat([pd.read_csv(f, **kw) for f in sorted(glob.glob("r_final/*.csv"))], ignore_index=True)
d["method"] = d.method.replace("", "None"); d = d.drop_duplicates(["dataset", "fold", "clf", "method"]); d = d[~((d.clf == "RF") & (d.fold >= 10))]
d = d[d.dataset.isin(names)]; assert d.dataset.nunique() == 59, d.dataset.nunique()
st = pd.read_csv("keel_stats.csv").drop_duplicates("name").set_index("name").loc[names].astype(float)
FAST = ["kNN", "DT", "SVM", "LR"]; rng = np.random.RandomState(1)
def P(sub, met): return sub.groupby(["dataset", "method"])[met].mean().unstack()
def boot(x, B=4000): x = np.asarray(x); return np.percentile([x[rng.randint(0, len(x), len(x))].mean() for _ in range(B)], [2.5, 97.5])
def cmp(x, y):
    z = (x - y).values; p = wilcoxon(z).pvalue if np.any(z != 0) else 1.0; lo, hi = boot(z)
    return z.mean(), lo, hi, int((z > 0).sum()), int((z < 0).sum()), p
out = []
def pr(*a): s = " ".join(str(x) for x in a); print(s); out.append(s)
# ---- A1 size-proportional
pr("=== A1: size-proportional allocation (dataset-level) ===")
for lab, sub in (("fast", d[d.clf.isin(FAST)]), ("RF", d[d.clf == "RF"])):
    for met in ("gmean", "ap", "hpn"):
        M = P(sub, met)
        for a, b in (("Alloc-size", "Alloc-uniform"), ("Alloc-size", "SMOTE"), ("SMOTE", "Alloc-uniform")):
            r = cmp(M[a], M[b]); pr(f"{lab} {met} {a}-{b}: {r[0]:+.4f} [{r[1]:+.4f},{r[2]:+.4f}] W{r[3]}/L{r[4]} p={r[5]:.3g}")
# does size recover the SMOTE-uniform gap? share
M = P(d[d.clf.isin(FAST)], "gmean"); gap = (M["SMOTE"] - M["Alloc-uniform"]); rec = (M["Alloc-size"] - M["Alloc-uniform"])
pr("share of SMOTE-uniform gap recovered by size (ratio of means):", round(rec.mean() / gap.mean(), 2), " spearman across datasets:", np.round(spearmanr(gap, rec)[0], 2))
Hn = d[d.method == "Alloc-size"].groupby("dataset")[["Hn", "maxshare"]].apply(lambda g: g.apply(pd.to_numeric, errors="coerce").mean()); pr("size alloc Hn/maxshare mean:", Hn.mean().round(3).to_dict())
# ---- A2 moderators
pr("\n=== A2: moderators (Spearman between dataset descriptor and G-mean difference to uniform; fast clfs) ===")
M = P(d[d.clf.isin(FAST)], "gmean"); k_med = None
desc = pd.DataFrame({"log10 IR": np.log10(st.IR), "log10 n_min": np.log10(st.nmin), "log10 n": np.log10(st.n), "d": st.d})
Dm = pd.DataFrame({"SMOTE": M["SMOTE"] - M["Alloc-uniform"], "Size": M["Alloc-size"] - M["Alloc-uniform"], "Difficulty": M["Alloc-difficulty"] - M["Alloc-uniform"],
                   "CAOS": M["CAOS-a0.5"] - M["Alloc-uniform"], "SMOTE-ENN": M["SMOTE-ENN"] - M["Alloc-uniform"]})
mod_rows = []
for c in Dm: 
    for v in desc:
        r, p = spearmanr(desc[v], Dm[c]); mod_rows.append((c, v, r, p)); pr(f"{c:11s} vs {v:12s}: rho={r:+.2f} p={p:.3f}")
mod = pd.DataFrame(mod_rows, columns=["m", "v", "r", "p"]); mod.to_csv("../figures/moderators.csv", index=False)
pr("(20 tests, unadjusted; expect ~1 at p<.05 by chance)")
ter = pd.qcut(desc["log10 n_min"], 3, labels=["small", "mid", "large"]); pr("by n_min tercile (mean diff to uniform):"); pr(Dm.groupby(ter, observed=True).mean().round(4).to_string())
ter2 = pd.qcut(desc["log10 IR"], 3, labels=["low", "mid", "high"]); pr("by IR tercile:"); pr(Dm.groupby(ter2, observed=True).mean().round(4).to_string())
# ---- A3 family robustness
def fam(n):
    for f in ("yeast", "ecoli", "abalone", "kr-vs-k", "winequality", "poker", "page-blocks", "glass", "vehicle", "shuttle", "kddcup", "car-", "new_thyroid", "led7"):
        if n.startswith(f): return f.strip("-")
    return n
F = pd.Series({n: fam(n) for n in names}); pr("\n=== A3: source-family robustness ===")
pr("families (size>1):", F.value_counts()[lambda s: s > 1].to_dict(), "| number of groups:", F.nunique())
for met in ("gmean", "ap"):
    M = P(d[d.clf.isin(FAST)], met)
    for a, b in (("SMOTE", "Alloc-uniform"), ("Alloc-difficulty", "Alloc-uniform"), ("CAOS-a0.5", "Alloc-uniform"), ("SMOTE-ENN", "Alloc-uniform"), ("Alloc-size", "Alloc-uniform")):
        z = (M[a] - M[b]); g = z.groupby(F.reindex(z.index)).mean()
        p = wilcoxon(g).pvalue; lo, hi = [np.percentile([g.values[rng.randint(0, len(g), len(g))].mean() for _ in range(4000)], q) for q in (2.5, 97.5)]
        pr(f"{met} {a}-{b}: group-mean {g.mean():+.4f} [{lo:+.4f},{hi:+.4f}] groups>0 {int((g>0).sum())}/{len(g)} Wilcoxon(groups) p={p:.3g} | dataset-level {z.mean():+.4f}")
M = P(d[d.clf.isin(FAST)], "gmean"); z = M["SMOTE"] - M["Alloc-uniform"]
loo = {f: z[F != f].mean() for f in F.value_counts()[lambda s: s > 1].index}; pr("SMOTE-uniform G-mean leaving out one family:", {k: round(v, 4) for k, v in loo.items()})
pr("range leave-one-family-out:", round(min(loo.values()), 4), round(max(loo.values()), 4))
fz = z.groupby(F.reindex(z.index)).agg(["mean", "count"]); pr(fz[fz["count"] > 1].round(4).to_string())
# ---- A4 rho / lambda
pr("\n=== A4: selected rho / lambda (all fast-classifier folds) ===")
fast = d[d.clf.isin(FAST)]
for m, col in (("SMOTE-rho-G", "rho"), ("V3-both-G", "rho"), ("V3-both-G", "lam"), ("V2-lam(rho=1)-G", "lam"), ("V1-rho(unif)-G", "rho")):
    v = pd.to_numeric(fast[fast.method == m][col], errors="coerce").dropna(); pr(f"{m} {col}:", v.round(2).value_counts(normalize=True).sort_index().round(3).to_dict())
open("../analysis/analysis_A.txt", "w").write("\n".join(out))
# ================= figures =================
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200, "font.family": "serif"})
# Fig1 dot plot of differences to uniform, G-mean vs AUC-PR
sets = [("Alloc-difficulty", "Difficulty"), ("Alloc-sparsity-1", "Sparsity-1"), ("Alloc-sparsity-d", "Sparsity-$d$"), ("CAOS-a0.5", "CAOS"), ("U-greedy", "U-greedy"), ("Alloc-size", "Size"),
        ("KMeansSMOTE", "K-means SMOTE"), ("SMOTE", "SMOTE"), ("SMOTE-Tomek", "SMOTE-Tomek"), ("SMOTE-ENN", "SMOTE-ENN"), ("ThreshTune", "Threshold tuning")]
fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
for ax, met, t in zip(axs, ("gmean", "ap"), ("G-mean", "AUC-PR")):
    Mx = P(d[d.clf.isin(FAST)], met)
    for i, (m, lab) in enumerate(sets):
        if m not in Mx: continue
        z = (Mx[m] - Mx["Alloc-uniform"]).dropna(); lo, hi = boot(z); c = "#1b6ca8" if z.mean() > 0 else "#b5472f"
        ax.plot([lo, hi], [i, i], color=c, lw=1.6); ax.plot(z.mean(), i, "o", color=c, ms=4)
    ax.axvline(0, color="k", lw=.6); ax.set_title(f"{t}, difference to uniform allocation"); ax.set_xlabel("mean difference (95% bootstrap CI)")
axs[0].set_yticks(range(len(sets))); axs[0].set_yticklabels([s[1] for s in sets]); axs[0].invert_yaxis()
plt.tight_layout(); plt.savefig("../figures/fig_diff_dot.pdf"); plt.close()
# Fig2 per-dataset SMOTE - uniform sorted
M = P(d[d.clf.isin(FAST)], "gmean"); z = (M["SMOTE"] - M["Alloc-uniform"]).sort_values()
fams = ["yeast", "ecoli", "abalone", "kr-vs-k"]; cols = {"yeast": "#c9822b", "ecoli": "#3a8f5b", "abalone": "#7a56b0", "kr-vs-k": "#1b6ca8"}
fig, ax = plt.subplots(figsize=(7.1, 2.6)); ax.bar(range(len(z)), z.values, color=[cols.get(F[n], "#9a9a9a") for n in z.index], width=.8); ax.axhline(0, color="k", lw=.5)
ax.set_xticks([]); ax.set_xlabel("59 data sets, sorted"); ax.set_ylabel("G-mean: SMOTE $-$ uniform")
from matplotlib.patches import Patch; ax.legend(handles=[Patch(color=cols[f], label=f) for f in fams] + [Patch(color="#9a9a9a", label="other")], frameon=False, ncol=5, loc="upper left")
plt.tight_layout(); plt.savefig("../figures/fig_per_dataset.pdf"); plt.close()
# Fig3 moderators
fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.4), sharey=True)
for ax, v in zip(axs, ("log10 IR", "log10 n_min", "d")):
    for c, col in (("SMOTE", "#1b6ca8"), ("Difficulty", "#b5472f")):
        ax.scatter(desc[v], Dm[c], s=7, alpha=.7, color=col, label=c)
        b = np.polyfit(desc[v], Dm[c], 1); xs = np.linspace(desc[v].min(), desc[v].max(), 20); ax.plot(xs, np.polyval(b, xs), color=col, lw=1)
    ax.axhline(0, color="k", lw=.5); ax.set_xlabel(v.replace("log10 ", "$\\log_{10}$ ").replace("n_min", "$n_{\\min}$"))
axs[0].set_ylabel("G-mean $-$ uniform"); axs[0].legend(frameon=False); plt.tight_layout(); plt.savefig("../figures/fig_moderators.pdf"); plt.close()
# Fig4 k-sensitivity
k = pd.concat([pd.read_csv(f) for f in glob.glob("ks_out/*.csv")]); k = k[k.dataset.isin(names)]
kk = k[k.method.isin(["uniform", "difficulty", "sparsity1"])].groupby(["dataset", "method", "ktag"]).gmean.mean().reset_index()
order = ["k2", "k3", "half", "def", "double", "quarter"]; lab = {"k2": "2", "k3": "3", "half": "½√n", "def": "√n", "double": "2√n", "quarter": "n/4"}
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for m, col, nm in (("uniform", "#1b6ca8", "Uniform"), ("difficulty", "#b5472f", "Difficulty"), ("sparsity1", "#7a56b0", "Sparsity-1")):
    v = [kk[(kk.method == m) & (kk.ktag == t)].gmean.mean() for t in order]; ax.plot(range(6), v, "o-", color=col, ms=3, lw=1.2, label=nm)
ax.set_xticks(range(6)); ax.set_xticklabels([lab[t] for t in order]); ax.set_xlabel("number of clusters $k$ ($n_{\\min}$-dependent settings)"); ax.set_ylabel("mean G-mean"); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig("../figures/fig_ksens.pdf"); plt.close(); pr("figs done")
