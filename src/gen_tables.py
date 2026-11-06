import glob, numpy as np, pandas as pd
def load(p): return pd.concat([pd.read_csv(f,keep_default_na=False) for f in sorted(glob.glob(p))],ignore_index=True)
ff=load('r_final/ff_*.csv'); sp=load('r_final/spf_*.csv'); rf=load('r_final/rff_*.csv')
for d in (ff,sp,rf):
    d['fold']=pd.to_numeric(d.fold)
    for c in ('gmean','f1','hpn','ap'): d[c]=pd.to_numeric(d[c])
sp=sp[(sp.clf!='RF')|(sp.fold<10)]; rf=rf[rf.fold<10]
fast=pd.concat([ff,sp[sp.clf!='RF']],ignore_index=True)           # kNN,DT,SVM,LR
allr=pd.concat([fast,rf,sp[sp.clf=='RF']],ignore_index=True)
st=pd.read_csv('keel_stats.csv',keep_default_na=False)
names=[x for x in open('names_keel.txt').read().replace('\n',',').split(',') if x]
st=st[st['name'].isin(names)]; st['n']=st.n.astype(int); st['IR']=st.IR.astype(float)
st=st.sort_values(['n','name']).reset_index(drop=True); assert len(st)==59,len(st)
def esc(s): return s.replace('_','\\_')
# ---------- Table: data sets
rows=[(esc(r['name']),int(r.n),int(r.d),int(r.nmin),float(r.IR)) for _,r in st.iterrows()]
h=(len(rows)+1)//2; L,R=rows[:h],rows[h:]
t=[r"\begin{table*}[t]",r"\centering",r"\caption{The 59 KEEL data sets: number of examples $n$, features $d$, minority examples $n_{\min}$ and imbalance ratio IR. Sorted by $n$.}",r"\label{tab:datalist}",r"\scriptsize",r"\setlength{\tabcolsep}{3pt}",
 r"\begin{tabular}{@{}lrrrr@{\hspace{14pt}}lrrrr@{}}",r"\toprule",r"Data set & $n$ & $d$ & $n_{\min}$ & IR & Data set & $n$ & $d$ & $n_{\min}$ & IR \\",r"\midrule"]
for i in range(h):
    a=L[i]; b=R[i] if i<len(R) else None
    s=f"{a[0]} & {a[1]} & {a[2]} & {a[3]} & {a[4]:.1f}"
    s+=(f" & {b[0]} & {b[1]} & {b[2]} & {b[3]} & {b[4]:.1f}" if b else " & & & & &")
    t.append(s+r" \\")
t+=[r"\bottomrule",r"\end{tabular}",r"\end{table*}"]; open('../tables/tab_datalist.tex','w').write('\n'.join(t)+'\n')
# ---------- Table: per-data-set G-mean
M=[('SMOTE-ENN','SMOTE-ENN'),('SMOTE','SMOTE'),('Alloc-uniform','Unif.'),('CAOS-a0.5','CAOS'),('Alloc-difficulty','Diff.'),('ThreshTune','Thr.')]
G=fast.groupby(['dataset','method']).gmean.mean().unstack()
order=list(st['name']); G=G.loc[order,[m for m,_ in M]]
def blk(df):
    out=[]
    for ds,r in df.iterrows():
        best=r.max(); cells=[(f"\\textbf{{{v:.3f}}}" if v==best else f"{v:.3f}") for v in r.values]
        out.append(esc(ds)+' & '+' & '.join(cells))
    return out
L=blk(G.iloc[:h]); R=blk(G.iloc[h:])
t=[r"\begin{table*}[t]",r"\centering",r"\caption{G-mean per data set (mean over folds and the four fast classifiers; best of the six columns in bold). Unif.: uniform allocation; Diff.: difficulty-proportional allocation; Thr.: threshold tuning on the original data.}",r"\label{tab:perdataset}",r"\scriptsize",r"\setlength{\tabcolsep}{2.5pt}",
 r"\begin{tabular}{@{}lcccccc@{\hspace{12pt}}lcccccc@{}}",r"\toprule"]
hd=' & '.join(['Data set']+[n for _,n in M]); t.append(hd+' & '+hd+r' \\'); t.append(r"\midrule")
for i in range(h):
    t.append(L[i]+' & '+(R[i] if i<len(R) else ' & '*6)+r" \\")
t+=[r"\bottomrule",r"\end{tabular}",r"\end{table*}"]; open('../tables/tab_perdataset.tex','w').write('\n'.join(t)+'\n')
# facts for text
d_su=(G['SMOTE']-G['Alloc-uniform']); d_cu=(G['CAOS-a0.5']-G['Alloc-uniform'])
print('SMOTE>=Unif',int((d_su>0).sum()),'ties',int((d_su==0).sum()),'of',len(d_su)); print('top SMOTE-Unif',d_su.sort_values().tail(3).round(3).to_dict()); print('bottom',d_su.sort_values().head(3).round(3).to_dict())
print('CAOS max |diff|',float(d_cu.abs().max()),d_cu.abs().idxmax(), 'wins',int((d_cu>0).sum()))
print('best counts',G.idxmax(axis=1).value_counts().to_dict())
# ---------- Table: all metrics per classifier
K=[('SMOTE-ENN','SMOTE-ENN'),('SMOTE','SMOTE'),('Alloc-uniform','Uniform'),('CAOS-a0.5','CAOS'),('Alloc-difficulty','Difficulty'),('Alloc-sparsity-1','Sparsity-1'),('KMeansSMOTE','K-means SMOTE'),('ThreshTune','Threshold tuning'),('None','None')]
cl=['kNN','DT','SVM','LR','RF']; mets=[('gmean','G-mean'),('ap','AUC-PR'),('f1','F1'),('hpn','HPN')]
A=allr.groupby(['clf','method',allr.dataset])[[m for m,_ in mets]].mean().groupby(['clf','method']).mean()
t=[r"\begin{table*}[t]",r"\centering",r"\caption{Mean over the 59 data sets of G-mean, AUC-PR, F1 and HPN for each classifier (random forest: 2$\times$5-fold, others 5$\times$5-fold). Best value of each column in bold.}",r"\label{tab:allmetrics}",r"\scriptsize",r"\setlength{\tabcolsep}{2.6pt}",
 r"\begin{tabular}{@{}l"+("ccccc@{\\hspace{8pt}}"*3)+"ccccc@{}}",r"\toprule"]
t.append(' & '+' & '.join(f"\\multicolumn{{5}}{{c}}{{{n}}}" for _,n in mets)+r" \\")
t.append(r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}\cmidrule(lr){12-16}\cmidrule(lr){17-21}")
t.append('Method & '+' & '.join(cl*4)+r" \\"); t.append(r"\midrule")
best={(m,c):max(A.loc[(c,mm),m] for mm,_ in K) for m,_ in mets for c in cl}
for mm,nm in K:
    cells=[]
    for m,_ in mets:
        for c in cl:
            v=A.loc[(c,mm),m]; cells.append(f"\\textbf{{{v:.3f}}}" if abs(v-best[(m,c)])<1e-12 else f"{v:.3f}")
    t.append(nm+' & '+' & '.join(cells)+r" \\")
t+=[r"\bottomrule",r"\end{tabular}",r"\end{table*}"]; open('../tables/tab_allmetrics.tex','w').write('\n'.join(t)+'\n')
print(A.loc[('RF','None')].round(3).to_dict())
