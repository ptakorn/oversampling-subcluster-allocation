import os, re, numpy as np, pandas as pd
ROOT=os.environ.get("KEEL_ROOT","data/imbalanced_databases/imbalanced_databases/data/")
def load_keel(name):
    path=f"{ROOT}{name}/{name}.dat"; attrs=[]; rows=[]; indata=False
    for ln in open(path,errors="ignore"):
        s=ln.strip()
        if not s: continue
        if s.lower().startswith("@attribute"):
            m=re.match(r"@attribute\s*('[^']*'|\S+)\s*(.*)",s,re.I); attrs.append((m.group(1),m.group(2).strip()))
        elif s.lower().startswith("@data"): indata=True
        elif indata and not s.startswith("%"): rows.append([t.strip() for t in s.split(",")])
    df=pd.DataFrame(rows); y=df.iloc[:,-1].str.strip().values; X=df.iloc[:,:-1]
    cols=[]
    for j in range(X.shape[1]):
        col=X.iloc[:,j]; num=pd.to_numeric(col,errors="coerce")
        if num.isna().mean()>0.5 or attrs[j][1].startswith("{"):         # nominal -> integer codes
            cols.append(pd.Categorical(col).codes.astype(float))
        else: cols.append(num.fillna(num.median()).values)
    X=np.column_stack(cols); lab=pd.Series(y).value_counts()
    pos=lab.index[-1]; yb=(y==pos).astype(int)                                  # KEEL: minority = positive
    return X.astype(float), yb
def all_names(): return sorted(d for d in os.listdir(ROOT) if os.path.isdir(ROOT+d))
if __name__=="__main__":
    rows=[]
    for n in all_names():
        try:
            X,y=load_keel(n); rows.append(dict(name=n,n=len(y),d=X.shape[1],nmin=int(y.sum()),IR=round((len(y)-y.sum())/y.sum(),1)))
        except Exception as e: rows.append(dict(name=n,n=-1,d=-1,nmin=-1,IR=str(e)[:40]))
    t=pd.DataFrame(rows); t.to_csv("keel_stats.csv",index=False)
    t["IR"]=pd.to_numeric(t.IR,errors="coerce"); print(t.describe().round(1).to_string()); print((t.n<0).sum(),"failed"); print(t[t.n<0].to_string())
    print("nmin<10:",(t.nmin<10).sum()," nmin>=20:",(t.nmin>=20).sum(), " n>5000:",(t.n>5000).sum())
