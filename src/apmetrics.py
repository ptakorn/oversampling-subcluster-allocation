"""Adds average precision (AUC-PR) to caos.metrics by recording the scores of every predict() call.
Must be imported BEFORE caos2/caos3/bench so that `from caos import metrics` binds the patched function."""
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import caos
LAST = None
def score_of(est, X):
    return est.decision_function(X) if isinstance(est, SVC) else est.predict_proba(X)[:, 1]
def _patch(cls):
    orig = cls.predict
    def predict(self, X):
        global LAST
        out = orig(self, X)
        try: LAST = score_of(self, X)
        except Exception: LAST = None
        return out
    cls.predict = predict
for _c in (KNeighborsClassifier, DecisionTreeClassifier, SVC, LogisticRegression, RandomForestClassifier): _patch(_c)
_orig_metrics = caos.metrics
def metrics(yt, yp):
    global LAST
    d = _orig_metrics(yt, yp); s, LAST = LAST, None                       # consume: never reuse stale scores
    yt = np.asarray(yt)
    d["ap"] = average_precision_score(yt, s) if (s is not None and len(s) == len(yt) and yt.sum() > 0) else np.nan
    return d
def set_scores(s):
    global LAST; LAST = s
caos.metrics = metrics
