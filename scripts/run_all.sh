#!/bin/bash
# Reproduce every result file. Run from the repository root:  bash scripts/run_all.sh [WORKERS]
# Needs the KEEL data (see README, step 2). Everything is resumable: finished data sets are skipped.
set -e
W=${1:-2}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1      # essential when running several workers
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export KEEL_ROOT=${KEEL_ROOT:-$ROOT/data/imbalanced_databases/imbalanced_databases/data/}
cd "$ROOT/results"; mkdir -p r_final ks_out gb_out
NAMES=$(tr ',' '\n' < names_keel.txt | grep -v '^$')
# E1/E2: fast classifiers (5 repeats), sparsity rules (5 repeats), random forest (2 repeats)
echo "$NAMES" | xargs -P "$W" -I{} bash -c '
  n={}
  [ -s r_final/ff_$n.csv ]  || python ../src/bench.py --names $n --clfs kNN,DT,SVM,LR --repeats 5 --out r_final/ff_$n.csv
  [ -s r_final/spf_$n.csv ] || python ../src/extra_sparsity.py --names $n --clfs kNN,DT,SVM,LR,RF --repeats 5 --out r_final/spf_$n.csv
  [ -s r_final/rff_$n.csv ] || python ../src/bench.py --names $n --clfs RF --repeats 2 --out r_final/rff_$n.csv'
# k-sensitivity and K-means SMOTE tuning (3 classifiers, 2 repeats)
echo "$NAMES" | xargs -P "$W" -I{} python ../src/ksens.py {} 2
# gradient boosting (5 repeats)
echo "$NAMES" | xargs -P "$W" -I{} python ../src/gb.py {} 5
# per-cluster recall (E3)
python ../src/run_pc_keel.py
echo ALLDONE
