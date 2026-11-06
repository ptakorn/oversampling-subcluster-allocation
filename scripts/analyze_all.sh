#!/bin/bash
# Re-create the analysis text files, figures and LaTeX tables from the stored results (minutes, no data needed
# except results/keel_stats.csv, which is included). Run from the repository root.
set -e
cd "$(dirname "$0")/../results"; mkdir -p ../analysis ../figures ../tables
python ../src/analyze_final.py > ../analysis/analysis_final.txt
python ../src/analyze_ks.py    > ../analysis/analysis_ks.txt
python ../src/analyze_pc_keel.py > ../analysis/analysis_pc_keel.txt
python ../src/analyze_A.py  > /dev/null
python ../src/analyze_gb.py > /dev/null
python ../src/gen_tables.py
echo done
