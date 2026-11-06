# Where should synthetic samples go? Allocation across minority sub-clusters in oversampling

Code, per-fold results and analysis scripts for the paper of the same title (Takorn Prexawanprasut,
submitted to IEEE Access). The study compares rules that split a synthetic-sample budget over
k-means sub-clusters of the minority class (uniform, size-, difficulty- and sparsity-proportional,
searched allocations, plain SMOTE, K-means SMOTE) with standard oversamplers, cleaning variants,
class weighting and decision-threshold tuning, on 59 KEEL data sets.

## Layout

| Path | Content |
|---|---|
| `src/` | Python code. `bench.py` (main benchmark), `caos*.py` (allocation rules and search), `extra_sparsity.py`, `ksens.py` (k-sensitivity, K-means SMOTE tuning), `gb.py` (gradient boosting), `per_cluster.py` / `run_pc_keel.py` (sub-cluster recall), `analyze_*.py`, `gen_tables.py` |
| `results/` | Per-fold results: `r_final/` (`ff_*` four fast classifiers, `rff_*` random forest, `spf_*` sparsity rules), `ks_out/`, `gb_out/`, `pc_keel_rows.csv`, plus `keel_stats.csv` and `names_keel.txt` (the 59 data sets) |
| `analysis/` | Text output of the analyses quoted in the paper |
| `figures/`, `tables/` | Figures (PDF) and generated LaTeX tables |
| `scripts/` | `run_all.sh` (re-run all experiments), `analyze_all.sh` (re-create analyses, figures and tables from stored results) |

## Reproducing the analyses (minutes)

```bash
pip install -r requirements.txt
bash scripts/analyze_all.sh
```

This regenerates `analysis/*.txt`, `figures/*.pdf` and `tables/*.tex` from the stored per-fold results.
It needs no data download. We checked that the regenerated text and tables are identical to the stored ones.

## Re-running the experiments (hours)

1. Install the requirements (same versions as above; other versions may change numbers).
2. Download the KEEL data. They are **not redistributed here**; we use the copy in
   `https://github.com/gykovacs/imbalanced_databases` (no longer maintained, mirrors the KEEL imbalanced collection;
   see also the KEEL repository, Alcalá-Fdez et al., 2011):
   ```bash
   mkdir -p data && git clone --depth 1 https://github.com/gykovacs/imbalanced_databases data/imbalanced_databases
   ```
   Alternatively set `KEEL_ROOT` to the folder that contains one sub-folder per data set.
3. `bash scripts/run_all.sh 2` (the argument is the number of parallel workers). The script is resumable.
   Single-threaded BLAS (`OMP_NUM_THREADS=1`, set in the script) is important with several workers.
   Reproduction check: for `iris0`, a fresh run of `bench.py` gives G-mean values equal to the stored ones (maximum difference 0).
4. Run `scripts/analyze_all.sh`.

Protocol: repeated stratified 5-fold CV (5 repeats for kNN, decision tree, SVM, logistic regression and gradient boosting;
2 repeats for the random forest and the k-sensitivity runs), standardization and all resampling inside the training fold,
the data set as the unit of analysis. The data sets in `results/names_keel.txt` have at least 25 minority examples.

## Notes

* `caos.py` also contains a legacy loader for UCI data sets (`UCI_DATA` environment variable); it is not used for any
  result reported in the paper.
* Methods called "CAOS" in the code are the searched allocation of the paper; `Alloc-*` are the fixed allocation rules;
  `KMeansSMOTE` is the imbalanced-learn implementation.
* Several secondary analyses are post hoc; see the Limitations section of the paper.

## License

Code: MIT (see `LICENSE`). The KEEL data keep their own terms and are not included.

## Citation

Prexawanprasut, T. "Where Should Synthetic Samples Go? An Empirical Study of Allocation Across Minority Sub-Clusters in Oversampling." Submitted to IEEE Access, 2026. (Update with the final reference.)
