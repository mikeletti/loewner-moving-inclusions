# loewner-moving-inclusions

Code and raw results for the article

> **Reusable Löwner Bounds for Elliptic Problems with Moving Inclusions**
> Stefano Micheletti (MOX, Dipartimento di Matematica, Politecnico di Milano).
> Manuscript being prepared for submission to *SIAM Journal on Scientific Computing* (2026).

This repository provides the reproducibility information for that article only. It contains:
- the Python scripts that produce every table and every numerical statement of the article and of its Supplementary Materials;
- the raw result files (JSON and text logs) of the runs reported there;
- randomized checks of the main algebraic statements.

Statement and section numbers below refer to the manuscript prepared for submission (Sections 1–10) and to its Supplementary Materials (Sections SM1–SM7).

Source repository: https://github.com/mikeletti/loewner-moving-inclusions. Archived v1.0.0 release: https://doi.org/10.5281/zenodo.23442339 (DOI: 10.5281/zenodo.23442339). The v1.0.0 release remains unchanged; this `main` branch may receive documentation updates. License: BSD 3-Clause (see `LICENSE`).

> **On the word "certified".** As in the article, every bound is a theorem in exact arithmetic. The scripts run in ordinary floating point, and the JSON fields `valid`, `certified` and `fp_contained` mean that the computed bounds contained an independently computed reference value. They do **not** denote interval-arithmetic validation.

## Requirements

- Python ≥ 3.10, NumPy, SciPy (`pip install -r requirements.txt`). No other dependencies.
- **Single-threaded BLAS for all timings:** `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. The `run_*.sh` and `run_*.bat` scripts set it.
- **Environments of the reported runs:**
  - Section 8 (n = 44, 52) and SM5 (n = 44, 52, 60): a 64 GB Windows workstation. Exact Python/NumPy/SciPy versions of the historical runs are not recorded in the deposited JSONs and should not be inferred from the VM smoke-test environment. To document the workstation's present versions, run `python -c "import sys,numpy,scipy;print(sys.version,numpy.__version__,scipy.__version__)"` there; please record them as *current* if versions may have changed since the reported runs.
  - The n = 36 comparison in Section 8.2 and the (k, r) study in SM5: a 2-core Linux virtual machine.
  - Smoke tests of this repository: Python 3.13, NumPy 2.5, SciPy 1.18 (Linux).
- **Memory:** peak memory of the largest runs was about 27 GB (single tile, n = 60) and 11 GB (many-query, n = 52). All other scripts need less than 2 GB.

Absolute times depend on hardware and BLAS; the ratios between methods are what the article compares.

## Repository layout

| Folder | Content |
|---|---|
| `experiments_3d/` | Three-dimensional finite-difference benchmarks (Section 8 and SM5); `loewner_online.py` is the online kernel (single factorization plus the damped iteration of Lemma 6.8 with the a posteriori bound of Proposition 6.7). |
| `fem2d/` | Mesh-agnostic Dirichlet–Neumann bracketing: unstructured P1 finite elements in 2D, and the 3D finite-difference test (Section 5). |
| `structured_2d/` | Structured-grid construction with the exact lattice kernel (SM4) and the oracle-level experiments (SM6). |
| `counterexamples/` | Remark 4.4 (gap τI not supported on range F), Remark 6.9 (damping is necessary), and the failure of the PR robust certificate under global-only admissibility (Section 7). |
| `strip_model/` | Exact modal spectrum for the periodic strip (Proposition SM3.1). |
| `comparison_2d_small/` | Small two-dimensional comparison with an optimized exact tile Woodbury (mentioned in SM5). |
| `theorem_checks/` | Randomized and adversarial checks of Theorems 3.4, 5.1, 5.3, Corollaries 3.5, 5.5, Lemmas 4.5, 6.1, 6.3, 6.8. |
| `results/` | Raw outputs of the reported runs (see the table below). |

Scripts write their outputs (JSON) into the current directory, so run each one from its own folder, as the run scripts do.

## Map: article → script → command → raw result

In the 3D scripts the last argument `r` is the rank of the randomized range finder; the truncation rank actually used is `r + 10` (oversampling). For example, `600` gives the r = 610 of the article.

### Main text

| Article | Command (from the script's folder) | Raw result in `results/` | Approx. time (2-core VM unless noted) |
|---|---|---|---|
| §5, table of relative widths (P1 FEM, collar 0.08, meshes 48², 64², 96²) and collar 0.04 | `python fem_bracketing_m.py 48`; `python fem_bracketing.py` (64², both collars); `python fem_bracketing_m.py 96` | `logs_2d/fem_refine.out` (48², 96²), `logs_2d/fem_bracketing.out` (64²) | 1–3 min each |
| §5, "Mesh-independence and d = 3": d_k/d_0, energy-optimal vs Euclidean W | `python fem_optimal.py 48` (also 64, 96, 128) | `logs_2d/fem_optimal.out` | < 1 min each |
| §5, 3D test (32³ nodes, k = 0, 8, 16, 32) | `python fd3d_bracketing.py 32` | `logs_2d/fd3d.out` | ~2 min |
| Remark 4.4, 12×12 grid, factors 5.9 and 12 | `python tauI_example.py` | printed | ~2 min |
| Remark 4.4, saturation in the contrast range | `python tauI_example_contrast_range.py` | printed | ~4 min |
| Remark 4.4, 20×20 grid, factor ≈ 9 | `python tauI_example_20x20.py` | printed | < 1 min |
| Remark 6.9, damping is necessary (eigenvalue 10/3) | `python damping_counterexample.py` | printed | < 1 s |
| §7, PR robust certificate fails under global-only admissibility (72/200 queries, factor 1.17) | `python pr_robust_counterexample.py` | printed | ~20 s |
| §8.1, many-query tables, n = 44 (27 tiles, 3240 queries) | `python bench3d_multitile.py 44 3 20 16 200 64 600` | `multitile/multitile_n44_t3_Q20_k16r200_k64r600.json`, `multitile/log_mt_n44.txt` | ~1 h (workstation) |
| §8.1, many-query tables, n = 52 (8 tiles, 1200 queries) | `python bench3d_multitile.py 52 2 25 16 200 64 600` | `multitile/multitile_n52_t2_Q25_k16r200_k64r600.json`, `multitile/log_mt_n52.txt` | ~1 h (workstation) |
| §8.2, comparison table, n = 44 (Löwner, PR robust, PR coercive, exact) | `python bench3d_portred.py 44 2 10 600 16 32 64 128` | `portred/portred_n44_t2_Q10_r600_k16-32-64-128.json`, `portred/log_pr_n44.txt` | ~45 min (workstation) |
| §8.2, comparison table, n = 36 (including PR-Gram) | `python bench3d_portred_v2.py 36 2 10 600 16 32 64 128` | `portred/container_portred2_n36_t2_Q10.json` | ~25 min (2-core VM) |
| §8.2, rank M = 458 and Gram dimension 586 at n = 36 | `python gram_rank.py 36` | printed | ~1 min |

### Supplementary Materials

| Supplement | Command | Raw result in `results/` | Approx. time |
|---|---|---|---|
| SM3, Proposition SM3.1 (mode-by-mode check, m = 32, 64, 128) | `python monotonicity_strip_spectrum.py` | printed | seconds |
| SM4, tile tables (8×8 and 16×16 tiles, r = 10, 16, 24) and single compression | `python certified_offline.py && python certified_online.py && python certified_patches.py` | `logs_2d/certified_online.out`, `logs_2d/certified_patches.out` | ~4 min |
| SM5, single-tile timing table (n = 44, 52, 60; k = 64, r = 610) | `python bench3d_v3b.py N 64 600` with N = 44, 52, 60 | `single_tile_pc/bench3d_v3b_n{44,52,60}_k64_r600.json` | 5–40 min (workstation) |
| SM5, widths relative to D and signed margins | `python bench3d_v3c.py N 64 600` | `single_tile_pc/bench3d_v3c_n{52,60}_k64_r600.json`, `single_tile_vm/bench3d_v3c_n44_k64_r600.json` | as above |
| SM5, current implementation (bounds directly on D, Lemma 6.8) | `python bench3d_v3d.py N 64 600` with N = 36, 44 | `single_tile_vm/bench3d_v3d_n{36,44}_k64_r600.json` | minutes |
| SM5, η/σ of the indefinite queries at n = 52, 60 | `python eta_sigma_check.py 52 ../results/single_tile_pc/bench3d_v3c_n52_k64_r600.json` | printed | seconds |
| SM5, convergence in (k, r) at n = 36 | `python bench3d_v3b.py 36 K R` with (K, R) = (16,200), (32,200), (32,600), (64,600) | `single_tile_vm/bench3d_v3b_n36_*.json` | minutes |
| SM5, small 2D comparison (opposite ranking) | `BENCH_M=32 BENCH_REPS=12 python monotonicity_manyquery_fair.py` (also 48, 64); `BENCH_M=64 python monotonicity_exact_tile_optimized.py` | `comparison_2d_small/*.json` | minutes |
| SM6, summary table and soft/near-void/mixed tests | `python lft_test.py`; `python hierarchy_test.py`; `python note_tests.py`; `python inertia_test.py` | `logs_2d/note_tests.out`, `logs_2d/inertia_test.out` | 1–5 min each |

In `logs_2d/certified_patches.out` the field `valid` counts agreement with a *sparse direct solve* of the perturbed system. As explained in SM4, the only disagreements occur in the rigid class, where the direct solve itself is less accurate than the certified width; against the Woodbury reference no bound is violated.

### Checks of the theory

`python theorem_checks/randomized_theorem_checks.py` takes about 20 s. It prints worst-case values over a few thousand random and adversarial instances. All of them must be at round-off level, or below the bound stated in the article.

## Run scripts

- `run_quick.sh`: smoke test with small parameters, about 10 minutes on one core. It detects failing commands and explicit `FAIL` / Python traceback markers, returning a nonzero exit code on failure. It does not reproduce the reported numbers.
- `run_paper.sh`: the full set of runs underlying the tables listed above (several hours; peak memory about 27 GB for the n = 60 single tile). `run_paper.bat` reproduces **only the main Windows 3D experiments**, not all the 2D, theorem-check, and supplemental runs. It stops on command failure.

## Notes on the raw results

- **Section 8.1.** Each JSON stores, per tile, the offline timings (`geometry_s`, `exact_offline_s`, and `offline_s`, `tau`, `eta` of each Löwner configuration). Per query it stores:
  - the class `cl`, the size `p`, the exact Woodbury value `D` and the exact online time `t_exact`;
  - for each configuration, in `lw`: the bounds `lo` and `up`, the signed margins `mlo` and `mup` relative to |D|, the width `wid`, the online time `t`, and the branch and iteration data `info`.

  `summary` holds the aggregated numbers of the tables, including the measured end-to-end totals and the linear cost model.
- **Section 8.2.** Per query and level, `lw` stores the Löwner bounds and the branch taken. `pr` stores:
  - the port-reduced value `Dr`;
  - the coercive and robust estimators with their times;
  - the checks `check_Dr_vs_Dminus` and `check_energy_identity`;
  - the true Galerkin error.

  The adaptive search at each tolerance is evaluated from these records, and the results are stored in `summary`.
- Inline comments in some scripts were written with an earlier numbering of the statements; the tables above use the article's numbering.
