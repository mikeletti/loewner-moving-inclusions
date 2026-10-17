#!/bin/bash
# Full runs behind the tables of the article (several hours; up to ~27 GB of memory for n = 60).
# Outputs are written next to the scripts; compare them with the files in results/.
set -eu
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Section 5
( cd fem2d && python fem_bracketing_m.py 48 && python fem_bracketing.py && python fem_bracketing_m.py 96 \
  && for m in 48 64 96 128; do python fem_optimal.py $m; done && python fd3d_bracketing.py 32 )
# Remark 4.4, Remark 6.9, Section 7 counterexample
( cd counterexamples && python tauI_example.py && python tauI_example_contrast_range.py && python tauI_example_20x20.py \
  && python damping_counterexample.py && python pr_robust_counterexample.py )
# Section 8.1 and 8.2
( cd experiments_3d && python bench3d_multitile.py 44 3 20 16 200 64 600 && python bench3d_multitile.py 52 2 25 16 200 64 600 \
  && python bench3d_portred.py 44 2 10 600 16 32 64 128 && python bench3d_portred_v2.py 36 2 10 600 16 32 64 128 && python gram_rank.py 36 )
# Supplementary Materials
( cd strip_model && python monotonicity_strip_spectrum.py )
( cd structured_2d && python certified_offline.py && python certified_online.py && python certified_patches.py \
  && python lft_test.py && python hierarchy_test.py && python note_tests.py && python inertia_test.py )
( cd experiments_3d && for N in 44 52 60; do python bench3d_v3b.py $N 64 600; python bench3d_v3c.py $N 64 600; done \
  && for N in 36 44; do python bench3d_v3d.py $N 64 600; done \
  && python bench3d_v3b.py 36 16 200 && python bench3d_v3b.py 36 32 200 && python bench3d_v3b.py 36 32 600 && python bench3d_v3b.py 36 64 600 )
( cd comparison_2d_small && for m in 32 48 64; do BENCH_M=$m python monotonicity_manyquery_fair.py; done && BENCH_M=64 python monotonicity_exact_tile_optimized.py )
( cd theorem_checks && python randomized_theorem_checks.py )
