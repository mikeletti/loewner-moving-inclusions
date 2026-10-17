#!/usr/bin/env bash
# Smoke test: smaller problems, not a reproduction of the published timing tables.
set -u
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p quick_logs
failures=0
run () {
  name=$1; dir=$2; shift 2
  # Save output and examine BOTH the process exit status and explicit failure markers.
  if ( cd "$dir" && "$@" ) > "quick_logs/$name.log" 2>&1; then
    if grep -Eqi '(^|[^[:alpha:]])FAIL([[:space:]:.!]|$)|Traceback \(most recent call last\)' "quick_logs/$name.log"; then
      echo "FAIL $name (failure marker in quick_logs/$name.log)"
      failures=$((failures+1))
    else
      echo "OK   $name"
    fi
  else
    echo "FAIL $name (nonzero exit; see quick_logs/$name.log)"
    failures=$((failures+1))
  fi
}
run v3b        experiments_3d  python bench3d_v3b.py 20 16 200
run v3c        experiments_3d  python bench3d_v3c.py 20 16 200
run v3d        experiments_3d  python bench3d_v3d.py 20 16 200
run eta        experiments_3d  python eta_sigma_check.py 52 ../results/single_tile_pc/bench3d_v3c_n52_k64_r600.json
run multitile  experiments_3d  python bench3d_multitile.py 20 1 2 16 200
run portred    experiments_3d  python bench3d_portred.py 20 1 2 200 8 16
run portred2   experiments_3d  python bench3d_portred_v2.py 20 1 2 200 8 16
run gram       experiments_3d  python gram_rank.py 20
run fem2d      fem2d           python fem_bracketing_m.py 32
run fem_opt    fem2d           python fem_optimal.py 32
run fd3d       fem2d           python fd3d_bracketing.py 16 quick
run strip      strip_model     python monotonicity_strip_spectrum.py
run checks     theorem_checks  python randomized_theorem_checks.py
run damping    counterexamples python damping_counterexample.py
run pr_robust  counterexamples python pr_robust_counterexample.py
run structured structured_2d   sh -c "python certified_offline.py && python certified_online.py"
run note       structured_2d   sh -c "python note_tests.py && python inertia_test.py"
run fair2d     comparison_2d_small env BENCH_M=32 BENCH_REPS=1 python monotonicity_manyquery_fair.py
if [ "$failures" -gt 0 ]; then
  echo "$failures smoke test(s) failed; see quick_logs/*.log"
  exit 1
fi
echo 'All quick smoke tests passed.'
