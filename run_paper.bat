@echo off
rem Windows 3D subset of the full paper runs, with single-threaded BLAS.
rem Run run_paper.sh on Unix for the complete 2D/3D/theorem-check collection.
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
cd /d "%~dp0experiments_3d"
python bench3d_multitile.py 44 3 20 16 200 64 600
if errorlevel 1 exit /b 1
python bench3d_multitile.py 52 2 25 16 200 64 600
if errorlevel 1 exit /b 1
python bench3d_portred.py 44 2 10 600 16 32 64 128
if errorlevel 1 exit /b 1
for %%N in (44 52 60) do (
  python bench3d_v3b.py %%N 64 600
  if errorlevel 1 exit /b 1
  python bench3d_v3c.py %%N 64 600
  if errorlevel 1 exit /b 1
)
echo Windows 3D subset finished.
