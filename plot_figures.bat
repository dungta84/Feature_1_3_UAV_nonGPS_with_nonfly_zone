@echo off
REM Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
REM Institute of Mathematical and Computational Sciences (IMACS)
REM Contact: Prof. Phan Thanh An, thanhan@hcmut.edu.vn
REM          Tran Anh Dung, trananhdung@iuh.edu.vn, tadung.sdh231@hcmut.edu.vn
REM Released under the MIT License (see LICENSE).
REM Figures of Examples 4.2 and 4.3. Double-click this file.
REM The figures need the feature files written by run_examples.py; if they are
REM missing (for example right after cloning the repository), they are computed first.
cd /d "%~dp0"
chcp 65001 > nul
set MPLBACKEND=Agg
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist results_examples mkdir results_examples
echo Started %date% %time% > results_examples\plot_log.txt
set NEED_RUN=0
if not exist results_examples\ex4_2\references\reference_n5_k5.csv set NEED_RUN=1
if not exist results_examples\ex4_2\aerials\features_grid\aerial_n5_k5.csv set NEED_RUN=1
if not exist results_examples\ex4_3\references\reference_n5_k5.csv set NEED_RUN=1
if not exist results_examples\ex4_2\aerials\test_05.png set NEED_RUN=1
if "%NEED_RUN%"=="1" (
  echo Feature files not found: running run_examples.py first, this takes several minutes...
  echo Feature files not found: running run_examples.py first >> results_examples\plot_log.txt
  python run_examples.py > results_examples\run_log.txt 2>&1
  if errorlevel 1 (
    echo run_examples.py failed, see results_examples\run_log.txt >> results_examples\plot_log.txt
    echo run_examples.py failed, see results_examples\run_log.txt
    pause
    exit /b 1
  )
)
echo Drawing figures...
python plot_reference_maps.py >> results_examples\plot_log.txt 2>&1
python plot_descriptor.py ex4_2 test_04.png 5 5 4 >> results_examples\plot_log.txt 2>&1
python plot_localization.py ex4_2 5 5 >> results_examples\plot_log.txt 2>&1
python plot_localization.py ex4_2 3 3 >> results_examples\plot_log.txt 2>&1
python plot_localization.py ex4_3 5 5 >> results_examples\plot_log.txt 2>&1
python plot_localization.py ex4_3 3 3 >> results_examples\plot_log.txt 2>&1
REM compact version of Figure 5 of the paper
python plot_localization.py paper ex4_2 test_04.png 5 5 >> results_examples\plot_log.txt 2>&1
echo Exit code %errorlevel% >> results_examples\plot_log.txt
echo Finished %date% %time% >> results_examples\plot_log.txt
echo.
echo Done. Figures are in results_examples\ex4_2 and results_examples\ex4_3 (see plot_log.txt)
pause
