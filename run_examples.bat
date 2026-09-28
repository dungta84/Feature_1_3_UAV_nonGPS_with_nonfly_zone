@echo off
REM Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
REM Institute of Mathematical and Computational Sciences (IMACS)
REM Contact: Prof. Phan Thanh An, thanhan@hcmut.edu.vn
REM          Tran Anh Dung, trananhdung@iuh.edu.vn, tadung.sdh231@hcmut.edu.vn
REM Released under the MIT License (see LICENSE).
REM Recompute Examples 4.2 and 4.3 on this computer. Double-click this file.
cd /d "%~dp0"
chcp 65001 > nul
if not exist results_examples mkdir results_examples
echo Started %date% %time% > results_examples\run_log.txt
python --version >> results_examples\run_log.txt 2>&1
python -c "import shapely, numpy, pandas, cv2, scipy; print('shapely', shapely.__version__, 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'opencv', cv2.__version__)" >> results_examples\run_log.txt 2>&1
set MPLBACKEND=Agg
REM Vietnamese messages in the scripts: force UTF-8 output when printing to the log file
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python run_examples.py >> results_examples\run_log.txt 2>&1
echo Exit code %errorlevel% >> results_examples\run_log.txt
echo Finished %date% %time% >> results_examples\run_log.txt
echo.
echo Done. See results_examples\run_log.txt
pause
