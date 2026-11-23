@echo off
setlocal EnableExtensions
rem Train tren may GPU bang du lieu CO SAN tren may (chi doc), khong can chep goi 8 GB.
rem Dat repo o D:\LVTS_TriLe\Thesis_Sym. Moi thu ghi vao D:\LVTS_TriLe (thu muc cha cua repo).
rem Chay lai file nay de lam tiep: lan 1 cai dat + kiem du lieu + chay thu 1 epoch, lan 2 tro di train.
set "SRC=D:\KLTN_Khang\DISFA_Data"
set "SEEDS=2"
rem Neu list trong SRC khac ban chuan: chep thu muc DISFA\list cua may ca nhan thanh <thu muc lam viec>\list (file nay tu nhan).

for %%I in ("%~dp0..\..") do set "REPO=%%~fI"
for %%I in ("%REPO%\..") do set "ROOT=%%~fI"
set "TEMP=%ROOT%\tmp"
set "TMP=%ROOT%\tmp"
set "PIP_CACHE_DIR=%ROOT%\cache\pip"
set "CUDA_CACHE_PATH=%ROOT%\cache\cuda"
set "PYTHONUTF8=1"
set "DATA=%ROOT%\DISFA"
set "VPY=%REPO%\.venv-pc\Scripts\python.exe"
if not exist "%ROOT%\tmp" mkdir "%ROOT%\tmp"
if not exist "%REPO%\pc\train_18_run.py" (echo [LOI] Khong thay repo tai %REPO% & goto :fail)
cd /d "%REPO%"
echo Repo: %REPO%   Thu muc lam viec: %ROOT%   Du lieu nguon (chi doc): %SRC%

echo === 1/5 Python va moi truong dung phien ban da khoa
if exist "%ROOT%\python\python.exe" (set "PYEXE=%ROOT%\python\python.exe" & set "PYARG=" & goto :havepy)
for %%V in (3.12 3.13 3.11 3.14 3.10) do (
  py -%%V -c "import sys" >nul 2>&1 && (set "PYEXE=py" & set "PYARG=-%%V" & goto :havepy)
)
python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 14) else 1)" >nul 2>&1 && (set "PYEXE=python" & set "PYARG=" & goto :havepy)
echo [LOI] Khong tim thay Python 3.10-3.14. Cai vao %ROOT%\python, khong can admin:
echo   1. Tai https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe vao %ROOT%
echo   2. Chay: python-3.12.10-amd64.exe /quiet InstallAllUsers=0 TargetDir=%ROOT%\python Include_launcher=0 PrependPath=0 Shortcuts=0 Include_test=0
echo   3. Chay lai file nay
goto :fail
:havepy
nvidia-smi
"%PYEXE%" %PYARG% pc\setup_env.py || goto :fail

echo === 2/5 Du lieu: doi chieu va chep tu %SRC% (chi doc)
set "LISTDIR="
if exist "%ROOT%\list\DISFA_train_img_path_fold1.txt" set "LISTDIR=%ROOT%\list"
if defined LISTDIR (
  echo Dung list tai %LISTDIR%
  "%VPY%" pc\train_18_run.py from-folder --src "%SRC%" --data "%DATA%" --lists "%LISTDIR%" || goto :fail
) else (
  "%VPY%" pc\train_18_run.py from-folder --src "%SRC%" --data "%DATA%" || goto :fail
)

echo === 3/5 Kiem khoa
"%VPY%" pc\train_18_run.py check --data "%DATA%" || goto :fail

if exist "%ROOT%\smoke_ok.txt" goto :train
echo === 4/5 Chay thu 1 epoch
"%VPY%" pc\train_18_run.py smoke --data "%DATA%" || goto :fail
echo ok> "%ROOT%\smoke_ok.txt"
echo.
echo SMOKE OK. Gui Claude cac dong ket qua o tren, roi chay lai file nay de bat dau train.
pause
exit /b 0

:train
echo === 5/5 Train seed %SEEDS%. Bi ngat thi chay lai file nay: run do se resume.
echo Ngat UltraViewer khong sao, nhung KHONG dong cua so nay va KHONG sign out may.
"%VPY%" pc\train_18_run.py train --data "%DATA%" --seeds %SEEDS% || goto :fail
echo === Refit (dong b): train lai tren du 18 subject, dung o epoch E cua run (a)
"%VPY%" pc\train_18_run.py refit --data "%DATA%" --seeds %SEEDS% || goto :fail
"%VPY%" pc\train_18_run.py status
echo.
echo XONG. Zip ket qua: %REPO%\pc_runs\upload
echo Chep cac zip ve may ca nhan, keo vao MyDrive/Sym/upload/, roi chay notebook Nhap_run_PC tren Colab.
pause
exit /b 0

:fail
echo.
echo [DUNG] Co loi o tren. Chep nguyen thong bao gui Claude.
pause
exit /b 1
