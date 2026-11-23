@echo off
setlocal EnableExtensions
rem Moi thu nam trong thu muc chua file nay (vi du D:\LVTS_TriLe): moi truong, cache pip, file tam, du lieu, ket qua.
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "TEMP=%ROOT%\tmp"
set "TMP=%ROOT%\tmp"
set "PIP_CACHE_DIR=%ROOT%\cache\pip"
set "CUDA_CACHE_PATH=%ROOT%\cache\cuda"
set "PYTHONUTF8=1"
set "VPY=%ROOT%\Thesis_Sym\.venv-pc\Scripts\python.exe"
set "DATA=%ROOT%\DISFA"
if not exist "%ROOT%\tmp" mkdir "%ROOT%\tmp"
if not exist "%ROOT%\Thesis_Sym\pc\train_18_run.py" (
  echo [LOI] Khong thay %ROOT%\Thesis_Sym. Chep NGUYEN thu muc goi, giu dung cau truc.
  goto :fail
)
cd /d "%ROOT%\Thesis_Sym"

rem Seed giao cho may nay. Colab chay cac seed con lai (cell G, SEEDS = [0, 1]). Khong de 2 may trung seed.
set "SEEDS=2"
if not exist "%VPY%" (echo [LOI] Chua cai dat: chay 1_cai_dat.bat truoc & goto :fail)
echo Train seed %SEEDS%. Bi ngat thi chay lai file nay: run do se resume.
echo Ngat UltraViewer khong sao, nhung KHONG dong cua so nay va KHONG sign out may.
"%VPY%" pc\train_18_run.py train --data "%DATA%" --seeds %SEEDS% || goto :fail
"%VPY%" pc\train_18_run.py status
echo.
echo XONG. Zip ket qua: %ROOT%\Thesis_Sym\pc_runs\upload
echo Chep cac zip ve may ca nhan, keo vao MyDrive/Sym/upload/, roi chay notebook Nhap_run_PC tren Colab.
pause
exit /b 0

:fail
echo.
echo [DUNG] Co loi o tren. Chep nguyen thong bao gui Claude.
pause
exit /b 1
