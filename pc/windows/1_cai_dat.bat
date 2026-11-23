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

echo === 1/5 Tim Python 3.10-3.14
if exist "%ROOT%\python\python.exe" (set "PYEXE=%ROOT%\python\python.exe" & set "PYARG=" & goto :havepy)
for %%V in (3.12 3.13 3.11 3.14 3.10) do (
  py -%%V -c "import sys" >nul 2>&1 && (set "PYEXE=py" & set "PYARG=-%%V" & goto :havepy)
)
python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 14) else 1)" >nul 2>&1 && (set "PYEXE=python" & set "PYARG=" & goto :havepy)
echo [LOI] Khong tim thay Python 3.10-3.14. Cai Python vao ngay thu muc nay, khong can quyen admin:
echo   1. Tai https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe vao %ROOT%
echo   2. Mo cmd tai %ROOT% va chay:
echo      python-3.12.10-amd64.exe /quiet InstallAllUsers=0 TargetDir=%ROOT%\python Include_launcher=0 PrependPath=0 Shortcuts=0 Include_test=0
echo   3. Chay lai 1_cai_dat.bat
goto :fail
:havepy
echo Dung Python: %PYEXE% %PYARG%

echo === 2/5 Tao moi truong dung phien ban da khoa (tai khoang 3 GB lan dau)
nvidia-smi
"%PYEXE%" %PYARG% pc\setup_env.py || goto :fail

echo === 3/5 Trong so ImageNet (ban chinh thuc cua PyTorch, prepare se kiem SHA-256)
if not exist "%ROOT%\sym_download\resnet50-19c8e357.pth" (
  curl -L --fail -o "%ROOT%\sym_download\resnet50-19c8e357.pth.part" https://download.pytorch.org/models/resnet50-19c8e357.pth || goto :fail
  move /y "%ROOT%\sym_download\resnet50-19c8e357.pth.part" "%ROOT%\sym_download\resnet50-19c8e357.pth" >nul || goto :fail
)

echo === 4/5 Giai nen du lieu va kiem khoa
"%VPY%" pc\train_18_run.py prepare --downloads "%ROOT%\sym_download" --data "%DATA%" || goto :fail
"%VPY%" pc\train_18_run.py check --data "%DATA%" || goto :fail

echo === 5/5 Chay thu 1 epoch (phai bao SMOKE OK va khoi tao trung pilot)
"%VPY%" pc\train_18_run.py smoke --data "%DATA%" || goto :fail

echo.
echo XONG cai dat. Gui Claude cac dong SMOKE OK o tren, roi chay 2_train.bat.
pause
exit /b 0

:fail
echo.
echo [DUNG] Co loi o tren. Chup hoac chep nguyen thong bao gui Claude. Chay lai file nay sau khi sua.
pause
exit /b 1
