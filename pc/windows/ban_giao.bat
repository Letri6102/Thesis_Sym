@echo off
setlocal
rem Ban giao cho Colab: dong goi run do (partial_...) va run xong (aug1_/refit1_) vao pc_runs\upload.
rem Run do duoc danh dau HANDED_OFF: may nay KHONG train tiep run do cho toi khi nhan ban moi ve.
for %%I in ("%~dp0..\..") do set "REPO=%%~fI"
set "PYTHONUTF8=1"
set "VPY=%REPO%\.venv-pc\Scripts\python.exe"
cd /d "%REPO%"
if not exist "%VPY%" (echo [LOI] Chua cai dat moi truong: chay train_data_co_san.bat truoc & pause & exit /b 1)
echo Phai DONG cua so train truoc khi ban giao (bam X hoac Ctrl+C trong cua so do).
choice /c YN /m "Da dong cua so train chua"
if errorlevel 2 (echo Dong cua so train roi chay lai file nay. & pause & exit /b 1)
"%VPY%" pc\train_18_run.py send --old-session-dead || goto :fail
echo.
echo Chep TAT CA file trong %REPO%\pc_runs\upload ve may ca nhan, keo vao MyDrive/Sym/upload/,
echo roi chay notebook Train_Colab tren Colab (cell I roi cell G2). Chi xoa zip o day sau khi cell I bao DA NHAP.
explorer "%REPO%\pc_runs\upload"
pause
exit /b 0

:fail
echo.
echo [DUNG] Co loi o tren. Chep nguyen thong bao gui Claude.
pause
exit /b 1
