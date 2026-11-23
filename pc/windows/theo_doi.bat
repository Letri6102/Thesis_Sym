@echo off
setlocal
rem Xem tien do train: %% hoan thanh va gio xong du kien, lam moi moi 60 giay.
rem Chi doc, KHONG anh huong train dang chay. Dong cua so nay luc nao cung duoc.
for %%I in ("%~dp0..\..") do set "REPO=%%~fI"
set "PYTHONUTF8=1"
cd /d "%REPO%"
if not exist "%REPO%\.venv-pc\Scripts\python.exe" (echo [LOI] Chua cai dat moi truong: chay train_data_co_san.bat truoc & pause & exit /b 1)
"%REPO%\.venv-pc\Scripts\python.exe" pc\train_18_run.py progress --watch 60
pause
