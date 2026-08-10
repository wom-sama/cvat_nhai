@echo off
setlocal
cd /d "%~dp0"

set "VENV_PYTHON=.venv\Scripts\python.exe"
set "NEED_SETUP=0"

if not exist "%VENV_PYTHON%" (
  set "NEED_SETUP=1"
) else (
  "%VENV_PYTHON%" -c "import PIL, PySide6, yaml" >nul 2>&1
  if errorlevel 1 set "NEED_SETUP=1"
)

if "%NEED_SETUP%"=="1" (
  echo Dang tao lai moi truong Python cho CVAT Nhai...
  py -m venv --clear .venv
  if errorlevel 1 goto :setup_failed
  "%VENV_PYTHON%" -m pip install --upgrade pip
  if errorlevel 1 goto :setup_failed
  "%VENV_PYTHON%" -m pip install -r requirements.txt
  if errorlevel 1 goto :setup_failed
  "%VENV_PYTHON%" -m pip install -e .
  if errorlevel 1 goto :setup_failed
)

set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
"%VENV_PYTHON%" -m cvat_nhai.app
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" pause
exit /b %APP_EXIT%

:setup_failed
echo.
echo Khong the cai moi truong CVAT Nhai. Kiem tra Python va ket noi mang.
pause
exit /b 1
