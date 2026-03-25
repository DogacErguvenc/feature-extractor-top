@echo off
setlocal

cd /d "%~dp0"

if not exist ".\logs" mkdir ".\logs"
if not exist ".\incoming" mkdir ".\incoming"

set "ALLOWED_IMAGE_DIR=%~dp0incoming"
if exist "%~dp0butcher_config.yaml" set "BUTCHER_CONFIG_PATH=%~dp0butcher_config.yaml"
if exist "%~dp0tray_roi.json" set "TRAY_ROI_PATH=%~dp0tray_roi.json"
if exist "%~dp0plu_budgets.json" set "PLU_BUDGETS_PATH=%~dp0plu_budgets.json"

"%~dp0terazi_backend.exe" >> "%~dp0logs\backend.log" 2>&1
