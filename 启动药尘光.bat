@echo off
chcp 65001 >nul
setlocal EnableExtensions
title 药尘光 - 本地启动器

rem ============================================================
rem  NOTE (keep this file's comments ASCII-only!)
rem  cmd.exe + chcp 65001 mis-parses .bat files that contain
rem  non-ASCII text inside "rem" comment lines: after a few such
rem  lines the parser drifts by a byte, then tries to execute a
rem  fragment of a Chinese comment ("xx" is not recognized).
rem  Chinese text in echo/title lines is fine and is what users see.
rem  So: comments = English/ASCII, user-visible output = Chinese.
rem ============================================================

rem ---- Silence duplicated OpenMP runtimes (shap/torch crash 0xC06D007F without this) ----
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "OMP_DUPLICATE_LIB_OK=TRUE"

rem ---- Overridable via environment variables (also used by automated tests) ----
rem      YCG_PORT=8600            change port (default 8501)
rem      YCG_NO_BROWSER=1         do not open a browser
rem      YCG_PYTHON=<python.exe>  force an interpreter
set "PORT=8501"
if not "%YCG_PORT%"=="" set "PORT=%YCG_PORT%"
set "URL=http://localhost:%PORT%"
set "OPEN_BROWSER=1"
if "%YCG_NO_BROWSER%"=="1" set "OPEN_BROWSER=0"

cd /d "%~dp0"

echo.
echo   药尘光 - 靶点成药性与设计教学平台
echo   ==============================================
echo.

if not exist "app.py" (
    echo   [X] 当前目录里没有 app.py：
    echo       %CD%
    echo       请把 启动药尘光.bat 放在仓库根目录（与 app.py 同层）。
    echo.
    pause
    exit /b 1
)

rem ---- 1. Find the interpreter: use egfr-md's python.exe directly, no conda-on-PATH needed ----
set "PY="
for %%c in (
    "%YCG_PYTHON%"
    "%APPDATA%\mamba\envs\egfr-md\python.exe"
    "%USERPROFILE%\miniforge3\envs\egfr-md\python.exe"
    "%USERPROFILE%\miniconda3\envs\egfr-md\python.exe"
    "%USERPROFILE%\anaconda3\envs\egfr-md\python.exe"
    "%LOCALAPPDATA%\mamba\envs\egfr-md\python.exe"
) do if not defined PY if exist "%%~c" set "PY=%%~c"

if not defined PY (
    echo   [环境] 没找到 egfr-md 的 python.exe，回退 conda activate ...
    call conda activate egfr-md 2>nul
    if errorlevel 1 (
        echo   [X] conda 环境 egfr-md 也不可用。
        echo       手动建环境：conda env create -f environment_md.yml
        echo.
        pause
        exit /b 1
    )
    set "PY=python"
)

echo   [环境] %PY%
"%PY%" -c "import streamlit" 2>nul
if errorlevel 1 (
    echo   [X] 这个解释器里没有 streamlit：
    echo       %PY%
    echo       手动修复：conda activate egfr-md ^&^& pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

rem ---- 2. Already running? just open the browser; never start a 2nd instance ----
set "UP=0"
curl.exe -f -s -o nul --max-time 2 "%URL%/_stcore/health" >nul 2>&1
if not errorlevel 1 set "UP=1"

if "%UP%"=="1" (
    echo   [运行中] %URL% 已有服务在跑，直接打开浏览器。
    if "%OPEN_BROWSER%"=="1" start "" "%URL%"
    ping -n 3 127.0.0.1 >nul
    exit /b 0
)

rem ---- 3. Wait for the health endpoint in the background, then open the browser ----
rem  (curl, not PowerShell: Invoke-WebRequest honours the system proxy and times out on localhost)
if "%OPEN_BROWSER%"=="1" start "" /b cmd /c "curl.exe --retry-connrefused --retry 60 --retry-delay 1 --retry-max-time 120 -f -s -o nul %URL%/_stcore/health && start %URL%/"

echo   [启动] Streamlit 正在启动，首屏大约 5-20 秒 ...
echo          地址：%URL%
echo          停止：在本窗口按 Ctrl+C，或直接关闭本窗口。
echo.

"%PY%" -m streamlit run app.py --server.port %PORT% --server.headless true
set "RC=%errorlevel%"

echo.
if not "%RC%"=="0" (
    echo   [X] Streamlit 退出，代码 %RC%。常见原因：
    echo       - 端口 %PORT% 被别的程序占用：先 set YCG_PORT=8600 再双击本文件
    echo       - 缺依赖：conda activate egfr-md ^&^& pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo   已停止。
ping -n 3 127.0.0.1 >nul
exit /b 0
