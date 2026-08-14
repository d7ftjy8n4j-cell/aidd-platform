@echo off
chcp 65001 >nul
title 药尘光 · EGFR 平台启动器
echo ============================================
echo   药尘光 · EGFR抑制剂智能发现与设计平台
echo ============================================
echo.

REM ===== 解决 Windows OpenMP 运行时冲突 =====
REM conda MKL numpy/scipy (libiomp5md.dll) 与 pip numba/llvmlite (libomp.dll)
REM 共存会触发 OMP Error #15，进而导致 import shap 时进程崩溃 (0xC06D007F)。
REM 此环境变量允许两个 OpenMP 运行时共存（OpenMP 官方建议）。
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_DUPLICATE_LIB_OK=TRUE

cd /d "%~dp0"

where conda >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 conda，请先安装 Miniconda 并加入 PATH。
    pause
    exit /b 1
)

REM ===== 方式1：激活环境后直接运行 =====
call conda activate egfr-md 2>nul
if not errorlevel 1 (
    echo [环境] 已激活 egfr-md
    echo [启动] 正在启动 Streamlit，浏览器将自动打开 http://localhost:8501 ...
    start "" http://localhost:8501
    streamlit run app.py
    goto :done
)

REM ===== 方式2：conda run 兜底 =====
echo [环境] 直接激活失败，改用 conda run 方式...
echo [启动] 正在启动 Streamlit，浏览器将自动打开 http://localhost:8501 ...
start "" http://localhost:8501
conda run -n egfr-md python -m streamlit run app.py

:done
pause
