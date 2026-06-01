@echo off
REM 科学视频质量评分系统 - Windows启动脚本

setlocal enabledelayedexpansion

echo.
echo ╔════════════════════════════════════════════════════════╗
echo ║      科学视频质量评分系统 - Web推理服务               ║
echo ║    Science Video Quality Ranker - Inference Server    ║
echo ╚════════════════════════════════════════════════════════╝
echo.

REM 设置项目路径
set PROJECT_PATH=%~dp0science_video_project
set VENV_PATH=%PROJECT_PATH%\venv

echo [1/5] 检查项目路径...
if not exist "%PROJECT_PATH%" (
    echo ✗ 项目路径不存在: %PROJECT_PATH%
    echo 请从正确的目录运行此脚本
    pause
    exit /b 1
)
echo ✓ 项目路径正确

echo.
echo [2/5] 检查虚拟环境...
if not exist "%VENV_PATH%" (
    echo ⚠ 虚拟环境不存在，正在创建...
    cd /d "%PROJECT_PATH%"
    python -m venv venv
    if !errorlevel! neq 0 (
        echo ✗ 创建虚拟环境失败
        pause
        exit /b 1
    )
    echo ✓ 虚拟环境创建完成
) else (
    echo ✓ 虚拟环境已存在
)

echo.
echo [3/5] 激活虚拟环境...
call "%VENV_PATH%\Scripts\activate.bat"
if !errorlevel! neq 0 (
    echo ✗ 激活虚拟环境失败
    pause
    exit /b 1
)
echo ✓ 虚拟环境已激活

echo.
echo [4/5] 检查依赖...
python -c "import flask; import flask_cors" >nul 2>&1
if !errorlevel! neq 0 (
    echo ⚠ 检测到缺失的依赖，正在安装...
    pip install -q flask flask-cors
    if !errorlevel! neq 0 (
        echo ✗ 依赖安装失败
        pause
        exit /b 1
    )
    echo ✓ 依赖已安装
) else (
    echo ✓ 所有依赖都已安装
)

echo.
echo [5/5] 检查模型和配置...
python -c "from pipeline.config import CFG; print('✓ 配置加载成功')" 2>nul
if !errorlevel! neq 0 (
    echo ⚠ 配置加载警告 - 某些模型路径可能不可用
    echo   但不影响使用随机特征的推理功能
)

echo.
echo ════════════════════════════════════════════════════════
echo.
echo ✓ 所有检查完成！
echo.
echo 启动参数：
echo   主机: 0.0.0.0
echo   端口: 5000
echo   URL: http://localhost:5000
echo.
echo 按任意键启动服务...
pause >nul

echo.
echo 🚀 启动推理服务...
echo.

cd /d "%PROJECT_PATH%"
python app.py

pause
