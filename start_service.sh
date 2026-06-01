#!/bin/bash
# 科学视频质量评分系统 - Linux/Mac启动脚本

set -e

PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/science_video_project"
VENV_PATH="$PROJECT_PATH/venv"

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║      科学视频质量评分系统 - Web推理服务               ║"
echo "║    Science Video Quality Ranker - Inference Server    ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# 检查项目路径
echo "[1/5] 检查项目路径..."
if [ ! -d "$PROJECT_PATH" ]; then
    echo "✗ 项目路径不存在: $PROJECT_PATH"
    exit 1
fi
echo "✓ 项目路径正确"

echo ""
echo "[2/5] 检查虚拟环境..."
if [ ! -d "$VENV_PATH" ]; then
    echo "⚠ 虚拟环境不存在，正在创建..."
    python3 -m venv "$VENV_PATH"
    echo "✓ 虚拟环境创建完成"
else
    echo "✓ 虚拟环境已存在"
fi

echo ""
echo "[3/5] 激活虚拟环境..."
source "$VENV_PATH/bin/activate"
echo "✓ 虚拟环境已激活"

echo ""
echo "[4/5] 检查依赖..."
python -c "import flask; import flask_cors" 2>/dev/null || {
    echo "⚠ 检测到缺失的依赖，正在安装..."
    pip install -q flask flask-cors
    echo "✓ 依赖已安装"
}
echo "✓ 所有依赖都已安装"

echo ""
echo "[5/5] 检查模型和配置..."
python -c "from pipeline.config import CFG; print('✓ 配置加载成功')" 2>/dev/null || {
    echo "⚠ 配置加载警告 - 某些模型路径可能不可用"
    echo "  但不影响使用随机特征的推理功能"
}

echo ""
echo "════════════════════════════════════════════════════════"
echo ""
echo "✓ 所有检查完成！"
echo ""
echo "启动参数："
echo "  主机: 0.0.0.0"
echo "  端口: 5000"
echo "  URL: http://localhost:5000"
echo ""

cd "$PROJECT_PATH"
echo "🚀 启动推理服务..."
echo ""
python app.py
