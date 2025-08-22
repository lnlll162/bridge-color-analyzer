@echo off
chcp 65001 >nul
echo ========================================
echo 桥梁颜色分析器 - 启动脚本
echo ========================================
echo.

echo 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到Python环境！
    echo 请先安装Python 3.7或更高版本
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python环境检查通过！
echo.

echo 检查并安装依赖库...
echo 正在安装必要的Python库...
pip install opencv-python numpy matplotlib pillow

if errorlevel 1 (
    echo 警告：某些依赖库安装失败，尝试使用国内镜像源...
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ opencv-python numpy matplotlib pillow
)

echo.
echo 启动桥梁颜色分析器...
echo.
python "桥梁颜色分析器_最终版.py"

if errorlevel 1 (
    echo.
    echo 程序运行出错！
    echo 请检查错误信息并确保所有依赖库已正确安装
    pause
)

echo.
echo 程序已退出
pause
