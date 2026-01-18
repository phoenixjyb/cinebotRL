#!/bin/bash
# PyBullet 环境快速配置脚本 - Ubuntu 24.04
# 适用于 cinebotRL 项目的 zq/hist/2123 分支

set -e  # 遇到错误立即退出

echo "=========================================="
echo "CinebotRL PyBullet 环境配置"
echo "Ubuntu 24.04 + RTX 5060 Ti"
echo "=========================================="
echo ""

# 检查当前目录
if [ ! -f "linux_env_dev/environment.yml" ]; then
    echo "❌ 错误：请在项目根目录运行此脚本"
    echo "当前目录：$(pwd)"
    echo "期望目录：/home/bolin/Projects/cinebotRL"
    exit 1
fi

# 1. 检查 conda
echo "📦 检查 conda..."
if ! command -v conda &> /dev/null; then
    echo "❌ 未找到 conda，请先安装 Miniconda 或 Anaconda"
    echo ""
    echo "安装命令："
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi
echo "✅ conda 已安装：$(conda --version)"
echo ""

# 2. 创建 conda 环境
echo "🔨 创建 conda 环境：cinebotrl-linux (Python 3.11)..."
if conda env list | grep -q "^cinebotrl-linux "; then
    echo "⚠️  环境已存在，是否重新创建？(y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        echo "删除旧环境..."
        conda env remove -n cinebotrl-linux -y
    else
        echo "跳过环境创建"
        conda activate cinebotrl-linux
        echo "✅ 已激活现有环境"
        exit 0
    fi
fi

echo "创建环境（基础依赖）..."
conda env create -f linux_env_dev/environment.yml
echo "✅ conda 环境创建成功"
echo ""

# 3. 激活环境
echo "🔄 激活环境..."
eval "$(conda shell.bash hook)"
conda activate cinebotrl-linux
echo "✅ 环境已激活：$(which python)"
echo ""

# 4. 安装 PyTorch (CUDA 版本)
echo "🚀 安装 PyTorch (CUDA 12.4 for RTX 5060 Ti)..."
echo "   这将需要几分钟..."

# 检查 CUDA 版本
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
echo "   检测到 NVIDIA Driver CUDA 版本: ${CUDA_VERSION}"

# 根据 CUDA 版本安装对应的 PyTorch
if [[ "$CUDA_VERSION" == "12."* ]]; then
    echo "   安装 PyTorch with CUDA 12.4..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "   安装 PyTorch with CUDA 11.8 (fallback)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi
echo "✅ PyTorch 安装完成"
echo ""

# 5. 验证安装
echo "🔍 验证安装..."

# 检查 Python 包
python -c "import pybullet; print('✅ PyBullet:', pybullet.__version__)"
python -c "import stable_baselines3; print('✅ Stable-Baselines3:', stable_baselines3.__version__)"
python -c "import gymnasium; print('✅ Gymnasium:', gymnasium.__version__)"
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import torch; print('✅ CUDA Available:', torch.cuda.is_available())"
python -c "import torch; print('✅ CUDA Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo ""

# 6. 设置项目路径
echo "📁 添加项目到 Python 路径..."
PYTHONPATH_LINE="export PYTHONPATH=\"/home/bolin/Projects/cinebotRL:\$PYTHONPATH\""
if ! grep -q "$PYTHONPATH_LINE" ~/.bashrc; then
    echo "$PYTHONPATH_LINE" >> ~/.bashrc
    echo "✅ 已添加到 ~/.bashrc"
else
    echo "✅ 已存在于 ~/.bashrc"
fi
export PYTHONPATH="/home/bolin/Projects/cinebotRL:$PYTHONPATH"
echo ""

# 7. 完成提示
echo "=========================================="
echo "✅ 环境配置完成！"
echo "=========================================="
echo ""
echo "📝 下一步操作："
echo ""
echo "1. 激活环境（每次使用前）："
echo "   conda activate cinebotrl-linux"
echo ""
echo "2. 测试环境（小规模训练 1000 步）："
echo "   cd linux_env_dev"
echo "   python train_pybullet_sb3.py --timesteps 1000 --n_envs 4"
echo ""
echo "3. 开始训练（推荐配置）："
echo "   python train_pybullet_sb3.py --timesteps 5000000 --n_envs 16"
echo ""
echo "4. 可视化训练过程："
echo "   tensorboard --logdir models/ --port 6006"
echo "   然后访问：http://localhost:6006"
echo ""
echo "📖 详细文档："
echo "   cat linux_env_dev/README.md"
echo ""
