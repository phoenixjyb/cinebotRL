# Ubuntu 24.04 Native Setup Guide

**为 Ubuntu 24.04 原生环境配置 Isaac Sim + Isaac Lab 训练环境**

## 你的系统信息

- ✅ Ubuntu 24.04.3 LTS (Noble Numbat)
- ✅ NVIDIA GeForce RTX 5060 Ti (16GB)
- ✅ Driver 575.57.08
- ✅ 当前目录：`/home/bolin/Projects/cinebotRL`

---

## 安装步骤

### 1. 系统依赖安装

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础依赖
sudo apt install -y \
    build-essential \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglu1-mesa \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxi6 \
    libxcursor1 \
    libxinerama1 \
    libxkbcommon-x11-0 \
    libsm6 \
    libice6 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libvulkan1 \
    mesa-vulkan-drivers \
    vulkan-tools \
    libegl1

# 验证 Vulkan（Isaac Sim 需要）
vulkaninfo | grep "Vulkan Instance Version" || echo "⚠️ Vulkan 配置可能有问题"
```

### 2. 下载并安装 Isaac Sim

Isaac Sim 5.0+ 支持 Ubuntu 22.04/24.04。

#### 方法 A：使用 Omniverse Launcher（推荐新手）

```bash
# 下载 Omniverse Launcher
cd ~/Downloads
wget https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.AppImage

# 添加执行权限
chmod +x omniverse-launcher-linux.AppImage

# 运行 Launcher
./omniverse-launcher-linux.AppImage

# 在 Launcher 中：
# 1. 登录 NVIDIA 账号
# 2. 进入 "Exchange" 标签
# 3. 搜索 "Isaac Sim"
# 4. 安装最新版本（建议 5.0.0 或更新版本）
```

**默认安装路径**：`~/.local/share/ov/pkg/isaac-sim-*`

#### 方法 B：直接下载压缩包（高级用户）

```bash
# 创建安装目录
mkdir -p ~/isaac-sim
cd ~/isaac-sim

# 从 NVIDIA NGC 下载（需要 NGC API Key）
# 访问：https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim
# 或使用 Docker 方式（见下文 Docker 部分）

# 如果下载了 tar.gz：
tar -xzf isaac-sim-*.tar.gz
```

### 3. 配置环境变量

```bash
# 找到 Isaac Sim 安装路径（根据实际情况调整）
ISAAC_SIM_PATH=~/.local/share/ov/pkg/isaac-sim-5.0.0

# 添加到 ~/.bashrc 或 ~/.zshrc
cat >> ~/.bashrc << 'EOF'

# Isaac Sim Environment
export ISAAC_SIM_PATH="$HOME/.local/share/ov/pkg/isaac-sim-5.0.0"
export PATH="$ISAAC_SIM_PATH:$PATH"
export LD_LIBRARY_PATH="$ISAAC_SIM_PATH/kit/libs:$LD_LIBRARY_PATH"

# Vulkan 配置
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"

# PhysX 性能优化
export PHYSX_GPU_MAX_RIGID_CONTACT_COUNT=524288

# 禁用 Gymnasium 插件（避免 Windows 兼容问题）
export GYMNASIUM_DISABLE_PLUGIN_ENTRYPOINTS=1

EOF

# 重新加载配置
source ~/.bashrc
```

### 4. 验证 Isaac Sim 安装

```bash
# 测试 Isaac Sim Python
$ISAAC_SIM_PATH/python.sh --version

# 应该输出类似：Python 3.10.x 或 3.11.x
```

### 5. 安装 Isaac Lab

Isaac Lab 2.2.0+ 作为 pip 包安装：

```bash
cd /home/bolin/Projects/cinebotRL

# 使用 Isaac Sim 的 Python 安装 Isaac Lab
$ISAAC_SIM_PATH/python.sh -m pip install --upgrade pip

# 安装 Isaac Lab（如果 pip 包可用）
$ISAAC_SIM_PATH/python.sh -m pip install isaaclab==2.2.0

# 或者从源码安装（如果 pip 包不可用）
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
git checkout v2.2.0
$ISAAC_SIM_PATH/python.sh -m pip install -e .
```

### 6. 安装项目依赖

```bash
cd /home/bolin/Projects/cinebotRL

# 安装 Stable-Baselines3 和其他依赖
$ISAAC_SIM_PATH/python.sh -m pip install \
    stable-baselines3[extra] \
    tensorboard \
    matplotlib \
    scipy

# 如果需要 ROS2 支持（可选）
# sudo apt install ros-jazzy-desktop  # Ubuntu 24.04 使用 ROS2 Jazzy
```

### 7. 注册项目任务

创建 Python 包装脚本（简化命令）：

```bash
# 创建启动脚本
cat > ~/cinebotrl.sh << 'EOF'
#!/bin/bash
# CinebotRL Training Launcher for Ubuntu

ISAAC_SIM_PATH="$HOME/.local/share/ov/pkg/isaac-sim-5.0.0"
PROJECT_ROOT="/home/bolin/Projects/cinebotRL"

cd "$PROJECT_ROOT"
"$ISAAC_SIM_PATH/python.sh" "$@"
EOF

chmod +x ~/cinebotrl.sh

# 添加到 PATH
echo 'export PATH="$HOME:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 8. 测试环境

```bash
cd /home/bolin/Projects/cinebotRL

# 测试任务注册
~/cinebotrl.sh -c "
from src.task_spec import register_isaac_lab_tasks
register_isaac_lab_tasks()
print('✅ Task registration successful')
"

# 测试环境创建（使用较少环境数）
~/cinebotrl.sh scripts/reinforcement_learning/sb3/train.py \
    --task MobileMMTrackEE-v0 \
    --num_envs 64 \
    --headless \
    --total_timesteps 100000
```

---

## 运行训练

### 基础训练命令

```bash
cd /home/bolin/Projects/cinebotRL

# 使用 Python 包装脚本
~/cinebotrl.sh scripts/reinforcement_learning/sb3/train.py \
    --task MobileMMTrackEE-v0 \
    --num_envs 512 \
    --headless \
    --total_timesteps 5000000

# 或者直接使用 Isaac Sim Python
$ISAAC_SIM_PATH/python.sh scripts/reinforcement_learning/sb3/train.py \
    --task MobileMMTrackEE-v0 \
    --num_envs 512 \
    --headless
```

### 推荐配置（RTX 5060 Ti 16GB）

```bash
# 中等规模训练（推荐起步）
~/cinebotrl.sh scripts/reinforcement_learning/sb3/train.py \
    --task MobileMMTrackEE-v0 \
    --num_envs 512 \
    --headless \
    --total_timesteps 10000000

# 大规模训练（充分利用 16GB）
~/cinebotrl.sh scripts/reinforcement_learning/sb3/train.py \
    --task MobileMMTrackEE-v0 \
    --num_envs 2048 \
    --headless \
    --total_timesteps 100000000
```

### 监控训练

```bash
# 打开新终端，启动 TensorBoard
tensorboard --logdir logs/sb3/ --port 6006

# 浏览器访问：http://localhost:6006
```

---

## 替代方案：Docker 部署

如果不想在系统上直接安装 Isaac Sim，可以使用 Docker：

```bash
cd /home/bolin/Projects/cinebotRL

# 安装 Docker 和 NVIDIA Container Toolkit
sudo apt install -y docker.io
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# 构建 Docker 镜像（如果 Dockerfile 存在）
# docker build -t cinebotrl:latest .

# 或使用 NVIDIA 官方镜像
docker pull nvcr.io/nvidia/isaac-sim:4.0.0
```

---

## 常见问题

### 1. Vulkan 初始化失败

```bash
# 检查 Vulkan
vulkaninfo | grep "deviceName"

# 如果没有输出，重新配置
sudo apt install --reinstall mesa-vulkan-drivers vulkan-tools
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"
```

### 2. CUDA 版本不匹配

```bash
# 检查 CUDA 版本
nvcc --version || echo "CUDA toolkit not installed"

# Isaac Sim 5.0 内置 CUDA，通常不需要系统 CUDA
# 如果需要，安装 CUDA 12.x：
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
# sudo dpkg -i cuda-keyring_1.1-1_all.deb
# sudo apt update
# sudo apt install cuda-toolkit-12-8
```

### 3. 内存不足（OOM）

```bash
# 减少并行环境数量
--num_envs 256  # 从 512 降到 256
```

### 4. 找不到 Isaac Sim

```bash
# 手动指定路径
export ISAAC_SIM_PATH="/path/to/your/isaac-sim"
```

### 5. Ubuntu 24.04 特定问题

Ubuntu 24.04 较新，可能遇到：
- **libssl 版本问题**：`sudo apt install libssl3`
- **Python 版本问题**：Isaac Sim 要求 Python 3.10 或 3.11

---

## 性能对比

| 平台 | GPU | 并行环境数 | 预估速度 |
|------|-----|-----------|----------|
| Windows (原项目) | RTX 3090 24GB | 8,192 | ~12M steps/sec |
| Ubuntu (你的) | RTX 5060 Ti 16GB | 2,048-4,096 | ~4-8M steps/sec |

RTX 5060 Ti 虽然显存较小，但架构更新，预计训练效率接近。

---

## 下一步

安装完成后：

1. ✅ 运行测试训练（100K 步，约 5 分钟）
2. ✅ 查看 TensorBoard 确认指标记录正常
3. ✅ 启动正式训练（10M-100M 步）
4. ✅ 参考 [docs/README.md](../README.md) 了解训练策略

**需要帮助？** 查看项目文档或提问。
