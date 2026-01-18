# PyBullet 环境快速入门 - Ubuntu 24.04

**为 Ubuntu 24.04 配置 PyBullet + Stable-Baselines3 训练环境**

适用于 `zq/hist/2123` 分支（PyBullet 原型）

---

## 系统要求

- ✅ Ubuntu 24.04 LTS
- ✅ NVIDIA GPU (RTX 5060 Ti 16GB)
- ✅ Driver 575.57+
- ✅ 10GB+ 磁盘空间

---

## 一键安装

```bash
cd /home/bolin/Projects/cinebotRL

# 切换到 PyBullet 分支
git checkout zq/hist/2123

# 运行安装脚本
bash setup_pybullet_ubuntu.sh
```

脚本会自动：
1. 创建 conda 环境 `cinebotrl-linux`
2. 安装 Python 3.11
3. 安装 PyBullet, Stable-Baselines3, PyTorch (CUDA)
4. 配置项目路径
5. 验证所有依赖

---

## 手动安装（可选）

如果自动脚本失败，按以下步骤手动安装：

### 1. 安装 Miniconda（如果没有）

```bash
cd ~/Downloads
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### 2. 创建 conda 环境

```bash
cd /home/bolin/Projects/cinebotRL
conda env create -f linux_env_dev/environment.yml
conda activate cinebotrl-linux
```

### 3. 安装 PyTorch (CUDA)

```bash
# 对于 RTX 5060 Ti (CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 或者使用 conda
# conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 4. 验证安装

```bash
python -c "import pybullet; print('PyBullet:', pybullet.__version__)"
python -c "import stable_baselines3; print('SB3:', stable_baselines3.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

---

## 快速开始训练

### 1. 测试运行（1000 步，约 1 分钟）

```bash
conda activate cinebotrl-linux
cd /home/bolin/Projects/cinebotRL/linux_env_dev

# 小规模测试
python train_pybullet_sb3.py \
    --timesteps 1000 \
    --n_envs 4 \
    --save_freq 500
```

**期望输出**：
```
Using cuda device
Logging to models/20260118_XXXXXX
Policy parameter counts: total=XXX,XXX
Training PPO for 1000 timesteps...
...
✅ Training completed!
```

### 2. 正式训练（500万步，推荐配置）

```bash
cd /home/bolin/Projects/cinebotRL/linux_env_dev

# Stage 1: 初始训练
python train_pybullet_sb3.py \
    --timesteps 5000000 \
    --n_envs 16 \
    --save_freq 50000 \
    --log_interval 10
```

**训练参数说明**：
- `--timesteps`: 总训练步数（500 万步约 2-4 小时）
- `--n_envs`: 并行环境数量（16 推荐，RTX 5060 Ti 可尝试 32）
- `--save_freq`: 每 N 步保存检查点
- `--log_interval`: 每 N 次更新打印日志

### 3. 监控训练（新终端）

```bash
conda activate cinebotrl-linux
cd /home/bolin/Projects/cinebotRL/linux_env_dev

# 启动 TensorBoard
tensorboard --logdir models/ --port 6006
```

浏览器访问：http://localhost:6006

**关键指标**：
- `rollout/ep_rew_mean`: 平均回报（应上升）
- `train/loss`: 损失（应下降）
- `train/entropy_loss`: 熵（探索程度）

---

## 配置选项

### 训练脚本参数

编辑 `linux_env_dev/train_pybullet_sb3.py` 的关键参数：

```python
# 第 184-185 行：轨迹文件路径
json_txt = "linux_env_dev/new_json_50/train_stage1.txt"  # Stage 1
# json_txt = "linux_env_dev/new_json_50/train.txt"       # Stage 2 (全量)

# 第 110-120 行：奖励权重
reward_distance_weight = 1.0
reward_yaw_weight = 1.0

# 第 145-155 行：PPO 超参数
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
gamma = 0.99
```

### 环境数量选择

| GPU | 推荐环境数 | 预估速度 |
|-----|----------|---------|
| RTX 5060 Ti 16GB | 16-32 | 1-2K steps/sec |
| RTX 3090 24GB | 32-64 | 2-4K steps/sec |
| RTX 4090 24GB | 64-128 | 4-8K steps/sec |

### 分阶段训练策略

根据 README，推荐两阶段训练：

**Stage 1**（简化轨迹集）：
```bash
# 修改 train_pybullet_sb3.py 第 184 行：
json_txt = "linux_env_dev/new_json_50/train_stage1.txt"

python train_pybullet_sb3.py --timesteps 5000000 --n_envs 16
```

**Stage 2**（完整轨迹集）：
```bash
# 修改为完整训练集
json_txt = "linux_env_dev/new_json_50/train.txt"

# 从 Stage 1 检查点继续（可选）
python train_pybullet_sb3.py \
    --timesteps 10000000 \
    --n_envs 16 \
    --load_model "models/ppo_mobile_mm_stage1.zip"
```

---

## 测试训练好的模型

### 1. 配置测试脚本

编辑 `linux_env_dev/test_trained_model_traj.py`：

```python
# 第 20 行：指定模型路径
model_path = 'linux_env_dev/models/your_timestamp/ppo_mobile_mm_final.zip'

# 第 23 行：是否可视化所有轨迹
should_vis = True  # True=显示所有, False=仅显示关键轨迹
```

### 2. 运行测试

```bash
conda activate cinebotrl-linux
cd /home/bolin/Projects/cinebotRL

python linux_env_dev/test_trained_model_traj.py
```

**期望输出**：
- PyBullet GUI 窗口显示机器人执行轨迹
- 终端打印每条轨迹的成功/失败信息
- 生成统计报告

---

## 输入输出说明

### 观察空间（53 维）

来自 `linux_env_dev/pybullet_envs/mobile_mm_traj.py` 的 `_get_obs()`：

```
[0-9]   底盘状态：x, y, cos(θ), sin(θ), 6个机械臂关节角度
[10-18] 速度：底盘 vx, vy, ωz, 6个关节角速度
[19-24] 末端执行器位置姿态
[25-30] 目标位置姿态
[31-36] 位置误差
[37-42] 上一步动作
[43-52] 其他辅助信息（时间步、距离等）
```

### 动作空间（8 维）

```
[0-5] 6个机械臂关节的增量控制
[6]   底盘 x 方向速度
[7]   底盘旋转角速度
```

---

## 性能优化

### 1. 增加并行环境数

```bash
# 从 16 增加到 32（如果 GPU 内存充足）
python train_pybullet_sb3.py --n_envs 32
```

### 2. 调整 PPO 超参数

```python
# train_pybullet_sb3.py
learning_rate = 1e-4  # 降低学习率提升稳定性
n_steps = 4096        # 增加样本量（需要更多内存）
```

### 3. 使用 GPU 监控

```bash
# 另开终端
watch -n 1 nvidia-smi
```

---

## 常见问题

### 1. ImportError: No module named 'pybullet_envs'

```bash
# 添加项目路径
export PYTHONPATH="/home/bolin/Projects/cinebotRL:$PYTHONPATH"

# 或在代码中添加
import sys
sys.path.insert(0, "/home/bolin/Projects/cinebotRL")
```

### 2. CUDA out of memory

```bash
# 减少环境数
python train_pybullet_sb3.py --n_envs 8
```

### 3. 找不到轨迹文件

```bash
# 检查路径
ls linux_env_dev/new_json_50/train_stage1.txt

# 确保在项目根目录运行
cd /home/bolin/Projects/cinebotRL
```

### 4. PyBullet 无法显示 GUI

Ubuntu 24.04 可能需要：
```bash
sudo apt install -y libgl1-mesa-glx libglu1-mesa
```

---

## 与 Isaac Lab 对比

| 特性 | PyBullet (当前) | Isaac Lab (未来) |
|------|----------------|-----------------|
| 安装复杂度 | ⭐ 简单 | ⭐⭐⭐ 复杂 |
| 并行环境数 | 16-32 | 2048-8192 |
| 训练速度 | 1-2K steps/sec | 4-12M steps/sec |
| 物理精度 | 中等 | 高 |
| GPU 加速 | 有限 | 完全 |
| 适用场景 | 算法验证、快速迭代 | 大规模训练、生产部署 |

**推荐路径**：
1. **先用 PyBullet** 验证算法、调试奖励函数（几小时到几天）
2. **再迁移到 Isaac Lab** 进行大规模训练（100M+ 步）

---

## 下一步

训练完成后：

1. ✅ 分析 TensorBoard 指标
2. ✅ 测试模型性能
3. ✅ 调整奖励函数（如需要）
4. ✅ 准备迁移到 Isaac Lab（参考 [UBUNTU_NATIVE_SETUP.md](UBUNTU_NATIVE_SETUP.md)）

**需要帮助？** 查看 `linux_env_dev/README.md` 或提问。
