Linux PyBullet + Stable-Baselines3 minimal training setup for cinebotRL
==============================================================

Quick start
-----------
1. Create and activate env (conda):

```bash
conda env create -f linux_env_dev/environment.yml
conda activate cinebotrl-linux
```

2. Install matching PyTorch (CUDA) for your system. Example (change cuda version to match drivers):

```bash
# Example for CUDA 12.8 using pip wheels is not yet widely published; use conda if possible:
conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.8
```

3. Alternative: Python `venv` (pip) workflow (no conda):

```bash
python3 -m venv .venv_rl
source .venv_rl/bin/activate
pip install -U pip setuptools wheel

# Install a CUDA-enabled PyTorch wheel matching your driver (example for cu121):
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch

# Core deps for linux_env_dev
pip install stable-baselines3==2.0.0 gymnasium==0.28.1 pybullet tensorboard wandb matplotlib pandas pillow
```

4. Run stage1 training:

4.1 start training (stage1 list)

```bash
python linux_env_dev/train_pybullet_sb3.py --n_envs 16 --train_txt linux_env_dev/new_json_50/train_stage1.txt
```

The config of 16 envs is strongly recomended. Otherwise 8 is also accepted.

Robot selection (mobile_mm vs recomo):

```bash
# mobile_mm (default)
python linux_env_dev/train_pybullet_sb3.py --robot mobile_mm --n_envs 16

# recomo (uses /home/converge/data/yanbo/gikWBC9DOF by default)
python linux_env_dev/train_pybullet_sb3.py --robot recomo --n_envs 16

# override URDF explicitly
python linux_env_dev/train_pybullet_sb3.py --robot recomo --urdf_path /home/converge/data/yanbo/gikWBC9DOF/models/recomoDemo1/recomoDemo1.urdf --n_envs 16
```

Useful env knobs:

- `--frame_skip` (default `24`, i.e. 0.1s per RL step at 240Hz)
- `--max_steps` episode horizon

Useful training knobs (PPO):

- `--device auto|cuda|cpu` (use `cuda` to force GPU)
- `--gamma`, `--gae_lambda`, `--clip_range`, `--clip_range_vf`
- `--ent_coef`, `--vf_coef`, `--max_grad_norm`, `--target_kl`
- `--pi_layers`, `--vf_layers` (actor/critic MLP sizes)

Optional normalization:

- `--vec_normalize` enables `VecNormalize` (obs normalization); stats saved to `linux_env_dev/models/<run>/vecnormalize.pkl`
- If you enable `--vec_normalize`, deployment/inference must apply the same normalization stats to observations.


5. Run stage2 training:

5.1 start training (stage2 list)

```bash
python linux_env_dev/train_pybullet_sb3.py --n_envs 16 --train_txt linux_env_dev/new_json_50/train.txt
```

Curriculum (single run)
-----------------------

Instead of manually running stage1 then stage2, you can enable an automatic curriculum:

```bash
# Switch: train only stage1 for the first N timesteps, then stage2
python linux_env_dev/train_pybullet_sb3.py --robot recomo --n_envs 16 \
  --curriculum switch --curriculum_stage1_steps 1000000

# Mix: stage2 sampling probability ramps 0->1 over N timesteps
python linux_env_dev/train_pybullet_sb3.py --robot recomo --n_envs 16 \
  --curriculum mix --curriculum_stage1_steps 1000000
```

6. Test

6.1 set model_path = 'linux_env_dev/models/your_path_to/ppo_mobile_mm_final.zip'

6.2 set should_vis = True if you want to visualize all trajectories, set False if you just want to visualize critical trajectories

6.3 start test

```bash
python linux_env_dev/test_trained_model_traj.py
```

TensorBoard / metrics
---------------------

Training writes TensorBoard logs under `linux_env_dev/models/<run>/tensorboard_logs/`.

```bash
tensorboard --logdir linux_env_dev/models
```

Optional evaluation during training (uses `linux_env_dev/new_json_50/test.txt` by default):

```bash
python linux_env_dev/train_pybullet_sb3.py --n_envs 16 --eval_freq 20000 --eval_episodes 10
```

Basics
---------

1. Model input

输入53维（mobile_mm），详见linux_env_dev/pybullet_envs/mobile_mm_traj.py的_get_obs()函数

recomoDemo1: 输入51维（2DOF机械臂 + 全向底盘），输出5维（vx, vy, wz + 2个机械臂关节增量）

- 底盘x方向的位移
- 底盘y方向的位移
- cos(底盘theta的旋转角)
- sin(底盘theta的旋转角)
- 机械臂第一个关节的角度
- 机械臂第二个关节的角度
- 机械臂第三个关节的角度
- 机械臂第四个关节的角度
- 机械臂第五个关节的角度
- 机械臂第六个关节的角度

- 底盘x方向的速度
- 底盘y方向的速度
- 底盘theta的旋转角速度
- 机械臂第一个关节的角速度
- 机械臂第二个关节的角速度
- 机械臂第三个关节的角速度
- 机械臂第四个关节的角速度
- 机械臂第五个关节的角速度
- 机械臂第六个关节的角速度

（底盘过去3个时刻的速度）
- 前0.1s的底盘x/y/theta的（角）速度
- 前0.2s的底盘x/y/theta的（角）速度
- 前0.3s的底盘x/y/theta的（角）速度

- 底盘x/y/z位置
- 末端执行器x/y/z位置

- 目标点x/y/z位置
（间隔0.1m，预瞄5个点的位置）
- 0.1m后的预瞄点x/y/z位置
- 0.2m后的预瞄点x/y/z位置
- 0.3m后的预瞄点x/y/z位置
- 0.4m后的预瞄点x/y/z位置
- 0.5m后的预瞄点x/y/z位置

- 当前进度（剩余待跟踪点数 / 总点数，即初始时为1.0，到终点时为0.0）

2. Model output

输出8维（mobile_mm），详见linux_env_dev/pybullet_envs/mobile_mm_traj.py的line542-550
分别如下：

- 底盘加速度系数
- 底盘角速度系数
- 机械臂第一个关节角度系数
- 机械臂第二个关节角度系数
- 机械臂第三个关节角度系数
- 机械臂第四个关节角度系数
- 机械臂第五个关节角度系数
- 机械臂第六个关节角度系数

使用方法如下：
定义时间间隔delta_t = 0.1
- 底盘位移 = delta_t * 上一个时刻的底盘速度 + 0.5 * 底盘加速度系数 * 1.5 * delta_t^2
- 底盘朝向 = 上一个时刻的底盘朝向 + 0.01 * 底盘角速度系数
- 机械臂第一个关节的角度 = 上一个时刻的机械臂第一个关节的角度 + 0.02 * 机械臂第一个关节角度系数
...
- 机械臂第六个关节的角度 = 上一个时刻的机械臂第六个关节的角度 + 0.02 * 机械臂第六个关节角度系数
