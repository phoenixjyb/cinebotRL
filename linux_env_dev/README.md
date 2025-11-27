Linux PyBullet + Stable-Baselines3 minimal training setup for cinebotRL
==============================================================

Quick start
-----------
1. Create and activate conda env:

```bash
conda env create -f linux_env_dev/environment.yml
conda activate cinebotrl-linux
```

2. Install matching PyTorch (CUDA) for your system. Example (change cuda version to match drivers):

```bash
# Example for CUDA 12.8 using pip wheels is not yet widely published; use conda if possible:
conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.8
```

4. Run stage1 training:

4.1 set json_txt = linux_env_dev/new_json_50/train_stage1.txt in 'linux_env_dev/train_pybullet_sb3.py'

4.2 start training

```bash
python linux_env_dev/train_pybullet_sb3.py --n_envs 16
```

The config of 16 envs is strongly recomended. Otherwise 8 is also accepted.


5. Run stage2 training:

5.1 set json_txt = linux_env_dev/new_json_50/train.txt in 'linux_env_dev/train_pybullet_sb3.py'

5.2 start training

```bash
python linux_env_dev/train_pybullet_sb3.py --n_envs 16
```

6. Test

6.1 set model_path = 'linux_env_dev/models/your_path_to/ppo_mobile_mm_final.zip'

6.2 set should_vis = True if you want to visualize all trajectories, set False if you just want to visualize critical trajectories

6.3 start test

```bash
python linux_env_dev/test_trained_model_traj.py
```

Basics
---------

1. Model input

输入53维，详见linux_env_dev/pybullet_envs/mobile_mm_traj.py的_get_obs()函数

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

输出8维，详见linux_env_dev/pybullet_envs/mobile_mm_traj.py的line542-550
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