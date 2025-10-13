# 机器人规格 - 移动底盘 + 左臂

## 文档背景
- 资产来源：`assets_own/mobile_arm_whole_body`
- 解析的 URDF：`assets_own/mobile_arm_whole_body/urdf/arm_on_car_center_rotZ_neg90.urdf`
- ROS 包元数据：`assets_own/mobile_arm_whole_body/package.xml`（`mobile_arm_whole_body` 版本 0.1.0）
- 网格库根目录：`assets_own/mobile_arm_whole_body/meshes`（STL，CAD 原始单位为毫米，Z 轴向上）

## 底盘（Base）
- 主体链接：`chassis_center_link`（当前 URDF 中仅占位，无惯性/可视/碰撞定义）。
- 实体底盘网格：`meshes/cr_no_V.stl`，尚未在 URDF 中引用——准备好碰撞几何后再挂载。
- 状态：**待补充** —— 需要测量整机质量、惯性矩阵、轮距/轴距、驱动限制及差速底盘 PID 参数。
- 实测质量：30 kg（仅底盘，不含负载）。
- 轮距 / 轴距：轴距 0.35 m，轮距 0.60 m（轮毂中心到轮毂中心）。
- 速度限制：线速度暂封顶 1.5 m/s；角速度待测；车轮转速数据暂缺。
- 安装接口（`arm_mount_joint`）：
  - 父 -> 子：`chassis_center_link` -> `left_arm_base_link`。
  - 变换（单位：米/弧度）：平移 `(0.150, -0.0675, 1.050)`；旋转 `(roll=0, pitch=0, yaw=-1.5708)`，即绕 Z 轴 -90°。
  - 说明：机械臂基座位于底盘原点前方 150 mm、右侧 67.5 mm、上方 1.05 m，并绕 Z 轴顺时针 90° 朝前。

## 机械臂（左臂，6 自由度）
- 运动链：`chassis_center_link` -> `left_arm_base_link` -> `left_arm_link1` -> `left_arm_link2` -> `left_arm_link3` -> `left_arm_link4` -> `left_arm_link5` -> `left_arm_link6` -> `left_gripper_link`。
- 关节极限（弧度 / rad*s^-1 / N*m）：

| Joint             | Type     | Lower | Upper | Max Velocity | Max Effort |
|-------------------|----------|-------|-------|--------------|------------|
| left_arm_joint1   | Revolute | -2.8798 |  2.8798 | 1.6 | 40 |
| left_arm_joint2   | Revolute | 0.0    |  3.2289 | 1.6 | 40 |
| left_arm_joint3   | Revolute | -3.3161 | 0.0   | 4.0 | 27 |
| left_arm_joint4   | Revolute | -2.8798 | 2.8798 | 4.0 | 7  |
| left_arm_joint5   | Revolute | -1.6581 | 1.6581 | 4.0 | 7  |
| left_arm_joint6   | Revolute | -2.8798 | 2.8798 | 4.0 | 7  |

- 各连杆惯性参数（kg / m / kg*m^2）：

| Link                | Mass | CoM (x y z) [m]             | Inertia (ixx, iyy, izz, ixy, ixz, iyz) |
|---------------------|------|-----------------------------|-----------------------------------------|
| left_arm_base_link  | 1.658 | -0.0005634 0.038934 0.0000032 | (0.0010597, 0.0011787, 0.0010647, 1.9821E-05, -1.6752E-07, -1.9146E-07) |
| left_arm_link1      | 1.164 | 0.000015 0.105259 -0.001954 | (0.001125, 0.001084, 0.001158, 0, 0, -2.3E-05) |
| left_arm_link2      | 1.300 | -0.23622 0.016352 -0.000133 | (0.00060638, 0.0075936, 0.0075712, 0.00041817, 0.00014956, -8.0916E-06) |
| left_arm_link3      | 0.818 | 0.045114 0.054616 -0.000456 | (0.00060107, 0.0013959, 0.0015027, -0.00022467, -7.1194E-06, -9.7503E-06) |
| left_arm_link4      | 0.698 | 0.24285 0.0023784 0.0000013 | (8.45E-05, 0.00010174, 9.7044E-05, -8.2627E-07, -2.2607E-09, 5.3612E-09) |
| left_arm_link5      | 0.417 | 0.054309 0.0041811 0.0000041 | (8.3999E-05, 9.8498E-05, 0.00011333, 1.6234E-05, 7.4127E-08, -1.3811E-08) |
| left_arm_link6      | 0.037 | 0.028138 0.00000012 0.00000005 | (3.5662E-06, 2.0238E-06, 2.0238E-06, 6.6514E-12, 2.9628E-12, -4.1666E-12) |
| left_gripper_link   | 0.604 | -0.031107 -0.00000014 -0.00000014 | (0.00017588, 9.8637E-05, 0.00016512, 4.1789E-10, -5.3493E-10, -8.1856E-08) |

（科学计数法直接取自 URDF；下游工具需要可转化为十进制形式。）

## 末端执行器 / 摄影平台
- 末端链接：`left_gripper_link`。
- 与腕关节（`left_arm_joint6`）的安装变换：平移 `(0.1039, 0.0, 0.0)` m；旋转 `(0, 0, 0)` rad。
- 惯性特性：质量 0.604 kg；惯性张量同上表。
- 标定状态：**待补充** —— 需加入相机外参、负载重心修正与布线示意。

## 电气 / 控制
- MoveIt 配置：`assets_own/mobile_arm_whole_body/config/*.yaml`（运动学、控制器、规划管线）。
- 启动入口：`assets_own/mobile_arm_whole_body/launch/whole_body_demo.launch.py`。
- 当前控制器：FakeController（仅用于规划验证）。集成实体硬件时需替换为机械臂 + 差速底盘控制器。
- 底盘与机械臂协调：示例假设差速底盘指令来源于 `world_joint` 轨迹（参见 `scripts/world_joint_to_cmd_vel.py`）。

## 参考资产
- 网格：`meshes/*.STL`（臂部连杆与底盘外壳）。
- URDF：`urdf/arm_on_car_center_rotZ_neg90.urdf`。
- SRDF 与 MoveIt 配置：`srdf/arm_on_car_center_whole_body.srdf` 及 `config/` 下的文件。
- 支撑文档：本规格文件（`assets/raw/robot_spec.md`）与模板（`assets/raw/robot_spec_template.md`）。

## 未完成事项
- 补充底盘惯性张量及车轮转速 / 角速度限制。
- 汇总电气架构（供电母线、驱动器、安全互锁等）与网络拓扑。
- 在完成 CAD 与轨迹坐标系对齐测量后，记录 URDF Home 位姿与期望轨迹帧的偏差（`docs/tracking/ee_frame_alignment.md`）。
- 待选定传感器负载后补充相机/云台型号、标定与数据链路。
