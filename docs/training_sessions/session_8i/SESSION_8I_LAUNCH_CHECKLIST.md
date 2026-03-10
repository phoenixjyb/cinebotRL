# Session 8i - 启动前检查清单

**日期**: 2025-11-06  
**状态**: ✅ 代码完成，⚠️ 配套设施待完善

---

## ✅ 已完成项

### 1. 核心代码实现 (100% 完成)
- [x] 观测增强 (70→73 dims)：轴角误差 +3 dims
- [x] 距离门控奖励：远距权重 4.0，近距权重 30.0
- [x] 姿态进步奖励：orientation_progress_bonus
- [x] 角速度平滑惩罚：angular_velocity_penalty
- [x] 环境集成：prev_ee_ori_error 缓冲区和参数传递
- [x] 配置参数：RewardWeights 中的 Session 8i 参数
- [x] 启动脚本：launch_session_8i.ps1（已更新为灵活版本）

### 2. Bug 修复 (100% 完成)
- [x] 角速度惩罚的关节切片问题（已移除错误切片）
- [x] 观测空间注释更新（7 dims → 10 dims）
- [x] Reachability "bell curve" 注释更新

---

## ⚠️ 待完善项

### 1. Checkpoint 准备 (可选)

#### 选项 A: 从头训练（推荐）
**优势**:
- 观测空间变化 (70→73 dims)，从头训练避免适应期
- VecNormalize 统计数据完全匹配新观测空间
- 干净的基线，易于分析

**启动命令**:
```powershell
.\scripts\launch_session_8i.ps1 -Phase short  # 不指定 checkpoint
```

#### 选项 B: 从 Session 8h 继续（需适应期）
**查找可用 checkpoint**:
```powershell
# 方法 1: 使用脚本内置的查找功能
.\scripts\launch_session_8i.ps1 -Phase short  # 会自动显示可用 checkpoint

# 方法 2: 手动查找
Get-ChildItem -Path "logs\sb3\mobilemmtrackee_v0" -Recurse -Filter "*.zip" | 
    Where-Object { $_.Name -match "40000000|100000000" } | 
    Select-Object FullName
```

**启动命令** (示例):
```powershell
# 假设找到了 100M 的 checkpoint
$cp = "logs\sb3\mobilemmtrackee_v0\20251103_235918\checkpoints\ppo_mobile_mm_100663296_steps.zip"
.\scripts\launch_session_8i.ps1 -Phase short -CheckpointPath $cp
```

**注意事项**:
- 观测空间不匹配 (70 vs 73 dims)，策略需要 10-20M 步适应
- VecNormalize 统计数据也不匹配，可能导致初期性能下降
- 权重会自动填充/截断，但可能影响学习稳定性

---

### 2. 轨迹课程 (未启用)

#### 当前状态
- ✅ 目录结构已创建：`trajectoryToLearn/stage0/`, `stage1/`, `stage2/`, `stage3/`
- ❌ 各目录仅有 README，无实际轨迹 JSON 文件
- ❌ launcher 使用 `--use_chassis_only`，未启用分阶段加载

#### 启用方案

**方案 1: 手动准备轨迹文件**
```bash
# 1. 将轨迹按难度分类到各阶段目录
# stage0: 静止或慢速目标 (< 30°/s, < 1.5m)
# stage1: 中等动态目标 (< 60°/s, < 3.0m)
# stage2: 高动态目标
# stage3: 所有轨迹

# 2. 修改 launcher 使用阶段课程
# 在 launch_session_8i.ps1 中设置:
# TrajectoryStage = "stage0_easy"  # 或按步数切换
```

**方案 2: 暂时跳过轨迹课程**
- 保持当前设置（`--use_chassis_only`）
- 专注验证距离门控奖励策略本身
- 轨迹课程作为后续优化（Session 8i.1）

**建议**: 采用方案 2，先验证核心距离门控机制

---

### 3. 姿态监控回调 (未实现)

#### 需求
在训练过程中自动监控姿态指标，满足条件时停止训练：
- 姿态误差相比前一个里程碑恶化 >15%
- 位置误差 >250cm

#### 实现方案

**方案 1: 在 train.py 中添加自定义回调**
```python
# scripts/reinforcement_learning/sb3/train.py
from stable_baselines3.common.callbacks import BaseCallback

class OrientationMonitorCallback(BaseCallback):
    def __init__(self, eval_freq=10_000_000, max_ori_degradation=0.15, max_pos_error=2.5):
        super().__init__()
        self.eval_freq = eval_freq
        self.max_ori_degradation = max_ori_degradation
        self.max_pos_error = max_pos_error
        self.prev_ori_error = None
        
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # 运行评估
            mean_ori_error = evaluate_orientation(self.model, self.training_env)
            mean_pos_error = evaluate_position(self.model, self.training_env)
            
            # 检查终止条件
            if self.prev_ori_error is not None:
                degradation = (mean_ori_error - self.prev_ori_error) / self.prev_ori_error
                if degradation > self.max_ori_degradation:
                    print(f"Stopping: Orientation degraded {degradation*100:.1f}%")
                    return False
            
            if mean_pos_error > self.max_pos_error:
                print(f"Stopping: Position error {mean_pos_error:.2f}m > threshold")
                return False
            
            self.prev_ori_error = mean_ori_error
        return True
```

**方案 2: 手动评估（临时方案）**
```powershell
# 每 10M 步手动运行评估
.\scripts\evaluation\evaluate_checkpoint.ps1 -CheckpointPath <path> -NumEnvs 256

# 检查结果，决定是否继续训练
```

**建议**: 采用方案 2（手动评估），方案 1 留作后续优化

---

## 🚀 推荐启动流程

### Phase 1: 冒烟测试 (必需)
```powershell
# 验证 73 维观测空间和距离门控逻辑
.\scripts\launch_session_8i.ps1 -Phase smoke -Test
```
**预期**: 无崩溃，观测空间 73 维，距离门控参数正确加载

---

### Phase 2: 短期验证 (40M 步，~6 小时)
```powershell
# 从头训练到 40M（0→40M）
.\scripts\launch_session_8i.ps1 -Phase short
```

**监控指标** (每 10M 步评估一次):
- 姿态误差趋势：期望从 ~135° 下降
- 位置误差：期望维持 ~237cm
- 距离门控组件：
  - `orientation_progress_bonus`: 应逐渐增加
  - `angular_velocity_penalty`: 应逐渐减少
  - `orientation_tracking`: 检查远距/近距环境的不同行为

**成功标准** @ 40M:
- ✅ 姿态: <110° (20% 改善)
- ✅ 位置: <250cm (保持基线)
- ✅ 工作空间: 0.50-0.65m
- ✅ 不可达 %: <10%

**决策点** - 评估 40M 结果:
1. **如果所有标准通过** ✅
   - 继续到 Phase 3: `.\scripts\launch_session_8i.ps1 -Phase continuation`
   - 将自动继承 40M checkpoint，节省 6 小时
   
2. **如果任何标准失败** ❌
   - ⚠️ **不要运行 continuation phase** - 会继承坏的训练状态
   - 诊断问题：检查 TensorBoard 日志，分析失败原因
   - 调整配置：修改 `config.py` 中的权重/门控参数
   - 从头开始：运行 `.\scripts\launch_session_8i.ps1 -Phase full`

---

### Phase 3: 连续训练 (40M→120M，~10 小时)
**⚠️ 前提条件: Phase 2 必须通过所有成功标准！**

```powershell
# 自动查找 Phase 2 的 40M checkpoint，继续训练到 120M
.\scripts\launch_session_8i.ps1 -Phase continuation
```

**特性**:
- ✅ 自动查找最近的 40M checkpoint（±5M 容差）
- ✅ 继承 Phase 2 的训练成果
- ✅ 节省 6 小时训练时间
- ⚠️ **重要**: 只有在 Phase 2 成功时才使用！

**安全检查**:
- 如果 Phase 2 失败任何标准，此 phase 会继承坏的训练状态
- 坏的策略会继续恶化，浪费 10 小时训练时间
- 正确做法：诊断问题 → 修复配置 → 运行 `full` phase 从头开始

**监控策略**:
- 每 10M 步评估一次 (50M, 60M, 70M, ...)
- 手动检查姿态和位置指标
- 如果姿态恶化 >15% 或位置 >250cm，手动停止训练

**目标** @ 120M:
- 🎯 姿态: 80-100° (40% 改善)
- 🎯 位置: ~237cm (保持基线)

---

### 备选: 独立完整训练 (0→120M，~16 小时)

**使用场景**:
1. Phase 2 (short) 失败了成功标准，已修复配置，需要从头开始
2. 你想要一个独立的 120M 基线对比
3. 配置发生了重大变化（如门控阈值、权重大幅调整）

```powershell
# 从头训练完整 120M（不继承 Phase 2）
.\scripts\launch_session_8i.ps1 -Phase full
```

**区别**:
- ✅ 从头开始，保证没有坏状态继承
- ✅ 适用于配置修复后的清洁重启
- ❌ 没有 40M 验证检查点（如果再次失败，浪费 16 小时）
- ❌ 相比 short→continuation 额外 6 小时

**推荐**: 只在以下情况使用 full phase:
- Phase 2 已经失败，并且你修复了配置
- 你希望对比不同配置的独立 120M 运行
- 其他情况，优先使用 short → continuation 流程（有验证检查点）

---

## 📊 监控和评估

### 关键指标
1. **主要性能指标**:
   - `ee_pos_error_mean`: 位置误差均值（目标: ~2.37m）
   - `ee_ori_error_mean`: 姿态误差均值（目标: 80-100° = 1.4-1.75 rad）
   - `workspace_distance_mean`: 工作空间距离（目标: 0.50-0.65m）
   - `unreachable_percentage`: 不可达百分比（目标: <10%）

2. **距离门控组件**:
   - `orientation_progress_bonus`: 应逐步增加（姿态改善信号）
   - `angular_velocity_penalty`: 应逐步减少（动作更平滑）
   - `orientation_tracking`: 观察不同距离下的行为

3. **训练稳定性**:
   - `policy/kl_divergence`: 应保持 <0.1
   - `policy/explained_variance`: 应保持 >0.3
   - `policy/entropy`: 应逐渐下降

### WandB 可视化
```python
# 重点关注的图表
- Episode Rewards vs Steps
- Mean Orientation Error vs Steps (自定义指标)
- Mean Position Error vs Steps (自定义指标)
- Distance-Gated Reward Components vs Steps
```

---

## 🔍 调试检查清单

### 如果姿态没有改善
1. 检查距离门控是否生效:
   ```python
   # 在评估时打印距离门控统计
   in_comfort_zone = distance < 0.7
   print(f"Comfort zone %: {in_comfort_zone.float().mean()*100:.1f}%")
   print(f"Mean ori weight: {ori_weight_current.mean():.2f}")
   ```

2. 检查姿态进步奖励是否激活:
   ```python
   # 应该看到非零的 orientation_progress_bonus
   print(f"Ori progress bonus: {ori_progress_reward.mean():.4f}")
   ```

3. 检查观测空间是否正确:
   ```python
   # 73 维，包含轴角误差
   print(f"Obs shape: {obs.shape}")  # 应该是 [N, 73]
   ```

### 如果位置恶化
1. 检查基础移动奖励是否受影响:
   ```python
   # base_mobilization 应该仍然积极
   print(f"Base mobilization: {base_mob_reward.mean():.4f}")
   ```

2. 检查距离门控阈值是否合适:
   ```python
   # 0.7m 是否太保守？
   print(f"Mean distance to target: {base_target_distance.mean():.2f}m")
   ```

---

## 📝 文档和日志

### 需要创建的文档
- [ ] `docs/training_sessions/session_8i/SESSION_8I_TRAINING_LOG.md` - 训练日志
- [ ] `docs/training_sessions/session_8i/SESSION_8I_EVALUATION_RESULTS.md` - 评估结果
- [ ] `scripts/evaluation/evaluate_session_8i.ps1` - 评估脚本

### 需要更新的文档
- [ ] `TRAINING_SESSIONS_MASTER_LOG.md` - 添加 Session 8i 条目
- [ ] `README.md` - 更新最新进展

---

## ✅ 准备就绪检查

在启动训练前，确认：

- [ ] **冒烟测试通过**: `.\scripts\launch_session_8i.ps1 -Phase smoke -Test`
- [ ] **理解观测空间变化**: 70→73 dims，可能需要从头训练
- [ ] **决定 checkpoint 策略**: 从头训练 vs 从 8h 继续
- [ ] **准备监控方案**: 手动评估 vs 自动回调
- [ ] **接受轨迹课程缺失**: 当前仅用 chassis 轨迹
- [ ] **有足够时间**: short (6h), full (16h)

---

## 🎯 成功标准总结

| 阶段 | 姿态目标 | 位置目标 | 工作空间 | 不可达% |
|------|---------|---------|----------|---------|
| Baseline (8h@40M) | 135.1° | 237.3cm | 0.554m | 3.1% |
| **Short (40M)** | **<110°** | **<250cm** | **0.50-0.65m** | **<10%** |
| **Full (120M)** | **80-100°** | **~237cm** | **0.50-0.65m** | **<10%** |

---

## 🔄 训练流程决策树

```
START
  │
  ├─[1]─► Run smoke test
  │       └─► Pass? ──NO──► Fix issues → Retry
  │              │
  │             YES
  │              │
  ├─[2]─► Run Phase 2 (short): 0→40M (~6h)
  │       └─► Evaluate @ 40M
  │              │
  │       ┌──────┴───────┐
  │       │              │
  │      PASS          FAIL
  │       │              │
  │       │         ┌────┴────┐
  │       │         │         │
  │       │    [Option A]  [Option B]
  │       │    Diagnose    Give up
  │       │    issue       Session 8i
  │       │       │
  │       │    Adjust
  │       │    config
  │       │       │
  │       │    Run Phase
  │       │    'full' from
  │       │    scratch
  │       │    (0→120M)
  │       │       │
  │       │       └─► [END]
  │       │
  ├─[3]─► Run Phase 3 (continuation): 40M→120M (~10h)
  │       │   ⚠️ Auto-inherits Phase 2 checkpoint
  │       │
  │       └─► Evaluate @ 120M
  │              │
  │           SUCCESS
  │              │
  │          [END] ✅
  │
  └─[Alternative]─► Run Phase 'full': 0→120M (~16h)
                    • Use if you want independent baseline
                    • Use after fixing config from Phase 2 failure
                    • No validation checkpoint, all-or-nothing
                    └─► [END]

KEY DECISION RULES:
• Phase 2 PASS → Run continuation (saves 6h, has validation)
• Phase 2 FAIL → Run full from scratch (clean state, no bad inheritance)
• Never run continuation after Phase 2 failure - wastes 10h with bad policy
```

---

## 📞 需要帮助？

如果遇到问题，检查：
1. `SESSION_8I_IMPLEMENTATION_SUMMARY.md` - 实现细节
2. `SESSION_8I_IMPLEMENTATION_PLAN.md` - 原始计划
3. 相关代码文件中的 "SESSION 8i" 注释

**祝训练顺利！** 🚀
