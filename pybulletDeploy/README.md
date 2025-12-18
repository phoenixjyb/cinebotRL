# PyBullet → Orin Deployment (ONNX / TensorRT)

This folder exports the **PyBullet PPO policy** trained in `linux_env_dev` to ONNX and provides ready-to-run TensorRT commands for NVIDIA Orin.

## Contents
- `export_pybullet_policy.py` – SB3 checkpoint → ONNX (opset 14) + `<output_stem>_stats.npz`
- `test_pybullet_onnx.py` – quick ONNXRuntime sanity check (CPU/GPU), robot-aware mapping demo
- (Artifacts after export) `policy_*.onnx`, `policy_*_stats.npz`, optional TensorRT engine

## Quick Start (export on dev machine)
```bash
# Recommended local venv in this repo (already set up): 
#   source .venv/bin/activate
# (Torch 2.5.1+cu124, SB3 2.0.0, gymnasium 0.28.1, onnxruntime-gpu 1.22.0)

# Export: mobile_mm (obs_dim=53, action_dim=8)
python pybulletDeploy/export_pybullet_policy.py \
  --checkpoint linux_env_dev/models/logs_20251126_112856/ppo_mobile_mm_final.zip \
  --output pybulletDeploy/policy_mobile_mm.onnx

# Export: recomoDemo1 (obs_dim=51, action_dim=5)
python pybulletDeploy/export_pybullet_policy.py \
  --checkpoint linux_env_dev/models/logs_20251217_184738_recomo/best_model/best_model.zip \
  --output pybulletDeploy/policy_recomo.onnx
```
Outputs in `pybulletDeploy/`:
- `policy_mobile_mm.onnx` (deterministic mean actions)
- `policy_mobile_mm_stats.npz` (identity stats; VecNormalize not used)
- `policy_recomo.onnx`
- `policy_recomo_stats.npz`

## Validate (optional, desktop or Orin)
```bash
python pybulletDeploy/test_pybullet_onnx.py \
  --model pybulletDeploy/policy_mobile_mm.onnx \
  --iterations 100

python pybulletDeploy/test_pybullet_onnx.py \
  --model pybulletDeploy/policy_recomo.onnx \
  --robot recomo --iterations 100
```
To force GPU execution locally (onnxruntime-gpu, CUDA 12.x, cuDNN 9 from the venv wheels), export the CUDA libs before running:
```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cublas/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cufft/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/curand/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cusolver/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cusparse/lib:$PWD/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
python pybulletDeploy/test_pybullet_onnx.py --model pybulletDeploy/policy_mobile_mm.onnx --stats pybulletDeploy/policy_mobile_mm_stats.npz --iterations 50
```
Note: `test_pybullet_onnx.py` will also self-reexec once to prepend these paths automatically when it detects the repo-local `.venv`.

If CUDA EP still fails, it will automatically fall back to CPU.
Notes: on this machine we installed onnxruntime-gpu 1.22.0; it expects CUDA 12.x + cuDNN 9. If `CUDAExecutionProvider` fails with `libcudnn.so.9` missing, install cuDNN 9 or temporarily fall back to the CPU provider (the test script does this automatically).

## TensorRT Engine on Orin
Run on the Orin (JetPack with TensorRT installed):
```bash
trtexec --onnx=policy_mobile_mm.onnx --saveEngine=policy_mobile_mm_fp16.engine \
  --explicitBatch \
  --minShapes=observation:1x53 --optShapes=observation:4x53 --maxShapes=observation:16x53 \
  --fp16 --workspace=2048
```
For recomo:
```bash
trtexec --onnx=policy_recomo.onnx --saveEngine=policy_recomo_fp16.engine \
  --explicitBatch \
  --minShapes=observation:1x51 --optShapes=observation:4x51 --maxShapes=observation:16x51 \
  --fp16 --workspace=2048
```
Notes:
- Keep `--explicitBatch` since export uses dynamic batch axis.
- Use `--int8` only with calibration data; `--workspace` can be increased if memory allows.
- The engine is device-specific; build it on the target Orin.

## I/O specification
- **mobile_mm**
  - **Observation (53 floats):** `[q_with_cos_sin_theta(10), qdot(9), chassis_hist(9), base_xyz(3), ee_xyz(3), target_xyz(3), lookahead_xyz*5(15), remain_ratio(1)]`
  - **Action (8 floats):** `[base_accel_coeff, base_yaw_coeff, arm_joint1..6_coeff]`
  - Control step uses `delta_t=0.1s` with scaling in `linux_env_dev/pybullet_envs/mobile_mm_traj.py`
  - Example: `ds = 0.1*v_prev + 0.5*(1.5*action[0])*0.1^2`, `dtheta = 0.01*action[1]`, joints: `dq = 0.02*action[i]`
- **recomoDemo1**
  - **Observation (51 floats):** `[q_with_cos_sin_theta(9), qdot(8), chassis_hist(9), base_xyz(3), ee_xyz(3), target_xyz(3), lookahead_xyz*5(15), remain_ratio(1)]`
  - **Action (5 floats):** `[vx_cmd, vy_cmd, wz_cmd, arm_yaw_delta_coeff, arm_pitch_delta_coeff]`
  - Example: `vx = 0.5*action[0] (m/s)`, `vy = 0.5*action[1] (m/s)`, `wz = 0.1*action[2] (rad/s)`, arm: `dq = 0.02*action[3:5] (rad/step @ 20Hz)`

## Files to copy to Orin
- `policy_mobile_mm.onnx`
- `policy_mobile_mm_stats.npz`
- `policy_recomo.onnx`
- `policy_recomo_stats.npz`
- Optional: `policy_mobile_mm_fp16.engine` (if you build it on Orin)
- Optional: `policy_recomo_fp16.engine` (if you build it on Orin)

After copying, load the ONNX with onnxruntime/tensorrt and feed normalized observations (identity stats by default).
