
# Mobile Manipulation RL — Review & Next‑Step Playbook

**Context.** 9‑DoF robot (6‑DoF arm + differential base vx,wz). Goal: EE tracks 3D reference trajectories (≈1,000). We trained PPO (SB3) with an MLP policy, several long runs up to ~110M steps. We observed multiple regimes across runs.

---

## 1) What we learned from the logs (condensed)

### Early runs (≤0.5M steps)
- **Explained variance** near/under 0, then oscillating negative → critic initially unstable.
- **Entropy high; std ≈ 1.0** → wide exploration.
- **KL** ~0.006–0.01; **clip_fraction** ~0.06–0.16 → healthy/benign updates.

### Mid runs (1–10M steps)
- **Explained variance** climbs toward 1.0 (good critic fit).
- **Frequent early-stop on KL** (when `target_kl`/max KL was set): indicates **updates too big** given current LR/epochs.
- With KL disabled: **approx_kl ≈ 0.06–0.09** and **clip_fraction ≈ 0.40–0.47** → too many samples are clipped → **instability & inefficiency** despite high value accuracy.

### A special run (std explosion)
- **std** rose into the **20–38** range and **entropy_loss ≈ -35 to -40** → action distribution blew up. This usually comes from unconstrained log‑std or strong entropy pressure.
- Despite that, EV remained high → critic still fit, but actor overly stochastic.

### Latest long run (~110M steps)
- **approx_kl** often **0.4–2.5**, **clip_fraction ≈ 0.68–0.75** (very high).
- **Entropy_loss ≈ -6.2** and **std ≈ 0.54** (much tighter than the “explosion” run).
- **Explained variance ~ 1.0; value_loss ≈ 1e‑6** → critic saturated/very accurate.
- Takeaway: **policy step size is too large** (very high KL, most updates are clipped). We are learning, but **wasting gradient** and risking oscillations.

---

## 2) Root causes (most likely)

1) **No/loose KL guardrail** late in training + **constant LR** → policy jumps grow, PPO relies on clipping (clip_fraction ↑), causing inefficiency.
2) **Clip range fixed at 0.2** with large nets and long training → update mismatch late in training.
3) **Unbounded/large action std at times** (one run) → excessive exploration. SB3’s Gaussian can drift without bounds.
4) **MLP only, single head for all actions** → base and arm share the same variance/step size; different control scales/curvatures favor **branched heads**.
5) **Observation design** lacks temporal context (no LSTM): needs either **stacking** (K past obs + actions) or **look‑ahead reference points** for pure MLP to anticipate the next trajectory segment.
6) **Reward & safety shaping** could better constrain base tilt/tipover and coordinate base vs arm.

---

## 3) Targets (“green zone” telemetry)

- **approx_kl:** 0.005–0.02 per update epoch
- **clip_fraction:** < 0.30 (ideally 0.15–0.25)
- **entropy trend:** decays over training
- **std (per‑dim):** within a task‑sensible band (e.g., joints ~0.1–0.5 rad step, base vx/wz ~0.05–0.3 in normalized units)
- **explained_variance:** ≥ 0.9 and stable (we already meet this)

---

## 4) Hyperparameter plan (PPO, SB3)

### 4.1 Scheduling (critical)
- **KL guardrail** (use SB3 `target_kl`):
  - 0–10% steps: **0.07**
  - 10–85%: **0.02**
  - 85–100%: **0.008**

- **Learning rate** (linear decay): **3e‑4 → 3e‑5**
- **Clip range** (linear decay): **0.2 → 0.1**
- **Entropy coef** (linear decay): **1e‑2 → 0**

> These four together prevent the late‑stage high‑KL/high‑clip‑fraction regime we saw.

### 4.2 Core PPO knobs
- `n_steps`: **4096–8192** (bigger batches → steadier GAE)
- `batch_size`: **1024–2048** (if memory allows)
- `n_epochs`: **5–10** (start 10; drop to 5 beyond 70% training when KL rises)
- `max_grad_norm`: **0.5** (keep it)
- `gamma`: **0.99**, `gae_lambda`: **0.95** (keep)

### 4.3 Policy distribution hygiene
- `use_sde=True`, `sde_sample_freq=4`  (stabilizes exploration for Box actions)
- `log_std_init=-1.0`
- **Bound log‑std**: `log_std_bounds=(-3, 1)`  # std ∈ [~0.05, ~2.72]
- (Optional) `squash_output=True` (SB3≥2.2) to map actions via Tanh

---

## 5) Policy architecture

### 5.1 MLP (if staying non‑recurrent)
- Shared trunk: **[256, 256]** with **SiLU** and **ortho_init=True**
- **Branched heads**:
  - **Arm head**: [128] → 6 actions (separate log‑std)
  - **Base head**: [128] → 2 actions (separate log‑std)

This lets the base and arm adopt different step sizes & variances.

### 5.2 Add temporal context (no LSTM)
- **Stack K=3–5 past obs** and **last action** (per‑dim normalization). Or
- Append **look‑ahead** of the next **H=3–5** trajectory waypoints (pos/orient deltas).

Either option markedly improves MLP tracking of time‑indexed goals.

---

## 6) Reward & safety shaping (surgical)

- **EE tracking**: position Huber loss + quaternion geodesic error (angle distance).
- **Smoothness**: action L2 + delta‑action L2 (jerk) with small weights; decay their weight over training.
- **Base safety**: roll/pitch penalty (quadratic), terminate if |tilt| > threshold; small **base‑motion tax** encouraging arm use when reachable; allow base to move more when EE error stays above a band.
- **Limits**: joint limit proximity penalty; workspace bound penalty.
- **Success bonus**: dense shaping + per‑step small success band reward to lock‑in tracking.

---

## 7) Vectorization & throughput

- Favor **>=512 envs** if CPU physics allows; match `batch_size` to `n_envs*n_steps`.
- Ensure **VecNormalize** is used; **freeze stats** for evaluation.
- Keep **GPU pinned memory** and **num_workers** tuned so the policy step, not IPC, is the bottleneck.

---

## 8) Monitoring & automated guardrails

Track per update:
- `approx_kl`, `clip_fraction`, `entropy`, **per‑dim std**, EV, mean EE error.
Add callbacks:
- **KL scheduler** (as above).
- **LR/clip decay** tied to progress.
- **Alarm** if `clip_fraction > 0.4` or `approx_kl > 0.05` for 3 consecutive rollouts → reduce LR ×0.5 and set tighter `target_kl`.

Compute validation KPIs (eval env, deterministic):
- EE **RMSE** (pos/orient), **max error**, **% timesteps within tolerance**, **trajectory completion rate**, **action RMS**, **base distance traveled**.

---

## 9) Concrete SB3 snippet

```python
import torch.nn as nn
from stable_baselines3 import PPO

def linear_schedule(v0, v1):
    return lambda frac: v1 + (v0 - v1) * frac  # frac: 1->0

policy_kwargs = dict(
    activation_fn=nn.SiLU,
    net_arch=dict(pi=[256, 256], vf=[384, 384]),
    ortho_init=True,
    log_std_init=-1.0,
    log_std_bounds=(-3, 1),
)

model = PPO(
    "MlpPolicy", env,
    learning_rate=linear_schedule(3e-4, 3e-5),
    clip_range=linear_schedule(0.2, 0.1),
    ent_coef=linear_schedule(1e-2, 0.0),
    n_steps=4096, batch_size=1024, n_epochs=10,
    gamma=0.99, gae_lambda=0.95,
    use_sde=True, sde_sample_freq=4,
    target_kl=0.07,  # warm-up, then adjust via callback
    clip_range_vf=0.5,  # reduce large value updates
    verbose=1,
)
```

Implement the **DynamicKLSchedule** callback to switch `target_kl` to 0.02 mid‑training and 0.008 for fine‑tuning.

---

## 10) Short experiment plan (order matters)

1. **Stability pass (no code changes to env):**
   - Add **KL, LR, clip, entropy schedules** (Section 4.1).
   - Add **log‑std bounds** + `use_sde=True`.
   - Reduce `clip_range_vf` to **0.5**.
   - Expect: **approx_kl → 0.01–0.03**, **clip_fraction < 0.3**, **returns smoother**.

2. **Architecture pass:**
   - Switch to **branched heads** (base vs arm).
   - Add **K=3 look‑ahead waypoints** to obs OR stack **past K=3 obs + last action**.
   - Expect: lower tracking error, fewer base “thrashes”.

3. **Reward pass:**
   - Add **tilt penalty + termination**, **base tax**, **jerk penalty**; convert orientation error to **geodesic angle**.
   - Expect: safer, smoother, fewer large KL spikes (policy won’t discover crazy maneuvers that the critic rates well).

4. **Curriculum pass (if needed):**
   - Start with short, slow trajectories and tighter tolerances on action magnitudes; gradually lengthen/impose faster references and harder initializations.

5. **Scale‑up pass:**
   - Increase `n_steps` to **8192** and `batch_size` to **2048** once stable.

---

## 11) What to watch for next

- If **KL stays >0.05** after schedules → **halve LR** and **drop n_epochs to 5**.
- If **std drifts** toward bounds frequently → lower upper log‑std bound to **0.5**.
- If **clip_fraction remains >0.4** → tighten `target_kl` (e.g., 0.015) or lower clip to 0.08–0.12.
- If **EV collapses** suddenly → value target/normalization issue; verify rewards, `VecNormalize`, and `clip_range_vf` not too small.

---

## 12) Go/no‑go criteria for this round

- **Green:** approx_kl 0.01–0.03, clip_fraction <0.3, std within task bands, EV ≥0.95, EE RMSE below spec.
- **Yellow:** approx_kl 0.03–0.08 or clip_fraction 0.3–0.45 → apply LR/epoch reduction.
- **Red:** approx_kl >0.1 or clip_fraction >0.5 for 3+ rollouts, or std exploding → tighten KL, lower LR, verify rewards/normalization.

---

### TL;DR
You’re very close on the critic; the actor is stepping **too far** late in training. Add **KL/LR/clip/entropy schedules**, **bound log‑std** (prefer `use_sde`), **branch the policy heads**, and give the MLP minimal temporal context (stacking or look‑ahead). Then tune rewards for safety and smoothness. This will pull KL and clip_fraction into the green zone while keeping your strong value baseline.
