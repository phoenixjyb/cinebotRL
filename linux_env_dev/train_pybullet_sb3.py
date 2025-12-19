import datetime
import os
import argparse
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization
from pybullet_envs.mobile_mm import MobileMMBulletEnv
from pybullet_envs.mobile_mm_traj import MobileMMTrajEnv
from pybullet_envs.target_generator import (
    FixedTarget,
    RandomTargetForEpisode,
    JSONNearestTargetGenerator,
    CurriculumJSONNearestTargetGenerator,
)
import torch
from pybullet_envs.transformer_extractor import TransformerFeaturesExtractor

# parse custom network architectures
def parse_layers(s: str):
    if s is None or s.strip() == "":
        return []
    return [int(x) for x in s.split(",") if x.strip()]

def calcaulate_time_stamp():
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("Asia/Shanghai")
    ts = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
    return ts

def get_policy_kwargs(pi_layers_str, vf_layers_str):
    pi_layers = parse_layers(pi_layers_str)
    vf_layers = parse_layers(vf_layers_str)
    policy_kwargs = {"net_arch": dict(pi=pi_layers, vf=vf_layers)}
    return policy_kwargs

def calc_param_num(policy):
    # 参数统计：总参数、actor、critic、共享

    def num_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    total = num_params(policy)
    actor_mlp = num_params(policy.mlp_extractor.policy_net)
    value_mlp = num_params(policy.mlp_extractor.value_net)
    actor_params = num_params(policy.action_net)
    value_params = num_params(policy.value_net)

    print(f"Policy parameter counts: total={total:,d}, "
          f"actor_mlp={actor_mlp:,d}, value_mlp={value_mlp:,d}, "
          f"actor_head={actor_params:,d}, value_head={value_params:,d}")


class TrainInfoMetricsCallback(BaseCallback):
    """Logs rolling means for env info keys to TensorBoard."""

    def __init__(self, log_freq: int = 2000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_freq = int(log_freq)
        self._ee_dist = []
        self._rew_dist = []
        self._rew_collision = []
        self._rew_close = []
        self._base_vx = []
        self._base_vy = []
        self._base_wz = []
        self._base_lin_norm = []
        self._remain_ratio = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True

        for info in infos:
            if not isinstance(info, dict):
                continue
            if "ee_distance" in info:
                self._ee_dist.append(float(info["ee_distance"]))
            if "reward_dist" in info:
                self._rew_dist.append(float(info["reward_dist"]))
            if "reward_collision" in info:
                self._rew_collision.append(float(info["reward_collision"]))
            if "reward_close" in info:
                self._rew_close.append(float(info["reward_close"]))
            if "base_vx" in info:
                self._base_vx.append(float(info["base_vx"]))
            if "base_vy" in info:
                self._base_vy.append(float(info["base_vy"]))
            if "base_wz" in info:
                self._base_wz.append(float(info["base_wz"]))
            if "base_lin_vel_norm" in info:
                self._base_lin_norm.append(float(info["base_lin_vel_norm"]))
            if "remain_traj_ratio" in info:
                self._remain_ratio.append(float(info["remain_traj_ratio"]))

        if self.log_freq > 0 and (self.n_calls % self.log_freq) == 0:
            if self._ee_dist:
                self.logger.record("train/ee_distance_mean", float(np.mean(self._ee_dist)))
                self._ee_dist.clear()
            if self._rew_dist:
                self.logger.record("train/reward_dist_mean", float(np.mean(self._rew_dist)))
                self._rew_dist.clear()
            if self._rew_collision:
                self.logger.record("train/reward_collision_mean", float(np.mean(self._rew_collision)))
                self._rew_collision.clear()
            if self._rew_close:
                self.logger.record("train/reward_close_mean", float(np.mean(self._rew_close)))
                self._rew_close.clear()
            if self._base_vx:
                self.logger.record("train/base_vx_mean", float(np.mean(self._base_vx)))
                self._base_vx.clear()
            if self._base_vy:
                self.logger.record("train/base_vy_mean", float(np.mean(self._base_vy)))
                self._base_vy.clear()
            if self._base_wz:
                self.logger.record("train/base_wz_mean", float(np.mean(self._base_wz)))
                self._base_wz.clear()
            if self._base_lin_norm:
                self.logger.record("train/base_lin_vel_norm_mean", float(np.mean(self._base_lin_norm)))
                self._base_lin_norm.clear()
            if self._remain_ratio:
                self.logger.record("train/remain_traj_ratio_mean", float(np.mean(self._remain_ratio)))
                self._remain_ratio.clear()
        return True


class EvalInfoMetricsCallback(EvalCallback):
    """EvalCallback that also logs extra info keys at episode end.

    The env must put desired scalars into `info` on the terminal step of each eval episode.
    """

    def __init__(self, *args, info_keys: tuple[str, ...] = (), **kwargs):
        super().__init__(*args, **kwargs)
        self.info_keys = tuple(info_keys or ())
        self._metric_buffer = {k: [] for k in self.info_keys}
        self.evaluations_metrics = {k: [] for k in self.info_keys}

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        info = locals_["info"]
        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)
            for k in self.info_keys:
                v = info.get(k, np.nan)
                try:
                    self._metric_buffer[k].append(float(v))
                except Exception:
                    self._metric_buffer[k].append(np.nan)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset buffers
            self._is_success_buffer = []
            self._metric_buffer = {k: [] for k in self.info_keys}

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            # Persist eval history
            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs["successes"] = self.evaluations_successes

                # Save per-episode terminal info metrics
                for k in self.info_keys:
                    self.evaluations_metrics[k].append(self._metric_buffer.get(k, []))
                    kwargs[f"{k}s"] = self.evaluations_metrics[k]

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = float(np.mean(episode_rewards)), float(np.std(episode_rewards))
            mean_ep_length, std_ep_length = float(np.mean(episode_lengths)), float(np.std(episode_lengths))
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            # Add to current Logger
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = float(np.mean(self._is_success_buffer))
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Extra metrics (terminal step values)
            for k in self.info_keys:
                arr = np.array(self._metric_buffer.get(k, []), dtype=float)
                if arr.size == 0:
                    continue
                self.logger.record(f"eval/final_{k}_mean", float(np.nanmean(arr)))

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            # Save best model (same logic as SB3 EvalCallback) + keep VecNormalize stats in sync if present
            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    try:
                        venv = self.model.get_vec_normalize_env()
                        if venv is not None and hasattr(venv, "save"):
                            venv.save(os.path.join(self.best_model_save_path, "vecnormalize.pkl"))
                    except Exception:
                        pass
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


class CurriculumCallback(BaseCallback):
    """Broadcasts curriculum stage2 mixing probability to all envs.

    - mode='switch': stage2_prob = 0 until `stage1_steps`, then 1
    - mode='mix': stage2_prob ramps linearly 0->1 over `stage1_steps`
    """

    def __init__(self, mode: str, stage1_steps: int, update_freq: int = 50_000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.mode = (mode or "none").strip().lower()
        self.stage1_steps = int(stage1_steps)
        self.update_freq = int(update_freq)
        self._last_update_at = -1
        self._last_prob = None

    def _compute_prob(self, num_timesteps: int) -> float:
        if self.mode == "switch":
            if self.stage1_steps <= 0:
                return 1.0
            return 0.0 if int(num_timesteps) < int(self.stage1_steps) else 1.0
        if self.mode == "mix":
            if self.stage1_steps <= 0:
                return 1.0
            return float(np.clip(float(num_timesteps) / float(self.stage1_steps), 0.0, 1.0))
        return 1.0

    def _apply(self, prob: float):
        try:
            env = self.model.get_env()
            env.env_method("set_curriculum_stage2_prob", float(prob))
        except Exception:
            # best-effort only; curriculum is optional
            pass
        self.logger.record("curriculum/stage2_prob", float(prob))
        self._last_prob = float(prob)

    def _on_training_start(self) -> None:
        if self.mode in {"switch", "mix"}:
            self._apply(self._compute_prob(self.num_timesteps))
            self._last_update_at = int(self.num_timesteps)

    def _on_step(self) -> bool:
        if self.mode not in {"switch", "mix"}:
            return True
        if self.update_freq <= 0:
            return True
        if (int(self.num_timesteps) - int(self._last_update_at)) < int(self.update_freq):
            return True
        self._last_update_at = int(self.num_timesteps)
        prob = self._compute_prob(self.num_timesteps)
        if self._last_prob is None or abs(float(prob) - float(self._last_prob)) > 1e-9:
            self._apply(prob)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        type=str,
        default="none",
        choices=["none", "recomo_accuracy"],
        help="Convenience hyperparameter preset for new training runs.",
    )
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--render", action="store_true", help="Use GUI")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device for SB3 policy network. 'auto' picks CUDA if available.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--robot",
        type=str,
        default="mobile_mm",
        choices=["mobile_mm", "recomo"],
        help="Robot model for the PyBullet env",
    )
    parser.add_argument("--urdf_path", type=str, default=None, help="Override URDF path (robot-specific)")
    parser.add_argument("--frame_skip", type=int, default=24, help="Physics steps per RL step (240Hz base)")
    parser.add_argument("--max_steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument(
        "--obs_frame",
        type=str,
        default="auto",
        choices=["auto", "world", "chassis"],
        help="Observation frame for target/ee/future points. 'chassis' is recommended for holonomic base.",
    )
    parser.add_argument("--train_txt", type=str, default="linux_env_dev/new_json_50/train.txt",
                        help="Training trajectory list (txt file with one json path per line)")
    parser.add_argument("--eval_txt", type=str, default="linux_env_dev/new_json_50/test.txt",
                        help="Evaluation trajectory list (txt file) for --eval_freq > 0")
    parser.add_argument(
        "--curriculum",
        type=str,
        default="none",
        choices=["none", "switch", "mix"],
        help="Curriculum over trajectory lists. 'switch': stage1 -> stage2 at --curriculum_stage1_steps. "
             "'mix': stage2 sampling prob ramps 0->1 over --curriculum_stage1_steps.",
    )
    parser.add_argument("--curriculum_stage1_txt", type=str, default="linux_env_dev/new_json_50/train_stage1.txt",
                        help="Stage1 trajectory list for curriculum (used when --curriculum != none)")
    parser.add_argument("--curriculum_stage1_steps", type=int, default=1_000_000,
                        help="Timesteps for stage1-only (switch) or ramp length (mix)")
    parser.add_argument("--curriculum_update_freq", type=int, default=50_000,
                        help="How often to update curriculum prob (timesteps)")
    parser.add_argument("--policy", type=str, default="Transformer",
                        choices=["MlpPolicy", "LargeMlp", "Transformer"],
                        help="Policy architecture to use. Transformer = Transformer-based feature extractor")
    # transformer-specific options
    parser.add_argument("--tf_seq_len", type=int, default=8, help="Transformer token sequence length")
    parser.add_argument("--tf_embed_dim", type=int, default=256, help="Transformer embedding dimension")
    parser.add_argument("--tf_layers", type=int, default=3, help="Number of Transformer encoder layers")
    parser.add_argument("--tf_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--tf_dropout", type=float, default=0.1, help="Transformer dropout")
    parser.add_argument("--tf_ff_dim", type=int, default=2048, help="Transformer feedforward dimension")
    parser.add_argument("--warmup_frac", type=float, default=0.05,
                        help="Fraction of total training for linear warmup (0-1)")
    parser.add_argument("--save_dir", type=str, default="linux_env_dev/models")
    # PPO / env config
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel envs (vec env)")
    parser.add_argument("--n_steps", type=int, default=2048, help="PPO n_steps (per env)")
    parser.add_argument("--batch_size", type=int, default=64, help="PPO batch_size")
    parser.add_argument("--n_epochs", type=int, default=10, help="PPO n_epochs")
    parser.add_argument("--gamma", type=float, default=0.99, help="PPO discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="PPO GAE lambda")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--clip_range_vf", type=float, default=None, help="PPO clip range for value function (None=disabled)")
    parser.add_argument("--normalize_advantage", action="store_true", help="Enable advantage normalization (SB3 default=True)")
    parser.add_argument("--no_normalize_advantage", action="store_true", help="Disable advantage normalization")
    parser.add_argument("--ent_coef", type=float, default=0.0, help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.3, help="Max gradient norm (clipping)")
    parser.add_argument("--target_kl", type=float, default=None, help="Target KL divergence (None=disabled)")
    parser.add_argument("--pi_layers", type=str, default="256,256",
                        help="Comma-separated sizes for actor (pi) MLP, e.g. '256,256'")
    parser.add_argument("--vf_layers", type=str, default="256,256",
                        help="Comma-separated sizes for critic (vf) MLP, e.g. '256,256'")
    parser.add_argument("--save_interval", type=int, default=100_000,
                        help="Save a checkpoint every N timesteps (0 to disable)")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to a saved model to load and continue training")
    parser.add_argument("--eval_freq", type=int, default=10000,
                        help="Run evaluation every N timesteps (0 to disable)")
    parser.add_argument("--eval_episodes", type=int, default=5,
                        help="Number of eval episodes per evaluation run")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Initial learning rate for the optimizer (overrides SB3 default if set)")
    parser.add_argument("--vec_normalize", action="store_true", help="Enable VecNormalize (obs normalization)")
    parser.add_argument("--vecnorm_norm_reward", action="store_true", help="Also normalize rewards (VecNormalize)")
    parser.add_argument("--vecnorm_clip_obs", type=float, default=10.0, help="VecNormalize clip_obs")
    parser.add_argument("--vecnorm_clip_reward", type=float, default=10.0, help="VecNormalize clip_reward")
    parser.add_argument("--vecnorm_path", type=str, default=None, help="Path to VecNormalize stats to load when resuming")
    parser.add_argument("--reward_dist_weight", type=float, default=None, help="Override distance reward weight")
    parser.add_argument("--reward_collision_threshold", type=float, default=None, help="Override collision distance threshold")
    parser.add_argument("--reward_collision_ratio", type=float, default=None, help="Override collision penalty ratio")
    parser.add_argument("--reward_clip_abs", type=float, default=None, help="Override per-step reward clip abs value")
    parser.add_argument("--reward_close_bonus", type=float, default=None, help="Override close-to-target shaping bonus")
    parser.add_argument("--reward_close_threshold", type=float, default=None, help="Override close-to-target threshold (meters)")
    args = parser.parse_args()

    argv = sys.argv[1:]

    def _arg_set(flag: str) -> bool:
        if flag in argv:
            return True
        return any(a.startswith(flag + "=") for a in argv)

    if args.preset == "recomo_accuracy":
        # Only override values the user did not explicitly provide on the CLI.
        if not _arg_set("--robot"):
            args.robot = "recomo"
        if not _arg_set("--policy"):
            args.policy = "Transformer"
        if not _arg_set("--device"):
            args.device = "cuda"
        if not _arg_set("--n_envs"):
            args.n_envs = 16
        if not _arg_set("--n_steps"):
            args.n_steps = 1024
        if not _arg_set("--batch_size"):
            args.batch_size = 4096
        if not _arg_set("--n_epochs"):
            args.n_epochs = 5
        if not _arg_set("--gamma"):
            args.gamma = 0.995
        if not _arg_set("--learning_rate"):
            args.learning_rate = 1e-4
        if not _arg_set("--warmup_frac"):
            args.warmup_frac = 0.05
        if not _arg_set("--ent_coef"):
            args.ent_coef = 0.005
        if not _arg_set("--max_grad_norm"):
            args.max_grad_norm = 0.5
        if not _arg_set("--target_kl"):
            args.target_kl = 0.03
        if not _arg_set("--eval_freq"):
            args.eval_freq = 20000
        if not _arg_set("--eval_episodes"):
            args.eval_episodes = 10
        if not _arg_set("--save_interval"):
            args.save_interval = 200000
        if not _arg_set("--curriculum"):
            args.curriculum = "mix"
        if not _arg_set("--curriculum_stage1_steps"):
            args.curriculum_stage1_steps = 2_000_000
        if not _arg_set("--obs_frame"):
            args.obs_frame = "chassis"
        if not _arg_set("--tf_dropout"):
            args.tf_dropout = 0.0
        if not _arg_set("--reward_close_bonus"):
            args.reward_close_bonus = 0.5
        if not _arg_set("--reward_close_threshold"):
            args.reward_close_threshold = 0.05

    # Make "LargeMlp" a convenience alias (SB3 only knows "MlpPolicy").
    # If user didn't override layers, use a larger default net.
    if args.policy == "LargeMlp":
        if (args.pi_layers or "").strip() == "256,256" and (args.vf_layers or "").strip() == "256,256":
            args.pi_layers = "512,512,256"
            args.vf_layers = "512,512,256"

    if args.curriculum != "none":
        if not os.path.isfile(args.curriculum_stage1_txt):
            raise FileNotFoundError(f"--curriculum_stage1_txt not found: {args.curriculum_stage1_txt}")
        if not os.path.isfile(args.train_txt):
            raise FileNotFoundError(f"--train_txt (stage2) not found: {args.train_txt}")

    # build a warmup + linear decay learning rate schedule (callable accepted by SB3)
    try:
        initial_lr = float(args.learning_rate) if args.learning_rate is not None else 1e-4
    except Exception:
        initial_lr = 1e-4
    try:
        warmup_frac = float(args.warmup_frac)
        if warmup_frac < 0.0:
            warmup_frac = 0.0
        if warmup_frac >= 1.0:
            warmup_frac = min(0.99, warmup_frac)
    except Exception:
        warmup_frac = 0.05

    def lr_schedule(progress_remaining: float) -> float:
        """Progress-based LR schedule for SB3.

        progress_remaining: 1.0 -> 0.0 over the full training run.
        We compute progress = 1 - progress_remaining (0 -> 1).
        Warmup: linear from 0 -> initial_lr over `warmup_frac` of training.
        Decay: linear from initial_lr -> 0 over remaining fraction.
        Returns current learning rate (float).
        """
        progress = 1.0 - float(progress_remaining)
        # warmup phase
        if warmup_frac > 0.0 and progress < warmup_frac:
            return float(initial_lr * (progress / max(1e-12, warmup_frac)))
        # linear decay after warmup
        denom = max(1e-12, 1.0 - warmup_frac)
        t = (progress - warmup_frac) / denom
        return float(max(1e-7, initial_lr * (1.0 - t)))

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed is not None:
        set_random_seed(int(args.seed))
    env_fn = lambda: MobileMMTrajEnv(
        robot=args.robot,
        urdf_path=args.urdf_path,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        render=args.render,
        obs_frame=args.obs_frame,
        reward_dist_weight=args.reward_dist_weight,
        reward_collision_threshold=args.reward_collision_threshold,
        reward_collision_ratio=args.reward_collision_ratio,
        reward_clip_abs=args.reward_clip_abs,
        reward_close_bonus=args.reward_close_bonus,
        reward_close_threshold=args.reward_close_threshold,
        target_generator=(
            CurriculumJSONNearestTargetGenerator(
                stage1_txt=args.curriculum_stage1_txt,
                stage2_txt=args.train_txt,
                mode="random",
                stage2_prob=0.0,
            )
            if args.curriculum != "none"
            else JSONNearestTargetGenerator(
                json_paths=[],
                json_txt=args.train_txt,
                mode="random",
            )
        ),
    )
    if args.n_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        vec_env = SubprocVecEnv([env_fn for _ in range(args.n_envs)])
    else:
        vec_env = DummyVecEnv([env_fn])
    from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
    vec_env = VecMonitor(vec_env)
    if args.vec_normalize:
        stats_candidates = []
        if args.vecnorm_path:
            stats_candidates.append(args.vecnorm_path)
        if args.load_model:
            # Common layouts:
            # - <run_dir>/best_model/best_model.zip
            # - <run_dir>/checkpoints/xxx.zip
            # We try nearby parents for vecnormalize.pkl
            mdir = os.path.dirname(os.path.abspath(args.load_model))
            stats_candidates.extend(
                [
                    os.path.join(mdir, "vecnormalize.pkl"),
                    os.path.join(os.path.dirname(mdir), "vecnormalize.pkl"),
                    os.path.join(os.path.dirname(os.path.dirname(mdir)), "vecnormalize.pkl"),
                ]
            )

        stats_path = next((p for p in stats_candidates if p and os.path.isfile(p)), None)
        if stats_path:
            print(f"Loading VecNormalize stats from: {stats_path}")
            vec_env = VecNormalize.load(stats_path, vec_env)
        else:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=bool(args.vecnorm_norm_reward),
                clip_obs=float(args.vecnorm_clip_obs),
                clip_reward=float(args.vecnorm_clip_reward),
                gamma=float(args.gamma),
            )
        vec_env.training = True

    ts = calcaulate_time_stamp()
    run_name = f"logs_{ts}" if args.robot == "mobile_mm" else f"logs_{ts}_{args.robot}"
    log_dir = os.path.join(args.save_dir, f"{run_name}/tensorboard_logs")
    os.makedirs(log_dir, exist_ok=True)


    policy_kwargs = get_policy_kwargs(args.pi_layers, args.vf_layers)

    # select device explicitly
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested --device cuda but torch reports CUDA is not available. "
            "Check that your venv has a CUDA-enabled torch build and that the NVIDIA driver is visible."
        )
    if device == "cuda":
        try:
            print(f"Using CUDA device for training: {torch.cuda.get_device_name(0)} (torch cuda={torch.version.cuda})")
        except Exception:
            print("Using CUDA device for training")
    else:
        print(f"Using device='{device}' for training")

    # If Transformer policy requested, wire custom features extractor
    if args.policy == "Transformer":
        # Add Transformer feature extractor while keeping actor/critic net_arch.
        policy_kwargs.update(
            dict(
                features_extractor_class=TransformerFeaturesExtractor,
                features_extractor_kwargs={
                    "seq_len": int(args.tf_seq_len),
                    "embed_dim": int(args.tf_embed_dim),
                    "n_heads": int(args.tf_heads),
                    "n_layers": int(args.tf_layers),
                    "dropout": float(args.tf_dropout),
                    "ff_dim": int(args.tf_ff_dim),
                },
            )
        )
        print(f"Using Transformer feature extractor: seq_len={args.tf_seq_len}, embed_dim={args.tf_embed_dim}, layers={args.tf_layers}, heads={args.tf_heads}")


    # Decide actual SB3 policy class name to use. For a custom features extractor
    # we still pass a standard policy name (e.g. MlpPolicy) and provide
    # features_extractor_class via policy_kwargs.
    policy_name = args.policy
    if args.policy in {"Transformer", "LargeMlp"}:
        # SB3 does not know a policy called 'Transformer' — use MlpPolicy but
        # provide our Transformer features extractor via policy_kwargs.
        policy_name = "MlpPolicy"
        if args.policy == "Transformer":
            print("Mapping requested 'Transformer' to SB3 policy 'MlpPolicy' with TransformerFeaturesExtractor")
        else:
            print("Mapping requested 'LargeMlp' to SB3 policy 'MlpPolicy'")

    # load existing model if requested, else create a new one
    if args.no_normalize_advantage:
        normalize_advantage = False
    elif args.normalize_advantage:
        normalize_advantage = True
    else:
        # SB3 default
        normalize_advantage = True
    model_kwargs = dict(
        policy=policy_name,
        env=vec_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        clip_range=float(args.clip_range),
        clip_range_vf=args.clip_range_vf,
        normalize_advantage=bool(normalize_advantage),
        policy_kwargs=policy_kwargs,
        ent_coef=float(args.ent_coef),
        vf_coef=float(args.vf_coef),
        device=device,
        verbose=1,
        tensorboard_log=log_dir,
        seed=args.seed,
        target_kl=args.target_kl,
    )
    # use warmup + linear decay schedule for learning rate (callable accepted by SB3)
    model_kwargs['learning_rate'] = lr_schedule
    model_kwargs["max_grad_norm"] = float(args.max_grad_norm)

    if args.load_model:
        print(f"Loading model from: {args.load_model}")
        model = PPO.load(args.load_model, env=vec_env, device=device)
        # Ensure resumed runs log to the new run directory (otherwise it may keep the old path).
        try:
            model.tensorboard_log = log_dir
        except Exception:
            pass
        # if user requested a different learning rate, override optimizer param groups
        if args.learning_rate is not None:
            try:
                for pg in model.policy.optimizer.param_groups:
                    pg['lr'] = float(args.learning_rate)
                print(f"Overrode loaded model optimizer lr -> {args.learning_rate}")
            except Exception:
                print("Warning: failed to override optimizer lr on loaded model")
    else:
        model = PPO(**model_kwargs)

    # 打印模型信息
    policy = model.policy
    calc_param_num(policy) 
    print(policy)

    # print initial optimizer lr for visibility
    try:
        optim = model.policy.optimizer
        lrs = [pg.get('lr', None) for pg in optim.param_groups]
        print(f"Initial optimizer learning rates: {lrs}")
    except Exception:
        pass

    # prepare callbacks for periodic checkpointing
    callbacks = []
    if args.curriculum != "none":
        callbacks.append(
            CurriculumCallback(
                mode=args.curriculum,
                stage1_steps=args.curriculum_stage1_steps,
                update_freq=args.curriculum_update_freq,
            )
        )
    callbacks.append(TrainInfoMetricsCallback(log_freq=2000))
    if args.save_interval and args.save_interval > 0:
        from stable_baselines3.common.callbacks import CheckpointCallback
        ckpt_dir = os.path.join(args.save_dir, f"{run_name}/checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        name_prefix = "ppo_mobile_mm" if args.robot == "mobile_mm" else f"ppo_{args.robot}"
        # SB3's CheckpointCallback uses callback step count (vector env steps),
        # so convert user timesteps -> callback steps by dividing by n_envs.
        save_freq = max(int(args.save_interval) // max(int(args.n_envs), 1), 1)
        ckpt_cb = CheckpointCallback(save_freq=save_freq, save_path=ckpt_dir,
                                     name_prefix=name_prefix)
        callbacks.append(ckpt_cb)

    if args.eval_freq and args.eval_freq > 0:
        eval_env_fn = lambda: MobileMMTrajEnv(
            robot=args.robot,
            urdf_path=args.urdf_path,
            frame_skip=args.frame_skip,
            max_steps=args.max_steps,
            render=False,
            obs_frame=args.obs_frame,
            reward_dist_weight=args.reward_dist_weight,
            reward_collision_threshold=args.reward_collision_threshold,
            reward_collision_ratio=args.reward_collision_ratio,
            reward_clip_abs=args.reward_clip_abs,
            reward_close_bonus=args.reward_close_bonus,
            reward_close_threshold=args.reward_close_threshold,
            target_generator=JSONNearestTargetGenerator(
                json_paths=[],
                json_txt=args.eval_txt,
                mode="seq",
            ),
        )
        eval_env = DummyVecEnv([eval_env_fn])
        eval_env = VecMonitor(eval_env)
        if args.vec_normalize:
            from stable_baselines3.common.vec_env import VecNormalize

            eval_env = VecNormalize(
                eval_env,
                training=False,
                norm_obs=True,
                norm_reward=False,
                clip_obs=float(args.vecnorm_clip_obs),
                clip_reward=float(args.vecnorm_clip_reward),
                gamma=float(args.gamma),
            )
        eval_freq = max(int(args.eval_freq) // max(int(args.n_envs), 1), 1)
        eval_cb = EvalInfoMetricsCallback(
            eval_env,
            n_eval_episodes=int(args.eval_episodes),
            eval_freq=eval_freq,
            log_path=os.path.join(args.save_dir, run_name),
            best_model_save_path=os.path.join(args.save_dir, run_name, "best_model"),
            deterministic=True,
            render=False,
            verbose=1,
            info_keys=(
                "ee_distance",
                "traj_id",
                "remain_traj_ratio",
                "base_vx",
                "base_vy",
                "base_wz",
                "base_lin_vel_norm",
                "base_ang_vel_norm",
                "reward_close",
            ),
        )
        callbacks.append(eval_cb)

    model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks) if callbacks else None)
    name_prefix = "ppo_mobile_mm" if args.robot == "mobile_mm" else f"ppo_{args.robot}"
    model_filename = f"{run_name}/{name_prefix}_final"
    model.save(os.path.join(args.save_dir, model_filename))
    if hasattr(vec_env, "save") and args.vec_normalize:
        try:
            run_root = os.path.join(args.save_dir, run_name)
            os.makedirs(run_root, exist_ok=True)
            stats_path = os.path.join(run_root, "vecnormalize.pkl")
            vec_env.save(stats_path)
            print(f"Saved VecNormalize stats to: {stats_path}")
        except Exception as e:
            print(f"Warning: failed to save VecNormalize stats: {e}")


if __name__ == "__main__":
    main()
