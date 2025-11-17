import datetime
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pybullet_envs.mobile_mm import MobileMMBulletEnv
from pybullet_envs.mobile_mm_traj import MobileMMTrajEnv
from pybullet_envs.target_generator import FixedTarget, RandomTargetForEpisode, JSONNearestTargetGenerator
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
    policy_kwargs = {"net_arch": [{"pi": pi_layers, "vf": vf_layers}]}
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--render", action="store_true", help="Use GUI")
    parser.add_argument("--policy", type=str, default="Transformer",
                        choices=["MlpPolicy", "LargeMlp", "Transformer"],
                        help="Policy architecture to use. Transformer = Transformer-based feature extractor")
    # transformer-specific options
    parser.add_argument("--tf_seq_len", type=int, default=8, help="Transformer token sequence length")
    parser.add_argument("--tf_embed_dim", type=int, default=256, help="Transformer embedding dimension")
    parser.add_argument("--tf_layers", type=int, default=3, help="Number of Transformer encoder layers")
    parser.add_argument("--tf_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--tf_dropout", type=float, default=0.1, help="Transformer dropout")
    parser.add_argument("--warmup_frac", type=float, default=0.05,
                        help="Fraction of total training for linear warmup (0-1)")
    parser.add_argument("--save_dir", type=str, default="linux_env_dev/models")
    # PPO / env config
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel envs (vec env)")
    parser.add_argument("--n_steps", type=int, default=2048, help="PPO n_steps (per env)")
    parser.add_argument("--batch_size", type=int, default=64, help="PPO batch_size")
    parser.add_argument("--n_epochs", type=int, default=10, help="PPO n_epochs")
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
    args = parser.parse_args()

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
    env_fn = lambda: MobileMMTrajEnv(render=args.render,
                                       target_generator=JSONNearestTargetGenerator(
                                            json_paths=[
                                                #  "linux_env_dev/new_json_50/cinematic_db_arc_right_pull_arc_right_pull_004.json",
                                                #  "trajectoryToLearn/world_json/scene_1/traj_random_20251110_112441.json",
                                                #          "trajectoryToLearn/world_json/scene_1/traj_random_20251110_215950.json",
                                                #          "trajectoryToLearn/world_json/scene_1/traj_random_20251111_154427.json",
                                                #          "trajectoryToLearn/world_json/scene_1/traj_random_20251111_154646.json",
                                                #          "trajectoryToLearn/world_json/scene_1/traj_random_20251111_154810.json"
                                                    ],
                                            json_txt="linux_env_dev/new_json_50/train.txt",
                                            # json_txt="linux_env_dev/new_json_50/train_stage1.txt",
                                            mode="random"
                                       ))
    if args.n_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        vec_env = SubprocVecEnv([env_fn for _ in range(args.n_envs)])
    else:
        vec_env = DummyVecEnv([env_fn])

    ts = calcaulate_time_stamp()
    log_dir = os.path.join(args.save_dir, f"logs_{ts}/tensorboard_logs")
    os.makedirs(log_dir, exist_ok=True)


    policy_kwargs = get_policy_kwargs(args.pi_layers, args.vf_layers)

    # select device explicitly: prefer CUDA when available
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device != 'cuda':
            print(f"CUDA not available, using device='{device}'")
        else:
            print("Using CUDA device for training")
    except Exception:
        device = 'cpu'
        print("Warning: failed to query CUDA availability; falling back to 'cpu'")

    # If Transformer policy requested, wire custom features extractor
    if args.policy == "Transformer":
        # override policy_kwargs to use our features extractor
        policy_kwargs = dict(
            features_extractor_class=TransformerFeaturesExtractor,
            features_extractor_kwargs={
                'seq_len': int(args.tf_seq_len),
                'embed_dim': int(args.tf_embed_dim),
                'n_heads': int(args.tf_heads),
                'n_layers': int(args.tf_layers),
                'dropout': float(args.tf_dropout),
            }
        )
        print(f"Using Transformer feature extractor: seq_len={args.tf_seq_len}, embed_dim={args.tf_embed_dim}, layers={args.tf_layers}, heads={args.tf_heads}")


    # Decide actual SB3 policy class name to use. For a custom features extractor
    # we still pass a standard policy name (e.g. MlpPolicy) and provide
    # features_extractor_class via policy_kwargs.
    policy_name = args.policy
    if args.policy == "Transformer":
        # SB3 does not know a policy called 'Transformer' — use MlpPolicy but
        # provide our Transformer features extractor via policy_kwargs.
        policy_name = "MlpPolicy"
        print("Mapping requested 'Transformer' to SB3 policy 'MlpPolicy' with TransformerFeaturesExtractor")

    # load existing model if requested, else create a new one
    model_kwargs = dict(
        policy=policy_name,
        env=vec_env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        policy_kwargs=policy_kwargs,
        ent_coef=0.0,
        device=device,
        verbose=1,
        tensorboard_log=log_dir,
    )
    # use warmup + linear decay schedule for learning rate (callable accepted by SB3)
    model_kwargs['learning_rate'] = lr_schedule
    # reduce gradient clipping threshold from SB3 default (0.5) to be more conservative
    model_kwargs['max_grad_norm'] = 0.3

    if args.load_model:
        print(f"Loading model from: {args.load_model}")
        model = PPO.load(args.load_model, env=vec_env, device=device)
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
    if args.save_interval and args.save_interval > 0:
        from stable_baselines3.common.callbacks import CheckpointCallback
        ckpt_dir = os.path.join(args.save_dir, f"logs_{ts}/checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_cb = CheckpointCallback(save_freq=args.save_interval, save_path=ckpt_dir,
                                     name_prefix=f"ppo_mobile_mm")
        callbacks.append(ckpt_cb)

    model.learn(total_timesteps=args.timesteps, callback=callbacks or None)
    model_filename = f"logs_{ts}/ppo_mobile_mm_final"
    model.save(os.path.join(args.save_dir, model_filename))


if __name__ == "__main__":
    main()
