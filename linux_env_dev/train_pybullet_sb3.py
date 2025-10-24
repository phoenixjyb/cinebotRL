import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pybullet_envs.mobile_mm import MobileMMBulletEnv

# parse custom network architectures
def parse_layers(s: str):
    if s is None or s.strip() == "":
        return []
    return [int(x) for x in s.split(",") if x.strip()]

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
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--render", action="store_true", help="Use GUI")
    parser.add_argument("--save_dir", type=str, default="linux_env_dev/models")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name to include in logs/model filenames")
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
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    env_fn = lambda: MobileMMBulletEnv(render=args.render)
    if args.n_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        vec_env = SubprocVecEnv([env_fn for _ in range(args.n_envs)])
    else:
        vec_env = DummyVecEnv([env_fn])

    # create timestamped log/model folder to avoid overwriting previous runs
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_suffix = f"_{args.run_name}" if args.run_name else ""
    log_dir = os.path.join(args.save_dir, f"tensorboard_logs{run_suffix}_{ts}")
    os.makedirs(log_dir, exist_ok=True)


    policy_kwargs = get_policy_kwargs(args.pi_layers, args.vf_layers)

    # load existing model if requested, else create a new one
    if args.load_model:
        print(f"Loading model from: {args.load_model}")
        model = PPO.load(args.load_model, env=vec_env, device='auto')
        # overwrite policy_kwargs if user provided new arch? keep existing loaded arch
    else:
        model = PPO(
            "MlpPolicy", vec_env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            policy_kwargs=policy_kwargs,
            ent_coef=0.0,
            verbose=1,
            tensorboard_log=log_dir,
        )

    # 打印模型信息
    policy = model.policy
    calc_param_num(policy) 



    # prepare callbacks for periodic checkpointing
    callbacks = []
    if args.save_interval and args.save_interval > 0:
        from stable_baselines3.common.callbacks import CheckpointCallback
        ckpt_dir = os.path.join(args.save_dir, f"checkpoints{run_suffix}_{ts}")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_cb = CheckpointCallback(save_freq=args.save_interval, save_path=ckpt_dir,
                                     name_prefix=f"ppo_mobile_mm{run_suffix}")
        callbacks.append(ckpt_cb)

    model.learn(total_timesteps=args.timesteps, callback=callbacks or None)
    model_filename = f"ppo_mobile_mm{run_suffix}_{ts}"
    model.save(os.path.join(args.save_dir, model_filename))


if __name__ == "__main__":
    main()
