"""Behavioural Cloning (BC) pre-training for CinebotRL.

Trains a policy network on (observation, expert_action) pairs collected by
il_dataset.py.  No Isaac Lab / Isaac Sim dependency — pure PyTorch + SB3.

Usage::

    python scripts/reinforcement_learning/bc/pretrain_bc.py \\
        --demo_file data/il_demos/demos.npz \\
        --obs_dim 70 \\
        --output_path data/il_demos/bc_policy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (needed so we can reuse src/ utilities if desired)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent            # bc/
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent             # project root
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavioural Cloning pre-training")
    parser.add_argument(
        "--demo_file",
        type=str,
        required=True,
        help="Path to .npz demo file produced by il_dataset.py",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/il_demos/bc_policy",
        help="Output path for the SB3-compatible .zip policy (no extension needed)",
    )
    parser.add_argument("--obs_dim",  type=int, required=True, help="Observation dimension")
    parser.add_argument("--act_dim",  type=int, default=9,     help="Action dimension (default: 9)")
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--holdout_fraction", type=float, default=0.1)
    parser.add_argument(
        "--split_mode",
        choices=["row_random", "source_grouped"],
        default="row_random",
        help="Use source_grouped for trajectory-disjoint validation and holdout sets.",
    )
    parser.add_argument("--split_seed", type=int, default=20260710)
    parser.add_argument(
        "--hidden_sizes",
        type=str,
        default="256,256,128",
        help="Comma-separated hidden layer sizes matching train.py net_arch",
    )
    parser.add_argument(
        "--policy_arch",
        type=str,
        default="flat",
        choices=["flat", "grouped"],
        help="Output SB3 policy architecture: flat MlpPolicy or grouped arm/gimbal/base heads.",
    )
    parser.add_argument(
        "--grouped_shared_hidden_dims",
        type=str,
        default="256,256",
        help="Comma-separated shared encoder dims for --policy_arch grouped.",
    )
    parser.add_argument(
        "--grouped_head_hidden_dim",
        type=int,
        default=128,
        help="Hidden dim for each grouped action head.",
    )
    parser.add_argument(
        "--grouped_value_hidden_dims",
        type=str,
        default="256,128",
        help="Comma-separated value-network dims for grouped SB3 export.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--pretrained_policy",
        type=str,
        default=None,
        help="Optional grouped SB3 policy whose actor weights initialize this bounded BC update.",
    )
    parser.add_argument(
        "--use_action_mask",
        action="store_true",
        help="Use action_valid_mask from the demo file and compute MSE only on valid labels.",
    )
    parser.add_argument(
        "--action_loss_weights",
        type=str,
        default=None,
        help=(
            "Optional comma-separated per-action loss weights. Example: "
            "'1,1,1,1,1,1,2,2,1' to emphasize base vx/vy labels."
        ),
    )
    parser.add_argument(
        "--sample_weight_mode",
        type=str,
        default="none",
        choices=["none", "trajectory_tail"],
        help=(
            "Optional per-sample weighting. 'trajectory_tail' upweights late "
            "waypoints inside each trajectory so BC focuses on tail/final-step errors."
        ),
    )
    parser.add_argument(
        "--trajectory_length",
        type=int,
        default=0,
        help=(
            "Trajectory length for --sample_weight_mode trajectory_tail. "
            "If omitted, num_waypoints is read from demo metadata when present."
        ),
    )
    parser.add_argument(
        "--tail_start_fraction",
        type=float,
        default=0.65,
        help="Fraction of each trajectory after which tail sample weights ramp up.",
    )
    parser.add_argument(
        "--tail_weight",
        type=float,
        default=3.0,
        help="Maximum sample weight applied at the final waypoint for trajectory_tail mode.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BCDataset(Dataset):
    """Wraps numpy obs/act arrays as a PyTorch Dataset."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        action_mask: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
    ):
        self.obs  = torch.tensor(observations, dtype=torch.float32)
        self.acts = torch.tensor(actions,      dtype=torch.float32)
        self.mask = None if action_mask is None else torch.tensor(action_mask, dtype=torch.float32)
        self.sample_weights = (
            torch.ones(self.obs.shape[0], dtype=torch.float32)
            if sample_weights is None
            else torch.tensor(sample_weights, dtype=torch.float32)
        )

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx):
        mask = torch.ones_like(self.acts[idx]) if self.mask is None else self.mask[idx]
        return self.obs[idx], self.acts[idx], mask, self.sample_weights[idx]


def build_sample_weights(total_transitions: int, metadata: dict[str, object], args: argparse.Namespace) -> np.ndarray:
    """Build optional per-sample BC weights without changing default behavior."""

    weights = np.ones((total_transitions,), dtype=np.float32)
    if args.sample_weight_mode == "none":
        return weights
    if args.sample_weight_mode != "trajectory_tail":
        raise ValueError(f"unsupported sample_weight_mode: {args.sample_weight_mode}")
    if not (0.0 <= args.tail_start_fraction < 1.0):
        raise ValueError("--tail_start_fraction must be in [0, 1)")
    if args.tail_weight < 1.0:
        raise ValueError("--tail_weight must be >= 1.0")

    trajectory_length = int(args.trajectory_length or metadata.get("num_waypoints", 0) or 0)
    if trajectory_length <= 1:
        raise ValueError(
            "--sample_weight_mode trajectory_tail requires --trajectory_length or metadata.num_waypoints"
        )
    if total_transitions % trajectory_length != 0:
        raise ValueError(
            f"total transitions {total_transitions} is not divisible by trajectory length {trajectory_length}"
        )

    phase = (np.arange(total_transitions, dtype=np.float32) % trajectory_length) / float(trajectory_length - 1)
    ramp = np.clip(
        (phase - float(args.tail_start_fraction)) / max(1.0 - float(args.tail_start_fraction), 1e-6),
        0.0,
        1.0,
    )
    weights = 1.0 + ramp * (float(args.tail_weight) - 1.0)
    return weights.astype(np.float32)


# ---------------------------------------------------------------------------
# MLP builder
# ---------------------------------------------------------------------------

def build_mlp(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    """Build a ReLU MLP matching train.py's architecture convention."""
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class GroupedBCPolicyNet(nn.Module):
    """BC actor wrapper using the grouped SB3 extractor."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        shared_hidden_dims: list[int],
        head_hidden_dim: int,
        activation_fn: type[nn.Module] = nn.ELU,
    ) -> None:
        super().__init__()
        from scripts.reinforcement_learning.sb3.grouped_policy import GroupedActionMlpExtractor

        self.extractor = GroupedActionMlpExtractor(
            feature_dim=obs_dim,
            action_dim=act_dim,
            shared_hidden_dims=shared_hidden_dims,
            head_hidden_dim=head_hidden_dim,
            value_hidden_dims=(256, 128),
            activation_fn=activation_fn,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.extractor.forward_actor(obs)


# ---------------------------------------------------------------------------
# BC training
# ---------------------------------------------------------------------------

def train_bc(
    obs: np.ndarray,
    acts: np.ndarray,
    action_mask: np.ndarray | None,
    sample_weights: np.ndarray,
    args: argparse.Namespace,
    split_labels: np.ndarray | None = None,
) -> nn.Module:
    """Train a BC policy via MSE regression and return the trained network."""

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",") if x.strip()]
    if args.policy_arch == "grouped":
        shared_hidden_dims = [int(x) for x in args.grouped_shared_hidden_dims.split(",") if x.strip()]
        net = GroupedBCPolicyNet(
            obs_dim=args.obs_dim,
            act_dim=args.act_dim,
            shared_hidden_dims=shared_hidden_dims,
            head_hidden_dim=args.grouped_head_hidden_dim,
        ).to(device)
        if args.pretrained_policy:
            from stable_baselines3 import PPO

            pretrained_path = Path(args.pretrained_policy)
            if not pretrained_path.is_absolute():
                pretrained_path = PROJECT_ROOT / pretrained_path
            if not pretrained_path.exists():
                raise FileNotFoundError(f"pretrained policy not found: {pretrained_path}")
            pretrained = PPO.load(str(pretrained_path), device=device)
            net.extractor.load_state_dict(pretrained.policy.mlp_extractor.state_dict(), strict=True)
            print(f"[INFO] Initialized grouped actor from: {pretrained_path}")
    else:
        if args.pretrained_policy:
            raise ValueError("--pretrained_policy currently requires --policy_arch grouped")
        net = build_mlp(args.obs_dim, hidden_sizes, args.act_dim).to(device)

    # Split train / val
    dataset     = BCDataset(obs, acts, action_mask, sample_weights)
    if split_labels is None:
        val_size = max(1, int(len(dataset) * args.val_fraction))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
    else:
        train_indices = np.flatnonzero(split_labels == 0).tolist()
        val_indices = np.flatnonzero(split_labels == 1).tolist()
        if not train_indices or not val_indices:
            raise ValueError("source-grouped split must contain non-empty train and validation rows")
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
        train_size = len(train_indices)
        val_size = len(val_indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    action_weights = torch.ones(args.act_dim, dtype=torch.float32, device=device)
    if args.action_loss_weights:
        parsed_weights = [float(x) for x in args.action_loss_weights.split(",") if x.strip()]
        if len(parsed_weights) != args.act_dim:
            raise ValueError(
                f"--action_loss_weights expected {args.act_dim} values, got {len(parsed_weights)}"
            )
        action_weights = torch.tensor(parsed_weights, dtype=torch.float32, device=device)
        if torch.any(action_weights < 0.0):
            raise ValueError("--action_loss_weights must be non-negative")
    print(f"[INFO] Action loss weights: {action_weights.detach().cpu().numpy().round(4).tolist()}")

    def masked_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> torch.Tensor:
        weighted_mask = mask * action_weights.unsqueeze(0) * sample_weight.unsqueeze(1)
        sq = (pred - target).pow(2) * weighted_mask
        denom = torch.clamp(weighted_mask.sum(), min=1.0)
        return sq.sum() / denom

    best_val_loss  = float("inf")
    best_state     = None

    print(f"\n{'─'*60}")
    print(f"{'Epoch':>6}  {'Train MSE':>12}  {'Val MSE':>12}")
    print(f"{'─'*60}")

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        net.train()
        train_total = 0.0
        for obs_b, act_b, mask_b, weight_b in train_loader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            mask_b = mask_b.to(device)
            weight_b = weight_b.to(device)
            optimizer.zero_grad()
            pred = net(obs_b)
            loss = masked_mse(pred, act_b, mask_b, weight_b)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * obs_b.shape[0]
        train_loss = train_total / train_size

        # --- val ---
        net.eval()
        val_total = 0.0
        with torch.no_grad():
            for obs_b, act_b, mask_b, weight_b in val_loader:
                obs_b = obs_b.to(device)
                act_b = act_b.to(device)
                mask_b = mask_b.to(device)
                weight_b = weight_b.to(device)
                val_total += masked_mse(net(obs_b), act_b, mask_b, weight_b).item() * obs_b.shape[0]
        val_loss = val_total / val_size

        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in net.state_dict().items()}

    print(f"{'─'*60}")
    print(f"Best val MSE: {best_val_loss:.6f}")

    # Restore best weights
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def build_source_grouped_split(
    source_index: np.ndarray,
    *,
    val_fraction: float,
    holdout_fraction: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Return row labels 0=train, 1=validation, 2=holdout by whole source."""

    source_index = np.asarray(source_index, dtype=np.int64)
    if source_index.ndim != 1:
        raise ValueError(f"source_index must be 1D, got {source_index.shape}")
    if val_fraction <= 0.0 or holdout_fraction <= 0.0 or val_fraction + holdout_fraction >= 1.0:
        raise ValueError("val_fraction and holdout_fraction must be positive and sum to less than 1")
    groups = np.unique(source_index)
    if groups.size < 3:
        raise ValueError("source-grouped split requires at least three trajectories")
    rng = np.random.default_rng(seed)
    shuffled = groups.copy()
    rng.shuffle(shuffled)
    holdout_count = max(1, int(round(groups.size * holdout_fraction)))
    val_count = max(1, int(round(groups.size * val_fraction)))
    if holdout_count + val_count >= groups.size:
        raise ValueError("split fractions leave no training trajectories")
    holdout_groups = shuffled[:holdout_count]
    val_groups = shuffled[holdout_count : holdout_count + val_count]
    train_groups = shuffled[holdout_count + val_count :]
    labels = np.zeros(source_index.shape[0], dtype=np.int8)
    labels[np.isin(source_index, val_groups)] = 1
    labels[np.isin(source_index, holdout_groups)] = 2
    return labels, {
        "train": sorted(int(x) for x in train_groups),
        "validation": sorted(int(x) for x in val_groups),
        "holdout": sorted(int(x) for x in holdout_groups),
    }


# ---------------------------------------------------------------------------
# Save as SB3-compatible PPO policy
# ---------------------------------------------------------------------------

def save_as_sb3_policy(
    trained_net: nn.Module,
    obs_dim: int,
    act_dim: int,
    output_path: str,
    hidden_sizes: list[int],
    device: str,
    policy_arch: str = "flat",
    grouped_shared_hidden_dims: list[int] | None = None,
    grouped_head_hidden_dim: int = 128,
    grouped_value_hidden_dims: list[int] | None = None,
) -> None:
    """Transplant trained BC weights into a PPO MlpPolicy and save as .zip.

    A dummy Gymnasium Box environment is used solely to instantiate the PPO
    model with the correct observation / action spaces.  The real Isaac Lab
    environment must be provided when loading the policy for RL fine-tuning.
    """
    import gymnasium as gym
    from stable_baselines3 import PPO

    # --- Dummy env (placeholder only — real env used at RL stage) ---
    obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )
    act_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
    )

    class _DummyEnv(gym.Env):
        """Placeholder env — real Isaac Lab env must be provided at RL stage."""
        observation_space = obs_space
        action_space      = act_space

        def reset(self, *, seed=None, options=None):
            return obs_space.sample(), {}

        def step(self, action):
            return obs_space.sample(), 0.0, False, False, {}

    if policy_arch == "grouped":
        from scripts.reinforcement_learning.sb3.grouped_policy import GroupedActorCriticPolicy

        ppo_model = PPO(
            GroupedActorCriticPolicy,
            _DummyEnv(),
            policy_kwargs=dict(
                shared_hidden_dims=grouped_shared_hidden_dims or [256, 256],
                head_hidden_dim=grouped_head_hidden_dim,
                value_hidden_dims=grouped_value_hidden_dims or [256, 128],
                activation_fn=torch.nn.ELU,
                log_std_init=-2.0,
            ),
            device=device,
            verbose=0,
        )
        if not isinstance(trained_net, GroupedBCPolicyNet):
            raise TypeError(f"grouped export requires GroupedBCPolicyNet, got {type(trained_net)!r}")
        ppo_model.policy.mlp_extractor.copy_actor_from(trained_net.extractor)
        ppo_model.save(output_path)
        print(f"[OK] SB3-compatible grouped policy saved to: {output_path}.zip")
        return

    ppo_model = PPO(
        "MlpPolicy",
        _DummyEnv(),
        policy_kwargs=dict(
            net_arch=dict(pi=hidden_sizes, vf=hidden_sizes),
            activation_fn=torch.nn.ReLU,
        ),
        device=device,
        verbose=0,
    )

    # --- Transplant actor weights ---
    # The SB3 MlpExtractor builds policy_net as a Sequential matching net_arch["pi"].
    # The trained_net is: [Linear, ReLU, ..., Linear(output)] where the last Linear
    # is the action head.  We split at the last Linear for policy_net vs action_net.
    trained_net_cpu = trained_net.to("cpu")
    layers = list(trained_net_cpu.children())
    # All layers except the final Linear → policy_net
    # Final Linear → action_net
    action_layer = layers[-1]
    feature_layers = layers[:-1]

    # Build state dicts matching SB3 naming (Linear_0, Linear_1, …)
    policy_net_sd = {}
    lin_idx = 0
    for layer in feature_layers:
        if isinstance(layer, nn.Linear):
            policy_net_sd[f"{lin_idx}.weight"] = layer.weight.data.clone()
            policy_net_sd[f"{lin_idx}.bias"]   = layer.bias.data.clone()
            lin_idx += 2  # SB3 Sequential numbering: 0=Linear, 1=ReLU, 2=Linear, …

    action_net_sd = {
        "weight": action_layer.weight.data.clone(),
        "bias":   action_layer.bias.data.clone(),
    }

    ppo_model.policy.mlp_extractor.policy_net.load_state_dict(policy_net_sd, strict=False)
    ppo_model.policy.action_net.load_state_dict(action_net_sd)

    ppo_model.save(output_path)
    print(f"[OK] SB3-compatible policy saved to: {output_path}.zip")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    np.random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.split_seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load demo data
    demo_path = Path(args.demo_file)
    if not demo_path.exists():
        print(f"[ERROR] Demo file not found: {demo_path}")
        sys.exit(1)

    data = np.load(str(demo_path))
    observations = data["observations"]  # [N, obs_dim]
    actions      = data["actions"]       # [N, act_dim]
    metadata: dict[str, object] = {}
    if "metadata" in data:
        try:
            metadata = json.loads(str(data["metadata"]))
        except json.JSONDecodeError as exc:
            print(f"[WARN] Could not parse metadata JSON: {exc}")
    action_mask = None
    if args.use_action_mask:
        if "action_valid_mask" not in data:
            print("[ERROR] --use_action_mask requested, but demo file has no action_valid_mask")
            sys.exit(1)
        action_mask = data["action_valid_mask"].astype(np.float32)
        if action_mask.shape != actions.shape:
            print(f"[ERROR] action_valid_mask shape {action_mask.shape} does not match actions {actions.shape}")
            sys.exit(1)

    total_transitions = observations.shape[0]
    print(f"\n[INFO] Loaded demo file: {demo_path}")
    print(f"       Total transitions : {total_transitions:,}")
    print(f"       Obs dim           : {observations.shape[1]}")
    print(f"       Act dim           : {actions.shape[1]}")
    if action_mask is not None:
        print(f"       Masked action mean: {np.mean(action_mask, axis=0)}")

    sample_weights = build_sample_weights(total_transitions, metadata, args)
    print(
        "       Sample weights    : "
        f"mode={args.sample_weight_mode}, min={float(sample_weights.min()):.3f}, "
        f"mean={float(sample_weights.mean()):.3f}, max={float(sample_weights.max()):.3f}"
    )

    if observations.shape[1] != args.obs_dim:
        print(
            f"[WARN] obs_dim mismatch: file has {observations.shape[1]}, "
            f"--obs_dim={args.obs_dim}. Using file value."
        )
        args.obs_dim = observations.shape[1]

    hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
    grouped_shared_hidden_dims = [int(x) for x in args.grouped_shared_hidden_dims.split(",") if x.strip()]
    grouped_value_hidden_dims = [int(x) for x in args.grouped_value_hidden_dims.split(",") if x.strip()]

    split_labels = None
    split_groups = None
    if args.split_mode == "source_grouped":
        if "source_index" not in data:
            print("[ERROR] --split_mode source_grouped requires source_index in the demo file")
            sys.exit(1)
        split_labels, split_groups = build_source_grouped_split(
            data["source_index"],
            val_fraction=args.val_fraction,
            holdout_fraction=args.holdout_fraction,
            seed=args.split_seed,
        )
        print(
            "       Grouped split     : "
            f"train={len(split_groups['train'])}, validation={len(split_groups['validation'])}, "
            f"holdout={len(split_groups['holdout'])} trajectories"
        )

    # Train
    trained_net = train_bc(
        observations,
        actions,
        action_mask,
        sample_weights,
        args,
        split_labels=split_labels,
    )

    # Save as SB3 policy
    output_path = str(PROJECT_ROOT / args.output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_as_sb3_policy(
        trained_net=trained_net,
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        output_path=output_path,
        hidden_sizes=hidden_sizes,
        device=device,
        policy_arch=args.policy_arch,
        grouped_shared_hidden_dims=grouped_shared_hidden_dims,
        grouped_head_hidden_dim=args.grouped_head_hidden_dim,
        grouped_value_hidden_dims=grouped_value_hidden_dims,
    )
    if split_labels is not None and split_groups is not None:
        source_files = data["source_files"].astype(str).tolist() if "source_files" in data else []
        split_manifest = {
            "schema": "cinebotrl_source_grouped_split_v1",
            "seed": args.split_seed,
            "val_fraction": args.val_fraction,
            "holdout_fraction": args.holdout_fraction,
            "row_counts": {
                "train": int(np.sum(split_labels == 0)),
                "validation": int(np.sum(split_labels == 1)),
                "holdout": int(np.sum(split_labels == 2)),
            },
            "source_indices": split_groups,
            "source_files": {
                split: [source_files[index] for index in indices] if source_files else []
                for split, indices in split_groups.items()
            },
        }
        Path(output_path + ".split.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    # Summary
    if split_labels is None:
        val_size = max(1, int(total_transitions * args.val_fraction))
        train_size = total_transitions - val_size
        holdout_size = 0
    else:
        train_size = int(np.sum(split_labels == 0))
        val_size = int(np.sum(split_labels == 1))
        holdout_size = int(np.sum(split_labels == 2))
    print(f"\n{'='*60}")
    print("BEHAVIOURAL CLONING — SUMMARY")
    print(f"{'='*60}")
    print(f"  Total transitions : {total_transitions:,}")
    print(f"  Train / val split : {train_size:,} / {val_size:,}")
    if holdout_size:
        print(f"  Holdout rows       : {holdout_size:,}")
    print(f"  Epochs trained    : {args.epochs}")
    if args.policy_arch == "grouped":
        print(
            "  Architecture      : grouped "
            f"shared={grouped_shared_hidden_dims}, head={args.grouped_head_hidden_dim}, "
            f"value={grouped_value_hidden_dims}, act_dim={args.act_dim}"
        )
    else:
        print(f"  Architecture      : {args.obs_dim} -> {' -> '.join(str(h) for h in hidden_sizes)} -> {args.act_dim}")
    print(f"  Saved to          : {output_path}.zip")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
