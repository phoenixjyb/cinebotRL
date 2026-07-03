"""Distill base-assist teacher labels into selected PPO action-head rows.

This updates an existing PPO checkpoint in place conceptually, but writes a new
checkpoint.  The actor feature extractor, critic, and unselected action rows are
preserved.  Only selected rows of ``policy.action_net`` are trained.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, Dataset, random_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill selected PPO action-head rows.")
    parser.add_argument("--checkpoint", required=True, help="Input PPO checkpoint/final_model.zip.")
    parser.add_argument("--dataset", required=True, help=".npz with observations/actions/action_valid_mask.")
    parser.add_argument("--output", required=True, help="Output PPO checkpoint path, with or without .zip.")
    parser.add_argument("--copy_vec_normalize", default=None, help="Optional VecNormalize pkl to copy beside output.")
    parser.add_argument("--action_indices", default="6,7", help="Comma-separated action rows to train.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument(
        "--reset_policy_optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset SB3 policy optimizer state before saving so PPO continuation starts cleanly.",
    )
    return parser.parse_args()


def parse_indices(raw: str) -> list[int]:
    indices = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not indices:
        raise ValueError("at least one action index is required")
    return indices


class MaskedActionDataset(Dataset):
    def __init__(self, observations: np.ndarray, actions: np.ndarray, mask: np.ndarray, action_indices: list[int]):
        self.obs = torch.tensor(observations, dtype=torch.float32)
        self.actions = torch.tensor(actions[:, action_indices], dtype=torch.float32)
        self.mask = torch.tensor(mask[:, action_indices], dtype=torch.float32)

        keep = self.mask.sum(dim=1) > 0
        self.obs = self.obs[keep]
        self.actions = self.actions[keep]
        self.mask = self.mask[keep]
        if len(self.obs) == 0:
            raise ValueError("dataset has no valid labels for requested action indices")

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, index: int):
        return self.obs[index], self.actions[index], self.mask[index]


def output_zip_path(path: str) -> Path:
    out = Path(path)
    if out.suffix != ".zip":
        out = out.with_suffix(".zip")
    return out


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = Path(args.checkpoint)
    dataset_path = Path(args.dataset)
    if not checkpoint.exists():
        print(f"checkpoint not found: {checkpoint}")
        return 1
    if not dataset_path.exists():
        print(f"dataset not found: {dataset_path}")
        return 1

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    action_indices = parse_indices(args.action_indices)

    data = np.load(dataset_path, allow_pickle=False)
    observations = data["observations"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    mask = data["action_valid_mask"].astype(np.float32)
    metadata = data["metadata"].item() if "metadata" in data else "{}"
    dataset = MaskedActionDataset(observations, actions, mask, action_indices)

    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = PPO.load(str(checkpoint), device=device)
    policy = model.policy
    policy.set_training_mode(True)

    for param in policy.parameters():
        param.requires_grad_(False)
    policy.action_net.weight.requires_grad_(True)
    policy.action_net.bias.requires_grad_(True)

    # Preserve unselected rows exactly even with optimizer state updates.
    selected = torch.tensor(action_indices, dtype=torch.long, device=device)
    original_weight = policy.action_net.weight.detach().clone()
    original_bias = policy.action_net.bias.detach().clone()
    optimizer = torch.optim.Adam([policy.action_net.weight, policy.action_net.bias], lr=args.lr)

    def predict_selected(obs_batch: torch.Tensor) -> torch.Tensor:
        features = policy.extract_features(obs_batch)
        latent_pi, _ = policy.mlp_extractor(features)
        mean_actions = policy.action_net(latent_pi)
        return mean_actions.index_select(dim=1, index=selected)

    def masked_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        sq = (pred - target).pow(2) * valid
        denom = torch.clamp(valid.sum(), min=1.0)
        return sq.sum() / denom

    best_val = float("inf")
    best_weight = None
    best_bias = None

    print("=" * 72)
    print("Base-Assist Action-Head Distillation")
    print("=" * 72)
    print(f"checkpoint       : {checkpoint}")
    print(f"dataset          : {dataset_path}")
    print(f"dataset metadata : {metadata}")
    print(f"rows             : {action_indices}")
    print(f"samples train/val: {train_size:,}/{val_size:,}")
    print(f"device           : {device}")
    print("-" * 72)
    print(f"{'epoch':>6}  {'train_mse':>12}  {'val_mse':>12}")

    for epoch in range(1, args.epochs + 1):
        policy.set_training_mode(True)
        train_total = 0.0
        train_count = 0
        for obs_b, act_b, mask_b in train_loader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)
            mask_b = mask_b.to(device)
            optimizer.zero_grad()
            loss = masked_mse(predict_selected(obs_b), act_b, mask_b)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                unselected = torch.ones(policy.action_net.out_features, dtype=torch.bool, device=device)
                unselected[selected] = False
                policy.action_net.weight[unselected] = original_weight[unselected]
                policy.action_net.bias[unselected] = original_bias[unselected]
            train_total += loss.item() * obs_b.shape[0]
            train_count += obs_b.shape[0]

        policy.set_training_mode(False)
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for obs_b, act_b, mask_b in val_loader:
                obs_b = obs_b.to(device)
                act_b = act_b.to(device)
                mask_b = mask_b.to(device)
                loss = masked_mse(predict_selected(obs_b), act_b, mask_b)
                val_total += loss.item() * obs_b.shape[0]
                val_count += obs_b.shape[0]
        train_loss = train_total / max(train_count, 1)
        val_loss = val_total / max(val_count, 1)
        print(f"{epoch:6d}  {train_loss:12.6f}  {val_loss:12.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_weight = policy.action_net.weight.detach().clone()
            best_bias = policy.action_net.bias.detach().clone()

    if best_weight is not None and best_bias is not None:
        with torch.no_grad():
            policy.action_net.weight.copy_(best_weight)
            policy.action_net.bias.copy_(best_bias)
            unselected = torch.ones(policy.action_net.out_features, dtype=torch.bool, device=device)
            unselected[selected] = False
            policy.action_net.weight[unselected] = original_weight[unselected]
            policy.action_net.bias[unselected] = original_bias[unselected]

    for param in policy.parameters():
        param.requires_grad_(True)
    if args.reset_policy_optimizer:
        policy.optimizer = policy.optimizer_class(
            policy.parameters(),
            lr=float(args.lr),
            **policy.optimizer_kwargs,
        )

    out_zip = output_zip_path(args.output)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_zip))
    print("-" * 72)
    print(f"best val mse     : {best_val:.6f}")
    print(f"saved checkpoint : {out_zip}")

    if args.copy_vec_normalize:
        src = Path(args.copy_vec_normalize)
        if not src.exists():
            print(f"[WARN] VecNormalize source not found: {src}")
        else:
            dst = out_zip.parent / "vec_normalize.pkl"
            shutil.copy2(src, dst)
            print(f"copied vecnorm   : {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
