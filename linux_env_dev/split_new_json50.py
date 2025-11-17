#!/usr/bin/env python3
"""
Scan linux_env_dev/new_json_50 for .json files, count them, randomly split 9:1 (seeded),
and save two plain-text files listing the relative paths: train.txt and test.txt.
"""
import argparse
import os
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="linux_env_dev/new_json_50", help="Directory with JSON files")
    p.add_argument("--train_out", default="linux_env_dev/new_json_50/train.txt", help="Train list output file")
    p.add_argument("--test_out", default="linux_env_dev/new_json_50/test.txt", help="Test list output file")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility")
    p.add_argument("--ratio", type=float, default=0.1, help="Fraction to reserve as test (default 0.1)")
    args = p.parse_args()

    repo_root = Path.cwd()
    in_dir = repo_root / args.input_dir
    if not in_dir.exists():
        print(f"ERROR: input_dir does not exist: {in_dir}")
        raise SystemExit(1)

    files = sorted([p for p in in_dir.rglob("*.json") if p.is_file()])
    N = len(files)
    print(f"Found {N} JSON files in {in_dir}")

    if N == 0:
        print("No files to split. Exiting.")
        open(args.train_out, "w").close()
        open(args.test_out, "w").close()
        return

    rnd = random.Random(args.seed)
    items = [str(p.relative_to(repo_root)) for p in files]
    rnd.shuffle(items)

    n_test = int(round(N * args.ratio))
    # ensure sensible split
    if n_test < 1 and N >= 2:
        n_test = 1
    if n_test >= N:
        n_test = max(1, N // 10)

    test_items = items[:n_test]
    train_items = items[n_test:]

    # ensure train non-empty
    if len(train_items) == 0 and len(test_items) > 0:
        train_items, test_items = test_items[:-1], test_items[-1:]

    out_train = Path(args.train_out)
    out_test = Path(args.test_out)
    out_train.parent.mkdir(parents=True, exist_ok=True)

    out_train.write_text("\n".join(train_items) + ("\n" if train_items else ""))
    out_test.write_text("\n".join(test_items) + ("\n" if test_items else ""))

    print(f"Wrote {len(train_items)} entries to {out_train}")
    print(f"Wrote {len(test_items)} entries to {out_test}")
    print("Train/Test counts: ", len(train_items), len(test_items))


if __name__ == '__main__':
    main()
