"""
train_random_forest.py
======================

Train a Random‑Forest classifier **for either policy**:

    python -m src.models.train_random_forest \
        --csv data/survey_data.csv \
        --policy sdr \
        --model models/rf_sdr.pkl
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from . import utils


# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random‑Forest for SDR / RCV support.")
    p.add_argument("--csv", required=True, help="Path to survey_data.csv")
    p.add_argument(
        "--policy",
        choices=("sdr", "rcv"),
        required=True,
        help="Which policy to model",
    )
    p.add_argument(
        "--model",
        help="Where to pickle the trained model (optional)",
    )
    p.add_argument("--trees", type=int, default=300, help="# trees (default 300)")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    df = utils.load_dataframe(Path(args.csv))

    # clean & build target
    y = utils.prepare_target(df, args.policy).values
    df = df.dropna(subset=[utils.POLICY_CONFIG[args.policy]["outcome"]]).copy()

    # build features
    X, feature_names = utils.build_feature_matrix(df)
    print(f"[INFO] Feature matrix shape: {X.shape}")

    # train
    rf = utils.train_random_forest(X, y, trees=args.trees)
    print(utils.evaluate(rf, X, y))

    # save
    if args.model:
        Path(args.model).parent.mkdir(parents=True, exist_ok=True)
        with open(args.model, "wb") as f:
            pickle.dump({"model": rf, "features": feature_names}, f)
        print(f"[INFO] Model saved to {args.model}")


if __name__ == "__main__":
    main()