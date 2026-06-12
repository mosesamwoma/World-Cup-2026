#!/usr/bin/env python3
"""
Usage:
    python run.py --mode train
    python run.py --mode train_ensemble
    python run.py --mode simulate --n 1000000
    python run.py --mode dashboard
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="WC 2026 Forecasting System")
    parser.add_argument(
        "--mode",
        choices=["train", "train_ensemble", "simulate", "dashboard"],
        required=True
    )
    parser.add_argument(
        "--n", type=int, default=1_000_000,
        help="Monte Carlo iterations (default: 1,000,000)"
    )
    args = parser.parse_args()

    if args.mode == "train":
        from src.pipeline import run_pipeline
        run_pipeline()

    elif args.mode == "train_ensemble":
        from src.pipeline import train_ensemble
        train_ensemble()

    elif args.mode == "simulate":
        from src.pipeline import run_simulation
        run_simulation(n=args.n)

    elif args.mode == "dashboard":
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])


if __name__ == "__main__":
    main()