#!/usr/bin/env python3
"""
Entry point for FIFA WC 2026 Forecasting System.
Usage:
    python run.py --mode train
    python run.py --mode simulate --n 100000
    python run.py --mode dashboard
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="WC 2026 Forecasting System")
    parser.add_argument("--mode", choices=["train", "simulate", "dashboard"], required=True)
    parser.add_argument("--n", type=int, default=100_000, help="Monte Carlo iterations")
    args = parser.parse_args()

    if args.mode == "train":
        print("Training models... (implement pipeline here)")

    elif args.mode == "simulate":
        print(f"Running Monte Carlo with {args.n:,} iterations...")

    elif args.mode == "dashboard":
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"])


if __name__ == "__main__":
    main()
