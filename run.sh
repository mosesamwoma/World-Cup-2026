#!/bin/bash
# Run full pipeline from scratch
set -e

echo "⚽ FIFA World Cup 2026 — Full Pipeline"
echo "======================================"

source venv/bin/activate

echo ""
echo "Step 1: Train models"
python run.py --mode train

echo ""
echo "Step 2: Train ensemble weights"
python run.py --mode train_ensemble

echo ""
echo "Step 3: Run simulation (100K iterations)"
python run.py --mode simulate --n 1000000

echo ""
echo "✅ Done. Launch dashboard with:"
echo "   python run.py --mode dashboard"