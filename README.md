# World Cup 2026 Match Predictor

A probabilistic World Cup 2026 forecasting engine using an ensemble of XGBoost, Dixon–Coles Poisson modeling, dynamic Elo ratings, and Monte Carlo simulation to generate data-driven match outcomes and tournament probabilities.

## Dashboard

Live dashboard: https://world-cup-2026-ai.streamlit.app/

## Docs

For more detailed information, see [docs.md](docs.md)

---

## Setup

```bash
git clone https://github.com/mosesamwoma/World-Cup-2026.git
cd World-Cup-2026
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Download data

```bash
cd data/raw
curl -L "https://raw.githubusercontent.com/martj42/international_results/master/results.csv" -o results.csv
curl -L "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv" -o goalscorers.csv
curl -L "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv" -o shootouts.csv
cd ../..
```

## Run

```bash
# 1. Train Dixon-Coles + XGBoost on 49K historical matches
python run.py --mode train

# 2. Learn optimal ensemble weights from 2002 + 2022 WC backtest
python run.py --mode train_ensemble

# 3. Run Monte Carlo simulation (100K tournament iterations)
python run.py --mode simulate --n 1000000

# 4. Launch dashboard
python run.py --mode dashboard
```

> **Note:** Running the full pipeline (training, ensemble optimization, and large-scale simulation) can take a considerable amount of time depending on your hardware. For quick testing, use `--n 10000`
