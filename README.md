# World Cup 2026 Match Predictor
A probabilistic prediction engine for the FIFA World Cup 2026 that combines machine learning and statistical modeling—XGBoost, Dixon–Coles Poisson, dynamic Elo ratings, and Monte Carlo simulation—to estimate match outcomes and tournament probabilities from data, not guesswork.

---

## Run it

```bash
# Clone the repository
git clone [https://github.com/mosesamwoma/World-Cup-2026.git](https://github.com/mosesamwoma/World-Cup-2026.git)
cd World-Cup-2026

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run src/dashboard/app.py