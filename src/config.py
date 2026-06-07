# ============================================================
#  Global configuration
# ============================================================
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent

# Paths
DATA_RAW        = ROOT / "data" / "raw"
DATA_PROCESSED  = ROOT / "data" / "processed"
DATA_EXTERNAL   = ROOT / "data" / "external"
MODELS_DIR      = ROOT / "models"
OUTPUTS_DIR     = ROOT / "outputs"

# Elo
BASE_ELO        = 1500
HOME_ADVANTAGE  = 65       # Elo points for non-neutral home team
K_FACTOR_BASE   = 40

# Temporal decay
LAMBDA_DECAY    = 0.25

# Competition importance weights
COMPETITION_WEIGHTS = {
    "FIFA World Cup":              4.0,
    "Continental Championship":   3.0,
    "Nations League":             2.5,
    "WC Qualifier":               2.0,
    "Continental Qualifier":      1.5,
    "Confederations Cup":         1.5,
    "Friendly":                   0.5,
}

# Ensemble weights
ENSEMBLE_WEIGHTS = {
    "elo":         0.35,
    "dixon_coles": 0.35,
    "xgboost":     0.20,
    "market":      0.10,
}

# Simulation
N_SIMULATIONS   = int(os.getenv("N_SIMULATIONS", 100_000))
MAX_GOALS       = 8        # max goals per team in score matrix

# 2026 format
N_GROUPS        = 12
TEAMS_PER_GROUP = 4
THIRD_PLACE_SLOTS = 8      # best third-place teams advancing to R32
