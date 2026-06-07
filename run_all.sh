#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}🏆 WORLD CUP 2026 PREDICTOR${NC}"
echo -e "${BLUE}========================================${NC}"

# Go to project root
cd /home/mosesamwoma/projects/woldcup2026

# Set Python path
export PYTHONPATH=/home/mosesamwoma/projects/woldcup2026

echo -e "\n${GREEN}✅ Running Elo Rating System...${NC}"
python3 src/ratings/elo.py

echo -e "\n${GREEN}✅ Running Dixon-Coles Model...${NC}"
python3 src/models/dixon_coles.py

echo -e "\n${GREEN}✅ Running XGBoost Model...${NC}"
python3 src/models/xgboost_model.py

echo -e "\n${GREEN}✅ Running Monte Carlo Simulation...${NC}"
python3 src/simulation/monte_carlo.py

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}🎉 All models trained successfully!${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\n${GREEN}🚀 Launching Dashboard...${NC}"
echo -e "${BLUE}Opening browser at: http://localhost:8501${NC}"

# Launch dashboard
streamlit run dashboard/app.py --server.port 8501

