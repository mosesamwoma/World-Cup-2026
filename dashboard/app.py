# ============================================================
#  Streamlit dashboard — FIFA World Cup 2026 Forecasting
# ============================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

from src.config import MODELS_DIR, OUTPUTS_DIR
from src.models import dixon_coles
from src.ratings.elo import current_ratings

st.set_page_config(
    page_title="WC 2026 Forecast",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ FIFA World Cup 2026 — ML Forecasting System")
st.caption("XGBoost · Dixon-Coles · Elo ratings · Monte Carlo")

page = st.sidebar.radio("Navigate", [
    "Match Forecast",
    "Team Profile",
    "Group Stage Odds",
    "Tournament Probabilities",
])

# ── load artefacts ───────────────────────────────────────────
@st.cache_resource
def load_dc_params():
    path = MODELS_DIR / "dixon_coles_params.pkl"
    if path.exists():
        return joblib.load(path)
    return None

@st.cache_data
def load_tournament_probs():
    path = OUTPUTS_DIR / "tournament_probabilities" / "latest.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

dc_params = load_dc_params()
tournament_df = load_tournament_probs()

# ── Match Forecast page ──────────────────────────────────────
if page == "Match Forecast":
    st.subheader("Match Forecast")
    teams = dc_params["teams"] if dc_params else ["France", "Brazil", "Spain"]
    col1, col2 = st.columns(2)
    team_a = col1.selectbox("Team A", teams, index=0)
    team_b = col2.selectbox("Team B", teams, index=1)
    neutral = st.checkbox("Neutral venue", value=True)

    if dc_params and st.button("Generate forecast"):
        result = dixon_coles.predict(team_a, team_b, dc_params, neutral)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{team_a} win", f"{result['win_a']*100:.1f}%")
        c2.metric("Draw", f"{result['draw']*100:.1f}%")
        c3.metric(f"{team_b} win", f"{result['win_b']*100:.1f}%")
        st.info(f"Expected goals: {team_a} λ={result['lambda_a']}  |  {team_b} λ={result['lambda_b']}")
    else:
        st.info("Load a trained Dixon-Coles model to generate forecasts.")

# ── Tournament Probabilities page ────────────────────────────
elif page == "Tournament Probabilities":
    st.subheader("Tournament Probabilities")
    if tournament_df is not None:
        stage = st.selectbox("Stage", ["champion", "final", "sf", "qf", "r16", "r32", "group"])
        sorted_df = tournament_df.sort_values(stage, ascending=False)
        fig = px.bar(
            sorted_df.head(20),
            x="team", y=stage,
            title=f"Probability of reaching: {stage.upper()}",
            labels={stage: "Probability", "team": ""},
            color=stage,
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sorted_df, use_container_width=True)
    else:
        st.warning("Run the Monte Carlo simulation first to generate tournament probabilities.")

else:
    st.info(f"'{page}' page — connect to your processed data to populate.")
