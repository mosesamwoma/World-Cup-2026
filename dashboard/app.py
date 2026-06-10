# ============================================================
#  Streamlit dashboard — FIFA World Cup 2026 Forecasting
#  Fixed: relative paths, works on Streamlit Cloud
# ============================================================
import sys
import os

# Fix paths so imports work both locally and on Streamlit Cloud
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# -- Relative paths -- works everywhere --
MODELS_DIR     = Path(ROOT) / "models"
OUTPUTS_DIR    = Path(ROOT) / "outputs"
DATA_PROCESSED = Path(ROOT) / "data" / "processed"
DATA_RAW       = Path(ROOT) / "data" / "raw"

from src.models.dixon_coles import score_matrix, match_probs, predict

st.set_page_config(
    page_title="WC 2026 Forecast",
    page_icon="⚽",
    layout="wide",
)

# -- Load artefacts --
@st.cache_resource
def load_dc_params():
    path = MODELS_DIR / "dixon_coles_params.pkl"
    if path.exists():
        return joblib.load(path)
    return None

@st.cache_resource
def load_ensemble_weights():
    path = MODELS_DIR / "ensemble_weights.pkl"
    if path.exists():
        return joblib.load(path)
    return {"elo": 0.35, "dixon_coles": 0.35, "xgboost": 0.20, "market": 0.10}

@st.cache_data
def load_tournament_probs():
    path = OUTPUTS_DIR / "tournament_probabilities" / "latest.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_elo_ratings():
    path = DATA_PROCESSED / "elo_ratings.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_ensemble_probs():
    path = OUTPUTS_DIR / "match_forecasts" / "ensemble_probs.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def get_2026_groups():
    """
    Derive groups from results.csv fixtures.
    Falls back to hardcoded groups if data/raw not available on Cloud.
    """
    try:
        from src.preprocessing.normalize import normalize_team
        df = pd.read_csv(DATA_RAW / "results.csv", parse_dates=["date"])
        fixtures = df[
            (df["tournament"] == "FIFA World Cup") &
            (df["date"].dt.year == 2026) &
            (df["home_score"].isna())
        ][["date", "home_team", "away_team"]].copy()

        fixtures["home_team"] = fixtures["home_team"].apply(normalize_team)
        fixtures["away_team"] = fixtures["away_team"].apply(normalize_team)

        played = set()
        for _, r in fixtures.iterrows():
            played.add(frozenset([r.home_team, r.away_team]))

        all_teams = sorted(set(fixtures.home_team) | set(fixtures.away_team))
        groups, assigned = {}, set()
        letter = "A"
        for t1 in all_teams:
            if t1 in assigned:
                continue
            group = [t1]
            for t2 in all_teams:
                if t2 == t1 or t2 in assigned:
                    continue
                if frozenset([t1, t2]) in played:
                    if all(frozenset([t2, gm]) in played for gm in group):
                        group.append(t2)
                if len(group) == 4:
                    break
            if len(group) == 4:
                groups[letter] = sorted(group)
                assigned.update(group)
                letter = chr(ord(letter) + 1)
        if len(groups) == 12:
            return groups
    except Exception:
        pass

    # Fallback -- hardcoded from official FIFA draw
    return {
        "A": ["Algeria",   "Argentina", "Austria",              "Jordan"],
        "B": ["Australia", "Paraguay",  "Turkey",               "United States"],
        "C": ["Belgium",   "Egypt",     "Iran",                 "New Zealand"],
        "D": ["Bosnia and Herzegovina", "Canada", "Qatar",      "Switzerland"],
        "E": ["Brazil",    "Haiti",     "Morocco",              "Scotland"],
        "F": ["Cape Verde","Saudi Arabia","Spain",              "Uruguay"],
        "G": ["Colombia",  "DR Congo",  "Portugal",             "Uzbekistan"],
        "H": ["Croatia",   "England",   "Ghana",                "Panama"],
        "I": ["Curacao",   "Ecuador",   "Germany",              "Ivory Coast"],
        "J": ["Czech Republic","Mexico","South Africa",         "South Korea"],
        "K": ["France",    "Iraq",      "Norway",               "Senegal"],
        "L": ["Japan",     "Netherlands","Sweden",              "Tunisia"],
    }

dc_params     = load_dc_params()
weights       = load_ensemble_weights()
tournament_df = load_tournament_probs()
elo_df        = load_elo_ratings()
ensemble_df   = load_ensemble_probs()
groups        = get_2026_groups()

# -- Sidebar --
st.sidebar.title("WC 2026 Forecast")
st.sidebar.caption("XGBoost · Dixon-Coles · Elo · Monte Carlo")
st.sidebar.divider()

# Navigation moved up
page = st.sidebar.radio("Navigate", [
    "Home",
    "Match Forecast",
    "Team Profile",
    "Group Stage Odds",
    "Tournament Probabilities",
])

st.sidebar.divider()

# Pipeline status for 3 models
st.sidebar.markdown("**Pipeline Status**")
status_dc = "✅ Ready" if dc_params else "❌ Missing"
status_elo = "✅ Ready" if elo_df is not None else "❌ Missing"
status_tournament = "✅ Ready" if tournament_df is not None else "❌ Missing"
st.sidebar.markdown(f"Dixon-Coles: {status_dc}")
st.sidebar.markdown(f"Elo Ratings: {status_elo}")
st.sidebar.markdown(f"Tournament: {status_tournament}")

st.sidebar.divider()
if weights:
    st.sidebar.markdown("**Ensemble Weights**")
    st.sidebar.markdown(f"Elo: `{weights.get('elo', 0):.2f}`")
    st.sidebar.markdown(f"Dixon-Coles: `{weights.get('dixon_coles', 0):.2f}`")
    st.sidebar.markdown(f"XGBoost: `{weights.get('xgboost', 0):.2f}`")

# ============================================================
#  HOME
# ============================================================
if page == "Home":
    st.title("FIFA World Cup 2026 - ML Forecasting System")
    st.markdown(
        "Probabilistic match & tournament predictions using "
        "**Dixon-Coles Poisson**, **XGBoost**, **Elo ratings**, and **Monte Carlo simulation** "
        "across 100,000 tournament iterations."
    )
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Teams", "48")
    c2.metric("Groups", "12")
    c3.metric("Matches", "104")
    c4.metric("MC iterations", "100,000")

    st.divider()

    if tournament_df is not None:
        st.subheader("Top 12 Championship Favourites")
        top12 = tournament_df.sort_values("champion", ascending=False).head(12).copy()
        top12["Champion %"] = (top12["champion"] * 100).round(1)
        fig = px.bar(
            top12, x="team", y="Champion %",
            color="Champion %",
            color_continuous_scale="Greens",
            labels={"Champion %": "Champion %", "team": ""},
            text="Champion %",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            height=400,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, width='stretch')

        st.divider()
        st.subheader("Full Tournament Odds Snapshot")
        snap = tournament_df.sort_values("champion", ascending=False).copy()
        for col in ["group", "r32", "r16", "qf", "sf", "final", "champion"]:
            snap[col] = (snap[col] * 100).round(1).astype(str) + "%"
        snap.columns = [
            c if c == "team" else c.upper()
            for c in snap.columns
        ]
        snap = snap.rename(columns={"team": "Team", "GROUP": "Group",
                                     "CHAMPION": "Champion"})
        st.dataframe(snap, width='stretch', hide_index=True)
    else:
        st.info("Run `python run.py --mode simulate` to generate tournament probabilities.")

# ============================================================
#  MATCH FORECAST
# ============================================================
elif page == "Match Forecast":
    st.title("Match Forecast")

    if not dc_params:
        st.error("No trained model found. Run `python run.py --mode train` first.")
        st.stop()

    teams = sorted(dc_params["teams"])
    col1, col2, col3 = st.columns([2, 1, 2])
    team_a = col1.selectbox(
        "Team A", teams,
        index=teams.index("France") if "France" in teams else 0
    )
    col2.markdown(
        "<div style='text-align:center;padding-top:30px;font-size:20px;'>VS</div>",
        unsafe_allow_html=True
    )
    team_b = col3.selectbox(
        "Team B", teams,
        index=teams.index("Brazil") if "Brazil" in teams else 1
    )
    neutral = st.checkbox("Neutral Venue", value=True)

    if st.button("Generate Forecast", type="primary"):

        # Dixon-Coles prediction
        result = predict(team_a, team_b, dc_params, neutral)

        # Pull ensemble probability if available
        ens_row = None
        if ensemble_df is not None:
            mask = (
                (ensemble_df["team_a"] == team_a) &
                (ensemble_df["team_b"] == team_b)
            )
            if mask.any():
                ens_row = ensemble_df[mask].iloc[0]

        st.divider()

        # Win probabilities -- use ensemble if available, else Dixon-Coles
        win_a = float(ens_row["win_a"]) if ens_row is not None else result["win_a"]
        draw  = float(ens_row["draw"])  if ens_row is not None else result["draw"]
        win_b = float(ens_row["win_b"]) if ens_row is not None else result["win_b"]

        source = "Ensemble (Elo + DC + XGB)" if ens_row is not None else "Dixon-Coles only"
        st.caption(f"Source: {source}")

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{team_a} Win", f"{win_a*100:.1f}%")
        c2.metric("Draw",          f"{draw*100:.1f}%")
        c3.metric(f"{team_b} Win", f"{win_b*100:.1f}%")

        fig_bar = go.Figure(go.Bar(
            x=[win_a*100, draw*100, win_b*100],
            y=[team_a, "Draw", team_b],
            orientation="h",
            marker_color=["#1a6fcc", "#888780", "#c0392b"],
            text=[f"{win_a*100:.1f}%", f"{draw*100:.1f}%", f"{win_b*100:.1f}%"],
            textposition="auto",
        ))
        fig_bar.update_layout(
            height=180,
            margin=dict(t=10, b=10),
            xaxis_title="Probability %",
        )
        st.plotly_chart(fig_bar, width='stretch')

        st.divider()

        col_a, col_b = st.columns(2)
        col_a.metric(f"{team_a} Expected Goals", result["lambda_a"])
        col_b.metric(f"{team_b} Expected Goals", result["lambda_b"])

        st.divider()

        st.subheader("Scoreline Probability Matrix")
        mat     = score_matrix(result["lambda_a"], result["lambda_b"],
                               dc_params["rho"], max_goals=6)
        mat_pct = np.round(mat * 100, 1)
        goals   = list(range(7))

        fig_heat = go.Figure(go.Heatmap(
            z=mat_pct,
            x=[f"{team_b} {g}" for g in goals],
            y=[f"{team_a} {g}" for g in goals],
            colorscale="Blues",
            text=mat_pct,
            texttemplate="%{text:.1f}%",
            showscale=False,
        ))
        fig_heat.update_layout(height=380, margin=dict(t=10))
        st.plotly_chart(fig_heat, width='stretch')

        st.subheader("Most Likely Scorelines")
        scores = []
        for i in range(7):
            for j in range(7):
                scores.append({
                    "Score":       f"{i} - {j}",
                    "Probability": f"{mat_pct[i][j]:.1f}%",
                    "_p":          mat_pct[i][j],
                })
        top8 = sorted(scores, key=lambda x: -x["_p"])[:8]
        cols = st.columns(8)
        for idx, s in enumerate(top8):
            cols[idx].metric(s["Score"], s["Probability"])

# ============================================================
#  TEAM PROFILE
# ============================================================
elif page == "Team Profile":
    st.title("Team Profile")

    if elo_df is None:
        st.error("No Elo ratings found. Run `python run.py --mode train` first.")
        st.stop()

    teams_list = sorted(elo_df["team"].tolist())
    team = st.selectbox(
        "Select Team", teams_list,
        index=teams_list.index("France") if "France" in teams_list else 0
    )

    elo_val  = float(elo_df[elo_df["team"] == team]["elo"].values[0])
    elo_rank = int(elo_df[elo_df["elo"] >= elo_val].shape[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Elo Rating", f"{elo_val:.0f}")
    c2.metric("World Rank",  f"#{elo_rank}")

    if tournament_df is not None and team in tournament_df["team"].values:
        row = tournament_df[tournament_df["team"] == team].iloc[0]
        c3.metric("Champion Odds", f"{row['champion']*100:.1f}%")

        st.divider()
        st.subheader("Tournament Stage Progression")

        stages = ["group", "r32", "r16", "qf", "sf", "final", "champion"]
        labels = ["Group Stage", "R32", "R16", "Quarterfinal",
                  "Semifinal", "Final", "Champion"]
        values = [row[s] * 100 for s in stages]
        colors = ["#888780", "#5b87c0", "#378add",
                  "#185fa5", "#0c447c", "#6fcf97", "#1a9a52"]

        fig_stages = go.Figure()
        for l, v, c in zip(labels, values, colors):
            fig_stages.add_trace(go.Bar(
                x=[v], y=[l],
                orientation="h",
                marker_color=c,
                text=f"{v:.1f}%",
                textposition="auto",
                name=l,
                showlegend=False,
            ))
        fig_stages.update_layout(
            height=350,
            xaxis_title="Probability %",
            margin=dict(t=10),
        )
        st.plotly_chart(fig_stages, width='stretch')

    st.divider()
    st.subheader("Elo Ranking - Top 20")
    top20 = elo_df.head(20).copy()
    top20["highlight"] = top20["team"] == team
    fig_elo = px.bar(
        top20, x="elo", y="team",
        orientation="h",
        color="highlight",
        color_discrete_map={True: "#1a9a52", False: "#378add"},
        labels={"elo": "Elo Rating", "team": ""},
        text="elo",
    )
    fig_elo.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_elo.update_layout(
        showlegend=False,
        height=520,
        margin=dict(t=10),
    )
    fig_elo.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_elo, width='stretch')

# ============================================================
#  GROUP STAGE ODDS
# ============================================================
elif page == "Group Stage Odds":
    st.title("Group Stage Odds")

    if tournament_df is None:
        st.error("Run `python run.py --mode simulate` first.")
        st.stop()

    group_letter = st.selectbox("Select Group", sorted(groups.keys()))
    group_teams  = groups[group_letter]

    st.subheader(f"Group {group_letter}")

    cols = st.columns(len(group_teams))
    for i, team in enumerate(group_teams):
        if team in tournament_df["team"].values:
            row     = tournament_df[tournament_df["team"] == team].iloc[0]
            elo_val = (
                float(elo_df[elo_df["team"] == team]["elo"].values[0])
                if elo_df is not None and team in elo_df["team"].values
                else None
            )
            delta = f"Elo {elo_val:.0f}" if elo_val else ""
            cols[i].metric(team, f"Qualify: {row['group']*100:.0f}%", delta)

    st.divider()

    group_data = []
    for team in group_teams:
        if team in tournament_df["team"].values:
            row = tournament_df[tournament_df["team"] == team].iloc[0]
            group_data.append({
                "team":       team,
                "Qualify %":  round(row["group"] * 100, 1),
                "R32 %":      round(row["r32"]   * 100, 1),
                "R16 %":      round(row["r16"]   * 100, 1),
            })

    if group_data:
        gdf = pd.DataFrame(group_data)
        fig_group = px.bar(
            gdf.melt(id_vars="team", var_name="Stage", value_name="Probability"),
            x="team", y="Probability", color="Stage",
            barmode="group",
            color_discrete_sequence=["#378add", "#185fa5", "#0c447c"],
            labels={"Probability": "Probability %", "team": ""},
            text="Probability",
        )
        fig_group.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_group.update_layout(height=420, margin=dict(t=10))
        st.plotly_chart(fig_group, width='stretch')

    # Group vs group comparison
    st.divider()
    st.subheader("Match Probabilities Within Group")
    if ensemble_df is not None:
        for i, ta in enumerate(group_teams):
            for tb in group_teams[i+1:]:
                mask = (
                    (ensemble_df["team_a"] == ta) &
                    (ensemble_df["team_b"] == tb)
                )
                if mask.any():
                    r = ensemble_df[mask].iloc[0]
                    st.markdown(
                        f"**{ta}** vs **{tb}** -- "
                        f"{ta}: `{r['win_a']*100:.1f}%` · "
                        f"Draw: `{r['draw']*100:.1f}%` · "
                        f"{tb}: `{r['win_b']*100:.1f}%`"
                    )

# ============================================================
#  TOURNAMENT PROBABILITIES
# ============================================================
elif page == "Tournament Probabilities":
    st.title("Tournament Probabilities")

    if tournament_df is None:
        st.error("Run `python run.py --mode simulate` first.")
        st.stop()

    stage_map = {
        "Champion":     "champion",
        "Final":        "final",
        "Semifinal":    "sf",
        "Quarterfinal": "qf",
        "Round of 16":  "r16",
        "Round of 32":  "r32",
        "Group Stage":  "group",
    }

    stage_label = st.selectbox("Stage", list(stage_map.keys()))
    stage_col   = stage_map[stage_label]

    sorted_df = tournament_df.sort_values(stage_col, ascending=False).copy()
    sorted_df[f"{stage_label} %"] = (sorted_df[stage_col] * 100).round(1)

    fig = px.bar(
        sorted_df.head(48),
        x="team",
        y=f"{stage_label} %",
        color=f"{stage_label} %",
        color_continuous_scale="Blues",
        text=f"{stage_label} %",
        labels={f"{stage_label} %": "Probability %", "team": ""},
        title=f"Probability of Reaching: {stage_label}",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        height=520,
        xaxis_tickangle=-45,
        margin=dict(t=40, b=100),
    )
    st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("Full Probability Table")

    display_df = sorted_df[
        ["team", "group", "r32", "r16", "qf", "sf", "final", "champion"]
    ].copy()
    for col in ["group", "r32", "r16", "qf", "sf", "final", "champion"]:
        display_df[col] = (
            display_df[col] * 100
        ).round(1).astype(str) + "%"
    display_df.columns = [
        "Team", "Group", "R32", "R16", "QF", "SF", "Final", "Champion"
    ]
    st.dataframe(display_df, width='stretch', hide_index=True)