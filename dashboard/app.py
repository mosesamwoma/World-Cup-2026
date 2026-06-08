# ============================================================
#  Streamlit dashboard — FIFA World Cup 2026 Forecasting
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

from src.config import MODELS_DIR, OUTPUTS_DIR, DATA_PROCESSED
from src.models import dixon_coles
from src.models.dixon_coles import score_matrix

st.set_page_config(
    page_title="WC 2026 Forecast",
    page_icon="⚽",
    layout="wide",
)

# ── Load artefacts ───────────────────────────────────────────
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

@st.cache_data
def load_elo_ratings():
    path = DATA_PROCESSED / "elo_ratings.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

dc_params      = load_dc_params()
tournament_df  = load_tournament_probs()
elo_df         = load_elo_ratings()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/en/thumb/e/e3/2026_FIFA_World_Cup.svg/200px-2026_FIFA_World_Cup.svg.png",
    width=120,
)
st.sidebar.title("⚽ WC 2026 Forecast")
st.sidebar.caption("XGBoost · Dixon-Coles · Elo · Monte Carlo")
st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "🏠  Home",
    "🔮  Match Forecast",
    "👤  Team Profile",
    "📊  Group Stage Odds",
    "🏆  Tournament Probabilities",
])

# ── Status indicators in sidebar ────────────────────────────
st.sidebar.divider()
st.sidebar.markdown("**Pipeline status**")
st.sidebar.markdown("✅ Dixon-Coles" if dc_params else "❌ Dixon-Coles — run `python run.py --mode train`")
st.sidebar.markdown("✅ Elo ratings"  if elo_df is not None else "❌ Elo ratings  — run `python run.py --mode train`")
st.sidebar.markdown("✅ Tournament"   if tournament_df is not None else "❌ Tournament  — run `python run.py --mode simulate`")

# ════════════════════════════════════════════════════════════
#  HOME
# ════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.title("⚽ FIFA World Cup 2026 — ML Forecasting System")
    st.markdown("Probabilistic match & tournament predictions using **Dixon-Coles Poisson**, **XGBoost**, and **Monte Carlo simulation**.")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Teams", "48")
    c2.metric("Groups", "12")
    c3.metric("Matches", "104")
    c4.metric("MC iterations", "100,000")

    st.divider()

    if tournament_df is not None:
        st.subheader("🏆 Top 10 championship favourites")
        top10 = tournament_df.sort_values("champion", ascending=False).head(10).copy()
        top10["champion_%"] = (top10["champion"] * 100).round(1)
        fig = px.bar(
            top10, x="team", y="champion_%",
            color="champion_%",
            color_continuous_scale="Greens",
            labels={"champion_%": "Champion %", "team": ""},
            text="champion_%",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python run.py --mode simulate` to generate tournament probabilities.")

# ════════════════════════════════════════════════════════════
#  MATCH FORECAST
# ════════════════════════════════════════════════════════════
elif page == "🔮  Match Forecast":
    st.title("🔮 Match Forecast")

    if not dc_params:
        st.error("No trained model found. Run `python run.py --mode train` first.")
        st.stop()

    teams = sorted(dc_params["teams"])
    col1, col2, col3 = st.columns([2, 1, 2])
    team_a  = col1.selectbox("Home team", teams, index=teams.index("France") if "France" in teams else 0)
    col2.markdown("<div style='text-align:center; padding-top:30px; font-size:20px;'>VS</div>", unsafe_allow_html=True)
    team_b  = col3.selectbox("Away team", teams, index=teams.index("Brazil") if "Brazil" in teams else 1)
    neutral = st.checkbox("Neutral venue", value=True)

    if st.button("⚡ Generate forecast", type="primary"):
        result = dixon_coles.predict(team_a, team_b, dc_params, neutral)

        st.divider()

        # Win probabilities
        c1, c2, c3 = st.columns(3)
        c1.metric(f"🏳 {team_a} win", f"{result['win_a']*100:.1f}%")
        c2.metric("🤝 Draw",          f"{result['draw']*100:.1f}%")
        c3.metric(f"🏳 {team_b} win", f"{result['win_b']*100:.1f}%")

        # Probability bar
        fig_bar = go.Figure(go.Bar(
            x=[result["win_a"]*100, result["draw"]*100, result["win_b"]*100],
            y=[team_a, "Draw", team_b],
            orientation="h",
            marker_color=["#1a6fcc", "#888780", "#c0392b"],
            text=[f"{result['win_a']*100:.1f}%", f"{result['draw']*100:.1f}%", f"{result['win_b']*100:.1f}%"],
            textposition="auto",
        ))
        fig_bar.update_layout(height=180, margin=dict(t=10, b=10), xaxis_title="Probability %")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # Expected goals
        col_a, col_b = st.columns(2)
        col_a.metric(f"{team_a} expected goals (λ)", result["lambda_a"])
        col_b.metric(f"{team_b} expected goals (λ)", result["lambda_b"])

        st.divider()

        # Scoreline probability heatmap
        st.subheader("Scoreline probability matrix")
        lam_a   = result["lambda_a"]
        lam_b   = result["lambda_b"]
        rho     = dc_params["rho"]
        mat     = score_matrix(lam_a, lam_b, rho, max_goals=6)
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
        fig_heat.update_layout(height=400, margin=dict(t=10))
        st.plotly_chart(fig_heat, use_container_width=True)

        # Top 8 scorelines
        st.subheader("Most likely scorelines")
        scores = []
        for i in range(7):
            for j in range(7):
                scores.append({"Score": f"{i} - {j}", "Probability": f"{mat_pct[i][j]:.1f}%", "_p": mat_pct[i][j]})
        top8 = sorted(scores, key=lambda x: -x["_p"])[:8]
        cols = st.columns(8)
        for idx, s in enumerate(top8):
            cols[idx].metric(s["Score"], s["Probability"])

# ════════════════════════════════════════════════════════════
#  TEAM PROFILE
# ════════════════════════════════════════════════════════════
elif page == "👤  Team Profile":
    st.title("👤 Team Profile")

    if elo_df is None:
        st.error("No Elo ratings found. Run `python run.py --mode train` first.")
        st.stop()

    teams_list = sorted(elo_df["team"].tolist())
    team = st.selectbox("Select team", teams_list, index=teams_list.index("France") if "France" in teams_list else 0)

    elo_val = elo_df[elo_df["team"] == team]["elo"].values[0]
    elo_rank = int(elo_df[elo_df["elo"] >= elo_val].shape[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Elo rating", f"{elo_val:.0f}")
    c2.metric("World rank", f"#{elo_rank}")

    if tournament_df is not None and team in tournament_df["team"].values:
        row = tournament_df[tournament_df["team"] == team].iloc[0]
        c3.metric("Champion odds", f"{row['champion']*100:.1f}%")

        st.divider()
        st.subheader("Tournament stage progression")

        stages   = ["group", "r32", "r16", "qf", "sf", "final", "champion"]
        labels   = ["Group stage", "Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final", "Champion"]
        values   = [row[s] * 100 for s in stages]
        colors   = ["#888780", "#5b87c0", "#378add", "#185fa5", "#0c447c", "#6fcf97", "#1a9a52"]

        fig_stages = go.Figure()
        for s, l, v, c in zip(stages, labels, values, colors):
            fig_stages.add_trace(go.Bar(
                x=[v], y=[l],
                orientation="h",
                marker_color=c,
                text=f"{v:.1f}%",
                textposition="auto",
                name=l,
                showlegend=False,
            ))
        fig_stages.update_layout(height=350, xaxis_title="Probability %", margin=dict(t=10))
        st.plotly_chart(fig_stages, use_container_width=True)

    st.divider()
    st.subheader(f"Elo ranking context")
    top20 = elo_df.head(20).copy()
    top20["rank"] = range(1, len(top20) + 1)
    top20["highlight"] = top20["team"] == team
    fig_elo = px.bar(
        top20, x="elo", y="team",
        orientation="h",
        color="highlight",
        color_discrete_map={True: "#1a9a52", False: "#378add"},
        labels={"elo": "Elo rating", "team": ""},
        text="elo",
    )
    fig_elo.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_elo.update_layout(showlegend=False, height=520, margin=dict(t=10))
    fig_elo.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_elo, use_container_width=True)

# ════════════════════════════════════════════════════════════
#  GROUP STAGE ODDS
# ════════════════════════════════════════════════════════════
elif page == "📊  Group Stage Odds":
    st.title("📊 Group Stage Odds")

    if tournament_df is None:
        st.error("No simulation data found. Run `python run.py --mode simulate` first.")
        st.stop()

    # Derive groups from fixture data
    try:
        from src.pipeline import get_2026_groups
        groups = get_2026_groups()
    except Exception:
        st.error("Could not load groups. Make sure data/raw/results.csv exists.")
        st.stop()

    group_letter = st.selectbox("Select group", sorted(groups.keys()))
    group_teams  = groups[group_letter]

    st.subheader(f"Group {group_letter}")

    cols = st.columns(len(group_teams))
    for i, team in enumerate(group_teams):
        if team in tournament_df["team"].values:
            row = tournament_df[tournament_df["team"] == team].iloc[0]
            elo_val = elo_df[elo_df["team"] == team]["elo"].values[0] if elo_df is not None and team in elo_df["team"].values else "N/A"
            cols[i].metric(team, f"Qual: {row['group']*100:.0f}%", f"Elo {elo_val:.0f}" if isinstance(elo_val, float) else "")

    st.divider()

    # Group qualification bar chart
    group_data = []
    for team in group_teams:
        if team in tournament_df["team"].values:
            row = tournament_df[tournament_df["team"] == team].iloc[0]
            group_data.append({
                "team": team,
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
        fig_group.update_layout(height=400, margin=dict(t=10))
        st.plotly_chart(fig_group, use_container_width=True)

# ════════════════════════════════════════════════════════════
#  TOURNAMENT PROBABILITIES
# ════════════════════════════════════════════════════════════
elif page == "🏆  Tournament Probabilities":
    st.title("🏆 Tournament Probabilities")

    if tournament_df is None:
        st.error("No simulation data found. Run `python run.py --mode simulate` first.")
        st.stop()

    stage_map = {
        "Champion":     "champion",
        "Final":        "final",
        "Semifinal":    "sf",
        "Quarterfinal": "qf",
        "Round of 16":  "r16",
        "Round of 32":  "r32",
        "Group stage":  "group",
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
        title=f"Probability of reaching: {stage_label}",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        height=500,
        xaxis_tickangle=-45,
        margin=dict(t=40, b=100),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Full probability table")

    display_df = sorted_df[["team", "group", "r32", "r16", "qf", "sf", "final", "champion"]].copy()
    for col in ["group", "r32", "r16", "qf", "sf", "final", "champion"]:
        display_df[col] = (display_df[col] * 100).round(1).astype(str) + "%"
    display_df.columns = ["Team", "Group", "R32", "R16", "QF", "SF", "Final", "Champion"]

    st.dataframe(display_df, use_container_width=True, hide_index=True)