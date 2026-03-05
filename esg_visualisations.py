"""
ESG Dashboard — Plotly Visualisations
======================================
Run this locally after installing: pip install plotly pandas

Generates 6 interactive charts saved as HTML files:
  1. ESG Leaderboard (horizontal bar chart)
  2. Pillar Breakdown — Stacked bars
  3. Environmental vs Social scatter (bubble = governance)
  4. Sector Heatmap
  5. Metric Radar Chart (top 5 vs bottom 5)
  6. Distribution of ESG Scores (histogram + KDE)
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

OUTPUT_DIR = "output/charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load scored data
df = pd.read_csv("output/esg_scores_master.csv")
df_long = pd.read_csv("output/esg_tableau_long.csv")
df_detail = pd.read_csv("output/esg_metric_detail.csv")
df_sector = pd.read_csv("output/esg_sector_summary.csv")

COLOUR_SCALE = {
    "Leader":   "#2ecc71",
    "Average":  "#f39c12",
    "Laggard":  "#e74c3c",
    "Critical": "#8e44ad",
}
PILLAR_COLOURS = {
    "Environmental": "#27ae60",
    "Social":        "#2980b9",
    "Governance":    "#8e44ad",
}


# ──────────────────────────────────────────────
# CHART 1: ESG Leaderboard
# ──────────────────────────────────────────────
def chart_leaderboard():
    df_sorted = df.sort_values("esg_score", ascending=True)
    colours = [COLOUR_SCALE[b] for b in df_sorted["esg_band"]]

    fig = go.Figure(go.Bar(
        x=df_sorted["esg_score"],
        y=df_sorted["company"],
        orientation="h",
        marker_color=colours,
        text=df_sorted["esg_score"].astype(str) + " (" + df_sorted["esg_rating"].astype(str) + ")",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>ESG Score: %{x}<br>Sector: " +
                      df_sorted["sector"].values + "<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="ESG Composite Scores — All Companies", font_size=20),
        xaxis=dict(title="ESG Score (0–100)", range=[0, 110]),
        yaxis=dict(title=""),
        height=700,
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        margin=dict(l=160, r=80, t=60, b=40),
    )

    path = f"{OUTPUT_DIR}/01_leaderboard.html"
    fig.write_html(path)
    print(f"✓ {path}")


# ──────────────────────────────────────────────
# CHART 2: Pillar Breakdown (grouped bars)
# ──────────────────────────────────────────────
def chart_pillar_breakdown():
    df_sorted = df.sort_values("esg_score", ascending=False)

    fig = go.Figure()
    for pillar, col, colour in [
        ("Environmental", "env_score", PILLAR_COLOURS["Environmental"]),
        ("Social",        "soc_score", PILLAR_COLOURS["Social"]),
        ("Governance",    "gov_score", PILLAR_COLOURS["Governance"]),
    ]:
        fig.add_trace(go.Bar(
            name=pillar,
            x=df_sorted["company"],
            y=df_sorted[col],
            marker_color=colour,
            hovertemplate=f"<b>%{{x}}</b><br>{pillar}: %{{y:.1f}}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        title=dict(text="ESG Pillar Scores by Company", font_size=20),
        xaxis=dict(tickangle=-35),
        yaxis=dict(title="Score (0–100)", range=[0, 110]),
        legend=dict(orientation="h", y=1.1),
        height=550,
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
    )

    path = f"{OUTPUT_DIR}/02_pillar_breakdown.html"
    fig.write_html(path)
    print(f"✓ {path}")


# ──────────────────────────────────────────────
# CHART 3: E vs S Scatter (bubble = Governance)
# ──────────────────────────────────────────────
def chart_scatter():
    fig = px.scatter(
        df,
        x="env_score",
        y="soc_score",
        size="gov_score",
        color="sector",
        text="ticker",
        hover_name="company",
        hover_data={"env_score": ":.1f", "soc_score": ":.1f", "gov_score": ":.1f", "esg_score": ":.1f"},
        size_max=40,
        title="Environmental vs Social Scores (bubble size = Governance)",
        labels={"env_score": "Environmental Score", "soc_score": "Social Score"},
        height=600,
    )

    # Add quadrant lines
    fig.add_hline(y=50, line_dash="dash", line_color="grey", opacity=0.4)
    fig.add_vline(x=50, line_dash="dash", line_color="grey", opacity=0.4)

    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(plot_bgcolor="#f8f9fa", paper_bgcolor="white")

    path = f"{OUTPUT_DIR}/03_env_vs_social_scatter.html"
    fig.write_html(path)
    print(f"✓ {path}")


# ──────────────────────────────────────────────
# CHART 4: Sector Heatmap
# ──────────────────────────────────────────────
def chart_sector_heatmap():
    pivot = df.groupby("sector")[["env_score", "soc_score", "gov_score", "esg_score"]].mean().round(1)
    pivot.columns = ["Environmental", "Social", "Governance", "Composite ESG"]
    pivot = pivot.sort_values("Composite ESG", ascending=False)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        zmin=0, zmax=100,
        text=pivot.values.round(1),
        texttemplate="%{text}",
        hovertemplate="Sector: %{y}<br>Pillar: %{x}<br>Score: %{z:.1f}<extra></extra>",
        colorbar=dict(title="Score"),
    ))

    fig.update_layout(
        title=dict(text="Average ESG Scores by Sector", font_size=20),
        xaxis=dict(title=""),
        yaxis=dict(title=""),
        height=450,
        paper_bgcolor="white",
    )

    path = f"{OUTPUT_DIR}/04_sector_heatmap.html"
    fig.write_html(path)
    print(f"✓ {path}")


# ──────────────────────────────────────────────
# CHART 5: Radar — Top 3 vs Bottom 3
# ──────────────────────────────────────────────
def chart_radar():
    metrics_env = ["carbon_emissions_mt", "renewable_energy_pct", "water_usage_intensity",
                   "waste_recycling_pct"]
    metrics_soc = ["employee_turnover_pct", "gender_diversity_pct", "safety_incident_rate",
                   "community_investment_mn"]
    metrics_gov = ["board_independence_pct", "exec_pay_ratio", "audit_committee_quality",
                   "shareholder_rights_score"]

    categories = ["Env Score", "Soc Score", "Gov Score", "ESG Score"]

    top3 = df.nlargest(3, "esg_score")
    bot3 = df.nsmallest(3, "esg_score")

    fig = go.Figure()

    for _, row in top3.iterrows():
        vals = [row["env_score"], row["soc_score"], row["gov_score"], row["esg_score"]]
        vals += [vals[0]]  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=categories + [categories[0]],
            fill="toself", name=row["company"],
            line=dict(color="green"), opacity=0.6,
        ))

    for _, row in bot3.iterrows():
        vals = [row["env_score"], row["soc_score"], row["gov_score"], row["esg_score"]]
        vals += [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=categories + [categories[0]],
            fill="toself", name=row["company"],
            line=dict(color="red"), opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title=dict(text="Radar: Top 3 vs Bottom 3 ESG Companies", font_size=18),
        height=550,
        paper_bgcolor="white",
    )

    path = f"{OUTPUT_DIR}/05_radar_top_vs_bottom.html"
    fig.write_html(path)
    print(f"✓ {path}")


# ──────────────────────────────────────────────
# CHART 6: ESG Score Distribution
# ──────────────────────────────────────────────
def chart_distribution():
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Distribution of ESG Scores", "Score by Sector (Box Plot)"),
    )

    # Histogram
    fig.add_trace(go.Histogram(
        x=df["esg_score"],
        nbinsx=10,
        marker_color="#3498db",
        name="All Companies",
        hovertemplate="Score range: %{x}<br>Count: %{y}<extra></extra>",
    ), row=1, col=1)

    # Box plot by sector
    for sector in df["sector"].unique():
        sub = df[df["sector"] == sector]
        fig.add_trace(go.Box(
            y=sub["esg_score"],
            name=sector,
            boxpoints="all",
            jitter=0.4,
            pointpos=-1.5,
            hovertemplate="%{text}<extra></extra>",
            text=sub["company"],
        ), row=1, col=2)

    fig.update_layout(
        height=500,
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa",
        title=dict(text="ESG Score Distribution Analysis", font_size=20),
    )

    path = f"{OUTPUT_DIR}/06_distribution.html"
    fig.write_html(path)
    print(f"✓ {path}")


# ──────────────────────────────────────────────
# RUN ALL
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── Generating Plotly Charts ──────────────────")
    chart_leaderboard()
    chart_pillar_breakdown()
    chart_scatter()
    chart_sector_heatmap()
    chart_radar()
    chart_distribution()
    print("\n✓ All charts saved to output/charts/")
    print("  Open any .html file in a browser to view interactively.")
