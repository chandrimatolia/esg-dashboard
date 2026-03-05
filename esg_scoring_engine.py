"""
ESG Scoring Engine
==================
Scores publicly listed companies across Environmental, Social, and Governance
pillars using publicly available proxy metrics.

Scoring methodology:
- Each pillar scored 0–100 (higher = better)
- Composite ESG score = weighted average of E, S, G pillars
- Percentile ranks computed within sector for fair comparison
- Output: scored CSV + Tableau-ready workbook
"""

import pandas as pd
import numpy as np
import json
import os

# ─────────────────────────────────────────────
# 1. DATA INGESTION
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load and clean the raw ESG metrics CSV."""
    df = pd.read_csv(path)
    # Strip stray whitespace from headers
    df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)
    df = df.dropna(subset=["ticker"])
    print(f"✓ Loaded {len(df)} companies across {df['sector'].nunique()} sectors.")
    return df


# ─────────────────────────────────────────────
# 2. NORMALISATION HELPERS
# ─────────────────────────────────────────────

def min_max_scale(series: pd.Series, invert: bool = False) -> pd.Series:
    """
    Scale a series to [0, 100].
    invert=True for 'lower is better' metrics (e.g. emissions, turnover).
    """
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([50.0] * len(series), index=series.index)
    scaled = (series - mn) / (mx - mn) * 100
    return (100 - scaled) if invert else scaled


def penalty(series: pd.Series, max_val: int = 4, weight: float = 10.0) -> pd.Series:
    """Convert a controversy/incident count into a score penalty (0 = best, max_val = worst)."""
    return (series.clip(0, max_val) / max_val) * weight


# ─────────────────────────────────────────────
# 3. PILLAR SCORING
# ─────────────────────────────────────────────

def score_environmental(df: pd.DataFrame) -> pd.Series:
    """
    Environmental pillar (weight: 40%)
    Metrics:
      - Carbon emissions (intensity proxy, inverted)         25 pts
      - Renewable energy usage %                             25 pts
      - Water usage intensity (inverted)                     20 pts
      - Waste recycling %                                    20 pts
      - Environmental controversies (penalty)               -10 pts max
    """
    carbon   = min_max_scale(df["carbon_emissions_mt"], invert=True) * 0.25
    renewabl = min_max_scale(df["renewable_energy_pct"])              * 0.25
    water    = min_max_scale(df["water_usage_intensity"], invert=True)* 0.20
    waste    = min_max_scale(df["waste_recycling_pct"])               * 0.20
    contro   = penalty(df["env_controversies"])                       * 0.10

    raw = carbon + renewabl + water + waste - contro
    # Rescale to 0–100
    return raw.clip(0, 100).rename("env_score")


def score_social(df: pd.DataFrame) -> pd.Series:
    """
    Social pillar (weight: 35%)
    Metrics:
      - Employee turnover % (inverted)                       20 pts
      - Gender diversity in leadership %                     20 pts
      - Safety incident rate (inverted)                      25 pts
      - Community investment (log-scaled)                    20 pts
      - Data breaches (penalty)                             -15 pts max
    """
    turnover   = min_max_scale(df["employee_turnover_pct"], invert=True) * 0.20
    diversity  = min_max_scale(df["gender_diversity_pct"])               * 0.20
    safety     = min_max_scale(df["safety_incident_rate"], invert=True)  * 0.25
    community  = min_max_scale(np.log1p(df["community_investment_mn"]))  * 0.20
    breaches   = penalty(df["data_breaches"], max_val=2)                 * 0.15

    raw = turnover + diversity + safety + community - breaches
    return raw.clip(0, 100).rename("soc_score")


def score_governance(df: pd.DataFrame) -> pd.Series:
    """
    Governance pillar (weight: 25%)
    Metrics:
      - Board independence %                                 25 pts
      - Exec pay ratio (inverted — lower ratio = better)     20 pts
      - Audit committee quality (1–10 scale)                 20 pts
      - Anti-corruption policy (binary 0/1)                  15 pts
      - Shareholder rights score (1–10)                      20 pts
    """
    board     = min_max_scale(df["board_independence_pct"])              * 0.25
    pay_ratio = min_max_scale(df["exec_pay_ratio"], invert=True)         * 0.20
    audit     = min_max_scale(df["audit_committee_quality"])             * 0.20
    anti_cor  = df["anti_corruption_policy"] * 15                        * 0.15 / 15 * 100 * 0.15
    # Normalise anti_cor properly
    anti_cor  = (df["anti_corruption_policy"] * 100)                     * 0.15
    sh_rights = min_max_scale(df["shareholder_rights_score"])            * 0.20

    raw = board + pay_ratio + audit + anti_cor + sh_rights
    return raw.clip(0, 100).rename("gov_score")


# ─────────────────────────────────────────────
# 4. COMPOSITE SCORE & RATINGS
# ─────────────────────────────────────────────

WEIGHTS = {"env_score": 0.40, "soc_score": 0.35, "gov_score": 0.25}

def compute_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Compute composite ESG score and letter ratings."""
    df["esg_score"] = (
        df["env_score"] * WEIGHTS["env_score"] +
        df["soc_score"] * WEIGHTS["soc_score"] +
        df["gov_score"] * WEIGHTS["gov_score"]
    ).round(1)

    # Letter rating bands
    bins   = [0, 30, 45, 55, 65, 75, 85, 100]
    labels = ["D", "C", "C+", "B", "B+", "A", "A+"]
    df["esg_rating"] = pd.cut(df["esg_score"], bins=bins, labels=labels, right=True)

    return df


def add_sector_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Add within-sector percentile rank for each pillar and composite score."""
    for col in ["env_score", "soc_score", "gov_score", "esg_score"]:
        pct_col = col.replace("_score", "_pct_rank").replace("esg", "esg")
        df[f"{col}_sector_pct"] = df.groupby("sector")[col].rank(pct=True).mul(100).round(1)
    return df


def add_score_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Add categorical band for heatmap / colour coding in Tableau."""
    def band(score):
        if score >= 75: return "Leader"
        elif score >= 55: return "Average"
        elif score >= 35: return "Laggard"
        else: return "Critical"

    df["esg_band"] = df["esg_score"].apply(band)
    df["env_band"] = df["env_score"].apply(band)
    df["soc_band"] = df["soc_score"].apply(band)
    df["gov_band"] = df["gov_score"].apply(band)
    return df


# ─────────────────────────────────────────────
# 5. LONG-FORMAT OUTPUT (for Tableau)
# ─────────────────────────────────────────────

def create_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt scores into long format — ideal for Tableau bar charts,
    heatmaps, and small multiples.
    """
    id_cols = ["ticker", "company", "sector", "country",
               "market_cap_bn", "esg_score", "esg_rating", "esg_band"]
    score_cols = {
        "env_score": "Environmental",
        "soc_score": "Social",
        "gov_score": "Governance"
    }
    frames = []
    for col, label in score_cols.items():
        tmp = df[id_cols + [col]].copy()
        tmp["pillar"] = label
        tmp = tmp.rename(columns={col: "pillar_score"})
        frames.append(tmp)

    long_df = pd.concat(frames, ignore_index=True)
    long_df["pillar_score"] = long_df["pillar_score"].round(1)
    return long_df


# ─────────────────────────────────────────────
# 6. METRIC DETAIL BREAKDOWN (for drill-down)
# ─────────────────────────────────────────────

def create_metric_detail(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per company per metric — useful for Tableau tooltip detail
    and individual metric bar charts.
    """
    metrics = {
        "carbon_emissions_mt":       ("Environmental", "Carbon Emissions (MT CO2e)", True),
        "renewable_energy_pct":      ("Environmental", "Renewable Energy (%)", False),
        "water_usage_intensity":     ("Environmental", "Water Usage Intensity", True),
        "waste_recycling_pct":       ("Environmental", "Waste Recycling (%)", False),
        "employee_turnover_pct":     ("Social",        "Employee Turnover (%)", True),
        "gender_diversity_pct":      ("Social",        "Gender Diversity (%)", False),
        "safety_incident_rate":      ("Social",        "Safety Incident Rate", True),
        "community_investment_mn":   ("Social",        "Community Investment ($M)", False),
        "board_independence_pct":    ("Governance",    "Board Independence (%)", False),
        "exec_pay_ratio":            ("Governance",    "Exec Pay Ratio", True),
        "audit_committee_quality":   ("Governance",    "Audit Committee Quality", False),
        "shareholder_rights_score":  ("Governance",    "Shareholder Rights Score", False),
    }
    id_cols = ["ticker", "company", "sector", "country"]
    frames = []
    for raw_col, (pillar, label, inverted) in metrics.items():
        tmp = df[id_cols + [raw_col]].copy()
        tmp["pillar"] = pillar
        tmp["metric_name"] = label
        tmp["raw_value"] = tmp[raw_col]
        tmp["normalised_score"] = min_max_scale(df[raw_col], invert=inverted).round(1).values
        tmp["lower_is_better"] = inverted
        tmp = tmp.drop(columns=[raw_col])
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────
# 7. SUMMARY STATS (for Tableau KPI cards)
# ─────────────────────────────────────────────

def create_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Sector-level aggregates for Tableau KPI / overview sheet."""
    agg = df.groupby("sector").agg(
        company_count=("ticker", "count"),
        avg_esg=("esg_score", "mean"),
        avg_env=("env_score", "mean"),
        avg_soc=("soc_score", "mean"),
        avg_gov=("gov_score", "mean"),
        top_company=("company", lambda x: x.loc[df.loc[x.index, "esg_score"].idxmax()]),
        top_esg=("esg_score", "max"),
    ).round(1).reset_index()
    return agg


# ─────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Load
    df = load_data(input_path)

    # Score each pillar
    df["env_score"] = score_environmental(df).round(1)
    df["soc_score"] = score_social(df).round(1)
    df["gov_score"] = score_governance(df).round(1)

    # Composite + enrichment
    df = compute_composite(df)
    df = add_sector_percentiles(df)
    df = add_score_bands(df)

    print("\n── Top 5 ESG Scores ──────────────────────────")
    top5 = df.nlargest(5, "esg_score")[["company", "sector", "env_score", "soc_score", "gov_score", "esg_score", "esg_rating"]]
    print(top5.to_string(index=False))

    print("\n── Bottom 5 ESG Scores ───────────────────────")
    bot5 = df.nsmallest(5, "esg_score")[["company", "sector", "env_score", "soc_score", "gov_score", "esg_score", "esg_rating"]]
    print(bot5.to_string(index=False))

    print("\n── Sector Averages ───────────────────────────")
    sector_avg = df.groupby("sector")[["env_score","soc_score","gov_score","esg_score"]].mean().round(1)
    print(sector_avg.to_string())

    # ── Outputs ──────────────────────────────────
    # 1. Master scored dataset
    master_path = os.path.join(output_dir, "esg_scores_master.csv")
    df.to_csv(master_path, index=False)
    print(f"\n✓ Master scores saved → {master_path}")

    # 2. Long format for Tableau
    long_df = create_long_format(df)
    long_path = os.path.join(output_dir, "esg_tableau_long.csv")
    long_df.to_csv(long_path, index=False)
    print(f"✓ Tableau long format → {long_path}")

    # 3. Metric detail breakdown
    detail_df = create_metric_detail(df)
    detail_path = os.path.join(output_dir, "esg_metric_detail.csv")
    detail_df.to_csv(detail_path, index=False)
    print(f"✓ Metric detail saved → {detail_path}")

    # 4. Sector summary
    summary_df = create_summary_stats(df)
    summary_path = os.path.join(output_dir, "esg_sector_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Sector summary saved → {summary_path}")

    # 5. Metadata / scoring framework (for documentation)
    framework = {
        "version": "1.0",
        "scoring_date": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "pillars": {
            "Environmental": {"weight": "40%", "metrics": ["Carbon Emissions", "Renewable Energy", "Water Usage", "Waste Recycling", "Env Controversies"]},
            "Social": {"weight": "35%", "metrics": ["Employee Turnover", "Gender Diversity", "Safety Incidents", "Community Investment", "Data Breaches"]},
            "Governance": {"weight": "25%", "metrics": ["Board Independence", "Exec Pay Ratio", "Audit Quality", "Anti-Corruption Policy", "Shareholder Rights"]},
        },
        "rating_bands": {"A+": "85–100", "A": "75–85", "B+": "65–75", "B": "55–65", "C+": "45–55", "C": "30–45", "D": "0–30"},
        "normalisation": "Min-max scaling within full universe; lower-is-better metrics inverted",
        "companies_scored": len(df),
    }
    framework_path = os.path.join(output_dir, "scoring_framework.json")
    with open(framework_path, "w") as f:
        json.dump(framework, f, indent=2)
    print(f"✓ Scoring framework metadata → {framework_path}")

    return df


if __name__ == "__main__":
    df = run_pipeline(
        input_path="data/esg_raw_data.csv",
        output_dir="output"
    )
