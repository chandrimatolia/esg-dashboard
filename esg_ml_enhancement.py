"""
ESG ML Enhancement
==================
Two upgrades to the base scoring engine:

  1. PCA-DERIVED WEIGHTS
     Uses Principal Component Analysis on the 15 raw metrics to let the
     data determine pillar weights statistically, replacing the hardcoded
     E:40% S:35% G:25% assumption.

  2. K-MEANS CLUSTERING — ESG ARCHETYPES
     Clusters companies by their three pillar scores to discover natural
     ESG archetypes beyond the simple high/low binary split.

Outputs:
  - output/esg_ml_scores.csv        (PCA-weighted scores + cluster labels)
  - output/pca_explained_variance.csv
  - output/cluster_profiles.csv
  - output/ml_methodology.json      (all parameters, for dashboard footer)
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

os.makedirs("output", exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────────
df = pd.read_csv("data/esg_raw_data.csv")
print(f"✓ Loaded {len(df)} companies\n")

# ─────────────────────────────────────────────
# 2. DEFINE METRIC GROUPS & DIRECTIONS
# ─────────────────────────────────────────────
# (metric_column, pillar, invert)
# invert=True means lower raw value = better ESG
METRICS = [
    # Environmental
    ("carbon_emissions_mt",     "Environmental", True),
    ("renewable_energy_pct",    "Environmental", False),
    ("water_usage_intensity",   "Environmental", True),
    ("waste_recycling_pct",     "Environmental", False),
    ("env_controversies",       "Environmental", True),
    # Social
    ("employee_turnover_pct",   "Social", True),
    ("gender_diversity_pct",    "Social", False),
    ("safety_incident_rate",    "Social", True),
    ("community_investment_mn", "Social", False),
    ("data_breaches",           "Social", True),
    # Governance
    ("board_independence_pct",  "Governance", False),
    ("exec_pay_ratio",          "Governance", True),
    ("audit_committee_quality", "Governance", False),
    ("anti_corruption_policy",  "Governance", False),
    ("shareholder_rights_score","Governance", False),
]

metric_cols  = [m[0] for m in METRICS]
metric_pillars = [m[1] for m in METRICS]
metric_inverts = [m[2] for m in METRICS]

# ─────────────────────────────────────────────
# 3. NORMALISE METRICS (min-max, direction-corrected)
# ─────────────────────────────────────────────
X_raw = df[metric_cols].copy().astype(float)

X_norm = pd.DataFrame(index=df.index, columns=metric_cols, dtype=float)
for col, invert in zip(metric_cols, metric_inverts):
    mn, mx = X_raw[col].min(), X_raw[col].max()
    scaled = (X_raw[col] - mn) / (mx - mn) if mx > mn else pd.Series(0.5, index=df.index)
    X_norm[col] = (1 - scaled) if invert else scaled

print("── Step 1: Normalised 15 metrics (0–1 scale, direction-corrected)")
print(X_norm.describe().round(3).to_string())
print()

# ─────────────────────────────────────────────
# 4. PCA ON ALL 15 METRICS
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_norm)

pca = PCA(n_components=3, random_state=42)
pca.fit(X_scaled)

explained = pca.explained_variance_ratio_
print("── Step 2: PCA — Explained Variance by Component")
print(f"  PC1 (Environmental proxy): {explained[0]*100:.1f}%")
print(f"  PC2 (Social proxy):        {explained[1]*100:.1f}%")
print(f"  PC3 (Governance proxy):    {explained[2]*100:.1f}%")
print(f"  Total explained:           {sum(explained)*100:.1f}%\n")

# Map PC loadings back to pillars
# The PC with the highest average loading on Environmental metrics = E pillar, etc.
loadings = pd.DataFrame(
    pca.components_.T,
    index=metric_cols,
    columns=["PC1", "PC2", "PC3"]
)
loadings["pillar"] = metric_pillars

# For each PC, compute the average absolute loading per pillar
pc_pillar_affinity = {}
for pc in ["PC1", "PC2", "PC3"]:
    affinities = loadings.groupby("pillar")[pc].apply(lambda x: x.abs().mean())
    pc_pillar_affinity[pc] = affinities.to_dict()

print("── PC → Pillar affinity (avg absolute loading):")
for pc, aff in pc_pillar_affinity.items():
    dominant = max(aff, key=aff.get)
    print(f"  {pc}: {aff}  →  dominant: {dominant}")
print()

# ─────────────────────────────────────────────
# 5. DERIVE PCA-BASED PILLAR WEIGHTS
# ─────────────────────────────────────────────
# Strategy: each PC's explained variance is allocated to the pillar
# it has the strongest affinity for. Weights are then the share of
# total variance attributed to each pillar.

pillar_variance = {"Environmental": 0.0, "Social": 0.0, "Governance": 0.0}
for pc, var in zip(["PC1", "PC2", "PC3"], explained):
    dominant_pillar = max(pc_pillar_affinity[pc], key=pc_pillar_affinity[pc].get)
    pillar_variance[dominant_pillar] += var

total_var = sum(pillar_variance.values())
pca_weights = {k: v / total_var for k, v in pillar_variance.items()}

print("── Step 3: PCA-Derived Pillar Weights")
for pillar, w in pca_weights.items():
    print(f"  {pillar}: {w*100:.1f}%")
print()

# ─────────────────────────────────────────────
# 6. COMPUTE PILLAR SCORES (same as before, 0–100)
# ─────────────────────────────────────────────
def pillar_score(pillar_name):
    cols = [m[0] for m in METRICS if m[1] == pillar_name]
    return X_norm[cols].mean(axis=1) * 100

df["env_score"]  = pillar_score("Environmental").round(1)
df["soc_score"]  = pillar_score("Social").round(1)
df["gov_score"]  = pillar_score("Governance").round(1)

# ─────────────────────────────────────────────
# 7. PCA-WEIGHTED COMPOSITE SCORE
# ─────────────────────────────────────────────
w_env = pca_weights["Environmental"]
w_soc = pca_weights["Social"]
w_gov = pca_weights["Governance"]

df["esg_score_pca"] = (
    df["env_score"] * w_env +
    df["soc_score"] * w_soc +
    df["gov_score"] * w_gov
).round(1)

# Also keep the original hardcoded-weight score for comparison
df["esg_score_hardcoded"] = (
    df["env_score"] * 0.40 +
    df["soc_score"] * 0.35 +
    df["gov_score"] * 0.25
).round(1)

df["score_delta"] = (df["esg_score_pca"] - df["esg_score_hardcoded"]).round(1)

print("── Step 4: PCA vs Hardcoded Score Comparison")
comparison = df[["company","esg_score_hardcoded","esg_score_pca","score_delta"]].sort_values("score_delta", ascending=False)
print(comparison.to_string(index=False))
print()

# ─────────────────────────────────────────────
# 8. K-MEANS CLUSTERING — ESG ARCHETYPES
# ─────────────────────────────────────────────
print("── Step 5: K-Means Clustering")

# Features for clustering: the three pillar scores
X_cluster = df[["env_score","soc_score","gov_score"]].values
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# Find optimal K using silhouette score (test k=2..6)
sil_scores = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster_scaled)
    sil = silhouette_score(X_cluster_scaled, labels)
    sil_scores[k] = round(sil, 3)
    print(f"  k={k}: silhouette={sil:.3f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"\n  → Optimal k={best_k} (silhouette={sil_scores[best_k]})\n")

# Fit final model
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster_id"] = km_final.fit_predict(X_cluster_scaled)

# ─────────────────────────────────────────────
# 9. NAME THE ARCHETYPES
# ─────────────────────────────────────────────
cluster_profiles = df.groupby("cluster_id").agg(
    n_companies=("company","count"),
    avg_env=("env_score","mean"),
    avg_soc=("soc_score","mean"),
    avg_gov=("gov_score","mean"),
    avg_esg=("esg_score_pca","mean"),
    companies=("company", lambda x: ", ".join(sorted(x)))
).round(1).reset_index()

# Auto-name clusters based on their profile
def name_cluster(row):
    env, soc, gov = row["avg_env"], row["avg_soc"], row["avg_gov"]
    esg = row["avg_esg"]
    if esg >= 72:
        return "Balanced Leaders"
    elif env < 20 and soc < 40:
        return "Structural Laggards"
    elif gov >= 65 and env < 40:
        return "Governance-Strong, E-Weak"
    elif env >= 60 and soc < 45:
        return "Environmental Focus"
    elif soc >= 60 and env < 50:
        return "Social Focus"
    else:
        return "Mid-Tier Performers"

cluster_profiles["archetype"] = cluster_profiles.apply(name_cluster, axis=1)
df["archetype"] = df["cluster_id"].map(
    cluster_profiles.set_index("cluster_id")["archetype"]
)

print("── Step 6: Cluster Archetypes")
for _, row in cluster_profiles.iterrows():
    print(f"\n  Cluster {int(row['cluster_id'])}: {row['archetype']}")
    print(f"    Companies ({int(row['n_companies'])}): {row['companies']}")
    print(f"    Avg E:{row['avg_env']}  S:{row['avg_soc']}  G:{row['avg_gov']}  ESG:{row['avg_esg']}")

# ─────────────────────────────────────────────
# 10. OUTPUTS
# ─────────────────────────────────────────────
# Master ML-enhanced scores
out_cols = [
    "ticker","company","sector","country","market_cap_bn",
    "env_score","soc_score","gov_score",
    "esg_score_hardcoded","esg_score_pca","score_delta",
    "cluster_id","archetype"
]
df[out_cols].to_csv("output/esg_ml_scores.csv", index=False)
print(f"\n✓ ML scores saved → output/esg_ml_scores.csv")

# PCA variance breakdown
pca_var_df = pd.DataFrame({
    "component": ["PC1","PC2","PC3"],
    "explained_variance_pct": (explained * 100).round(1),
    "dominant_pillar": [
        max(pc_pillar_affinity[pc], key=pc_pillar_affinity[pc].get)
        for pc in ["PC1","PC2","PC3"]
    ]
})
pca_var_df.to_csv("output/pca_explained_variance.csv", index=False)
print(f"✓ PCA variance saved → output/pca_explained_variance.csv")

# Cluster profiles
cluster_profiles.to_csv("output/cluster_profiles.csv", index=False)
print(f"✓ Cluster profiles saved → output/cluster_profiles.csv")

# Methodology JSON (for dashboard footer)
methodology = {
    "scoring_method": "PCA-derived weights + K-Means clustering",
    "pca": {
        "n_components": 3,
        "total_variance_explained_pct": round(sum(explained)*100, 1),
        "derived_weights": {
            k: f"{v*100:.1f}%" for k, v in pca_weights.items()
        },
        "vs_hardcoded": {"Environmental":"40.0%","Social":"35.0%","Governance":"25.0%"},
        "component_breakdown": {
            pc: {
                "explained_pct": round(var*100, 1),
                "dominant_pillar": max(pc_pillar_affinity[pc], key=pc_pillar_affinity[pc].get)
            }
            for pc, var in zip(["PC1","PC2","PC3"], explained)
        }
    },
    "clustering": {
        "algorithm": "K-Means",
        "optimal_k": best_k,
        "silhouette_scores": sil_scores,
        "best_silhouette": sil_scores[best_k],
        "archetypes": cluster_profiles[["cluster_id","archetype","n_companies","avg_esg"]].to_dict(orient="records")
    },
    "metrics": 15,
    "normalisation": "Min-max per metric, direction-corrected",
    "companies": len(df),
    "fiscal_year": 2023
}

with open("output/ml_methodology.json", "w") as f:
    json.dump(methodology, f, indent=2)
print(f"✓ Methodology JSON saved → output/ml_methodology.json")

print("\n── Summary ─────────────────────────────────────")
print(f"  PCA weights:  E:{w_env*100:.1f}%  S:{w_soc*100:.1f}%  G:{w_gov*100:.1f}%")
print(f"  vs hardcoded: E:40.0%  S:35.0%  G:25.0%")
print(f"  Clusters:     k={best_k}, silhouette={sil_scores[best_k]}")
archetypes = cluster_profiles[["archetype","n_companies"]].values
for a, n in archetypes:
    print(f"    · {a} ({int(n)} companies)")
