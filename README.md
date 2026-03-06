# ESG Scoring Dashboard
### Data Science Portfolio Project

A Python-based ESG (Environmental, Social, Governance) scoring engine that evaluates
publicly listed companies using proxy metrics from public data, with outputs designed
for Tableau dashboarding.

---

## Project Structure

```
esg_dashboard/
├── data/
│   └── esg_raw_data.csv          # Raw ESG metrics (20 companies)
├── output/
│   ├── esg_scores_master.csv     # Scored dataset (primary)
│   ├── esg_tableau_long.csv      # Long format for Tableau
│   ├── esg_metric_detail.csv     # Metric drill-down
│   ├── esg_sector_summary.csv    # Sector-level KPIs
│   └── scoring_framework.json   # Methodology metadata
├── esg_scoring_engine.py         # Core scoring pipeline
├── esg_visualisations.py         # Plotly charts (run locally)
└── README.md
```

---

## Scoring Methodology

### Pillars & Weights

| Pillar | Weight | Key Metrics |
|---|---|---|
| 🌿 Environmental | 40% | Carbon emissions, renewables %, water intensity, waste recycling, controversies |
| 🤝 Social | 35% | Employee turnover, gender diversity, safety incidents, community investment, data breaches |
| 🏛 Governance | 25% | Board independence, exec pay ratio, audit quality, anti-corruption policy, shareholder rights |

### Normalisation
- All metrics scaled 0–100 via min-max normalisation
- "Lower is better" metrics (emissions, turnover, incident rate) are inverted
- Controversy/breach counts converted to score penalties
- Composite = weighted average of three pillar scores

### Rating Bands
| Rating | Score Range |
|---|---|
| A+ | 85–100 |
| A  | 75–85  |
| B+ | 65–75  |
| B  | 55–65  |
| C+ | 45–55  |
| C  | 30–45  |
| D  | 0–30   |

---

## Data Sources (for real-world extension)

To replace synthetic data with real public data:

| Metric | Free Source |
|---|---|
| Carbon emissions | CDP Scores (cdp.net), company sustainability reports |
| Renewables % | RE100 member data, annual reports |
| Workforce diversity | Company ESG/DEI reports, Glassdoor |
| Safety incidents | OSHA public data (US companies) |
| Board composition | SEC proxy filings (DEF 14A), BoardEx free tier |
| Executive pay ratio | SEC proxy filings (US public companies) |
| Controversies | RepRisk public ESG controversy data |
| General ESG data | Yahoo Finance ESG tab, sustainalytics.com |

---

## Sample Results (2023)

| Rank | Company | ESG Score | Rating |
|---|---|---|---|
| 1 | Microsoft | 84.1 | A |
| 2 | Novartis | 79.1 | A |
| 3 | SAP SE | 76.1 | A |
| 4 | Apple | 75.9 | A |
| 5 | J&J | 73.0 | B+ |
| ... | ... | ... | ... |
| 18 | Exxon Mobil | 28.6 | D |
| 19 | Shell | 29.7 | D |
| 20 | BP | 26.2 | D |

---

## Extensions for Portfolio

- Add time-series data (2019–2023) to show ESG trend analysis
- Scrape real data from SEC EDGAR / CDP using `requests` + `BeautifulSoup`
- Add a machine learning model predicting ESG score from financial features
- Build a Streamlit app as an alternative to Tableau
- Add a weighting sensitivity analysis ("what if governance weight = 40%?")
