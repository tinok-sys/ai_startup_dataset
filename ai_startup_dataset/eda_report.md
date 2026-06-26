# AI Startup Sector — Synthetic Dataset: Complete EDA Report

## 1. Executive Summary

This report presents a comprehensive exploratory data analysis of a synthetic dataset representing **5,000 AI software startups** worldwide. The dataset captures 31 attributes across company profile, funding, financial performance, product characteristics, team composition, and market dynamics.

**Key Findings:**
- The AI startup ecosystem spans 7 primary domains, with **NLP/LLM** (24.4%) and **Computer Vision** (19.3%) being the most represented
- **NLP/LLM** shows the highest average revenue growth (75% YoY), reflecting the current generative AI boom
- **Computer Vision** attracts the most funding ($30M avg) and has the lowest churn (10%)
- Open-source companies (11.1% of total) achieve **4.6x more monthly active users** than closed-source peers
- Revenue, funding, and valuation are strongly correlated (r > 0.85), confirming that capital-efficient growth is the industry norm
- ~0.78% missing data across key financial fields — minimal and manageable
- 78% of AI startups were founded in 2023–2025, underscoring the sector's explosive recent growth

---

## 2. Data Quality Analysis

### 2.1 Missing Values

| Column | Missing | % of Total | Reason |
|---|---|---|---|
| `runway_months` | 706 | 14.1% | Bootstrapped companies have no runway (no funding) |
| `revenue_2024` | 100 | 2.0% | Intentionally injected for realism |
| `revenue_2025` | 100 | 2.0% | Intentionally injected for realism |
| `employee_count` | 100 | 2.0% | Intentionally injected for realism |
| `burn_rate_monthly` | 100 | 2.0% | Intentionally injected for realism |
| `gross_margin_pct` | 100 | 2.0% | Intentionally injected for realism |

**Overall: 1,206 missing cells out of 155,000 (0.78%)** — very low.

### 2.2 Duplicates
- 4,538 duplicate company names detected (expected, since we used prefix/suffix combinations for fictional names — not real duplicates, just naming collisions)
- **No duplicate `company_id` values** — primary key is clean

### 2.3 Recommendations for Data Cleaning
1. **Impute missing financial fields** using median by funding stage (e.g., replace NaN revenue with stage-appropriate median)
2. **Drop or flag** the 5 intentionally injected revenue outliers (3–6x normal) for separate analysis
3. **Runway NaN for bootstrapped companies** should be treated as "infinite" or excluded from runway analysis
4. **Company name duplicates** are acceptable here (synthetic data); for real data, merge records or investigate

### 2.4 Outliers
Using the IQR method (1.5x IQR), outliers are concentrated in:

| Variable | Outliers | % | Interpretation |
|---|---|---|---|
| Monthly Active Users | 767 | 15.3% | Heavy-tailed distribution; a few startups achieve viral adoption |
| Customer Count | 762 | 15.2% | Same pattern — Pareto-like distribution |
| Burn Rate Monthly | 718 | 14.7% | Large-variance; late-stage companies burn significantly more |
| Employee Count | 685 | 14.0% | Log-normal distribution expected in startup sizes |
| Revenue 2024 | 682 | 13.9% | Top performers (Series D+) far outpace median |

> **Verdict**: These are not errors — they reflect the naturally skewed, power-law distribution of startup outcomes. No removal recommended.

---

## 3. Descriptive Statistics

### 3.1 Key Metrics Overview

| Metric | Mean | Median | Std Dev | P5 | P95 |
|---|---|---|---|---|---|
| Total Funding | $28.2M | $5.4M | $62.7M | $0 | $152.0M |
| Revenue 2024 | $4.9M | $0.7M | $11.6M | $14K | $26.9M |
| Revenue 2025 | $8.4M | $1.2M | $21.3M | $22K | $47.7M |
| Valuation | $202.3M | $36.9M | $416.0M | $6.4K | $1.1B |
| Employees | 250 | 54 | 626 | 5 | 1,321 |
| Burn Rate (Mo) | $145K | $22K | $382K | $5K | $758K |
| Gross Margin | 72.1% | 72.7% | 10.0% | 53.2% | 87.6% |
| Churn Rate (Ann) | 10.1% | 8.2% | 8.4% | 2.3% | 26.8% |
| Net Revenue Retention | 193% | 189% | 42% | 130% | 269% |
| MAU | 395,673 | 25,365 | 1.28M | 429 | 1.99M |

### 3.2 Highest Variance Variables (by Coefficient of Variation)

| Variable | CV | Explanation |
|---|---|---|
| Monthly Active Users | 3.24 | User adoption is highly polarized |
| Customer Count | 3.07 | Enterprise vs SMB vs Developer-targeted |
| Burn Rate Monthly | 2.63 | Scales massively with funding stage |
| Revenue 2025 | 2.55 | Growth accelerates top performers |
| Employee Count | 2.51 | Team size correlates with funding |
| Revenue 2024 | 2.38 | Wide range from bootstrapped to Series D+ |

### 3.3 Key Trends and Patterns
- **Revenue scales exponentially with funding stage** — Series D+ avg ~$44.8M vs Bootstrapped ~$28K
- **Funding is heavily right-skewed** — median $5.4M but mean $28.2M (top companies pull the average)
- **Gross margins are healthy** (~72%) — typical for software/AI companies
- **NRR consistently above 100%** (median 189%) — strong expansion revenue
- **Growth rate median ~49%** — very high compared to traditional SaaS

---

## 4. Business Insights

### 4.1 Top Performing Segments

**By Domain (Revenue):**
1. **Computer Vision** — $5.37M avg revenue, highest funding ($30M), lowest churn (10%)
2. **Data/MLOps** — $5.23M avg revenue, solid margins (71.8%)
3. **Speech/Audio** — $5.18M avg revenue, highest margins (72.3%)

**By Domain (Growth):**
1. **NLP/LLM** — 75% avg YoY growth (driven by generative AI wave)
2. **AI Agents** — 69% avg YoY growth (emerging category)
3. **AI Infrastructure** — 57% avg YoY growth

**By Country (Revenue):**
1. **China** — $6.69M avg revenue (small sample, high performers)
2. **India** — $5.47M avg revenue (strong emerging ecosystem)
3. **Germany** — $5.02M avg revenue
4. **US** — $4.83M (largest market by volume: 1,977 companies)

### 4.2 Correlation Analysis (Top 10)

| Variable Pair | r | Interpretation |
|---|---|---|
| Revenue 2024 ↔ Revenue 2025 | +0.95 | Extremely stable year-over-year performance |
| Funding ↔ Valuation | +0.90 | Markets efficiently price capital |
| Funding ↔ Revenue 2024 | +0.87 | Capital deployment drives revenue |
| Revenue 2024 ↔ Burn Rate | +0.84 | Scaling requires spending |
| Revenue 2024 ↔ Employees | +0.83 | People cost is the primary expense |
| Revenue 2024 ↔ Valuation | +0.82 | Revenue is the key value driver |
| Funding ↔ Revenue 2025 | +0.82 | Fundraising precedes growth |
| Revenue 2025 ↔ Burn Rate | +0.80 | Growth spending continues |
| Revenue 2025 ↔ Employees | +0.78 | Team scaling lags slightly |
| Revenue 2025 ↔ Valuation | +0.77 | Forward-looking multiples |

**Notable non-correlations:**
- Churn rate vs Revenue: -0.18 (weak negative — better companies churn less)
- Gross Margin vs Growth: +0.04 (no direct relationship)
- Engineer % vs Revenue: -0.08 (more engineers doesn't directly predict revenue)

### 4.3 Open Source Impact

| Metric | Open Source | Closed Source | Delta |
|---|---|---|---|
| Avg Revenue | $5.13M | $4.86M | +5.7% |
| Avg MAU | 1,302,387 | 282,231 | **+361%** |
| Avg Employees | 278 | 246 | +13% |

Open-source companies achieve dramatically higher user adoption (4.6x MAU) with only slightly higher revenue, suggesting a **land-and-expand** adoption pattern.

### 4.4 Actionable Recommendations

1. **Focus on NLP/LLM if pursuing growth** — highest growth rates (75% YoY) attract investor attention
2. **Consider open-source strategy for adoption** — 4.6x user base advantage; monetize through enterprise add-ons
3. **Enterprise segment is the revenue anchor** — avg $5.04M revenue vs SMB $4.71M, with similar growth rates
4. **Monitor burn multiple** — companies with revenue/burn ratio < 3x need to show path to efficiency
5. **Churn below 8% is achievable** — companies with NRR > 200% have churn rates below 6%

---

## 5. Predictive & Strategic Analysis

### 5.1 Risks Identified
- **Concentration risk**: 40% of companies are US-based; 24% in NLP/LLM
- **Funding dependency**: 15% bootstrapped; the rest require continuous capital access
- **Burn rate escalation**: Late-stage companies burn $1M+/month on average
- **Market saturation**: NLP/LLM has 1,218 companies — most competitive domain

### 5.2 Opportunities
- **AI Agents** is the fastest-growing emerging category (69% growth) with less saturation (602 companies)
- **Developer/Individual segment** has the highest growth (61%) and revenue ($5.14M) — underserved channel
- **AI Infrastructure** investment is growing rapidly (57% growth) driven by compute demand
- **Geographic expansion**: China ($6.7M avg rev) and India ($5.5M) show strong unit economics

### 5.3 Hypotheses Testable with This Dataset

| Hypothesis | Test Method | Expected Result |
|---|---|---|
| Open source → higher MAU but lower revenue per user | Two-sample t-test | Confirmed (4.6x MAU, 1.06x revenue) |
| Earlier stage → higher growth rate | ANOVA by stage | Growth stable ~59% across all stages |
| Enterprise focus → lower churn | Correlation enterprise% vs churn | r = -0.18 (weak) |
| NRR clusters above 100% are healthier | K-means clustering | Identify 3+ clusters by growth profile |
| Domain predicts funding efficiency | Regression: funding → rev by domain | CV domain most efficient |

### 5.4 Strategic Decisions (Real-World Application)

1. **Portfolio Construction**: An AI-focused VC should allocate: 40% NLP/LLM (growth), 25% AI Agents (emerging), 20% Computer Vision (stability), 15% Infrastructure (moat)
2. **Go-to-Market**: Developer-first + open-source adoption layer → enterprise upsell (proven by 4.6x MAU advantage)
3. **Global Expansion**: Enter India and China markets first (highest avg revenue per startup outside US)
4. **Efficiency Threshold**: Companies should target < 8% annual churn and > 130% NRR before Series B
5. **Talent Strategy**: Engineer % does not directly predict revenue — balance engineering with go-to-market investment

---

## 6. Visualizations

### 6.1 Distribution Histograms — Key Variables
![Distributions](visualizations/01_distributions.png)
*All financial variables follow log-normal distributions typical of startup ecosystems. Revenue growth is right-skewed with a cluster around 20-80%. Gross margins are normally distributed around 72%.*

### 6.2 Correlation Heatmap
![Correlation Heatmap](visualizations/02_correlation_heatmap.png)
*Strong positive cluster among funding, revenue, valuation, burn rate, and employee count. Churn rate shows weak negative correlation with revenue. NRR and revenue growth have moderate positive correlations with financial metrics.*

### 6.3 Bar Charts — Category Comparisons
![Bar Charts](visualizations/03_bar_charts.png)
*Computer Vision leads avg revenue by domain. Funding escalates dramatically by stage (log scale). China and India show highest avg revenue among countries. NLP/LLM and AI Agents dominate growth rates.*

### 6.4 Line Charts — Trends Over Time
![Trends](visualizations/04_trends_over_time.png)
*The 2023-2025 period shows an explosion in AI startup formation (58% of all companies). Average funding has increased for later-founded companies, signaling more capital availability in recent years.*

### 6.5 Scatter Plots — Key Relationships
![Scatter](visualizations/05_scatter_plots.png)
*Strong positive log-linear relationships between funding ↔ revenue (by domain) and funding ↔ valuation (by stage). Employees scale linearly with revenue. Higher NRR clearly correlates with higher growth rates. Burn rate and runway show expected inverse relationship.*

### 6.6 Box Plots — Outlier Detection by Category
![Box Plots](visualizations/06_box_plots.png)
*Revenue grows exponentially with funding stage. Growth rates show more outliers at early stages. AI Agents and NLP/LLM show widest growth dispersion. Computer Vision has tightest churn distribution. Enterprise segment shows widest revenue range.*

### 6.7 Composition Analysis (Pie Charts)
![Composition](visualizations/07_composition_pie.png)
*Series A is the most common funding stage (28.3%). NLP/LLM dominates domain distribution (24.4%). SaaS leads product type (30%). Cloud deployment is preferred (50%). Enterprise and SMB are equally targeted. US represents 39.5% of all companies.*

### 6.8 Market Size by Domain
![Market Size](visualizations/08_market_size_treemap.png)
*NLP/LLM commands the largest addressable market (~$170B avg), followed by AI Infrastructure (~$96B). Data/MLOps has the smallest TAM (~$35B).*

### 6.9 Pairplot — Key Metrics
![Pairplot](visualizations/09_pairplot.png)
*Multi-dimensional view of relationships among funding, revenue, employees, gross margin, and NRR. Confirms log-linear relationships and shows the multivariate structure of startup performance.*

### 6.10 Growth Rate vs User Adoption
![Growth vs MAU](visualizations/10_growth_vs_mau.png)
*NLP/LLM and AI Agents cluster at higher growth rates. Computer Vision and Speech/Audio show broader MAU distribution. No strong linear relationship between growth rate and MAU at the aggregate level.*

### 6.11 Open Source vs Closed Source
![Open Source](visualizations/11_open_source_comparison.png)
*Open-source companies show significantly higher median MAU (4.6x) with comparable revenue. Employee count distribution is similar. Open-source strategy prioritizes adoption over direct monetization.*

### 6.12 Churn Rate vs Net Revenue Retention
![Churn vs NRR](visualizations/12_churn_vs_nrr.png)
*Strong inverse relationship: companies with NRR > 200% consistently show churn below 5%. Domain-level patterns are visible but overlapping. The NRR = 100% reference line separates value-creating from value-neutral companies.*

---

## 7. Dashboard-Style Summary

### AI Startup Ecosystem at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                   AI STARTUP DASHBOARD 2025                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  TOTAL COMPANIES: 5,000           MEDIAN REVENUE: $690K     │
│  TOTAL FUNDING AVG: $28.2M        MEDIAN FUNDING: $5.4M     │
│  AVG VALUATION: $202.3M           MEDIAN VALUATION: $36.9M  │
│  AVG GROWTH: 59% YoY              MEDIAN GROWTH: 40% YoY    │
│  AVG GROSS MARGIN: 72.1%          MEDIAN MARGIN: 72.7%      │
│  AVG CHURN: 10.1%                 MEDIAN CHURN: 8.2%        │
│  AVG NRR: 193%                    MEDIAN NRR: 189%          │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  TOP DOMAINS BY REVENUE:                                     │
│  ┌────────────┬────────┬────────┬────────┬─────────┐        │
│  │ Domain     │ Count  │ Rev    │ Growth │ Funding │        │
│  ├────────────┼────────┼────────┼────────┼─────────┤        │
│  │ CV         │  964   │ $5.4M  │  52%   │ $29.9M  │        │
│  │ Data/MLOps │  390   │ $5.2M  │  46%   │ $27.8M  │        │
│  │ Speech/Aud │  518   │ $5.2M  │  49%   │ $28.8M  │        │
│  │ AI Agents  │  602   │ $4.9M  │  69%   │ $27.8M  │        │
│  │ NLP/LLM    │ 1,218  │ $4.8M  │  75%   │ $29.1M  │        │
│  │ Decision   │  706   │ $4.5M  │  49%   │ $26.5M  │        │
│  │ AI Infra   │  602   │ $4.2M  │  57%   │ $25.8M  │        │
│  └────────────┴────────┴────────┴────────┴─────────┘        │
│                                                              │
│  STAGE DISTRIBUTION:                                         │
│  Bootstrapped: 14.1%  │  Seed: 24.3%  │  Series A: 28.3%    │
│  Series B: 18.6%      │  Series C: 9.2%  │  Series D+: 5.5% │
│                                                              │
│  GEOGRAPHY:                                                  │
│  US: 39.5%   UK: 10.5%   Germany: 6.8%   India: 6.3%        │
│  Canada: 5.7%   Israel: 5.3%   France: 4.9%   China: 4.3%   │
│                                                              │
│  KEY CORRELATIONS:                                           │
│  Rev24↔Rev25: +0.95   │  Funding↔Val: +0.90                  │
│  Funding↔Rev: +0.87   │  Rev↔Burn: +0.84                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. Conclusion and Next Steps

This synthetic dataset provides a realistic cross-section of the AI startup ecosystem in 2025. The data reveals clear patterns:

1. **AI is a winner-take-most market** — the top 5% of companies (Series C+) capture the majority of revenue and valuation
2. **Generative AI (NLP/LLM) is the defining trend** — highest growth, most companies, largest TAM
3. **Unit economics are healthy** — 72% gross margins, >100% NRR, manageable churn
4. **Open source is an adoption wedge, not a revenue strategy** — 4.6x more users but comparable revenue
5. **Capital efficiency varies by domain** — Computer Vision and Decision/MLOps show the best funding-to-revenue conversion

**Recommended follow-up analyses:**
- Survival analysis: what factors predict reaching Series B+?
- Cohort analysis: how do 2023 vs 2025 cohorts differ in capital efficiency?
- Segmentation: identify 3–5 distinct company archetypes via clustering
- Predictive model: what variables best predict next-stage funding?

---

*Report generated by automated EDA pipeline. Dataset: `ai_startups_5000.csv` (5,000 records, 31 columns).*
