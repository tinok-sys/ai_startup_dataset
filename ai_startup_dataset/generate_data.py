"""
Synthetic AI Startup Dataset Generator
Generates 5,000 realistic records for the AI software/startup sector.
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 5000

countries_pool = {
    "United States": 0.40, "United Kingdom": 0.10, "Germany": 0.07,
    "Canada": 0.06, "Israel": 0.05, "India": 0.06, "France": 0.05,
    "China": 0.04, "Singapore": 0.03, "Netherlands": 0.03,
    "Switzerland": 0.03, "Spain": 0.03, "Sweden": 0.02, "South Korea": 0.02,
    "Australia": 0.01
}
countries = list(countries_pool.keys())
country_weights = list(countries_pool.values())

cities_map = {
    "United States": ["San Francisco", "New York", "Palo Alto", "Seattle", "Boston", "Los Angeles", "Austin", "Chicago"],
    "United Kingdom": ["London", "Cambridge", "Oxford", "Edinburgh", "Manchester"],
    "Germany": ["Berlin", "Munich", "Hamburg", "Stuttgart", "Cologne"],
    "Canada": ["Toronto", "Vancouver", "Montreal", "Waterloo", "Ottawa"],
    "Israel": ["Tel Aviv", "Haifa", "Jerusalem", "Herzliya"],
    "India": ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune"],
    "France": ["Paris", "Lyon", "Grenoble", "Toulouse", "Nice"],
    "China": ["Beijing", "Shanghai", "Shenzhen", "Hangzhou", "Guangzhou"],
    "Singapore": ["Singapore"],
    "Netherlands": ["Amsterdam", "Rotterdam", "Eindhoven", "Utrecht"],
    "Switzerland": ["Zurich", "Geneva", "Lausanne", "Basel"],
    "Spain": ["Madrid", "Barcelona", "Valencia", "Seville"],
    "Sweden": ["Stockholm", "Gothenburg", "Malmo", "Uppsala"],
    "South Korea": ["Seoul", "Busan", "Daejeon", "Pangyo"],
    "Australia": ["Sydney", "Melbourne", "Brisbane", "Canberra"]
}

domains = ["NLP/LLM", "Computer Vision", "Speech/Audio", "Decision/MLOps", "AI Infrastructure", "AI Agents", "Data/MLOps"]
domain_weights = [0.25, 0.18, 0.10, 0.15, 0.12, 0.12, 0.08]

product_types = ["SaaS", "API-first", "Platform", "Hybrid", "Open-source"]
product_weights = [0.30, 0.25, 0.20, 0.15, 0.10]

deployment_types = ["Cloud", "Hybrid", "On-premise", "Edge"]
deployment_weights = [0.50, 0.25, 0.15, 0.10]

stages = ["Bootstrapped", "Seed", "Series A", "Series B", "Series C", "Series D+"]
stage_weights = [0.15, 0.25, 0.28, 0.18, 0.09, 0.05]

target_segments = ["Enterprise", "SMB", "Both", "Developer/Individual"]
seg_weights = [0.30, 0.30, 0.25, 0.15]

company_prefixes = [
    "Neural", "Cogni", "Deep", "Synth", "Quantum", "Axiom", "Voxel", "Tensor",
    "Pulse", "Flux", "Nova", "Zen", "Apt", "Lumina", "Catalyst", "Fusion",
    "Stellar", "Orbit", "Atlas", "Nexus", "Cortex", "Synthetik", "Echo", "Prism"
]
company_suffixes = [
    "AI", "Mind", "Labs", "Core", "Dynamics", "Systems", "Technologies",
    "Intelligence", "Analytics", "Works", "Solutions", "Data", "Platform",
    "Robotics", "Vision", "Logic", "Sense", "Wave"
]

# Generate enough names (cross-product may be less than N)
name_pool = [f"{p}{s}" for p in company_prefixes for s in company_suffixes]
if len(name_pool) < N:
    extras = [f"{p}AI" for p in ["Pixel", "Vector", "Matrix", "Binary", "Digital", "Cyber", "Data", "Logic", "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa"][:max(0, N - len(name_pool))]]
    name_pool.extend(extras)
    extras2 = [f"AI{np.random.choice(['Hub','Gen','Flow','Edge','Nest','Link','Mesh','Stack','Scope','Ware','Grid','Sync'])}" for _ in range(max(0, N - len(name_pool)))]
    name_pool.extend(extras2)
np.random.shuffle(name_pool)
name_pool = name_pool[:N]

data = {
    "company_id": [f"AI-{i:04d}" for i in range(1, N + 1)],
    "company_name": name_pool[:N],
}

countries_arr = np.random.choice(countries, size=N, p=country_weights)
data["country"] = countries_arr
data["city"] = [np.random.choice(cities_map[c]) for c in countries_arr]

year_weights = np.array([0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.58])
year_weights = year_weights / year_weights.sum()
data["year_founded"] = np.random.choice(range(2015, 2026), size=N, p=year_weights)

data["funding_stage"] = np.random.choice(stages, size=N, p=stage_weights)

stage_order = {"Bootstrapped": 0, "Seed": 1, "Series A": 2, "Series B": 3, "Series C": 4, "Series D+": 5}
stage_idx = np.array([stage_order[s] for s in data["funding_stage"]])

max_funding = np.array([0, 2e6, 15e6, 60e6, 200e6, 500e6])
min_funding = np.array([0, 0.1e6, 1e6, 5e6, 15e6, 50e6])
funding_median = np.array([0, 0.8e6, 6e6, 25e6, 80e6, 200e6])

funding = np.zeros(N)
for i in range(N):
    s = stage_idx[i]
    if s == 0:
        funding[i] = 0
    else:
        low, high = min_funding[s], max_funding[s]
        mu = np.log(funding_median[s])
        sigma = 0.5
        val = np.random.lognormal(mu, sigma)
        funding[i] = np.clip(val, low, high)

data["total_funding_usd"] = funding
data["funding_rounds"] = np.clip(np.round(stage_idx * np.random.uniform(0.8, 1.5, N) + np.random.randint(0, 2, N)).astype(int), 0, 8)

# Valuation: roughly 5-15x funding (with noise), cap at $2B
valuation_mult = np.random.lognormal(mean=2.0, sigma=0.5, size=N)
data["valuation_usd"] = np.clip(funding * valuation_mult, 0, 2e9)
data["valuation_usd"] = np.where(data["funding_stage"] == "Bootstrapped",
                                 np.random.lognormal(10, 1.0, N), data["valuation_usd"])

# Revenue: correlated with stage and domain
base_revenue = np.array([0.02e6, 0.1e6, 0.8e6, 4e6, 15e6, 50e6])
revenue_mu = np.log(np.maximum(base_revenue[stage_idx], 0.01))
revenue_sigma = np.where(stage_idx >= 3, 0.6, 0.8)
rev2024 = np.exp(np.random.normal(revenue_mu, revenue_sigma))
rev2024 = np.where(funding > 0, np.clip(rev2024, 0, funding * 0.3), rev2024)
rev2024 = np.clip(rev2024, 0, 80e6)

# Revenue growth: correlated with domain (LLM grows faster)
domain_growth_bonus = {"NLP/LLM": 0.3, "AI Agents": 0.25, "AI Infrastructure": 0.15,
                       "Computer Vision": 0.10, "Speech/Audio": 0.05, "Decision/MLOps": 0.05, "Data/MLOps": 0.0}
domains_arr = np.random.choice(domains, size=N, p=domain_weights)
growth_bonus = np.array([domain_growth_bonus[d] for d in domains_arr])
revenue_growth = np.random.lognormal(mean=-0.2, sigma=0.5, size=N) - 0.5  # centered near 20%
revenue_growth = np.clip(revenue_growth + growth_bonus, -0.2, 5.0)
rev2025 = rev2024 * (1 + revenue_growth)

data["revenue_2024"] = rev2024
data["revenue_2025"] = rev2025
data["revenue_growth_yoy"] = revenue_growth
data["primary_domain"] = domains_arr
data["product_type"] = np.random.choice(product_types, size=N, p=product_weights)
data["deployment"] = np.random.choice(deployment_types, size=N, p=deployment_weights)
data["is_open_source"] = np.random.choice([True, False], size=N, p=[0.12, 0.88])

data["employee_count"] = np.round(np.clip(
    np.random.lognormal(mean=np.log(np.maximum(rev2024 * 0.00005 + 5, 5)), sigma=0.7), 1, 5000)).astype(int)
data["engineers_pct"] = np.clip(np.random.normal(55, 15, N), 15, 90)
data["founders_count"] = np.random.choice([1, 2, 3, 4], size=N, p=[0.10, 0.55, 0.25, 0.10])
data["avg_experience_years"] = np.clip(np.random.exponential(4, N) + 3, 2, 30).round(1)

# Monthly active users: more for open-source, less for enterprise
os_boost = np.where(data["is_open_source"], 2.0, 0.0)
enterprise_penalty = np.where(data["product_type"] == "SaaS", 0.0, 0.5)
mau = np.exp(np.random.normal(np.log(np.maximum(rev2024 * 0.05 + 100, 100)) + os_boost - enterprise_penalty, 1.0))
data["monthly_active_users"] = np.clip(mau, 0, 1e7).astype(int)

data["customer_count"] = np.round(np.clip(
    rev2024 / np.random.uniform(1000, 50000, N), 0, 10000)).astype(int)

data["enterprise_pct"] = np.clip(np.random.normal(40, 20, N), 0, 100).round(1)
data["churn_rate_annual"] = np.clip(np.random.exponential(0.08, N) + 0.02, 0.01, 0.60).round(3)
data["net_revenue_retention"] = np.clip(np.random.lognormal(mean=0.1, sigma=0.25, size=N) + 0.8, 0.80, 2.50).round(3)

# Burn rate & runway
burn = np.clip(np.random.lognormal(
    mean=np.log(np.maximum(rev2024 * 0.02 * (1 + (1 / (1 + stage_idx))), 10000)), sigma=0.6), 5000, 5e6)
data["burn_rate_monthly"] = burn
data["runway_months"] = np.clip(
    np.where(funding > 0, funding / burn * np.random.uniform(0.7, 1.0, N), 0),
    0, 48).round(1)

data["gross_margin_pct"] = np.clip(np.random.normal(72, 10, N), 35, 95).round(1)
data["target_segment"] = np.random.choice(target_segments, size=N, p=seg_weights)

data["market_size_billion"] = np.where(
    data["primary_domain"] == "NLP/LLM", np.random.uniform(100, 250, N),
    np.where(data["primary_domain"] == "Computer Vision", np.random.uniform(30, 80, N),
    np.where(data["primary_domain"] == "AI Infrastructure", np.random.uniform(50, 150, N),
    np.random.uniform(5, 60, N)))).round(1)

data["competitor_count"] = np.clip(np.round(np.random.lognormal(3.5, 0.7, N)), 0, 500).astype(int)

df = pd.DataFrame(data)

# Inject ~2% missing values in key columns at random
for col in ["revenue_2024", "revenue_2025", "employee_count", "burn_rate_monthly", "gross_margin_pct"]:
    missing_idx = np.random.choice(df.index, size=int(N * 0.02), replace=False)
    df.loc[missing_idx, col] = np.nan

# Enforce runway = NaN where funding = 0 (bootstrapped)
df.loc[df["funding_stage"] == "Bootstrapped", "runway_months"] = np.nan

# Add a few outliers in revenue for realism
outlier_idx = np.random.choice(df.index, size=5, replace=False)
df.loc[outlier_idx, "revenue_2025"] = df.loc[outlier_idx, "revenue_2025"].values * np.random.uniform(3, 6, 5)

out_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(out_dir, "ai_startups_5000.csv")
df.to_csv(csv_path, index=False)
print(f"Dataset saved to {csv_path}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")

# Quick summary stats
print(f"\n--- Summary Stats ---")
print(df[["total_funding_usd", "revenue_2024", "revenue_2025", "employee_count",
           "burn_rate_monthly", "valuation_usd"]].describe())
