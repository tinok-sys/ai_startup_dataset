"""
Complete EDA + Visualization for AI Startup Dataset
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os, warnings, sys, json
warnings.filterwarnings('ignore')

# ── Config ──
BASE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE, "ai_startups_5000.csv")
OUT_DIR = os.path.join(BASE, "visualizations")
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({'figure.max_open_warning': 0, 'font.size': 10})

# ── Load ──
df = pd.read_csv(CSV_PATH)
N = len(df)
print(f"Loaded {N} records, {len(df.columns)} columns\n")

# ── 1. DATA QUALITY ──
print("=" * 60)
print("1. DATA QUALITY ANALYSIS")
print("=" * 60)

null_report = df.isnull().sum()
null_report = null_report[null_report > 0].sort_values(ascending=False)
print(f"\nMissing values by column:\n{null_report}")
print(f"\nTotal missing: {df.isnull().sum().sum()} / {N * len(df.columns)} ({100*df.isnull().sum().sum()/(N*len(df.columns)):.2f}%)")

dupes = df.duplicated(subset=['company_name']).sum()
print(f"Duplicate company names: {dupes}")

# Outlier detection via IQR for numeric cols
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
outlier_report = {}
for col in numeric_cols:
    col_data = df[col].dropna()
    if len(col_data) == 0:
        continue
    Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = ((col_data < lower) | (col_data > upper)).sum()
    outlier_report[col] = {'outliers': int(outliers), 'pct': round(100*outliers/len(col_data), 1), 'lower': round(lower, 2), 'upper': round(upper, 2)}
outlier_df = pd.DataFrame(outlier_report).T.sort_values('pct', ascending=False)
print(f"\nOutliers detected (IQR method, top 10):\n{outlier_df.head(10)}")

# ── 2. DESCRIPTIVE STATISTICS ──
print("\n" + "=" * 60)
print("2. DESCRIPTIVE STATISTICS")
print("=" * 60)

key_metrics = ['total_funding_usd', 'revenue_2024', 'revenue_2025', 'revenue_growth_yoy',
               'valuation_usd', 'employee_count', 'burn_rate_monthly', 'gross_margin_pct',
               'churn_rate_annual', 'net_revenue_retention', 'monthly_active_users', 'customer_count']
desc = df[key_metrics].describe(percentiles=[.05, .25, .5, .75, .95]).round(2)
print(f"\nKey metrics summary:\n{desc}")

# Variance (CV = std/mean)
cv = {}
for col in key_metrics:
    m = df[col].mean()
    s = df[col].std()
    cv[col] = s / m if m != 0 else np.nan
cv_series = pd.Series(cv).sort_values(ascending=False)
print(f"\nHighest variance (CV) variables:\n{cv_series.head(8)}")

# ── 3. SEGMENT ANALYSES ──
print("\n" + "=" * 60)
print("3. SEGMENT ANALYSIS")
print("=" * 60)

# By funding stage
print("\n--- By Funding Stage ---")
stage_agg = df.groupby('funding_stage').agg(
    count=('company_id', 'count'),
    avg_revenue=('revenue_2024', 'mean'),
    avg_funding=('total_funding_usd', 'mean'),
    avg_valuation=('valuation_usd', 'mean'),
    avg_employees=('employee_count', 'mean'),
    avg_growth=('revenue_growth_yoy', 'mean'),
    avg_nrr=('net_revenue_retention', 'mean')
).round(2)
print(stage_agg.to_string())

# By domain
print("\n--- By Primary Domain ---")
domain_agg = df.groupby('primary_domain').agg(
    count=('company_id', 'count'),
    avg_revenue=('revenue_2024', 'mean'),
    avg_growth=('revenue_growth_yoy', 'mean'),
    avg_funding=('total_funding_usd', 'mean'),
    avg_employees=('employee_count', 'mean'),
    avg_mau=('monthly_active_users', 'mean'),
    avg_margin=('gross_margin_pct', 'mean'),
    avg_churn=('churn_rate_annual', 'mean')
).round(2).sort_values('avg_revenue', ascending=False)
print(domain_agg.to_string())

# By country
print("\n--- By Country (top 10) ---")
country_agg = df.groupby('country').agg(
    count=('company_id', 'count'),
    avg_revenue=('revenue_2024', 'mean'),
    avg_funding=('total_funding_usd', 'mean')
).round(2).sort_values('count', ascending=False).head(10)
print(country_agg.to_string())

# ── 4. CORRELATION ANALYSIS ──
print("\n" + "=" * 60)
print("4. CORRELATION ANALYSIS")
print("=" * 60)
corr_cols = ['total_funding_usd', 'revenue_2024', 'revenue_2025', 'revenue_growth_yoy',
             'valuation_usd', 'employee_count', 'burn_rate_monthly', 'gross_margin_pct',
             'churn_rate_annual', 'net_revenue_retention', 'monthly_active_users',
             'engineers_pct', 'enterprise_pct', 'funding_rounds']
corr_matrix = df[corr_cols].corr()
# Find strongest correlations
corr_pairs = []
for i in range(len(corr_cols)):
    for j in range(i+1, len(corr_cols)):
        corr_pairs.append((corr_cols[i], corr_cols[j], corr_matrix.iloc[i, j]))
corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print("\nStrongest correlations (top 12):")
for a, b, r in corr_pairs[:12]:
    print(f"  {a:30s} vs {b:30s}: {r:+.4f}")

# ── 5. KEY INSIGHTS ──
print("\n" + "=" * 60)
print("5. KEY INSIGHTS")
print("=" * 60)

# Most funded domain
print(f"\nMost funded domain: {domain_agg['avg_funding'].idxmax()} (${domain_agg['avg_funding'].max():,.0f})")
# Highest growth domain
print(f"Highest growth domain: {domain_agg['avg_growth'].idxmax()} ({domain_agg['avg_growth'].max()*100:.1f}%)")
# Best margin
print(f"Best margin domain: {domain_agg['avg_margin'].idxmax()} ({domain_agg['avg_margin'].max():.1f}%)")
# Lowest churn
print(f"Lowest churn domain: {domain_agg['avg_churn'].idxmin()} ({domain_agg['avg_churn'].min():.1%})")
# Open source vs closed
os_agg = df.groupby('is_open_source').agg(avg_rev=('revenue_2024', 'mean'), avg_mau=('monthly_active_users', 'mean'), count=('company_id', 'count'))
print(f"\nOpen source companies: {os_agg.loc[True, 'count']} ({100*os_agg.loc[True, 'count']/N:.1f}%)")
print(f"  Avg revenue: ${os_agg.loc[True, 'avg_rev']:,.0f} vs closed: ${os_agg.loc[False, 'avg_rev']:,.0f}")
print(f"  Avg MAU: {os_agg.loc[True, 'avg_mau']:,.0f} vs closed: {os_agg.loc[False, 'avg_mau']:,.0f}")

# Enterprise vs SMB
seg_agg = df.groupby('target_segment').agg(avg_rev=('revenue_2024', 'mean'), avg_growth=('revenue_growth_yoy', 'mean'), avg_churn=('churn_rate_annual', 'mean'), count=('company_id', 'count'))
print(f"\nBy Target Segment:")
print(seg_agg.round(2).to_string())


# ═══════════════════════════════════════════════════════════
# 6. VISUALIZATIONS
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. GENERATING VISUALIZATIONS")
print("=" * 60)

viz_results = {}
p = lambda name: os.path.join(OUT_DIR, name)

# ─── 6a. Distribution Histograms ───
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle("Distribution Histograms — Key Variables", fontsize=16, fontweight='bold')

hist_cols = ['total_funding_usd', 'revenue_2024', 'revenue_2025', 'valuation_usd',
             'employee_count', 'revenue_growth_yoy', 'gross_margin_pct',
             'churn_rate_annual', 'net_revenue_retention']
for ax, col in zip(axes.flatten(), hist_cols):
    d = df[col].dropna()
    ax.hist(d, bins=50, color='#2563eb', alpha=0.8, edgecolor='white', linewidth=0.3)
    ax.set_title(col.replace('_', ' ').title(), fontsize=10)
    ax.set_ylabel('Frequency')
    if col in ['total_funding_usd', 'revenue_2024', 'revenue_2025', 'valuation_usd']:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(p('01_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
viz_results['distributions'] = '01_distributions.png'
print("  ✓ Distributions")

# ─── 6b. Correlation Heatmap ───
fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap — AI Startup Metrics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(p('02_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
viz_results['correlation'] = '02_correlation_heatmap.png'
print("  ✓ Correlation heatmap")

# ─── 6c. Bar Charts — Category Comparisons ───
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# avg_revenue by domain
domain_order = domain_agg.sort_values('avg_revenue', ascending=False).index
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(domain_order)))
axes[0,0].barh(domain_order, domain_agg.loc[domain_order, 'avg_revenue'], color=colors)
axes[0,0].set_title('Avg Revenue 2024 by Domain', fontweight='bold')
axes[0,0].set_xlabel('Revenue (USD)')
axes[0,0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# avg_funding by stage
stage_order = ['Bootstrapped', 'Seed', 'Series A', 'Series B', 'Series C', 'Series D+']
stage_vals = stage_agg.loc[stage_order, 'avg_funding']
axes[0,1].bar(stage_order, stage_vals, color='#10b981', alpha=0.85)
axes[0,1].set_title('Avg Total Funding by Stage', fontweight='bold')
axes[0,1].set_ylabel('Funding (USD)')
axes[0,1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
axes[0,1].tick_params(axis='x', rotation=30)

# avg_revenue by country (top 10)
top_countries = country_agg.head(10)
colors2 = plt.cm.Oranges(np.linspace(0.3, 0.9, len(top_countries)))
axes[1,0].barh(top_countries.index[::-1], top_countries['avg_revenue'][::-1], color=colors2[::-1])
axes[1,0].set_title('Avg Revenue 2024 by Country (Top 10)', fontweight='bold')
axes[1,0].set_xlabel('Revenue (USD)')
axes[1,0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

# avg_growth by domain
growth_order = domain_agg.sort_values('avg_growth', ascending=False).index
colors3 = plt.cm.Greens(np.linspace(0.3, 0.9, len(growth_order)))
axes[1,1].bar(growth_order, domain_agg.loc[growth_order, 'avg_growth'], color=colors3, alpha=0.85)
axes[1,1].set_title('Avg Revenue Growth YoY by Domain', fontweight='bold')
axes[1,1].set_ylabel('Growth Rate')
axes[1,1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0%}'))
axes[1,1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(p('03_bar_charts.png'), dpi=150, bbox_inches='tight')
plt.close()
viz_results['bar_charts'] = '03_bar_charts.png'
print("  ✓ Bar charts")

# ─── 6d. Line Chart — Trends Over Time ───
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

yearly = df.groupby('year_founded').agg(
    avg_funding=('total_funding_usd', 'mean'),
    avg_revenue=('revenue_2024', 'mean'),
    count=('company_id', 'count'),
    avg_val=('valuation_usd', 'mean')
).reset_index()

axes[0].plot(yearly['year_founded'], yearly['avg_funding'], marker='o', linewidth=2, color='#2563eb', label='Avg Funding')
axes[0].plot(yearly['year_founded'], yearly['avg_revenue'], marker='s', linewidth=2, color='#10b981', label='Avg Revenue')
axes[0].set_title('Avg Funding & Revenue by Founding Year', fontweight='bold')
axes[0].set_xlabel('Year Founded')
axes[0].set_ylabel('USD')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(yearly['year_founded'], yearly['count'], color='#8b5cf6', alpha=0.8, label='Companies Founded')
axes[1].set_title('Number of AI Startups Founded by Year', fontweight='bold')
axes[1].set_xlabel('Year Founded')
axes[1].set_ylabel('Count')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(p('04_trends_over_time.png'), dpi=150, bbox_inches='tight')
plt.close()
viz_results['trends'] = '04_trends_over_time.png'
print("  ✓ Line charts")

# ─── 6e. Scatter Plots ───
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Scatter Plots — Key Relationships', fontsize=16, fontweight='bold')

domains_unique = df['primary_domain'].unique()
domain_colors_map = {d: c for d, c in zip(domains_unique,
    plt.cm.tab10(np.linspace(0, 1, len(domains_unique))))}
stages_unique = df['funding_stage'].unique()
stage_colors_map = {s: c for s, c in zip(stages_unique,
    plt.cm.Set2(np.linspace(0, 1, len(stages_unique))))}

def scatter(ax, x, y, color_by=None, logx=False, logy=False, title=''):
    valid = df[[x, y]].dropna()
    if color_by:
        cmap = domain_colors_map if color_by == 'primary_domain' else stage_colors_map
        for group, grp in df.loc[valid.index].groupby(color_by):
            ax.scatter(grp[x], grp[y], alpha=0.4, s=12, label=group,
                       c=[cmap.get(group, '#333')], edgecolors='none')
        ax.legend(fontsize=7, loc='best', markerscale=2, ncol=2)
    else:
        ax.scatter(valid[x], valid[y], alpha=0.3, s=10, c='#2563eb', edgecolors='none')
    if logx: ax.set_xscale('log')
    if logy: ax.set_yscale('log')
    ax.set_xlabel(x.replace('_',' ').title(), fontsize=9)
    ax.set_ylabel(y.replace('_',' ').title(), fontsize=9)
    ax.set_title(title or f'{x} vs {y}', fontsize=10, fontweight='bold')
    if logx: ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    if logy: ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'${y:,.0f}'))
    ax.grid(True, alpha=0.2)

scatter(axes[0,0], 'total_funding_usd', 'revenue_2024', color_by='primary_domain', logx=True, logy=True,
        title='Funding vs Revenue (by Domain)')
scatter(axes[0,1], 'employee_count', 'revenue_2024', logx=True, logy=True,
        title='Employees vs Revenue')
scatter(axes[0,2], 'total_funding_usd', 'valuation_usd', color_by='funding_stage', logx=True, logy=True,
        title='Funding vs Valuation (by Stage)')
scatter(axes[1,0], 'net_revenue_retention', 'revenue_growth_yoy', title='NRR vs Revenue Growth')
scatter(axes[1,1], 'churn_rate_annual', 'net_revenue_retention', title='Churn vs NRR')
scatter(axes[1,2], 'burn_rate_monthly', 'runway_months', logx=True, title='Burn Rate vs Runway')

plt.tight_layout()
plt.savefig(p('05_scatter_plots.png'), dpi=150, bbox_inches='tight')
plt.close()
viz_results['scatter'] = '05_scatter_plots.png'
print("  ✓ Scatter plots")

# ─── 6f. Box Plots ───
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Box Plots — Outlier Detection by Category', fontsize=16, fontweight='bold')

sns.boxplot(data=df, x='funding_stage', y='revenue_2024', ax=axes[0,0],
            order=stage_order, palette='Blues', hue='funding_stage', legend=False)
axes[0,0].set_title('Revenue by Funding Stage')
axes[0,0].set_yscale('log')
axes[0,0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
axes[0,0].tick_params(axis='x', rotation=30)

sns.boxplot(data=df, x='primary_domain', y='revenue_growth_yoy', ax=axes[0,1],
            palette='Greens', hue='primary_domain', legend=False)
axes[0,1].set_title('Revenue Growth by Domain')
axes[0,1].tick_params(axis='x', rotation=30)
axes[0,1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0%}'))

sns.boxplot(data=df, x='primary_domain', y='gross_margin_pct', ax=axes[0,2],
            palette='Oranges', hue='primary_domain', legend=False)
axes[0,2].set_title('Gross Margin by Domain')
axes[0,2].tick_params(axis='x', rotation=30)

sns.boxplot(data=df, x='primary_domain', y='churn_rate_annual', ax=axes[1,0],
            palette='Reds', hue='primary_domain', legend=False)
axes[1,0].set_title('Churn Rate by Domain')
axes[1,0].tick_params(axis='x', rotation=30)
axes[1,0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0%}'))

sns.boxplot(data=df, x='target_segment', y='revenue_2024', ax=axes[1,1],
            palette='Purples', hue='target_segment', legend=False)
axes[1,1].set_title('Revenue by Target Segment')
axes[1,1].set_yscale('log')
axes[1,1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

sns.boxplot(data=df, x='funding_stage', y='employee_count', ax=axes[1,2],
            order=stage_order, palette='Blues', hue='funding_stage', legend=False)
axes[1,2].set_title('Employees by Funding Stage')
axes[1,2].set_yscale('log')
axes[1,2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(p('06_box_plots.png'), dpi=150, bbox_inches='tight')
plt.close()
viz_results['boxplots'] = '06_box_plots.png'
print("  ✓ Box plots")

# ─── 6g. Pie Charts — Composition Analysis ───
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

def pie(ax, series, title, colors_cm='Blues'):
    counts = series.value_counts()
    cmap = plt.colormaps[colors_cm]
    c = cmap(np.linspace(0.3, 0.9, len(counts)))
    wedges, texts, autotexts = ax.pie(counts.values, labels=None, autopct='%1.1f%%',
                                       startangle=90, colors=c, pctdistance=0.85)
    ax.legend(wedges, [f'{k} ({v})' for k, v in zip(counts.index, counts.values)],
              title=title, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)
    ax.set_title(title, fontweight='bold', fontsize=11)

pie(axes[0,0], df['funding_stage'], 'Funding Stage Distribution')
pie(axes[0,1], df['primary_domain'], 'Domain Distribution')
pie(axes[0,2], df['product_type'], 'Product Type')
pie(axes[1,0], df['deployment'], 'Deployment Model')
pie(axes[1,1], df['target_segment'], 'Target Segment')
pie(axes[1,2], df['country'].value_counts().head(8), 'Top 8 Countries')

plt.tight_layout()
plt.savefig(p('07_composition_pie.png'), dpi=150, bbox_inches='tight')
plt.close()
viz_results['pie'] = '07_composition_pie.png'
print("  ✓ Pie charts")

# ─── 6h. Treemap (market size by domain) ───
try:
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(14, 6))
    market_by_domain = df.groupby('primary_domain')['market_size_billion'].mean().sort_values(ascending=False)
    colors_t = plt.cm.viridis(np.linspace(0.1, 0.9, len(market_by_domain)))
    ax.bar(market_by_domain.index, market_by_domain.values, color=colors_t, edgecolor='white', width=0.7)
    ax.set_title('Avg Market Size by AI Domain (USD Billions)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Market Size ($B)')
    for i, (k, v) in enumerate(market_by_domain.items()):
        ax.text(i, v + 1, f'${v:.0f}B', ha='center', fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig(p('08_market_size_treemap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    viz_results['treemap'] = '08_market_size_treemap.png'
    print("  ✓ Market size bar chart")
except:
    print("  ✗ Market size chart skipped")

# ─── 6i. Pairplot (subset) ───
pair_cols = ['total_funding_usd', 'revenue_2024', 'employee_count', 'gross_margin_pct', 'net_revenue_retention']
pair_df = df[pair_cols].dropna().sample(min(500, len(df)), random_state=42)
pair_grid = sns.pairplot(pair_df, diag_kind='kde', plot_kws={'alpha': 0.5, 's': 15, 'color': '#2563eb'})
pair_grid.fig.suptitle('Pairplot — Key Metrics (500-sample)', fontsize=14, fontweight='bold', y=1.02)
pair_grid.savefig(p('09_pairplot.png'), dpi=150, bbox_inches='tight')
plt.close()
viz_results['pairplot'] = '09_pairplot.png'
print("  ✓ Pairplot")

# ─── 6j. Growth vs MAU scatter (insight) ───
fig, ax = plt.subplots(figsize=(12, 7))
for domain in df['primary_domain'].unique():
    subset = df[df['primary_domain'] == domain].dropna(subset=['revenue_growth_yoy', 'monthly_active_users'])
    ax.scatter(subset['revenue_growth_yoy'], subset['monthly_active_users'],
               alpha=0.4, s=10, label=domain, c=[domain_colors_map[domain]])
ax.set_xlabel('Revenue Growth YoY')
ax.set_ylabel('Monthly Active Users')
ax.set_title('Growth Rate vs User Adoption by Domain', fontweight='bold', fontsize=14)
ax.legend(fontsize=8, markerscale=3, ncol=2)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0%}'))
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(p('10_growth_vs_mau.png'), dpi=150, bbox_inches='tight')
plt.close()
viz_results['growth_mau'] = '10_growth_vs_mau.png'
print("  ✓ Growth vs MAU scatter")

# ─── 6k. Open source vs Closed comparison ───
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Open Source vs Closed Source — Company Comparison', fontweight='bold', fontsize=14)

os_df = df.copy()
os_df['is_open_source'] = os_df['is_open_source'].map({True: 'Open Source', False: 'Closed Source'})

for ax, col, title in zip(axes, ['revenue_2024', 'monthly_active_users', 'employee_count'],
                           ['Revenue 2024', 'Monthly Active Users', 'Employee Count']):
    d = os_df[[col, 'is_open_source']].dropna()
    bp = ax.boxplot([d.loc[d['is_open_source'] == 'Open Source', col].values,
                     d.loc[d['is_open_source'] == 'Closed Source', col].values],
                    tick_labels=['Open Source', 'Closed Source'], patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 2})
    bp['boxes'][0].set_facecolor('#10b981')
    bp['boxes'][1].set_facecolor('#2563eb')
    ax.set_title(title, fontweight='bold')
    if col in ['revenue_2024']:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    elif col == 'employee_count':
        ax.set_yscale('log')
    elif col == 'monthly_active_users':
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

plt.tight_layout()
plt.savefig(p('11_open_source_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
viz_results['open_source'] = '11_open_source_comparison.png'
print("  ✓ Open source comparison")

# ─── 6l. Churn vs NRR scatter by domain ───
fig, ax = plt.subplots(figsize=(12, 7))
for domain in df['primary_domain'].unique():
    subset = df[df['primary_domain'] == domain].dropna(subset=['churn_rate_annual', 'net_revenue_retention'])
    ax.scatter(subset['churn_rate_annual'], subset['net_revenue_retention'],
               alpha=0.4, s=10, label=domain, c=[domain_colors_map[domain]])
ax.set_xlabel('Annual Churn Rate')
ax.set_ylabel('Net Revenue Retention')
ax.set_title('Churn Rate vs Net Revenue Retention by Domain', fontweight='bold', fontsize=14)
ax.legend(fontsize=8, markerscale=3, ncol=2)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0%}'))
ax.grid(True, alpha=0.2)
# Add reference lines
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='NRR = 100%')
plt.tight_layout()
plt.savefig(p('12_churn_vs_nrr.png'), dpi=150, bbox_inches='tight')
plt.close()
viz_results['churn_nrr'] = '12_churn_vs_nrr.png'
print("  ✓ Churn vs NRR scatter")

print(f"\n✓ All {len(viz_results)} visualizations saved to {OUT_DIR}")

# ── Generate JSON summary for report ──
summary = {
    "total_records": N,
    "total_columns": len(df.columns),
    "missing_pct": round(100*df.isnull().sum().sum()/(N*len(df.columns)), 2),
    "duplicate_companies": int(dupes),
    "outliers_top": {k: v for k, v in outlier_df.head(5).to_dict('index').items()},
    "descriptive": {col: {"mean": round(df[col].mean(), 2), "median": round(df[col].median(), 2),
                          "std": round(df[col].std(), 2), "p5": round(df[col].quantile(0.05), 2),
                          "p95": round(df[col].quantile(0.95), 2)}
                    for col in key_metrics},
    "top_correlations": [(a, b, round(r, 4)) for a, b, r in corr_pairs[:10]],
    "domain_insights": json.loads(domain_agg.to_json()),
    "stage_insights": json.loads(stage_agg.to_json()),
    "country_top": json.loads(country_agg.head(5).to_json()),
    "visualizations": viz_results
}

with open(os.path.join(BASE, "eda_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("\n✓ EDA summary saved to eda_summary.json")
print("\n=== EDA COMPLETE ===")
