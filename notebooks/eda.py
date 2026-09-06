import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.loader import build_full_dataset
from src.data.preprocessor import DAYS_SENTINEL
from src.utils.config import config
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)

def save_fig(fig, name: str):
    path = os.path.join(FIGURES_DIR, f'{name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')

def plot_default_rate_by_bins(df, col, bins, labels, title, xlabel, fig_name):
    temp = df[[col, 'TARGET']].dropna(subset=[col]).copy()
    temp['bracket'] = pd.cut(temp[col], bins=bins, labels=labels, include_lowest=True)
    rates = temp.groupby('bracket', observed=False)['TARGET'].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    rates.plot(kind='bar', ax=ax, color=sns.color_palette('Blues_d', len(rates)), edgecolor='black')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Default Rate')
    ax.set_ylim(0, max(rates.max() * 1.3, 0.01))
    for i, v in enumerate(rates):
        ax.text(i, v + 0.003, f'{v:.1%}', ha='center', fontsize=9)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    save_fig(fig, fig_name)
    return rates

def plot_default_rate_by_quantile(df, col, q, title, xlabel, fig_name):
    temp = df[[col, 'TARGET']].dropna(subset=[col]).copy()
    temp['bracket'] = pd.qcut(temp[col], q=q, duplicates='drop')
    rates = temp.groupby('bracket', observed=False)['TARGET'].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    rates.plot(kind='bar', ax=ax, color=sns.color_palette('Oranges_d', len(rates)), edgecolor='black')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Default Rate')
    ax.set_ylim(0, max(rates.max() * 1.3, 0.01))
    for i, v in enumerate(rates):
        ax.text(i, v + 0.003, f'{v:.1%}', ha='center', fontsize=9)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    save_fig(fig, fig_name)
    return rates

def main():
    print('Loading full dataset...')
    df = build_full_dataset(config.DATA_DIR)
    print(f'Dataset shape: {df.shape}')
    for col in df.columns:
        if col.startswith('DAYS_'):
            df[col] = df[col].replace(DAYS_SENTINEL, pd.NA)
    print('\n=== Dataset Summary ===')
    print(df.describe(include='all').T.to_string())
    print('\n=== Top 20 Columns by Missing Rate ===')
    missing = df.isna().mean().sort_values(ascending=False)
    print(missing[missing > 0].head(20).to_string())
    print('\n=== Insight 1: Default Rate by Income Bracket ===')
    income_bins = [0, 100000, 150000, 200000, 300000, float('inf')]
    income_labels = ['<100K', '100-150K', '150-200K', '200-300K', '300K+']
    rates1 = plot_default_rate_by_bins(df, 'AMT_INCOME_TOTAL', income_bins, income_labels, 'Default Rate by Income Bracket', 'Income Bracket', 'insight1_income_default')
    print(rates1.to_string())
    print('\n=== Insight 2: Default Rate by Employment Length ===')
    temp_emp = df[['DAYS_EMPLOYED', 'TARGET']].dropna(subset=['DAYS_EMPLOYED']).copy()
    temp_emp['employment_years'] = temp_emp['DAYS_EMPLOYED'].astype(float) / -365.25
    emp_bins = [-1, 1, 3, 5, 10, 20, 100]
    emp_labels = ['<1y', '1-3y', '3-5y', '5-10y', '10-20y', '20y+']
    temp_emp['bracket'] = pd.cut(temp_emp['employment_years'], bins=emp_bins, labels=emp_labels, include_lowest=True)
    rates2 = temp_emp.groupby('bracket', observed=False)['TARGET'].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    rates2.plot(kind='bar', ax=ax, color=sns.color_palette('Greens_d', len(rates2)), edgecolor='black')
    ax.set_title('Default Rate by Employment Length', fontsize=14, fontweight='bold')
    ax.set_xlabel('Employment Length')
    ax.set_ylabel('Default Rate')
    ax.set_ylim(0, max(rates2.max() * 1.3, 0.01))
    for i, v in enumerate(rates2):
        ax.text(i, v + 0.003, f'{v:.1%}', ha='center', fontsize=9)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    save_fig(fig, 'insight2_employment_default')
    print(rates2.to_string())
    print('\n=== Insight 3: Default Rate by Education Type ===')
    rates3 = df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().sort_values(ascending=False)
    counts3 = df.groupby('NAME_EDUCATION_TYPE')['TARGET'].count()
    fig, ax = plt.subplots(figsize=(10, 5))
    rates3.plot(kind='bar', ax=ax, color=sns.color_palette('Reds_d', len(rates3)), edgecolor='black')
    ax.set_title('Default Rate by Education Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Education Type')
    ax.set_ylabel('Default Rate')
    ax.set_ylim(0, max(rates3.max() * 1.3, 0.01))
    for i, v in enumerate(rates3):
        ax.text(i, v + 0.003, f'{v:.1%}', ha='center', fontsize=9)
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    save_fig(fig, 'insight3_education_default')
    print(rates3.to_string())
    print('\n=== Insight 4a: Default Rate by Bureau Active Credit Ratio ===')
    rates4a = plot_default_rate_by_quantile(df, 'bureau_credit_active_ratio', q=4, title='Default Rate by Bureau Active Credit Ratio (Quartiles)', xlabel='Bureau Active Credit Ratio Quartile', fig_name='insight4a_bureau_active_ratio')
    print(rates4a.to_string())
    print('\n=== Insight 4b: Default Rate by Bureau Avg DPD Ratio ===')
    rates4b = plot_default_rate_by_quantile(df, 'bureau_avg_dpd_ratio', q=4, title='Default Rate by Bureau Avg DPD Ratio (Quartiles)', xlabel='Bureau Avg DPD Ratio Quartile', fig_name='insight4b_bureau_dpd_ratio')
    print(rates4b.to_string())
    print('\n=== Insight 5: Default Rate by Avg Payment Delay ===')
    rates5 = plot_default_rate_by_quantile(df, 'avg_payment_delay', q=5, title='Default Rate by Avg Payment Delay (Quintiles)', xlabel='Avg Payment Delay Quintile', fig_name='insight5_payment_delay')
    print(rates5.to_string())
    print('\n=== Insight 6: Default Rate by Credit Card Avg Utilization ===')
    rates6 = plot_default_rate_by_quantile(df, 'cc_avg_utilization', q=5, title='Default Rate by Credit Card Avg Utilization (Quintiles)', xlabel='CC Avg Utilization Quintile', fig_name='insight6_cc_utilization')
    print(rates6.to_string())
    print('\n=== Insight 7: Default Rate by POS Avg DPD ===')
    rates7 = plot_default_rate_by_quantile(df, 'pos_avg_dpd', q=5, title='Default Rate by POS Cash Avg DPD (Quintiles)', xlabel='POS Avg DPD Quintile', fig_name='insight7_pos_dpd')
    print(rates7.to_string())
    print('\n=== Feature Coverage Check ===')
    engineered_prefixes = ['bureau_', 'prev_', 'pos_', 'cc_', 'avg_payment', 'max_payment', 'avg_underpayment', 'bb_']
    engineered_cols = [c for c in df.columns if any((c.startswith(p) for p in engineered_prefixes))]
    coverage = df[engineered_cols].isna().mean().sort_values(ascending=False)
    print(coverage.to_string())
    highly_null = coverage[coverage > 0.5]
    if not highly_null.empty:
        print(f'\n[WARNING] Columns with >50% null (may need special handling):')
        print(highly_null.to_string())
    else:
        print('\n[OK] All engineered columns have <=50% null coverage.')
    print(f'\nDone. All figures saved to {FIGURES_DIR}/')
if __name__ == '__main__':
    main()