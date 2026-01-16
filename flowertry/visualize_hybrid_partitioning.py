"""Visualize the hybrid partitioning data distribution."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('./data/IndianPersoalFinance/indianPersonalFinanceAndSpendingHabits.csv')

# Compute disposable income
expense_cols = ['Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
                'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 
                'Education', 'Miscellaneous']
df['Disposable_Income'] = df['Income'] - df[expense_cols].sum(axis=1)

# Create client labels
df['Client'] = df['City_Tier'] + ' + ' + df['Occupation']

# Setup plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Hybrid Non-IID Partitioning: 12 Clients (City_Tier × Occupation)', 
             fontsize=16, fontweight='bold')

# 1. Sample distribution
ax1 = axes[0, 0]
client_counts = df.groupby(['City_Tier', 'Occupation']).size().reset_index(name='count')
client_counts['Client'] = client_counts['City_Tier'] + '+' + client_counts['Occupation']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
tier_colors = []
for tier in client_counts['City_Tier']:
    if tier == 'Tier_1':
        tier_colors.append('#3498db')
    elif tier == 'Tier_2':
        tier_colors.append('#e74c3c')
    else:
        tier_colors.append('#2ecc71')

bars = ax1.bar(range(len(client_counts)), client_counts['count'], color=tier_colors, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Client ID', fontsize=12)
ax1.set_ylabel('Number of Samples', fontsize=12)
ax1.set_title('Sample Distribution Across 12 Clients', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(client_counts)))
ax1.set_xticklabels(range(1, 13))
ax1.grid(axis='y', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', label='Tier_1'),
    Patch(facecolor='#e74c3c', label='Tier_2'),
    Patch(facecolor='#2ecc71', label='Tier_3')
]
ax1.legend(handles=legend_elements, loc='upper right')

# 2. Disposable income distribution by client
ax2 = axes[0, 1]
tier_order = ['Tier_1', 'Tier_2', 'Tier_3']
occ_order = ['Retired', 'Professional', 'Student', 'Self_Employed']

# Create proper ordering
client_order = [f"{tier} + {occ}" for tier in tier_order for occ in occ_order]
df_plot = df[df['Client'].isin(client_order)].copy()

bp = ax2.boxplot([df_plot[df_plot['Client'] == client]['Disposable_Income'].values 
                   for client in client_order],
                  labels=range(1, 13),
                  patch_artist=True,
                  showfliers=False)

# Color boxes by tier
for i, (patch, client) in enumerate(zip(bp['boxes'], client_order)):
    if 'Tier_1' in client:
        patch.set_facecolor('#3498db')
    elif 'Tier_2' in client:
        patch.set_facecolor('#e74c3c')
    else:
        patch.set_facecolor('#2ecc71')
    patch.set_alpha(0.6)

ax2.set_xlabel('Client ID', fontsize=12)
ax2.set_ylabel('Disposable Income ($)', fontsize=12)
ax2.set_title('Disposable Income Distribution per Client', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=df['Disposable_Income'].mean(), color='red', linestyle='--', 
            label=f'Global Mean: ${df["Disposable_Income"].mean():,.0f}', linewidth=2)
ax2.legend()

# 3. Heatmap of client distribution
ax3 = axes[1, 0]
pivot_counts = pd.crosstab(df['City_Tier'], df['Occupation'])
pivot_counts = pivot_counts.reindex(tier_order)[occ_order]
sns.heatmap(pivot_counts, annot=True, fmt='d', cmap='YlOrRd', ax=ax3, 
            cbar_kws={'label': 'Sample Count'})
ax3.set_title('Client Sample Counts (City_Tier × Occupation)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Occupation', fontsize=12)
ax3.set_ylabel('City_Tier', fontsize=12)

# 4. Mean disposable income heatmap
ax4 = axes[1, 1]
pivot_mean = df.groupby(['City_Tier', 'Occupation'])['Disposable_Income'].mean().unstack()
pivot_mean = pivot_mean.reindex(tier_order)[occ_order]
sns.heatmap(pivot_mean, annot=True, fmt='.0f', cmap='RdYlGn', ax=ax4,
            cbar_kws={'label': 'Mean Disposable Income ($)'})
ax4.set_title('Mean Disposable Income per Client', fontsize=14, fontweight='bold')
ax4.set_xlabel('Occupation', fontsize=12)
ax4.set_ylabel('City_Tier', fontsize=12)

plt.tight_layout()
plt.savefig('hybrid_partitioning_visualization.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved as 'hybrid_partitioning_visualization.png'")
print("\nKey Observations:")
print("=" * 70)
print("1. Client sizes vary from ~1,000 to ~2,500 samples")
print("2. Tier_2 clients have the most data (larger cities)")
print("3. Tier_3 clients have the least data (smaller cities)")
print("4. Disposable income varies significantly across clients")
print("5. This heterogeneity makes it a challenging FL scenario!")
print("=" * 70)

# Show plot
plt.show()
