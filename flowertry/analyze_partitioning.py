import pandas as pd
import numpy as np

df = pd.read_csv('./data/IndianPersoalFinance/indianPersonalFinanceAndSpendingHabits.csv')

print('='*70)
print('DATA PARTITIONING ANALYSIS FOR MORE CLIENTS')
print('='*70)
print()

total_samples = len(df)
test_ratio = 0.1
train_val_samples = int(total_samples * (1 - test_ratio))

print(f'Total samples: {total_samples:,}')
print(f'After reserving {test_ratio*100:.0f}% for test: ~{train_val_samples:,} samples')
print()

# Current Non-IID by City_Tier (3 clients)
print('OPTION 1: Non-IID by City_Tier (Current - 3 clients)')
print('-' * 70)
city_tier_counts = df['City_Tier'].value_counts().sort_index()
for tier, count in city_tier_counts.items():
    train_count = int(count * (1 - test_ratio))
    print(f'  {tier}: ~{train_count:,} training samples')
print()

# Non-IID by Occupation (4 clients)
print('OPTION 2: Non-IID by Occupation (4 clients)')
print('-' * 70)
occupation_counts = df['Occupation'].value_counts().sort_values(ascending=False)
for occ, count in occupation_counts.items():
    train_count = int(count * (1 - test_ratio))
    print(f'  {occ}: ~{train_count:,} training samples')
print()

# Hybrid: City_Tier x Occupation (12 clients)
print('OPTION 3: Hybrid Non-IID (12 clients: City_Tier × Occupation)')
print('-' * 70)
cross_tab = pd.crosstab(df['City_Tier'], df['Occupation'])
print(cross_tab)
print()

# IID partitioning (any number)
print('OPTION 4: IID Random Split (Flexible)')
print('-' * 70)
for num_clients in [5, 10, 15, 20]:
    samples_per_client = train_val_samples // num_clients
    print(f'  {num_clients} clients: ~{samples_per_client:,} samples/client')
print()

print('='*70)
print('RECOMMENDATIONS:')
print('='*70)
print('• For 3 clients: Use City_Tier (current Non-IID setup)')
print('• For 4 clients: Could partition by Occupation')  
print('• For 5-20 clients: Use IID (set iid: true in config)')
print('• For 12 clients: Could use City_Tier × Occupation hybrid')
print()
print('NOTE: IID partitioning is most flexible and works with ANY number')
print('      of clients. Just set num_clients to your desired value!')
