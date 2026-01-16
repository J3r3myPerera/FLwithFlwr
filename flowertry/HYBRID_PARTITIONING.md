# Hybrid Non-IID Partitioning Implementation

## Overview

Successfully implemented **hybrid non-IID partitioning** for heterogeneous federated learning with **12 clients**, each representing a unique combination of City_Tier and Occupation.

---

## ✨ What Was Implemented

### 1. **Enhanced Dataset Module** ([dataset.py](dataset.py))

Added support for multiple partitioning strategies:

#### Partitioning Strategies:

- **`city_tier`**: 3 clients (Tier_1, Tier_2, Tier_3) - Original
- **`occupation`**: 4 clients (Retired, Professional, Student, Self_Employed) - New
- **`hybrid`**: 12 clients (City_Tier × Occupation) - **New & Active**

#### Hybrid Partitioning Details:

Each of the 12 clients gets data from one specific combination:

```
Client  1: Tier_1 + Retired          (~1,220 train samples)
Client  2: Tier_1 + Professional     (~1,207 train samples)
Client  3: Tier_1 + Student          (~1,189 train samples)
Client  4: Tier_1 + Self_Employed    (~1,194 train samples)
Client  5: Tier_2 + Retired          (~2,036 train samples)
Client  6: Tier_2 + Professional     (~2,025 train samples)
Client  7: Tier_2 + Student          (~2,059 train samples)
Client  8: Tier_2 + Self_Employed    (~2,038 train samples)
Client  9: Tier_3 + Retired          (~811 train samples)
Client 10: Tier_3 + Professional     (~829 train samples)
Client 11: Tier_3 + Student          (~807 train samples)
Client 12: Tier_3 + Self_Employed    (~794 train samples)
```

### 2. **Updated Main Module** ([main.py](main.py))

- Added `partition_strategy` parameter to `run_simulation()`
- Updated `compare_strategies()` to support partition strategies
- Modified config reading to include partition strategy

### 3. **Updated Configuration** ([conf/base.yaml](conf/base.yaml))

```yaml
num_clients: 12 # Changed from 3
partition_strategy: hybrid # New parameter
iid: false
```

---

## 📊 Data Heterogeneity Benefits

### Why Hybrid Partitioning is Ideal for Your Project:

1. **Maximum Heterogeneity**: Each client has distinct demographic characteristics
   - Different city tiers → different income levels and living costs
   - Different occupations → different spending patterns
2. **Realistic Federated Scenario**: Mimics real-world FL where:

   - Data is naturally distributed across different demographics
   - No client has complete data distribution
   - Strong non-IID characteristics challenge the algorithm

3. **Better Algorithm Testing**:

   - Tests FL strategies under severe data heterogeneity
   - FedProx and SCAFFOLD are designed for exactly this scenario
   - Hybrid strategy should show significant benefits

4. **Varying Client Sizes**:
   - Tier_2 clients: ~2,000 samples (larger cities)
   - Tier_1 clients: ~1,200 samples (medium cities)
   - Tier_3 clients: ~800 samples (smaller cities)
   - This size variation adds another dimension of heterogeneity

---

## 🎯 Expected Impact on Training

### Challenges with 12 Heterogeneous Clients:

- **Higher gradient variance** between client updates
- **Slower convergence** compared to 3 clients
- **More client drift** due to diverse data distributions
- **Potential for unstable training** without proper regularization

### Why Your Strategies Should Perform Well:

1. **FedProx** (`mu=0.05`): Proximal term keeps clients from drifting too far
2. **SCAFFOLD**: Control variates reduce variance from heterogeneous clients
3. **Hybrid**: Combines both mechanisms for maximum stability

---

## 🚀 How to Run

### With Current Configuration (Hybrid - 12 Clients):

```bash
python main.py
```

### Switch Back to 3 Clients (City_Tier):

```yaml
# In conf/base.yaml
num_clients: 3
partition_strategy: city_tier
```

### Try 4 Clients (Occupation):

```yaml
# In conf/base.yaml
num_clients: 4
partition_strategy: occupation
```

### Try IID with Any Number:

```yaml
# In conf/base.yaml
num_clients: 10 # or any number
iid: true
```

---

## 📈 Monitoring Heterogeneity

The implementation provides detailed statistics for each client:

- Training sample count
- Mean target value (disposable income)
- Standard deviation

This allows you to:

1. Verify data heterogeneity
2. Understand which clients might struggle
3. Analyze convergence patterns per client group

---

## ✅ Verification

Run the test script to verify partitioning:

```bash
python test_hybrid_partitioning.py
```

Expected output shows:

- 12 clients with unique City_Tier + Occupation combinations
- Balanced test set (1,994 samples)
- Proper train/val split per client
- Clear heterogeneity in mean target values

---

## 📝 Next Steps

1. **Run training** with the current hybrid configuration
2. **Compare results** between:
   - 3 clients (city_tier) vs 12 clients (hybrid)
   - FedAvg vs FedProx vs Hybrid strategies
3. **Analyze convergence**:
   - Does FedProx/Hybrid handle 12 heterogeneous clients better?
   - How much does MAPE/R² change with more heterogeneity?
4. **Tune hyperparameters** if needed:
   - Increase `mu` if clients drift too much
   - Adjust `local_epochs` for balance
   - Consider increasing `num_rounds` for convergence

---

## 🎓 Research Significance

This hybrid partitioning setup is **excellent for your FL project** because:

- Demonstrates understanding of real-world FL challenges
- Tests algorithms under realistic heterogeneous conditions
- Allows comparison of different partitioning strategies
- Shows proper implementation of non-IID data distribution
- Provides quantitative metrics to evaluate FL algorithm robustness

Good luck with your federated learning experiments! 🚀
