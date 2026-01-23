# Stratified Client Selection Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FEDERATED LEARNING WITH STRATIFIED SELECTION              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Indian Personal Finance Dataset                                            │
│  ├── 19,483 samples                                                         │
│  ├── 19 features (12 numerical + 7 categorical)                             │
│  └── Target: Disposable_Income                                              │
│                                                                              │
│  Partitioning by City_Tier (Non-IID):                                       │
│  ┌──────────────┬──────────────┬──────────────┐                            │
│  │  Tier 1      │  Tier 2      │  Tier 3      │                            │
│  │  (4 clients) │  (4 clients) │  (4 clients) │                            │
│  │  High Income │  Med Income  │  Low Income  │                            │
│  │  Urban       │  Mixed       │  Rural       │                            │
│  └──────────────┴──────────────┴──────────────┘                            │
│                                                                              │
│  Heterogeneity Control: alpha = 0.3                                         │
│  ├── 70% primary tier data                                                  │
│  └── 30% mixed from other tiers                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        STRATIFIED SELECTION LAYER                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  StratifiedClientSelector                                                   │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │                                                              │           │
│  │  Input: client_strata = {                                   │           │
│  │    'Tier_1': [0, 1, 2, 3],                                  │           │
│  │    'Tier_2': [4, 5, 6, 7],                                  │           │
│  │    'Tier_3': [8, 9, 10, 11]                                 │           │
│  │  }                                                           │           │
│  │                                                              │           │
│  │  Configuration:                                              │           │
│  │  ├── clients_per_round: 6                                   │           │
│  │  └── min_per_stratum: 1 (fairness guarantee)               │           │
│  │                                                              │           │
│  │  Base Allocation (Proportional):                            │           │
│  │  ├── Tier_1: 2 clients (33.3%)                             │           │
│  │  ├── Tier_2: 2 clients (33.3%)                             │           │
│  │  └── Tier_3: 2 clients (33.3%)                             │           │
│  │                                                              │           │
│  │  Selection Process:                                          │           │
│  │  1. Allocate minimum from each stratum                      │           │
│  │  2. Distribute remaining slots proportionally               │           │
│  │  3. Random sampling within each stratum                     │           │
│  │  4. Track selection history                                 │           │
│  │                                                              │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                              │
│  Fairness Metrics:                                                          │
│  ├── Participation Equity (Gini): 0.0523                                    │
│  ├── Representation Ratios: ~1.0 for all strata                            │
│  └── Toxic Round Frequency: 5.0%                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         FEDERATED LEARNING LAYER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  StratifiedFedProx Strategy                                                 │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │                                                              │           │
│  │  Server Round Loop:                                          │           │
│  │                                                              │           │
│  │  1. configure_fit(round_num)                                │           │
│  │     ├── Call stratified_selector.select_clients(round_num)  │           │
│  │     ├── Map client IDs to ClientProxy objects               │           │
│  │     ├── Create FitIns with config (lr, mu, layer_mus)       │           │
│  │     └── Return [(ClientProxy, FitIns), ...]                 │           │
│  │                                                              │           │
│  │  2. Client Training (parallel)                               │           │
│  │     ├── Client 0 (Tier_1): Local training with FedProx      │           │
│  │     ├── Client 4 (Tier_2): Local training with FedProx      │           │
│  │     ├── Client 8 (Tier_3): Local training with FedProx      │           │
│  │     └── ... (6 clients total)                                │           │
│  │                                                              │           │
│  │  3. aggregate_fit(results)                                   │           │
│  │     ├── Call parent FedProx aggregation                      │           │
│  │     ├── Extract divergence metrics (if adaptive)             │           │
│  │     ├── Update adaptive controller (if enabled)              │           │
│  │     └── Return aggregated parameters                         │           │
│  │                                                              │           │
│  │  4. configure_evaluate(round_num)                            │           │
│  │     ├── Use stratified selection for evaluation too          │           │
│  │     └── Ensure balanced evaluation across strata            │           │
│  │                                                              │           │
│  │  5. Server-side evaluation                                   │           │
│  │     └── Evaluate on global test set                          │           │
│  │                                                              │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            MODEL LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DisposableIncomeModel (Neural Network)                                     │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │                                                              │           │
│  │  Input Layer (19 features)                                   │           │
│  │    ↓                                                          │           │
│  │  Linear(19 → 128) + BatchNorm + ReLU + Dropout(0.2)         │           │
│  │    ↓                                                          │           │
│  │  Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.2)         │           │
│  │    ↓                                                          │           │
│  │  Linear(64 → 32) + BatchNorm + ReLU + Dropout(0.1)          │           │
│  │    ↓                                                          │           │
│  │  Linear(32 → 1) [Output: Disposable Income]                 │           │
│  │                                                              │           │
│  │  FedProx Training:                                           │           │
│  │  Loss = MSE(y_pred, y_true) + (μ/2)||θ - θ_global||²       │           │
│  │                                                              │           │
│  │  Layer-specific μ values:                                    │           │
│  │  ├── Input:   μ = 0.003 (low - local adaptation)           │           │
│  │  ├── Hidden1: μ = 0.008 (moderate)                          │           │
│  │  ├── Hidden2: μ = 0.012 (higher)                            │           │
│  │  └── Output:  μ = 0.025 (highest - global consistency)     │           │
│  │                                                              │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        VISUALIZATION LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Generated Plots:                                                           │
│                                                                              │
│  1. stratified_selection_analysis.png                                       │
│     ├── Stacked area chart: Stratum participation over rounds              │
│     ├── Line plot: Actual vs expected representation                        │
│     ├── Heatmap: Deviation from expected (%)                                │
│     ├── Box plot: Client participation frequency                            │
│     └── Text: Fairness metrics summary                                      │
│                                                                              │
│  2. random_vs_stratified_comparison.png                                     │
│     ├── R² score: Random vs Stratified                                      │
│     ├── RMSE: Random vs Stratified                                          │
│     ├── MAE: Random vs Stratified                                           │
│     └── Summary: Performance improvements                                   │
│                                                                              │
│  3. comparison_metrics.png                                                  │
│     ├── R² evolution over rounds                                            │
│     ├── RMSE evolution over rounds                                          │
│     ├── MAE evolution over rounds                                           │
│     └── Loss evolution over rounds                                          │
│                                                                              │
│  4. summary_comparison.png                                                  │
│     └── Bar chart: Final metrics comparison                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING ROUND FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Server                          Stratified Selector              Clients
  │                                     │                           │
  │  1. Start Round N                   │                           │
  ├────────────────────────────────────>│                           │
  │                                     │                           │
  │  2. Select K=6 clients              │                           │
  │     (stratified sampling)           │                           │
  │                                     │                           │
  │  3. Return selected IDs             │                           │
  │<────────────────────────────────────┤                           │
  │     [0, 2, 4, 6, 8, 10]             │                           │
  │     Tier_1: 2, Tier_2: 2, Tier_3: 2 │                           │
  │                                     │                           │
  │  4. Send global model + config      │                           │
  ├─────────────────────────────────────────────────────────────────>│
  │     (parameters, lr, mu, layer_mus) │                           │
  │                                     │                           │
  │                                     │  5. Local training        │
  │                                     │     (5 epochs)            │
  │                                     │     FedProx loss:         │
  │                                     │     MSE + proximal term   │
  │                                     │                           │
  │  6. Return updated models           │                           │
  │<─────────────────────────────────────────────────────────────────┤
  │     + divergence metrics            │                           │
  │                                     │                           │
  │  7. Aggregate updates               │                           │
  │     (FedAvg weighted average)       │                           │
  │                                     │                           │
  │  8. Update global model             │                           │
  │                                     │                           │
  │  9. Server-side evaluation          │                           │
  │     (test set: R², RMSE, MAE)       │                           │
  │                                     │                           │
  │  10. Log metrics & fairness stats   │                           │
  │                                     │                           │
  │  11. Next round...                  │                           │
  │                                     │                           │
  ▼                                     ▼                           ▼
```

## Component Interaction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      COMPONENT INTERACTION DIAGRAM                           │
└─────────────────────────────────────────────────────────────────────────────┘

main.py
  │
  ├─> dataset.prepare_dataset()
  │     └─> Returns: trainloaders, valloaders, testloader, 
  │                  target_scaler, client_strata
  │
  ├─> stratified_selector.create_stratified_client_fn()
  │     └─> Creates: StratifiedClientSelector
  │           ├─> client_strata: {'Tier_1': [...], 'Tier_2': [...], 'Tier_3': [...]}
  │           ├─> clients_per_round: 6
  │           └─> min_per_stratum: 1
  │
  ├─> client.generate_client_fn()
  │     └─> Creates: FlowerClient factory
  │           ├─> trainloader
  │           ├─> valloader
  │           └─> target_scaler
  │
  ├─> server.get_on_fit_config()
  │     └─> Returns: fit_config_fn
  │           ├─> lr: 0.001
  │           ├─> mu: 0.01
  │           └─> layer_mus: {...}
  │
  ├─> server.get_evaluate_fn()
  │     └─> Returns: evaluate_fn
  │           └─> Evaluates on global test set
  │
  ├─> stratified_strategy.StratifiedFedProx()
  │     └─> Creates: FL strategy
  │           ├─> stratified_selector
  │           ├─> on_fit_config_fn
  │           ├─> evaluate_fn
  │           └─> proximal_mu
  │
  ├─> fl.simulation.start_simulation()
  │     └─> Runs: Federated learning
  │           ├─> client_fn
  │           ├─> num_clients: 12
  │           ├─> num_rounds: 20
  │           └─> strategy
  │
  └─> plotting.plot_*()
        └─> Generates: Visualization plots
              ├─> stratified_selection_analysis.png
              ├─> random_vs_stratified_comparison.png
              ├─> comparison_metrics.png
              └─> summary_comparison.png
```

## Stratified Selection Algorithm

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   STRATIFIED SELECTION ALGORITHM                             │
└─────────────────────────────────────────────────────────────────────────────┘

Input:
  - client_strata: {stratum_name: [client_ids]}
  - K: clients_per_round
  - min_per_stratum: minimum clients per stratum

Output:
  - selected_clients: List of K client IDs

Algorithm:

1. Compute Base Allocation:
   ┌─────────────────────────────────────────────────────────┐
   │ For each stratum h:                                      │
   │   N_h = number of clients in stratum h                   │
   │   W_h = N_h / N (stratum weight)                         │
   │   K_h = max(min_per_stratum, K * W_h)                    │
   │                                                          │
   │ Adjust to ensure Σ K_h = K                              │
   └─────────────────────────────────────────────────────────┘

2. Phase 1 - Allocate from Each Stratum:
   ┌─────────────────────────────────────────────────────────┐
   │ selected = []                                            │
   │ For each stratum h:                                      │
   │   Sample K_h clients randomly (without replacement)      │
   │   Add to selected list                                   │
   └─────────────────────────────────────────────────────────┘

3. Phase 2 - Fill Remaining Slots (if needed):
   ┌─────────────────────────────────────────────────────────┐
   │ While len(selected) < K:                                 │
   │   Sample stratum h with probability ∝ N_h                │
   │   Sample 1 client from stratum h (not yet selected)      │
   │   Add to selected list                                   │
   └─────────────────────────────────────────────────────────┘

4. Compute Statistics:
   ┌─────────────────────────────────────────────────────────┐
   │ For each stratum h:                                      │
   │   Count clients selected from h                          │
   │   Compute percentage: count / K * 100                    │
   │   Compare to expected: N_h / N * 100                     │
   └─────────────────────────────────────────────────────────┘

5. Track History:
   ┌─────────────────────────────────────────────────────────┐
   │ Store:                                                   │
   │   - round_num                                            │
   │   - selected_clients                                     │
   │   - stratum_counts                                       │
   │   - stratum_percentages                                  │
   └─────────────────────────────────────────────────────────┘

Example:
  Input:
    client_strata = {
      'Tier_1': [0, 1, 2, 3],      # 4 clients (33.3%)
      'Tier_2': [4, 5, 6, 7],      # 4 clients (33.3%)
      'Tier_3': [8, 9, 10, 11]     # 4 clients (33.3%)
    }
    K = 6
    min_per_stratum = 1

  Base Allocation:
    Tier_1: max(1, 6 * 0.333) = 2 clients
    Tier_2: max(1, 6 * 0.333) = 2 clients
    Tier_3: max(1, 6 * 0.333) = 2 clients
    Total: 2 + 2 + 2 = 6 ✓

  Phase 1 Selection:
    Tier_1: randomly select 2 from [0, 1, 2, 3] → [0, 2]
    Tier_2: randomly select 2 from [4, 5, 6, 7] → [4, 6]
    Tier_3: randomly select 2 from [8, 9, 10, 11] → [8, 10]
    
  Result: [0, 2, 4, 6, 8, 10]
  
  Statistics:
    Tier_1: 2 clients (33.3% vs 33.3% expected) ✓
    Tier_2: 2 clients (33.3% vs 33.3% expected) ✓
    Tier_3: 2 clients (33.3% vs 33.3% expected) ✓
```

## Configuration Hierarchy

```
conf/base.yaml
  │
  ├─> strategy: compare_stratified
  │
  ├─> Data Configuration
  │   ├─> num_clients: 12
  │   ├─> batch_size: 32
  │   ├─> non_iid: true
  │   └─> alpha: 0.3 (heterogeneity)
  │
  ├─> Training Configuration
  │   ├─> num_rounds: 20
  │   ├─> num_clients_per_round_fit: 6
  │   └─> num_clients_per_round_eval: 6
  │
  ├─> Stratified Selection Configuration
  │   ├─> use_stratified_selection: true
  │   └─> min_clients_per_stratum: 1
  │
  ├─> FedProx Configuration
  │   ├─> mu: 0.01 (base)
  │   ├─> lr: 0.001
  │   └─> layer_mus:
  │       ├─> input: 0.003
  │       ├─> hidden1: 0.008
  │       ├─> hidden2: 0.012
  │       └─> output: 0.025
  │
  └─> Plotting Configuration
      ├─> save_plots: true
      ├─> plot_format: png
      ├─> figsize: [12, 8]
      └─> dpi: 300
```

---

**Architecture Version**: 1.0  
**Last Updated**: January 23, 2026
