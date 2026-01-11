# Why Accuracy is Capped at ~60% - Root Cause Analysis

**Issue**: All strategies (FedAvg, FedProx, SCAFFOLD) plateau around 60% accuracy
**Expected**: Should reach 70-85% for a 3-class classification problem
**Date**: 2026-01-11

---

## 🎯 Summary

If **all three strategies** cap at ~60% accuracy, the problem is NOT with the federated learning algorithms. The issue is one of:

1. **Fundamental data limitations** (most likely)
2. **Model architecture insufficient** for task complexity
3. **Training configuration** preventing convergence
4. **Data preprocessing** losing critical information
5. **Class imbalance** causing biased predictions
6. **Intrinsic task difficulty** (the problem itself is hard)

---

## 🔍 Root Cause Categories

### Category 1: Data Quality Issues (60% Probability)

### Category 2: Model Architecture Issues (20% Probability)

### Category 3: Training Issues (15% Probability)

### Category 4: Intrinsic Task Difficulty (5% Probability)

---

## Category 1: Data Quality Issues

### 1A. 🔥 **Class Imbalance Creating Biased Predictions**

**Hypothesis**: One class dominates, model just predicts majority class

**Your current setup**:
```python
# In dataset.py discretize_savings()
labels[savings_percentage >= 7] = 1   # Medium
labels[savings_percentage > 12] = 2   # High
```

**Potential issue**: If class distribution is heavily skewed

**Example scenario**:
```
Class 0 (Low <7%):      100 samples (10%)
Class 1 (Medium 7-12%): 600 samples (60%)  ← Dominates!
Class 2 (High >12%):    300 samples (30%)
```

If the model just always predicts Class 1 (Medium), it gets **60% accuracy**!

**How to diagnose**:
```python
# Add to your code after training
predictions = model.predict(test_data)
unique, counts = np.unique(predictions, return_counts=True)
print(f"Prediction distribution: {dict(zip(unique, counts))}")

# Compare to true labels
print(f"True distribution: {dict(zip(*np.unique(test_labels, return_counts=True)))}")
```

**Expected signs**:
- Model predicts mostly one class (e.g., 90%+ predictions are Class 1)
- Confusion matrix shows diagonal is weak, one column dominates
- Per-class accuracy: one class is ~0%, another is ~100%

**Solutions**:

**Solution 1A-1: Balanced Class Weights**
```python
# In model.py train function
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = torch.FloatTensor(class_weights).to(device)

# Use in loss
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Solution 1A-2: Rebalance Class Thresholds**
```python
# In dataset.py, change discretization to create more balanced classes
def discretize_savings(savings_percentage: pd.Series) -> np.ndarray:
    # Use percentiles instead of fixed thresholds
    low_threshold = savings_percentage.quantile(0.33)   # 33rd percentile
    high_threshold = savings_percentage.quantile(0.67)  # 67th percentile

    labels = np.zeros(len(savings_percentage), dtype=np.int64)
    labels[savings_percentage >= low_threshold] = 1
    labels[savings_percentage >= high_threshold] = 2
    return labels
```

**Solution 1A-3: Stratified Sampling in FL**
```python
# Ensure each client has representative class distribution
# Modify dataset.py partition function to use stratified split
```

---

### 1B. **Insufficient Training Data per Client**

**Current setup**:
- 100 clients total
- 10 clients per round
- Assuming ~1000 total samples → ~10 samples per client

**Problem**: With only 10 samples per client, impossible to learn meaningful patterns

**Calculation**:
```
Total samples: ~1000 (typical for Indian Finance dataset)
Clients: 100
Samples per client: 1000/100 = 10 samples

With 3 classes: ~3-4 samples per class per client
With local_epochs=1, batch_size=32: Only 1 batch per epoch!
```

**How to diagnose**:
- Check average samples per client: should be > 50 for meaningful learning
- Check batches per client: should be > 5 for stable gradients

**Solutions**:

**Solution 1B-1: Reduce Number of Clients**
```yaml
# In base.yaml
num_clients: 20  # Down from 100
num_clients_per_round_fit: 5  # Down from 10
```

**Solution 1B-2: Use Larger Dataset**
- Augment data
- Combine multiple datasets
- Generate synthetic samples

**Solution 1B-3: Increase Samples per Client**
```python
# In dataset.py, allow unequal client sizes
# Give some clients more data
```

---

### 1C. **Data Standardization Losing Information**

**Current preprocessing**:
```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

**Problem**: StandardScaler on ENTIRE dataset before FL partitioning

**Issues**:
1. **Data leakage**: Test set statistics leak into training
2. **Non-IID broken**: Standardization uses global statistics, destroys heterogeneity
3. **Feature scales lost**: Some features might have important absolute values

**How to diagnose**:
- Try without standardization
- Use per-client standardization instead

**Solutions**:

**Solution 1C-1: Per-Client Standardization**
```python
# In dataset.py or client.py
# Each client standardizes their OWN data
def client_standardize(X_client):
    scaler = StandardScaler()
    return scaler.fit_transform(X_client)
```

**Solution 1C-2: Robust Scaling**
```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()  # Less sensitive to outliers
X = scaler.fit_transform(X)
```

**Solution 1C-3: Min-Max Scaling**
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()  # Preserves zero and boundedness
X = scaler.fit_transform(X)
```

---

### 1D. **Categorical Encoding Too Simplistic**

**Current encoding**:
```python
le = LabelEncoder()
encoded = le.fit_transform(df[col])
```

**Problem**:
- LabelEncoder creates ordinal relationship where none exists
- Occupation=1, Occupation=2 treated as ordered (1 < 2)
- City_Tier might be ordinal (Tier 1 > Tier 2), but Occupation is not

**Solutions**:

**Solution 1D-1: One-Hot Encoding**
```python
# In dataset.py
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)
X_categorical = ohe.fit_transform(df[categorical_features])

# This increases feature count from 16 to ~20-30 depending on categories
```

**Solution 1D-2: Embedding Layers**
```python
# In model.py, add embedding layers for categorical features
class Net(nn.Module):
    def __init__(self, num_classes=3, num_occupations=10, num_tiers=4):
        super().__init__()

        # Embeddings for categorical features
        self.occupation_embed = nn.Embedding(num_occupations, 8)
        self.tier_embed = nn.Embedding(num_tiers, 4)

        # MLP for numerical features (14) + embeddings (8+4=12) = 26 input
        self.fc1 = nn.Linear(26, 64)
        # ... rest of network
```

---

### 1E. **Feature Engineering Missing Important Interactions**

**Current features**: 16 raw features (14 numerical + 2 categorical)

**Problem**: Savings potential likely depends on RATIOS and INTERACTIONS

**Examples of missing features**:
```python
# Income-to-expenses ratio
savings_rate = (Income - Total_Expenses) / Income

# Rent-to-income ratio
rent_burden = Rent / Income

# Discretionary spending ratio
discretionary = (Eating_Out + Entertainment) / Income

# Total expenses
total_expenses = Rent + Loan_Repayment + Insurance + ... + Miscellaneous

# Dependency burden
per_capita_income = Income / (1 + Dependents)
```

**Solution 1E-1: Add Derived Features**
```python
# In dataset.py, add feature engineering
def engineer_features(df):
    # Total expenses
    expense_cols = ['Rent', 'Loan_Repayment', 'Insurance', 'Groceries',
                   'Transport', 'Eating_Out', 'Entertainment',
                   'Utilities', 'Healthcare', 'Education', 'Miscellaneous']
    df['Total_Expenses'] = df[expense_cols].sum(axis=1)

    # Savings rate (actual, not desired)
    df['Actual_Savings_Rate'] = (df['Income'] - df['Total_Expenses']) / df['Income']

    # Ratios
    df['Rent_to_Income'] = df['Rent'] / df['Income']
    df['Discretionary_to_Income'] = (df['Eating_Out'] + df['Entertainment']) / df['Income']
    df['Per_Capita_Income'] = df['Income'] / (1 + df['Dependents'])

    # Essential vs discretionary
    df['Essential_Expenses'] = df['Rent'] + df['Groceries'] + df['Utilities'] + df['Healthcare']
    df['Discretionary_Expenses'] = df['Eating_Out'] + df['Entertainment']

    return df
```

---

## Category 2: Model Architecture Issues

### 2A. 🔥 **Model Too Simple for Task Complexity**

**Current architecture**:
```
Input (16) → FC(64) → Dropout(0.3) → FC(32) → Dropout(0.3) → FC(16) → FC(3)
```

**Total parameters**: ~5,000

**Potential issues**:
1. **Network too narrow**: Max width is only 64 units
2. **Too shallow**: Only 3 hidden layers
3. **Bottleneck architecture**: 16→64→32→16 creates information bottleneck at layer 3

**How to diagnose**:
- Train centralized model (no FL) on full dataset
- If centralized model also caps at 60%, architecture is the issue
- If centralized model reaches 80%+, FL is the issue

**Solutions**:

**Solution 2A-1: Wider Network**
```python
class Net(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()

        # Wider layers
        self.fc1 = nn.Linear(16, 128)  # Was 64
        self.fc2 = nn.Linear(128, 128)  # Was 64→32
        self.fc3 = nn.Linear(128, 64)   # Was 32→16
        self.fc4 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(0.3)
```

**Solution 2A-2: Deeper Network**
```python
class Net(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()

        # More layers
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 64)    # Extra layer
        self.fc5 = nn.Linear(64, 32)    # Extra layer
        self.fc6 = nn.Linear(32, num_classes)

        self.dropout = nn.Dropout(0.3)
```

**Solution 2A-3: Residual Connections**
```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + residual  # Residual connection
        return F.relu(x)

class Net(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(16, 128)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.fc_out = nn.Linear(128, num_classes)
```

---

### 2B. **Dropout Too Aggressive**

**Current**: `Dropout(0.3)` applied after EVERY layer

**Problem**:
- In federated learning with small data per client, aggressive dropout can hurt
- 0.3 dropout = 30% of neurons zeroed out
- With small batches, this creates high variance

**Solution**:

**Solution 2B-1: Reduce Dropout**
```python
self.dropout = nn.Dropout(0.1)  # Down from 0.3
```

**Solution 2B-2: Remove Dropout from Earlier Layers**
```python
def forward(self, x):
    x = F.relu(self.fc1(x))
    # No dropout here
    x = F.relu(self.fc2(x))
    x = self.dropout(x)  # Only on later layers
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    return x
```

**Solution 2B-3: Batch Normalization Instead**
```python
class Net(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(16, 64)
        self.bn1 = nn.BatchNorm1d(64)  # Instead of dropout
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, num_classes)
```

---

### 2C. **No Non-Linearity After Final Layer**

**Current**:
```python
x = self.fc4(x)
return x  # No activation
```

**This is CORRECT for CrossEntropyLoss** (which applies softmax internally)

**But if using other losses, this could be an issue**

---

## Category 3: Training Issues

### 3A. 🔥 **Learning Rate Too Low**

**Current**: `lr: 0.01` with `local_epochs: 1`

**Problem**:
- With only 1 local epoch, model makes minimal progress per round
- lr=0.01 might be too conservative
- With 50 rounds × 1 epoch = effectively only 50 training epochs

**Centralized equivalent**:
```
50 rounds × 1 local epoch × ~10 clients = ~500 client updates
But only 50 global aggregations

In centralized training, this would be like training for only 50 epochs
For a neural network, need 100-500 epochs typically
```

**Solutions**:

**Solution 3A-1: Increase Learning Rate**
```yaml
lr: 0.05  # Up from 0.01
```

**Solution 3A-2: Increase Local Epochs**
```yaml
local_epochs: 5  # Up from 1 (but you changed this!)
```

**Solution 3A-3: Increase Global Rounds**
```yaml
num_rounds: 100  # Up from 50
```

---

### 3B. **Batch Size Too Large**

**Current**: `batch_size: 32`

**Problem**: With ~10 samples per client, batch_size=32 means only 1 batch!

**Calculation**:
```
Samples per client: ~10
Batch size: 32
Batches per epoch: 10/32 = 0.3 ≈ 1 batch

With 1 batch:
- Only 1 gradient update per epoch
- High variance in gradients
- Batch stats unreliable
```

**Solutions**:

**Solution 3B-1: Reduce Batch Size**
```yaml
batch_size: 4   # Down from 32
# Now: 10 samples / 4 = 2-3 batches per epoch
```

**Solution 3B-2: Use Full Batch**
```yaml
batch_size: -1  # Special value meaning "use all samples"
# Or in code: batch_size = len(trainloader.dataset)
```

---

### 3C. **Insufficient Training Rounds**

**Current**: 50 rounds

**Problem**: With `local_epochs=1`, this is very few effective training epochs

**Effective training**:
```
Total client updates: 50 rounds × 10 clients × 1 epoch × 1 batch = 500 updates
Centralized equivalent: ~50 epochs (with averaging dilution)
```

**For neural networks, typically need 100-500 epochs**

**Solutions**:

**Solution 3C-1: More Rounds**
```yaml
num_rounds: 100  # Double the rounds
```

**Solution 3C-2: More Local Epochs**
```yaml
local_epochs: 3  # With 50 rounds = 150 effective epochs
```

---

### 3D. **Optimizer Choice**

**Current**: SGD with momentum=0.9

**Problem**:
- SGD can be slow to converge
- Momentum=0.9 with small batches might overshoot

**Solutions**:

**Solution 3D-1: Use Adam**
```python
# In client.py
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Solution 3D-2: Use AdamW**
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01  # L2 regularization
)
```

**Solution 3D-3: Reduce Momentum**
```yaml
momentum: 0.5  # Down from 0.9
```

---

## Category 4: Intrinsic Task Difficulty

### 4A. **Task is Fundamentally Hard**

**Your task**: Predict desired savings percentage from spending habits

**Reality**: Desired savings is a SUBJECTIVE preference

**Problem**: The "desired" savings might not correlate with actual spending patterns

**Example**:
```
Person A: High income, low expenses → Should desire high savings
         But might desire low savings (wants to enjoy life)

Person B: Low income, high expenses → Should desire low savings
         But might desire high savings (aspirational)
```

**The disconnect**: You're predicting a PREFERENCE from BEHAVIOR

**Diagnosis**:
- Check correlation between features and target
- Try predicting ACTUAL savings instead of DESIRED savings

**Solutions**:

**Solution 4A-1: Change Target**
```python
# Instead of predicting desired_savings_percentage
# Predict actual_savings = (income - expenses) / income

df['Actual_Savings'] = (df['Income'] - df[expense_cols].sum(axis=1)) / df['Income']
y = discretize_savings(df['Actual_Savings'])  # This should be easier!
```

**Solution 4A-2: Add More Predictive Features**
- Demographic features (age, occupation)
- Financial stress indicators
- Past savings history (if available)

---

### 4B. **Class Boundaries Are Arbitrary**

**Current boundaries**:
- Low: < 7%
- Medium: 7-12%
- High: > 12%

**Problem**: These thresholds might not align with natural clusters in data

**Solution**:

**Solution 4B-1: Data-Driven Thresholds**
```python
# Use k-means or quantiles to find natural breakpoints
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(savings_percentage.values.reshape(-1, 1))

# Find cluster centers, use as thresholds
```

**Solution 4B-2: Regression Instead of Classification**
```python
# Predict continuous savings percentage
# Convert to class only for evaluation

class Net(nn.Module):
    # ...
    self.fc4 = nn.Linear(16, 1)  # Single output for regression

# Use MSE loss instead of CrossEntropy
criterion = nn.MSELoss()
```

---

## 🎯 Most Likely Causes (Ranked)

### 1. 🔥🔥🔥 **Class Imbalance** (90% probability)
- Model predicting majority class
- **Quick test**: Check prediction distribution
- **Quick fix**: Use class weights in loss

### 2. 🔥🔥 **Insufficient Samples per Client** (70% probability)
- 10 samples per client is too few
- **Quick fix**: Reduce to 20 clients instead of 100

### 3. 🔥🔥 **Batch Size Too Large** (70% probability)
- batch_size=32 with ~10 samples = only 1 batch
- **Quick fix**: batch_size=4

### 4. 🔥 **Model Too Simple** (50% probability)
- 16→64→32→16 might be insufficient
- **Quick fix**: Increase to 16→128→128→64→3

### 5. 🔥 **Missing Feature Engineering** (50% probability)
- Raw features don't capture savings potential
- **Quick fix**: Add income ratios, expense ratios

### 6. 🔥 **Training Insufficient** (40% probability)
- 50 rounds × 1 epoch is too little
- **Quick fix**: num_rounds=100 or local_epochs=3

---

## 🚀 Immediate Diagnostic Steps

### Step 1: Check Class Balance (1 minute)

Add to your test script:
```python
# After training
predictions = []
true_labels = []

for X, y in testloader:
    pred = model(X).argmax(dim=1)
    predictions.extend(pred.cpu().numpy())
    true_labels.extend(y.cpu().numpy())

print("Prediction distribution:", np.bincount(predictions))
print("True distribution:", np.bincount(true_labels))
```

**If predictions heavily skewed to one class → Class imbalance issue**

---

### Step 2: Check Centralized Performance (5 minutes)

Train model WITHOUT federated learning:
```python
# Quick centralized training script
model = Net(3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train on ALL data
for epoch in range(100):
    for X, y in full_dataloader:
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

# Test accuracy
test_acc = evaluate(model, testloader)
print(f"Centralized accuracy: {test_acc}")
```

**If centralized accuracy also ~60% → Data/model issue (not FL issue)**
**If centralized accuracy 75%+ → FL configuration issue**

---

### Step 3: Check Samples per Client

```python
# In your dataset partitioning code
for client_id, client_data in enumerate(client_datasets):
    print(f"Client {client_id}: {len(client_data)} samples")

avg_samples = sum(len(d) for d in client_datasets) / len(client_datasets)
print(f"Average samples per client: {avg_samples}")
```

**If < 20 samples per client → Reduce number of clients**

---

## 📋 Quick Fix Configuration

Try this config immediately:

```yaml
# conf/diagnostic.yaml
defaults:
  - base

# Address most likely issues
num_clients: 20          # Down from 100 (more data per client)
num_clients_per_round_fit: 5

batch_size: 8            # Down from 32 (more batches per client)
local_epochs: 3          # Up from 1 (more training)
num_rounds: 100          # Up from 50 (more convergence)

lr: 0.01
momentum: 0.0            # Disable momentum for stability

# Use class weights (need to implement in code)
use_class_weights: true
```

**Expected improvement**: 60% → 70-75%

---

## Summary

**Most likely issue**: Class imbalance causing model to predict majority class

**Second most likely**: Too few samples per client (100 clients is too many for this dataset)

**Third most likely**: Batch size too large relative to client data size

**Action**: Run the 3 diagnostic steps above to confirm root cause, then apply appropriate fix from this document.
