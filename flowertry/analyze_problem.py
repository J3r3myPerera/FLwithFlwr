"""
Analyze the root causes of poor model performance.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = './data/IndianPersoalFinance/indianPersonalFinanceAndSpendingHabits.csv'

def main():
    # Load data
    df = pd.read_csv(DATA_PATH)
    print("=" * 80)
    print("DATASET ANALYSIS")
    print("=" * 80)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    
    # Target analysis
    print("\n" + "=" * 80)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 80)
    print(f"\nDesired_Savings_Percentage statistics:")
    print(df['Desired_Savings_Percentage'].describe())
    
    # Current discretization
    print("\n\nCurrent discretization thresholds (<7%, 7-12%, >12%):")
    low = (df['Desired_Savings_Percentage'] < 7).sum()
    med = ((df['Desired_Savings_Percentage'] >= 7) & (df['Desired_Savings_Percentage'] <= 12)).sum()
    high = (df['Desired_Savings_Percentage'] > 12).sum()
    total = len(df)
    print(f"  Low (<7%):      {low} ({100*low/total:.1f}%)")
    print(f"  Medium (7-12%): {med} ({100*med/total:.1f}%)")
    print(f"  High (>12%):    {high} ({100*high/total:.1f}%)")
    
    # Better discretization using quantiles
    print("\n\nQuantile-based discretization (33%, 67% percentiles):")
    q33 = df['Desired_Savings_Percentage'].quantile(0.33)
    q67 = df['Desired_Savings_Percentage'].quantile(0.67)
    print(f"  33rd percentile: {q33:.2f}%")
    print(f"  67th percentile: {q67:.2f}%")
    
    # Feature correlations
    print("\n" + "=" * 80)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 80)
    
    numerical_cols = ['Income', 'Age', 'Dependents', 'Rent', 'Loan_Repayment', 'Insurance', 
                      'Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 
                      'Healthcare', 'Education', 'Miscellaneous']
    
    corr = df[numerical_cols + ['Desired_Savings_Percentage']].corr()['Desired_Savings_Percentage'].drop('Desired_Savings_Percentage')
    print("\nCorrelation with target (Desired_Savings_Percentage):")
    print(corr.sort_values(ascending=False))
    
    # Check if there are better predictive features available
    print("\n\nOther potentially useful columns in dataset:")
    other_cols = [c for c in df.columns if c not in numerical_cols + ['Occupation', 'City_Tier', 'Desired_Savings_Percentage']]
    print(other_cols)
    
    # Check Disposable_Income correlation
    if 'Disposable_Income' in df.columns:
        print(f"\nDisposable_Income correlation with target: {df['Disposable_Income'].corr(df['Desired_Savings_Percentage']):.4f}")
    
    # Feature engineering
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING")
    print("=" * 80)
    
    # Calculate expense columns
    expense_cols = ['Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
                    'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 
                    'Education', 'Miscellaneous']
    
    df['Total_Expenses'] = df[expense_cols].sum(axis=1)
    df['Savings_Rate'] = (df['Income'] - df['Total_Expenses']) / df['Income']
    df['Expense_to_Income'] = df['Total_Expenses'] / df['Income']
    df['Discretionary_Ratio'] = (df['Eating_Out'] + df['Entertainment']) / df['Income']
    df['Essential_Ratio'] = (df['Rent'] + df['Groceries'] + df['Utilities'] + df['Healthcare']) / df['Income']
    df['Per_Capita_Income'] = df['Income'] / (1 + df['Dependents'])
    
    # Check correlations of engineered features
    engineered_features = ['Total_Expenses', 'Savings_Rate', 'Expense_to_Income', 
                           'Discretionary_Ratio', 'Essential_Ratio', 'Per_Capita_Income']
    
    print("\nCorrelation of engineered features with target:")
    for feat in engineered_features:
        corr_val = df[feat].corr(df['Desired_Savings_Percentage'])
        print(f"  {feat}: {corr_val:.4f}")
    
    # Test with sklearn models first
    print("\n" + "=" * 80)
    print("SKLEARN BASELINE MODELS")
    print("=" * 80)
    
    # Prepare features
    all_features = numerical_cols + engineered_features
    X = df[all_features].values
    
    # Use quantile-based labels for better balance
    y = np.zeros(len(df), dtype=np.int64)
    y[df['Desired_Savings_Percentage'] >= q33] = 1
    y[df['Desired_Savings_Percentage'] >= q67] = 2
    
    print(f"\nQuantile-based class distribution:")
    for cls in range(3):
        print(f"  Class {cls}: {(y == cls).sum()} ({100*(y == cls).sum()/len(y):.1f}%)")
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Random Forest
    print("\n1. Random Forest Classifier:")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"   Accuracy: {rf_acc*100:.2f}%")
    
    # Feature importance
    print("\n   Top 10 important features:")
    importance = pd.DataFrame({'feature': all_features, 'importance': rf.feature_importances_})
    importance = importance.sort_values('importance', ascending=False)
    for _, row in importance.head(10).iterrows():
        print(f"     {row['feature']}: {row['importance']:.4f}")
    
    # Logistic Regression
    print("\n2. Logistic Regression:")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_acc = lr.score(X_test, y_test)
    print(f"   Accuracy: {lr_acc*100:.2f}%")
    
    # PyTorch Neural Network
    print("\n3. PyTorch Neural Network (same architecture as FL model):")
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Define model (matches the improved architecture)
    class ImprovedNet(nn.Module):
        def __init__(self, input_dim, num_classes=3):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.fc2 = nn.Linear(128, 128)
            self.bn2 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, 64)
            self.bn3 = nn.BatchNorm1d(64)
            self.fc4 = nn.Linear(64, 64)
            self.bn4 = nn.BatchNorm1d(64)
            self.fc5 = nn.Linear(64, 32)
            self.bn5 = nn.BatchNorm1d(32)
            self.fc6 = nn.Linear(32, num_classes)
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = torch.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = torch.relu(self.bn3(self.fc3(x)))
            x = self.dropout(x)
            x = torch.relu(self.bn4(self.fc4(x)))
            x = self.dropout(x)
            x = torch.relu(self.bn5(self.fc5(x)))
            x = self.fc6(x)
            return x
    
    model = ImprovedNet(len(all_features))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(100):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        predictions = outputs.argmax(dim=1)
        nn_acc = (predictions == y_test_t).float().mean().item()
    print(f"   Accuracy: {nn_acc*100:.2f}%")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print(f"""
Key Findings:
1. Random Forest accuracy: {rf_acc*100:.2f}%
2. Logistic Regression accuracy: {lr_acc*100:.2f}%
3. Neural Network accuracy: {nn_acc*100:.2f}%

If all models perform similarly (~40-50%):
  → The task is fundamentally difficult
  → 'Desired_Savings_Percentage' is subjective and poorly correlated with features
  → Consider predicting 'Savings_Rate' (actual savings) instead

If Random Forest > Neural Network:
  → Feature engineering helps significantly
  → Use the engineered features in FL model
  → Current 16 raw features may be insufficient

Recommendations:
1. Use quantile-based class thresholds for balanced classes
2. Add engineered features (Savings_Rate, ratios)
3. Consider predicting actual savings instead of desired savings
4. The low correlations suggest the prediction task itself may be too hard
""")


if __name__ == "__main__":
    main()
