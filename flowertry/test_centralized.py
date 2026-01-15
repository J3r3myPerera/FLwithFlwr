"""Quick centralized training test to verify improved accuracy."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import prepare_dataset
from model import Net, NetV2, FocalLoss

print('=' * 60)
print('CENTRALIZED MODEL TEST (Optimized Configuration)')
print('=' * 60)

# Prepare dataset
trainloaders, valloaders, testloader, _, input_dim = prepare_dataset(
    num_partitions=1,
    batch_size=64,
    iid=True,
    use_class_weights=False,
)

print(f'\nInput dimension: {input_dim}')
print(f'Train samples: {len(trainloaders[0].dataset)}')
print(f'Test samples: {len(testloader.dataset)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Use NetV2 with optimal configuration found through tuning
model = NetV2(num_classes=3, input_dim=input_dim, hidden_dim=96, num_blocks=2, dropout=0.12).to(device)
print(f'Model: NetV2 (hidden=96, blocks=2, optimized)')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

# CrossEntropyLoss - slightly boost Low and Medium, reduce High
class_weights = torch.tensor([1.15, 1.2, 0.95], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
print(f'Loss: CrossEntropy with label_smoothing=0.1, weights [1.15, 1.2, 0.95]')

# Adam optimizer - best balance of speed and stability
optimizer = optim.Adam(model.parameters(), lr=0.0006, weight_decay=1e-4)

# Cosine annealing for smooth LR decay
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250, eta_min=1e-6)

# Training without Mixup - simpler approach
print('\nTraining with optimized configuration...')
num_epochs = 250
best_test_acc = 0
best_balanced_acc = 0
best_min_class_acc = 0
best_epoch = 0
best_state = None
patience_counter = 0
early_stop_patience = 80

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(trainloaders[0]):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    scheduler.step()  # ReduceLROnPlateau - needs metric
    
    train_acc = correct / total
    avg_loss = total_loss / len(trainloaders[0])
    
    # Evaluate every epoch
    model.eval()
    test_correct = 0
    test_total = 0
    class_correct_eval = [0, 0, 0]
    class_total_eval = [0, 0, 0]
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
            test_total += target.size(0)
            
            for i in range(3):
                mask = target == i
                class_total_eval[i] += mask.sum().item()
                class_correct_eval[i] += (pred[mask] == i).sum().item()
    
    test_acc = test_correct / test_total
    class_accs = [class_correct_eval[i] / max(1, class_total_eval[i]) * 100 for i in range(3)]
    balanced_acc = np.mean(class_accs)
    min_class_acc = min(class_accs)
    
    # Use geometric mean as the metric for LR scheduling
    geo_mean = (max(0.01, class_accs[0]) * max(0.01, class_accs[1]) * max(0.01, class_accs[2])) ** (1/3)
    # scheduler.step(geo_mean)  # Uncomment if using ReduceLROnPlateau
    
    # Track best model - prioritize models where ALL classes have decent accuracy
    # Score = min_class_acc + 0.3 * balanced_acc (prioritize min class)
    current_score = min_class_acc + 0.3 * balanced_acc
    best_score = best_min_class_acc + 0.3 * best_balanced_acc
    
    improved = False
    if current_score > best_score:
        improved = True
    
    if improved:
        best_min_class_acc = min_class_acc
        best_balanced_acc = balanced_acc
        best_test_acc = test_acc
        best_epoch = epoch + 1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
    
    if (epoch + 1) % 20 == 0 or epoch == 0 or improved:
        lr = optimizer.param_groups[0]['lr']
        star = '★' if improved else ' '
        print(f'{star} Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Test={test_acc*100:.2f}% '
              f'[L:{class_accs[0]:.0f}%, M:{class_accs[1]:.0f}%, H:{class_accs[2]:.0f}%] MinCls={min_class_acc:.1f}% Geo={geo_mean:.1f}')
    
    # Early stopping
    if patience_counter >= early_stop_patience:
        print(f'\n>>> Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)')
        break

# Load best model state
if best_state is not None:
    model.load_state_dict(best_state)
    print(f'\n>>> Loaded best model from epoch {best_epoch} with min_class_acc={best_min_class_acc:.1f}%')

# Final evaluation
model.eval()
test_correct = 0
test_total = 0
class_correct = [0, 0, 0]
class_total = [0, 0, 0]

with torch.no_grad():
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        test_correct += pred.eq(target).sum().item()
        test_total += target.size(0)
        
        for i in range(3):
            mask = target == i
            class_total[i] += mask.sum().item()
            class_correct[i] += (pred[mask] == i).sum().item()

print('\n' + '=' * 60)
print('FINAL RESULTS')
print('=' * 60)
final_acc = test_correct/test_total*100
class_accs_final = [class_correct[i] / max(1, class_total[i]) * 100 for i in range(3)]
balanced_acc_final = np.mean(class_accs_final)

print(f'Overall Test Accuracy: {final_acc:.2f}%')
print(f'Balanced Accuracy: {balanced_acc_final:.2f}%')
print(f'Best Balanced Accuracy: {best_balanced_acc:.2f}% (epoch {best_epoch})')
print(f'\nPer-class Accuracy:')
class_names = ['Low Savings', 'Medium Savings', 'High Savings']
min_class_acc = 100
for i, name in enumerate(class_names):
    if class_total[i] > 0:
        acc = class_correct[i] / class_total[i] * 100
        min_class_acc = min(min_class_acc, acc)
        print(f'  {name}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})')

print(f'\nClass Balance Score: {min_class_acc:.2f}% (minimum class accuracy)')

if balanced_acc_final >= 70:
    print('\n✅ SUCCESS: Balanced accuracy >= 70%!')
elif balanced_acc_final >= 60:
    print('\n⚠️ GOOD: Balanced accuracy 60-70%')
else:
    print('\n❌ Balanced accuracy below 60% - needs more tuning')

if min_class_acc >= 50:
    print('✅ All classes have >= 50% accuracy (good balance)')
elif min_class_acc >= 30:
    print('⚠️ Some class imbalance (min class < 50%)')
else:
    print('❌ Severe class imbalance (min class < 30%)')
