"""Quick centralized training test to verify improved accuracy."""

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import prepare_dataset
from model import Net

print('=' * 60)
print('CENTRALIZED MODEL TEST (Baseline Check)')
print('=' * 60)

# Prepare dataset - use all data as one "client"
trainloaders, valloaders, testloader, class_weights, input_dim = prepare_dataset(
    num_partitions=1,  # Just one partition = centralized
    batch_size=64,
    iid=True,
    use_class_weights=True,
    class_weight_method='balanced',
    discretization_method='fixed'  # Use fixed thresholds for wider class boundaries
)

print(f'\nInput dimension: {input_dim}')
print(f'Train samples: {len(trainloaders[0].dataset)}')
print(f'Test samples: {len(testloader.dataset)}')

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(num_classes=4, input_dim=input_dim).to(device)

# Loss and optimizer
if class_weights is not None:
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    print(f'Using class weights: {class_weights}')
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Training
print('\nTraining centralized model...')
num_epochs = 100  # Train longer for better convergence

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
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    train_acc = correct / total
    avg_loss = total_loss / len(trainloaders[0])
    
    # Step the scheduler
    scheduler.step(avg_loss)
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)
        test_acc = test_correct / test_total
        
        print(f'Epoch {epoch+1:2d}: Train Loss={avg_loss:.4f}, Train Acc={train_acc*100:.2f}%, Test Acc={test_acc*100:.2f}%')

# Final evaluation
model.eval()
test_correct = 0
test_total = 0
class_correct = [0, 0, 0, 0]
class_total = [0, 0, 0, 0]

with torch.no_grad():
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        test_correct += pred.eq(target).sum().item()
        test_total += target.size(0)
        
        for i in range(4):
            mask = target == i
            class_total[i] += mask.sum().item()
            class_correct[i] += (pred[mask] == i).sum().item()

print('\n' + '=' * 60)
print('FINAL RESULTS')
print('=' * 60)
print(f'Overall Test Accuracy: {test_correct/test_total*100:.2f}%')
print(f'\nPer-class Accuracy:')
class_names = ['Low (<6.5%)', 'Lower-Middle (6.5-9%)', 'Upper-Middle (9-13%)', 'High (>13%)']
for i, name in enumerate(class_names):
    if class_total[i] > 0:
        acc = class_correct[i] / class_total[i] * 100
        print(f'  {name}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})')

if test_correct/test_total > 0.50:
    print('\n✅ Centralized model accuracy > 50% - Feature engineering is working!')
else:
    print('\n⚠️ Still below 50% - may need additional tuning')
