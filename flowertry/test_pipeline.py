"""Quick test script for the updated pipeline."""

from dataset import prepare_dataset
from cleint import generate_client_fn
from server import get_initial_parameters, get_evaluate_fn
from model import Net

print('Testing full pipeline...')

# Prepare dataset
trainloaders, valloaders, testloader, class_weights, input_dim = prepare_dataset(
    num_partitions=5,
    batch_size=32,
    iid=True
)

print(f'\nInput dimension: {input_dim}')
print(f'Number of clients: {len(trainloaders)}')

# Test client function creation
client_fn = generate_client_fn(trainloaders, valloaders, 3, class_weights=class_weights, input_dim=input_dim)
print('Client function created successfully')

# Test initial parameters
initial_params = get_initial_parameters(3, input_dim=input_dim)
print('Initial parameters created successfully')

# Test evaluation function
eval_fn = get_evaluate_fn(3, testloader, class_weights, input_dim=input_dim)
print('Evaluation function created successfully')

# Test a single forward pass
model = Net(num_classes=3, input_dim=input_dim)
for batch in testloader:
    x, y = batch
    output = model(x)
    print(f'Forward pass successful: input shape {x.shape}, output shape {output.shape}')
    break

print('\n✅ Full pipeline test passed!')
