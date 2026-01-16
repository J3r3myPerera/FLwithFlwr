"""Test strategy-specific configurations."""

import sys
import yaml

# Load and display config
with open('conf/base.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("="*80)
print("STRATEGY-SPECIFIC CONFIGURATION TEST")
print("="*80)

print("\n📋 Default Training Parameters:")
print(f"  Learning Rate: {config['learning_rate']}")
print(f"  Local Epochs: {config['local_epochs']}")
print(f"  Batch Size: {config['batch_size']}")
print(f"  Num Clients: {config['num_clients']}")
print(f"  Partitioning: {config['partition_strategy']} (iid={config['iid']})")

print("\n🎯 Strategy-Specific Configurations:")
print("-" * 80)

for strategy, params in config['strategy_configs'].items():
    print(f"\n{strategy.upper()}:")
    print(f"  Learning Rate: {params['lr']}")
    print(f"  Local Epochs: {params['local_epochs']}")
    print(f"  Max Grad Norm: {params['max_grad_norm']}")
    print(f"  Momentum: {params.get('momentum', 0.0)}")

print("\n" + "="*80)
print("✅ Configuration loaded successfully!")
print("\nKey Features:")
print("  • Each strategy can have custom learning rate")
print("  • Each strategy can have custom local epochs")
print("  • Each strategy can have custom gradient clipping norm")
print("  • Hybrid partitioning with 12 clients enabled")
print("="*80)
