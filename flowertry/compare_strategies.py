import hydra
from hydra.core.hydra_config import HydraConfig
import pickle
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from cleint import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
import flwr as fl
import time
from typing import Dict, List, Tuple
import torch
import numpy as np
from diagnostics import (
    analyze_class_distribution,
    collect_predictions,
    full_diagnostic_report
)


def get_strategy_config(cfg: DictConfig, strategy_name: str) -> DictConfig:
    """
    Get strategy-specific configuration, falling back to global config_fit if not specified.
    
    Args:
        cfg: Global configuration
        strategy_name: Name of strategy ('fedavg', 'fedprox', 'fedscaffold')
    
    Returns:
        Strategy-specific configuration
    """
    # Check if strategy_configs exists and has this strategy
    if hasattr(cfg, 'strategy_configs') and strategy_name in cfg.strategy_configs:
        strategy_cfg = cfg.strategy_configs[strategy_name]
        print(f"  Using strategy-specific config for {strategy_name}")
        print(f"    lr: {strategy_cfg.get('lr', cfg.config_fit.lr)}")
        print(f"    local_epochs: {strategy_cfg.get('local_epochs', cfg.config_fit.local_epochs)}")
        print(f"    momentum: {strategy_cfg.get('momentum', cfg.config_fit.momentum)}")
        print(f"    max_grad_norm: {strategy_cfg.get('max_grad_norm', cfg.config_fit.max_grad_norm)}")
        return strategy_cfg
    else:
        print(f"  Using global config_fit for {strategy_name}")
        return cfg.config_fit


def run_fedavg(cfg: DictConfig, trainloaders, validationloaders, testloader, client_fn, initial_parameters, class_weights=None, input_dim=None):
    """Run FedAvg strategy."""
    print("\n" + "=" * 60)
    print("RUNNING FedAvg")
    print("=" * 60)

    # Get strategy-specific config
    fedavg_config = get_strategy_config(cfg, 'fedavg')

    strategy = fl.server.strategy.FedAvg(
        # fraction_fit=0.00001,  # Use min_fit_clients instead
        min_fit_clients=cfg.num_clients_per_round_fit,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(fedavg_config),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader, class_weights, input_dim=input_dim),
        initial_parameters=initial_parameters
    )

    
    start_time = time.time()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 1.0, 'num_gpus': 0}
    )
    elapsed_time = time.time() - start_time
    
    return history, elapsed_time


def run_fedprox(cfg: DictConfig, trainloaders, validationloaders, testloader, client_fn, initial_parameters, class_weights=None, input_dim=None):
    """Run FedProx strategy (fixed mu, simple adaptive mu, or multi-signal adaptive mu)."""
    
    # Check if adaptive mu is enabled
    adaptive_cfg = cfg.get('adaptive_mu', {})
    use_adaptive = adaptive_cfg.get('enabled', False) if adaptive_cfg else False
    adaptive_mode = adaptive_cfg.get('mode', 'simple') if adaptive_cfg else 'simple'
    
    if use_adaptive and adaptive_mode == 'multi_signal':
        # Multi-signal adaptive mu (HAPI-based)
        print("\n" + "=" * 60)
        print("RUNNING FedProx with MULTI-SIGNAL ADAPTIVE MU")
        print("=" * 60)

        from adaptive_fedprox import MultiSignalAdaptiveFedProx

        # Get strategy-specific config
        fedprox_config = get_strategy_config(cfg, 'fedprox')

        # Get multi-signal mu parameters
        multi_cfg = cfg.get('multi_signal_mu', {})
        base_mu = multi_cfg.get('base_mu', 0.1)
        mu_min = multi_cfg.get('mu_min', 0.001)
        mu_max = multi_cfg.get('mu_max', 2.0)
        smoothing_factor = multi_cfg.get('smoothing_factor', 0.7)
        warmup_rounds = multi_cfg.get('warmup_rounds', 3)

        # Get signal weights
        weights_cfg = multi_cfg.get('weights', {})
        signal_weights = {
            'gradient_divergence': weights_cfg.get('gradient_divergence', 0.35),
            'loss_variance': weights_cfg.get('loss_variance', 0.25),
            'label_entropy': weights_cfg.get('label_entropy', 0.25),
            'feature_variance': weights_cfg.get('feature_variance', 0.15)
        }

        print(f"  Base mu: {base_mu}")
        print(f"  Mu range: [{mu_min}, {mu_max}]")
        print(f"  Smoothing factor: {smoothing_factor}")
        print(f"  Warmup rounds: {warmup_rounds}")
        print(f"  Signal weights: {signal_weights}")

        strategy = MultiSignalAdaptiveFedProx(
            base_mu=base_mu,
            mu_min=mu_min,
            mu_max=mu_max,
            signal_weights=signal_weights,
            smoothing_factor=smoothing_factor,
            warmup_rounds=warmup_rounds,
            # fraction_fit=0.00001,
            min_fit_clients=cfg.num_clients_per_round_fit,
            min_evaluate_clients=cfg.num_clients_per_round_eval,
            min_available_clients=cfg.num_clients,
            on_fit_config_fn=get_on_fit_config(fedprox_config),
            evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader, class_weights, input_dim=input_dim),
            initial_parameters=initial_parameters
        )
        
        start_time = time.time()
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy,
            client_resources={'num_cpus': 1.0, 'num_gpus': 0}
        )
        elapsed_time = time.time() - start_time
        
        # Store adaptation history
        mu_history = strategy.get_mu_history()
        final_mu = strategy.get_final_mu()
        adaptation_history = strategy.get_adaptation_history()
        
        print(f"\n  [MultiSignalAdaptiveFedProx] Final mu: {final_mu:.4f}")
        print(f"  [MultiSignalAdaptiveFedProx] Mu evolution: {[(r, f'{m:.4f}') for r, m in mu_history[-5:]]}")
        
        return history, elapsed_time, {
            'mu_history': mu_history, 
            'final_mu': final_mu,
            'adaptation_history': adaptation_history
        }
    
    elif use_adaptive:
        # Simple loss-based adaptive mu
        print("\n" + "=" * 60)
        print("RUNNING FedProx with SIMPLE ADAPTIVE MU")
        print("=" * 60)

        from adaptive_fedprox import AdaptiveFedProx

        # Get strategy-specific config
        fedprox_config = get_strategy_config(cfg, 'fedprox')

        # Get adaptive mu parameters with defaults
        initial_mu = adaptive_cfg.get('initial_mu', 0.1)
        mu_min = adaptive_cfg.get('mu_min', 0.001)
        mu_max = adaptive_cfg.get('mu_max', 1.0)
        increase_factor = adaptive_cfg.get('increase_factor', 1.5)
        decrease_factor = adaptive_cfg.get('decrease_factor', 0.9)
        loss_threshold = adaptive_cfg.get('loss_threshold', 0.0)
        warmup_rounds = adaptive_cfg.get('warmup_rounds', 3)

        print(f"  Initial mu: {initial_mu}")
        print(f"  Mu range: [{mu_min}, {mu_max}]")
        print(f"  Increase/Decrease factors: {increase_factor}/{decrease_factor}")
        print(f"  Warmup rounds: {warmup_rounds}")

        strategy = AdaptiveFedProx(
            initial_mu=initial_mu,
            mu_min=mu_min,
            mu_max=mu_max,
            mu_increase_factor=increase_factor,
            mu_decrease_factor=decrease_factor,
            loss_threshold=loss_threshold,
            warmup_rounds=warmup_rounds,
            # fraction_fit=0.00001,  # Use min_fit_clients instead
            min_fit_clients=cfg.num_clients_per_round_fit,
            min_evaluate_clients=cfg.num_clients_per_round_eval,
            min_available_clients=cfg.num_clients,
            on_fit_config_fn=get_on_fit_config(fedprox_config),
            evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader, class_weights, input_dim=input_dim),
            initial_parameters=initial_parameters
        )
        
        start_time = time.time()
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy,
            client_resources={'num_cpus': 1.0, 'num_gpus': 0}
        )
        elapsed_time = time.time() - start_time
        
        # Store mu history in the history object for later analysis
        mu_history = strategy.get_mu_history()
        final_mu = strategy.get_final_mu()
        print(f"\n  [AdaptiveFedProx] Final mu: {final_mu:.4f}")
        print(f"  [AdaptiveFedProx] Mu evolution: {[(r, f'{m:.4f}') for r, m in mu_history[-5:]]}")
        
        return history, elapsed_time, {'mu_history': mu_history, 'final_mu': final_mu}
    
    else:
        # Fixed mu
        print("\n" + "=" * 60)
        print("RUNNING FedProx (fixed mu)")
        print("=" * 60)

        # Get strategy-specific config
        fedprox_config = get_strategy_config(cfg, 'fedprox')

        fixed_mu = cfg.get('proximal_mu', 0.1)
        print(f"  Fixed mu: {fixed_mu}")

        strategy = fl.server.strategy.FedProx(
            # fraction_fit=0.00001,  # Use min_fit_clients instead
            min_fit_clients=cfg.num_clients_per_round_fit,
            min_evaluate_clients=cfg.num_clients_per_round_eval,
            min_available_clients=cfg.num_clients,
            # IMPORTANT: Pass proximal_mu to config so client applies the proximal term
            on_fit_config_fn=get_on_fit_config(fedprox_config, proximal_mu=fixed_mu),
            evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader, class_weights, input_dim=input_dim),
            proximal_mu=fixed_mu,
            initial_parameters=initial_parameters
        )
        
        start_time = time.time()
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=cfg.num_clients,
            config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy,
            client_resources={'num_cpus': 1.0, 'num_gpus': 0}
        )
        elapsed_time = time.time() - start_time
        
        return history, elapsed_time, {'final_mu': fixed_mu}


def run_fedscaffold(cfg: DictConfig, trainloaders, validationloaders, testloader, initial_parameters, class_weights=None, input_dim=None):
    """Run FedSCAFFOLD strategy with improved implementation."""
    print("\n" + "=" * 60)
    print("RUNNING FedSCAFFOLD (Improved)")
    print("=" * 60)

    # Import custom FedSCAFFOLD strategy
    from scaffold_strategy import FedScaffoldStrategy
    from cleint import generate_client_fn

    # Get strategy-specific config
    scaffold_config = get_strategy_config(cfg, 'fedscaffold')

    # Get SCAFFOLD parameters
    server_lr = cfg.get('scaffold_server_lr', 1.0)
    print(f"  Server learning rate: {server_lr}")

    # Create strategy
    strategy = FedScaffoldStrategy(
        # fraction_fit=0.00001,  # Use min_fit_clients instead
        min_fit_clients=cfg.num_clients_per_round_fit,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        total_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(scaffold_config),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader, class_weights, input_dim=input_dim),
        initial_parameters=initial_parameters
        # server_learning_rate=server_lr
    )

    # Create SCAFFOLD-aware client function with strategy reference
    client_fn = generate_client_fn(
        trainloaders,
        validationloaders,
        cfg.num_classes,
        strategy=strategy,
        class_weights=class_weights,
        input_dim=input_dim
    )

    start_time = time.time()
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 1.0, 'num_gpus': 0}
    )
    elapsed_time = time.time() - start_time

    print(f"\n  [SCAFFOLD] Training completed in {elapsed_time:.2f}s")

    # Extract c_global_norm history from metrics
    c_global_norm_history = []
    if hasattr(history, 'metrics_distributed_fit') and history.metrics_distributed_fit:
        c_global_norm_history = history.metrics_distributed_fit.get('c_global_norm', [])

        # Print c_global_norm evolution
        if c_global_norm_history:
            print(f"\n  [SCAFFOLD] Control Variate Norm Evolution:")
            print(f"    Initial: {c_global_norm_history[0][1]:.4f}")
            print(f"    Final:   {c_global_norm_history[-1][1]:.4f}")
            print(f"    Max:     {max(norm for _, norm in c_global_norm_history):.4f}")

    return history, elapsed_time, {'c_global_norm_history': c_global_norm_history}


def extract_metrics(history, strategy_name: str) -> Dict:
    """Extract key metrics from history."""
    metrics = {
        'strategy': strategy_name,
        'accuracies': [],
        'losses': [],
        'final_accuracy': None,
        'final_loss': None
    }
    
    if history.metrics_centralized:
        accuracies = history.metrics_centralized.get('accuracy', [])
        losses = history.losses_centralized
        
        if accuracies:
            metrics['accuracies'] = [(rnd, acc) for rnd, acc in accuracies]
            metrics['final_accuracy'] = accuracies[-1][1] if accuracies else None
        
        if losses:
            metrics['losses'] = [(rnd, loss) for rnd, loss in losses]
            metrics['final_loss'] = losses[-1][1] if losses else None
    
    return metrics


def print_comparison_summary(all_results: List[Dict]):
    """Print comparison summary of all strategies."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Strategy':<20} {'Final Accuracy':<18} {'Final Loss':<15} {'Time (s)':<12} {'Final Mu':<10}")
    print("-" * 80)
    
    for result in all_results:
        strategy = result['strategy']
        final_acc = result['final_accuracy']
        final_loss = result['final_loss']
        elapsed_time = result['elapsed_time']
        final_mu = result.get('final_mu', None)
        
        acc_str = f"{final_acc*100:.2f}%" if final_acc else "N/A"
        loss_str = f"{final_loss:.4f}" if final_loss else "N/A"
        time_str = f"{elapsed_time:.2f}"
        mu_str = f"{final_mu:.4f}" if final_mu else "-"
        
        print(f"{strategy:<20} {acc_str:<18} {loss_str:<15} {time_str:<12} {mu_str:<10}")
    
    # Print mu evolution for adaptive strategies
    for result in all_results:
        if 'mu_history' in result and result['mu_history']:
            print(f"\n  Mu evolution for {result['strategy']}:")
            mu_history = result['mu_history']
            # Show first 3, middle, and last 3 values
            if len(mu_history) <= 7:
                for rnd, mu in mu_history:
                    print(f"    Round {rnd}: mu = {mu:.4f}")
            else:
                for rnd, mu in mu_history[:3]:
                    print(f"    Round {rnd}: mu = {mu:.4f}")
                print(f"    ...")
                for rnd, mu in mu_history[-3:]:
                    print(f"    Round {rnd}: mu = {mu:.4f}")
        
        # Print signal breakdown for multi-signal adaptive
        if 'adaptation_history' in result and result['adaptation_history']:
            print(f"\n  Signal breakdown for {result['strategy']} (final round):")
            final_record = result['adaptation_history'][-1]
            signals = final_record['signals']
            score = final_record['heterogeneity_score']
            print(f"    Gradient divergence:  {signals['gradient_divergence']:.3f}")
            print(f"    Loss variance:        {signals['loss_variance']:.3f}")
            print(f"    Label entropy:        {signals['label_entropy']:.3f}")
            print(f"    Feature variance:     {signals['feature_variance']:.3f}")
            print(f"    Heterogeneity score:  {score:.3f}")
    
    # Find best strategy
    if all_results:
        best_acc = max(
            (r for r in all_results if r['final_accuracy'] is not None),
            key=lambda x: x['final_accuracy'],
            default=None
        )
        if best_acc:
            print(f"\n{'='*80}")
            print(f"BEST STRATEGY (by accuracy): {best_acc['strategy']} ({best_acc['final_accuracy']*100:.2f}%)")
            print(f"{'='*80}")


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    Compare FedAvg, FedProx, and FedSCAFFOLD strategies.
    
    Task: Classify users into savings potential categories:
    - Low (<7% savings)
    - Medium (7-12% savings)
    - High (>12% savings)
    """
    
    ## 1. Parse and display configuration
    print("=" * 80)
    print("FEDERATED LEARNING STRATEGY COMPARISON")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print(f"\nComparing strategies: FedAvg, FedProx, FedSCAFFOLD")
    
    # Display data distribution settings
    iid_setting = cfg.get('iid', True)
    alpha_setting = cfg.get('alpha', 0.5)
    print(f"\nData Distribution: {'IID' if iid_setting else 'Non-IID'}")
    if not iid_setting:
        print(f"  Non-IID alpha parameter: {alpha_setting}")

    ## 2. Prepare the dataset (once for all strategies)
    print("\n" + "=" * 80)
    print("PREPARING DATASET")
    print("=" * 80)
    
    # Check if class weights should be used
    use_class_weights = cfg.get('use_class_weights', False)
    class_weight_method = cfg.get('class_weight_method', 'balanced')
    
    # Feature engineering and discretization options
    use_engineered_features = cfg.get('use_engineered_features', True)
    discretization_method = cfg.get('discretization_method', 'quantile')
    
    trainloaders, validationloaders, testloader, class_weights, input_dim = prepare_dataset(
        num_partitions=cfg.num_clients,
        batch_size=cfg.batch_size,
        iid=cfg.get('iid', True),
        alpha=cfg.get('alpha', 0.5),
        use_class_weights=use_class_weights,
        class_weight_method=class_weight_method,
        use_engineered_features=use_engineered_features,
        discretization_method=discretization_method
    )
    
    print(f"\nNumber of clients: {len(trainloaders)}")
    print(f"Samples in first client's training set: {len(trainloaders[0].dataset)}")
    print(f"Input dimension: {input_dim}")

    ## 3. Define the client function
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes, class_weights=class_weights, input_dim=input_dim)

    ## 4. Create shared initial parameters (ONCE for all strategies)
    # This ensures fair comparison - all strategies start from the same weights
    from server import get_initial_parameters
    initial_parameters = get_initial_parameters(cfg.num_classes, input_dim=input_dim)
    print("\n[INFO] Created shared initial parameters for all strategies")

    ## 5. Run all strategies
    all_results = []
    
    strategies_to_run = cfg.get('strategies', ['fedavg', 'fedprox', 'fedscaffold'])
    
    if 'fedavg' in strategies_to_run:
        try:
            history, elapsed_time = run_fedavg(cfg, trainloaders, validationloaders, testloader, client_fn, initial_parameters, class_weights, input_dim=input_dim)
            metrics = extract_metrics(history, 'FedAvg')
            metrics['elapsed_time'] = elapsed_time
            all_results.append(metrics)
        except Exception as e:
            print(f"Error running FedAvg: {e}")
    
    if 'fedprox' in strategies_to_run:
        try:
            result = run_fedprox(cfg, trainloaders, validationloaders, testloader, client_fn, initial_parameters, class_weights, input_dim=input_dim)
            history, elapsed_time = result[0], result[1]
            adaptive_info = result[2] if len(result) > 2 else None
            
            # Use appropriate name based on adaptive mu mode
            adaptive_cfg = cfg.get('adaptive_mu', {})
            use_adaptive = adaptive_cfg.get('enabled', False) if adaptive_cfg else False
            adaptive_mode = adaptive_cfg.get('mode', 'simple') if adaptive_cfg else 'simple'
            
            if use_adaptive and adaptive_mode == 'multi_signal':
                strategy_name = 'FedProx (MultiSignal)'
            elif use_adaptive:
                strategy_name = 'FedProx (Adaptive)'
            else:
                strategy_name = 'FedProx'
            
            metrics = extract_metrics(history, strategy_name)
            metrics['elapsed_time'] = elapsed_time
            
            # Store adaptive mu info if available
            if adaptive_info:
                metrics['mu_history'] = adaptive_info['mu_history']
                metrics['final_mu'] = adaptive_info['final_mu']
                if 'adaptation_history' in adaptive_info:
                    metrics['adaptation_history'] = adaptive_info['adaptation_history']
            
            all_results.append(metrics)
        except Exception as e:
            print(f"Error running FedProx: {e}")
            import traceback
            traceback.print_exc()
    
    if 'fedscaffold' in strategies_to_run:
        try:
            # SCAFFOLD creates its own client function with strategy reference
            result = run_fedscaffold(cfg, trainloaders, validationloaders, testloader, initial_parameters, class_weights, input_dim=input_dim)
            history, elapsed_time = result[0], result[1]
            scaffold_info = result[2] if len(result) > 2 else None

            metrics = extract_metrics(history, 'FedSCAFFOLD')
            metrics['elapsed_time'] = elapsed_time

            # Store c_global_norm history if available
            if scaffold_info and 'c_global_norm_history' in scaffold_info:
                metrics['c_global_norm_history'] = scaffold_info['c_global_norm_history']

            all_results.append(metrics)
        except Exception as e:
            print(f"Error running FedSCAFFOLD: {e}")
            import traceback
            traceback.print_exc()

    ## 5. Run diagnostic analysis
    print("\n" + "=" * 80)
    print("RUNNING DIAGNOSTIC ANALYSIS")
    print("=" * 80)

    # Analyze true label distribution in test set
    print("\n[Step 1/3] Analyzing test set class distribution...")
    test_labels = []
    for _, labels in testloader:
        test_labels.extend(labels.numpy())
    test_labels = np.array(test_labels)

    true_dist = analyze_class_distribution(test_labels, "Test Set")

    # Collect predictions from each strategy and run diagnostics
    print("\n[Step 2/3] Collecting predictions from trained models...")

    # We need to get the final trained models from each strategy
    # For simplicity, we'll retrain a small model or use the server's evaluate_fn
    # Actually, we can use the final global model from each strategy's history

    # Import model for predictions
    from model import Net
    from flwr.common import parameters_to_ndarrays

    diagnostic_results = {}

    # Train a centralized model for diagnostic purposes
    print("\n[Step 3/4] Training centralized model for diagnostic comparison...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diagnostic_model = Net(cfg.num_classes).to(device)
    
    # Use the same testloader to gather all train data for centralized training
    # Create a simple dataloader from all client data
    all_train_data = []
    all_train_labels = []
    for train_loader in trainloaders:
        for X, y in train_loader:
            all_train_data.append(X)
            all_train_labels.append(y)
    
    full_train_X = torch.cat(all_train_data, dim=0)
    full_train_y = torch.cat(all_train_labels, dim=0)
    full_train_dataset = torch.utils.data.TensorDataset(full_train_X, full_train_y)
    full_train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=cfg.batch_size, shuffle=True)
    
    # Quick centralized training (30 epochs)
    optimizer = torch.optim.Adam(diagnostic_model.parameters(), lr=0.001)
    
    # Use class weights if enabled
    if class_weights is not None:
        weight_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    diagnostic_model.train()
    for epoch in range(30):
        for X, y in full_train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(diagnostic_model(X), y)
            loss.backward()
            optimizer.step()
    
    # Now check predictions vs true labels
    print("\n[Step 4/4] Analyzing prediction distributions...")
    diagnostic_model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for X, y in testloader:
            X = X.to(device)
            pred = diagnostic_model(X).argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(y.numpy())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Print diagnostic results
    print("\n" + "=" * 60)
    print("CLASS BALANCE DIAGNOSTIC")
    print("=" * 60)
    
    class_names = ['Low (<6.5%)', 'Lower-Middle (6.5-9%)', 'Upper-Middle (9-13%)', 'High (>13%)']
    
    print("\n📊 True Label Distribution (Test Set):")
    true_counts = np.bincount(true_labels, minlength=cfg.num_classes)
    for cls, count in enumerate(true_counts):
        pct = 100 * count / len(true_labels)
        print(f"  Class {cls} ({class_names[cls]}): {count} samples ({pct:.1f}%)")
    
    print("\n🎯 Model Prediction Distribution:")
    pred_counts = np.bincount(predictions, minlength=cfg.num_classes)
    for cls, count in enumerate(pred_counts):
        pct = 100 * count / len(predictions)
        print(f"  Class {cls} ({class_names[cls]}): {count} predictions ({pct:.1f}%)")
    
    # Calculate per-class accuracy
    print("\n✅ Per-Class Accuracy:")
    for cls in range(cfg.num_classes):
        cls_mask = true_labels == cls
        if cls_mask.sum() > 0:
            cls_acc = (predictions[cls_mask] == true_labels[cls_mask]).sum() / cls_mask.sum()
            print(f"  Class {cls} ({class_names[cls]}): {cls_acc*100:.1f}%")
    
    # Check for bias
    print("\n⚠️  Bias Analysis:")
    pred_entropy = -np.sum([(c/len(predictions)) * np.log(c/len(predictions) + 1e-10) for c in pred_counts])
    max_entropy = np.log(cfg.num_classes)
    pred_uniformity = pred_entropy / max_entropy
    
    if pred_uniformity < 0.7:
        print(f"  🔴 HIGH BIAS DETECTED (uniformity={pred_uniformity:.2f})")
        print(f"  → Model predictions are heavily skewed")
        print(f"  → Likely predicting majority class most of the time")
        print(f"  → Solution: Use class weights (use_class_weights=true)")
    elif pred_uniformity < 0.85:
        print(f"  🟡 MODERATE BIAS (uniformity={pred_uniformity:.2f})")
        print(f"  → Some class imbalance in predictions")
    else:
        print(f"  🟢 BALANCED PREDICTIONS (uniformity={pred_uniformity:.2f})")
    
    print("=" * 60)

    print("\n" + "=" * 80)
    print("DIAGNOSTIC ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"  • Class imbalance ratio: {true_dist['imbalance_ratio']:.2f}x")

    if true_dist['imbalance_ratio'] > 3.0:
        print(f"  • ⚠️  SEVERE CLASS IMBALANCE DETECTED")
        print(f"  • This is likely limiting model performance to majority class baseline")
        print(f"  • Recommendation: Add class weights to loss function")
    elif true_dist['imbalance_ratio'] > 2.0:
        print(f"  • ⚠️  Moderate class imbalance detected")
        print(f"  • Consider using class weights for better performance")
    else:
        print(f"  • ✅ Classes are reasonably balanced")

    ## 6. Save comparison results
    save_path = HydraConfig.get().runtime.output_dir
    comparison_path = Path(save_path) / "comparison_results.pkl"
    
    # Get adaptive mu config info
    adaptive_cfg = cfg.get('adaptive_mu', {})
    adaptive_mu_config = dict(adaptive_cfg) if adaptive_cfg else {}
    
    # Get multi-signal config info
    multi_signal_cfg = cfg.get('multi_signal_mu', {})
    multi_signal_config = dict(multi_signal_cfg) if multi_signal_cfg else {}
    
    comparison_data = {
        'results': all_results,
        'config': {
            'num_rounds': cfg.num_rounds,
            'num_clients': cfg.num_clients,
            'batch_size': cfg.batch_size,
            'num_classes': cfg.num_classes,
            'lr': cfg.config_fit.lr,
            'local_epochs': cfg.config_fit.local_epochs,
            'max_grad_norm': cfg.config_fit.get('max_grad_norm', 1.0),
            'strategies': strategies_to_run,
            'proximal_mu_fixed': cfg.get('proximal_mu', 0.1),
            'adaptive_mu_config': adaptive_mu_config,
            'multi_signal_config': multi_signal_config,
            'iid': cfg.get('iid', True),
            'alpha': cfg.get('alpha', 0.5),
            'task': 'Savings Potential Classification'
        }
    }

    with open(str(comparison_path), "wb") as f:
        pickle.dump(comparison_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    ## 6. Print comparison summary
    print_comparison_summary(all_results)
    
    print(f"\nComparison results saved to: {comparison_path}")


if __name__ == "__main__":
    main()

