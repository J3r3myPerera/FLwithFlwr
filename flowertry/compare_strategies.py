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


def run_fedavg(cfg: DictConfig, trainloaders, validationloaders, testloader, client_fn):
    """Run FedAvg strategy."""
    print("\n" + "=" * 60)
    print("RUNNING FedAvg")
    print("=" * 60)
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.00001,  # Use min_fit_clients instead
        min_fit_clients=cfg.num_clients_per_round_fit,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader)
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


def run_fedprox(cfg: DictConfig, trainloaders, validationloaders, testloader, client_fn):
    """Run FedProx strategy."""
    print("\n" + "=" * 60)
    print("RUNNING FedProx")
    print("=" * 60)
    
    strategy = fl.server.strategy.FedProx(
        fraction_fit=0.00001,  # Use min_fit_clients instead
        min_fit_clients=cfg.num_clients_per_round_fit,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
        proximal_mu=cfg.get('proximal_mu', 0.1)  # Default to 0.1 if not specified
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


def run_fedscaffold(cfg: DictConfig, trainloaders, validationloaders, testloader, client_fn):
    """Run FedSCAFFOLD strategy."""
    print("\n" + "=" * 60)
    print("RUNNING FedSCAFFOLD")
    print("=" * 60)
    
    # Import custom FedSCAFFOLD strategy
    from scaffold_strategy import FedScaffoldStrategy
    
    strategy = FedScaffoldStrategy(
        fraction_fit=0.00001,  # Use min_fit_clients instead
        min_fit_clients=cfg.num_clients_per_round_fit,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
        scaffold_lr=cfg.get('scaffold_lr', 1.0)  # Default to 1.0 if not specified
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
    
    print(f"\n{'Strategy':<15} {'Final Accuracy':<20} {'Final Loss':<20} {'Time (s)':<15}")
    print("-" * 80)
    
    for result in all_results:
        strategy = result['strategy']
        final_acc = result['final_accuracy']
        final_loss = result['final_loss']
        elapsed_time = result['elapsed_time']
        
        acc_str = f"{final_acc*100:.2f}%" if final_acc else "N/A"
        loss_str = f"{final_loss:.4f}" if final_loss else "N/A"
        time_str = f"{elapsed_time:.2f}"
        
        print(f"{strategy:<15} {acc_str:<20} {loss_str:<20} {time_str:<15}")
    
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
    
    trainloaders, validationloaders, testloader = prepare_dataset(
        num_partitions=cfg.num_clients,
        batch_size=cfg.batch_size,
        iid=cfg.get('iid', True),
        alpha=cfg.get('alpha', 0.5)
    )
    
    print(f"\nNumber of clients: {len(trainloaders)}")
    print(f"Samples in first client's training set: {len(trainloaders[0].dataset)}")

    ## 3. Define the client function
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    ## 4. Run all strategies
    all_results = []
    
    strategies_to_run = cfg.get('strategies', ['fedavg', 'fedprox', 'fedscaffold'])
    
    if 'fedavg' in strategies_to_run:
        try:
            history, elapsed_time = run_fedavg(cfg, trainloaders, validationloaders, testloader, client_fn)
            metrics = extract_metrics(history, 'FedAvg')
            metrics['elapsed_time'] = elapsed_time
            all_results.append(metrics)
        except Exception as e:
            print(f"Error running FedAvg: {e}")
    
    if 'fedprox' in strategies_to_run:
        try:
            history, elapsed_time = run_fedprox(cfg, trainloaders, validationloaders, testloader, client_fn)
            metrics = extract_metrics(history, 'FedProx')
            metrics['elapsed_time'] = elapsed_time
            all_results.append(metrics)
        except Exception as e:
            print(f"Error running FedProx: {e}")
    
    if 'fedscaffold' in strategies_to_run:
        try:
            history, elapsed_time = run_fedscaffold(cfg, trainloaders, validationloaders, testloader, client_fn)
            metrics = extract_metrics(history, 'FedSCAFFOLD')
            metrics['elapsed_time'] = elapsed_time
            all_results.append(metrics)
        except Exception as e:
            print(f"Error running FedSCAFFOLD: {e}")

    ## 5. Save comparison results
    save_path = HydraConfig.get().runtime.output_dir
    comparison_path = Path(save_path) / "comparison_results.pkl"
    
    comparison_data = {
        'results': all_results,
        'config': {
            'num_rounds': cfg.num_rounds,
            'num_clients': cfg.num_clients,
            'batch_size': cfg.batch_size,
            'num_classes': cfg.num_classes,
            'lr': cfg.config_fit.lr,
            'local_epochs': cfg.config_fit.local_epochs,
            'strategies': strategies_to_run,
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

