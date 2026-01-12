"""
Diagnostic utilities for federated learning model analysis.

Includes:
- Class imbalance detection
- Prediction distribution analysis
- Confusion matrix visualization
- Per-class accuracy metrics
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from collections import Counter


def analyze_class_distribution(labels: np.ndarray, dataset_name: str = "Dataset") -> Dict:
    """
    Analyze class distribution in a dataset.

    Args:
        labels: Array of class labels
        dataset_name: Name for printing

    Returns:
        Dictionary with class distribution statistics
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    print(f"\n{'='*60}")
    print(f"{dataset_name} Class Distribution")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print()

    class_names = ['Low (<7%)', 'Medium (7-12%)', 'High (>12%)']

    distribution = {}
    for cls, count in zip(unique, counts):
        percentage = 100 * count / total
        class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        print(f"  {class_name:20s}: {count:5d} samples ({percentage:5.1f}%)")
        distribution[int(cls)] = {
            'count': int(count),
            'percentage': float(percentage),
            'name': class_name
        }

    # Check for severe imbalance
    max_percentage = max(d['percentage'] for d in distribution.values())
    min_percentage = min(d['percentage'] for d in distribution.values())
    imbalance_ratio = max_percentage / min_percentage if min_percentage > 0 else float('inf')

    print()
    print(f"Imbalance ratio: {imbalance_ratio:.2f}x (max/min class)")

    if imbalance_ratio > 3.0:
        print("⚠️  WARNING: Severe class imbalance detected!")
        print("   → Consider using class weights in loss function")
        print("   → Or rebalance classes using percentile thresholds")
    elif imbalance_ratio > 2.0:
        print("⚠️  Moderate class imbalance detected")
        print("   → May benefit from class weights")
    else:
        print("✅ Classes are reasonably balanced")

    return {
        'distribution': distribution,
        'imbalance_ratio': float(imbalance_ratio),
        'total_samples': total
    }


def analyze_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strategy_name: str = "Model"
) -> Dict:
    """
    Analyze model predictions for class imbalance issues.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        strategy_name: Name of the strategy

    Returns:
        Dictionary with prediction analysis
    """
    print(f"\n{'='*60}")
    print(f"{strategy_name} Prediction Analysis")
    print(f"{'='*60}")

    # Prediction distribution
    pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
    true_unique, true_counts = np.unique(y_true, return_counts=True)

    total = len(y_pred)
    class_names = ['Low (<7%)', 'Medium (7-12%)', 'High (>12%)']

    print("\nPrediction Distribution vs True Distribution:")
    print(f"{'Class':<20s} {'Predicted':<15s} {'True':<15s} {'Difference':<15s}")
    print("-" * 65)

    all_classes = sorted(set(list(pred_unique) + list(true_unique)))
    pred_dict = dict(zip(pred_unique, pred_counts))
    true_dict = dict(zip(true_unique, true_counts))

    prediction_stats = {}

    for cls in all_classes:
        class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"

        pred_count = pred_dict.get(cls, 0)
        true_count = true_dict.get(cls, 0)

        pred_pct = 100 * pred_count / total
        true_pct = 100 * true_count / total
        diff_pct = pred_pct - true_pct

        print(f"{class_name:<20s} {pred_count:5d} ({pred_pct:4.1f}%)  "
              f"{true_count:5d} ({true_pct:4.1f}%)  "
              f"{diff_pct:+6.1f}%")

        prediction_stats[int(cls)] = {
            'predicted_count': int(pred_count),
            'predicted_pct': float(pred_pct),
            'true_count': int(true_count),
            'true_pct': float(true_pct),
            'difference_pct': float(diff_pct)
        }

    # Check if model is just predicting one class
    max_pred_pct = max(pred_dict.values()) / total * 100
    print()

    if max_pred_pct > 80:
        dominant_class = max(pred_dict, key=pred_dict.get)
        print(f"🔴 CRITICAL: Model predicts {class_names[dominant_class]} {max_pred_pct:.1f}% of the time!")
        print(f"   → This is a MAJORITY CLASS BASELINE strategy")
        print(f"   → The model has NOT learned meaningful patterns")
        print(f"   → ACTION REQUIRED: Use class weights in loss function")
        severity = "critical"
    elif max_pred_pct > 60:
        dominant_class = max(pred_dict, key=pred_dict.get)
        print(f"⚠️  WARNING: Model heavily favors {class_names[dominant_class]} ({max_pred_pct:.1f}%)")
        print(f"   → Model may be biased toward majority class")
        print(f"   → Consider using class weights")
        severity = "high"
    elif max_pred_pct > 45:
        print(f"⚠️  Model shows slight prediction bias ({max_pred_pct:.1f}% max)")
        print(f"   → Within acceptable range but could be improved")
        severity = "moderate"
    else:
        print(f"✅ Predictions are well-distributed ({max_pred_pct:.1f}% max)")
        severity = "none"

    return {
        'prediction_stats': prediction_stats,
        'max_predicted_percentage': float(max_pred_pct),
        'severity': severity
    }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes

    Returns:
        Confusion matrix (true × pred)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            cm[true, pred] += 1
    return cm


def print_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strategy_name: str = "Model",
    num_classes: int = 3
) -> np.ndarray:
    """
    Print confusion matrix with analysis.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        strategy_name: Name of the strategy
        num_classes: Number of classes

    Returns:
        Confusion matrix
    """
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)

    print(f"\n{'='*60}")
    print(f"{strategy_name} Confusion Matrix")
    print(f"{'='*60}")

    class_names = ['Low', 'Med', 'High']

    # Print header
    header = "True \\ Pred"
    print(f"\n{header:<12s}", end="")
    for cls_name in class_names[:num_classes]:
        print(f"{cls_name:>8s}", end="")
    print(f"{'Total':>8s}  {'Recall':>8s}")
    print("-" * 60)

    # Print matrix with recall
    for i in range(num_classes):
        print(f"{class_names[i]:<12s}", end="")
        row_sum = cm[i].sum()

        for j in range(num_classes):
            print(f"{cm[i, j]:8d}", end="")

        recall = cm[i, i] / row_sum * 100 if row_sum > 0 else 0
        print(f"{row_sum:8d}  {recall:7.1f}%")

    # Print precision
    print("-" * 60)
    print(f"{'Precision':<12s}", end="")
    for j in range(num_classes):
        col_sum = cm[:, j].sum()
        precision = cm[j, j] / col_sum * 100 if col_sum > 0 else 0
        print(f"{precision:7.1f}%", end=" ")
    print()

    # Overall accuracy
    accuracy = np.trace(cm) / cm.sum() * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    # Check for diagonal weakness
    print("\nPer-Class Performance:")
    for i in range(num_classes):
        row_sum = cm[i].sum()
        col_sum = cm[:, i].sum()

        recall = cm[i, i] / row_sum * 100 if row_sum > 0 else 0
        precision = cm[i, i] / col_sum * 100 if col_sum > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        status = "✅" if recall > 50 and precision > 50 else "⚠️" if recall > 30 or precision > 30 else "🔴"

        print(f"  {status} {class_names[i]:<10s}: Recall={recall:5.1f}%, "
              f"Precision={precision:5.1f}%, F1={f1:5.1f}%")

    return cm


def analyze_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 3
) -> Dict:
    """
    Compute detailed per-class metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes

    Returns:
        Dictionary with per-class metrics
    """
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)

    metrics = {}
    class_names = ['Low (<7%)', 'Medium (7-12%)', 'High (>12%)']

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        metrics[int(i)] = {
            'class_name': class_names[i],
            'recall': float(recall),
            'precision': float(precision),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn)
        }

    return metrics


def full_diagnostic_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strategy_name: str = "Model"
) -> Dict:
    """
    Generate complete diagnostic report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        strategy_name: Name of the strategy

    Returns:
        Dictionary with all diagnostic information
    """
    print(f"\n{'#'*60}")
    print(f"# DIAGNOSTIC REPORT: {strategy_name}")
    print(f"{'#'*60}")

    # 1. True label distribution
    true_dist = analyze_class_distribution(y_true, "True Labels")

    # 2. Prediction analysis
    pred_analysis = analyze_predictions(y_true, y_pred, strategy_name)

    # 3. Confusion matrix
    cm = print_confusion_matrix(y_true, y_pred, strategy_name)

    # 4. Per-class metrics
    per_class = analyze_per_class_metrics(y_true, y_pred)

    # 5. Overall summary
    print(f"\n{'='*60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")

    overall_accuracy = np.trace(cm) / cm.sum() * 100

    print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
    print(f"Class Imbalance Ratio: {true_dist['imbalance_ratio']:.2f}x")
    print(f"Prediction Bias Severity: {pred_analysis['severity'].upper()}")

    # Check if model is just doing majority class baseline
    if pred_analysis['severity'] == 'critical':
        print("\n🔴 CRITICAL ISSUE DETECTED:")
        print("   Model is using MAJORITY CLASS BASELINE strategy")
        print("   This means it has NOT learned meaningful patterns")
        print("\n   ROOT CAUSE: Class imbalance overwhelming the training")
        print("\n   IMMEDIATE FIXES REQUIRED:")
        print("   1. Add class weights to loss function")
        print("   2. Or rebalance dataset using percentile thresholds")
        print("   3. Or use oversampling/undersampling")
    elif pred_analysis['severity'] == 'high':
        print("\n⚠️  HIGH PRIORITY ISSUE:")
        print("   Model shows strong bias toward one class")
        print("   Consider using class weights or rebalancing data")
    elif pred_analysis['severity'] == 'moderate':
        print("\n⚠️  Moderate prediction bias detected")
        print("   Performance could be improved with class weights")
    else:
        print("\n✅ No major class imbalance issues detected")

    # Average metrics
    avg_recall = np.mean([m['recall'] for m in per_class.values()])
    avg_precision = np.mean([m['precision'] for m in per_class.values()])
    avg_f1 = np.mean([m['f1_score'] for m in per_class.values()])

    print(f"\nAverage Metrics (across classes):")
    print(f"  Recall:    {avg_recall*100:.2f}%")
    print(f"  Precision: {avg_precision*100:.2f}%")
    print(f"  F1 Score:  {avg_f1*100:.2f}%")

    return {
        'strategy_name': strategy_name,
        'overall_accuracy': float(overall_accuracy),
        'true_distribution': true_dist,
        'prediction_analysis': pred_analysis,
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': per_class,
        'avg_recall': float(avg_recall),
        'avg_precision': float(avg_precision),
        'avg_f1': float(avg_f1)
    }


def collect_predictions(model, dataloader, device='cpu') -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect predictions from a model on a dataloader.

    Args:
        model: PyTorch model
        dataloader: DataLoader with test data
        device: Device to run on

    Returns:
        Tuple of (true_labels, predictions)
    """
    model.eval()
    model.to(device)

    all_true = []
    all_pred = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)

            outputs = model(features)
            predictions = outputs.argmax(dim=1)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(predictions.cpu().numpy())

    return np.array(all_true), np.array(all_pred)
