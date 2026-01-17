"""
Model evaluation and visualization utilities.

This module contains functions for:
- Plotting confusion matrices
- Feature importance visualization
- Model comparison plots
- Performance metrics
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Optional


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: tuple = (14, 12),
    cmap: str = 'Blues'
) -> None:
    """
    Plot confusion matrix with annotations.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        figsize: Figure size
        cmap: Colormap
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str] = ['accuracy', 'f1_score', 'precision', 'recall'],
    figsize: tuple = (14, 10)
) -> None:
    """
    Plot comparison of multiple models across metrics.
    
    Args:
        comparison_df: DataFrame with model comparison
        metrics: List of metrics to plot
        figsize: Figure size
    """
    n_metrics = len(metrics)
    rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(rows, 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        data = comparison_df.sort_values(metric)
        ax.barh(data['Model'], data[metric], color='skyblue', edgecolor='navy')
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison', 
                    fontsize=12, fontweight='bold')
        ax.set_xlim([data[metric].min() - 0.02, 1.0])
        
        # Add value labels
        for i, v in enumerate(data[metric]):
            ax.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 10,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to show
        figsize: Figure size
    """
    if not hasattr(model, 'feature_importances_'):
        print("⚠️  Model doesn't support feature importance")
        return
    
    # Get importances
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.barh(importance_df['Feature'], importance_df['Importance'], 
            color='steelblue', edgecolor='navy')
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(importance_df['Importance']):
        plt.text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def plot_cv_results(
    cv_results: Dict,
    figsize: tuple = (12, 6)
) -> None:
    """
    Plot cross-validation results across folds.
    
    Args:
        cv_results: Dictionary of CV results from cross_validate_models
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    for model_name, results in cv_results.items():
        scores = results['scores']
        folds = range(1, len(scores) + 1)
        plt.plot(folds, scores, 'o-', label=model_name, linewidth=2, markersize=8)
    
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Cross-Validation Results Across Folds', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_model_summary(
    model_name: str,
    train_metrics: Dict,
    val_metrics: Dict,
    test_metrics: Optional[Dict] = None
) -> None:
    """
    Print formatted summary of model performance.
    
    Args:
        model_name: Name of the model
        train_metrics: Training metrics
        val_metrics: Validation metrics
        test_metrics: Test metrics (optional)
    """
    print("=" * 70)
    print(f"MODEL SUMMARY: {model_name}")
    print("=" * 70)
    
    print("\n📊 Training Performance:")
    for metric, value in train_metrics.items():
        print(f"   {metric.replace('_', ' ').title():12s}: {value:.4f}")
    
    print("\n📊 Validation Performance:")
    for metric, value in val_metrics.items():
        print(f"   {metric.replace('_', ' ').title():12s}: {value:.4f}")
    
    if test_metrics:
        print("\n📊 Test Performance:")
        for metric, value in test_metrics.items():
            print(f"   {metric.replace('_', ' ').title():12s}: {value:.4f}")
    
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    print("Model Evaluation Utilities Module")
    print("="*70)
    print("\nAvailable functions:")
    print("  - plot_confusion_matrix()")
    print("  - plot_model_comparison()")
    print("  - plot_feature_importance()")
    print("  - plot_cv_results()")
    print("  - print_model_summary()")