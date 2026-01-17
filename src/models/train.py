"""
Model training and evaluation utilities for Crop Recommendation project.

This module contains functions for:
- Training multiple models
- Model evaluation
- Cross-validation
- Model comparison
- Saving/loading models
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os


def get_default_models() -> Dict:
    """
    Get dictionary of default models with baseline hyperparameters.
    
    Returns:
        Dictionary of model_name: model_instance
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial',
            n_jobs=-1
        ),
        
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ),
        
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        ),
        
        'SVM': SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            random_state=42
        ),
        
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        )
    }
    
    return models


def train_single_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str = "Model"
) -> Dict:
    """
    Train a single model and evaluate on validation set.
    
    Args:
        model: Scikit-learn compatible model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_name: Name of the model for logging
        
    Returns:
        Dictionary containing model, predictions, and metrics
    """
    print(f"\nTraining: {model_name}")
    print("-" * 50)
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_val_pred = model.predict(X_val)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'f1_score': f1_score(y_val, y_val_pred, average='weighted'),
        'precision': precision_score(y_val, y_val_pred, average='weighted'),
        'recall': recall_score(y_val, y_val_pred, average='weighted')
    }
    
    print(f"✅ Validation Results:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   F1-Score:  {metrics['f1_score']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    
    return {
        'model': model,
        'predictions': y_val_pred,
        'metrics': metrics
    }


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    models: Optional[Dict] = None
) -> Dict:
    """
    Train multiple models and compare performance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        models: Dictionary of models (uses defaults if None)
        
    Returns:
        Dictionary of results for each model
    """
    if models is None:
        models = get_default_models()
    
    print("=" * 70)
    print("TRAINING MULTIPLE MODELS")
    print("=" * 70)
    
    results = {}
    
    for model_name, model in models.items():
        result = train_single_model(
            model, X_train, y_train, X_val, y_val, model_name
        )
        results[model_name] = result
    
    print("\n" + "=" * 70)
    print("✅ ALL MODELS TRAINED!")
    print("=" * 70)
    
    return results


def compare_models(results: Dict) -> pd.DataFrame:
    """
    Create comparison DataFrame from model results.
    
    Args:
        results: Dictionary of model results
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []
    
    for model_name, result in results.items():
        row = {
            'Model': model_name,
            **result['metrics']
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('accuracy', ascending=False)
    
    return comparison_df


def cross_validate_models(
    models: Dict,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = 'accuracy'
) -> Dict:
    """
    Perform cross-validation on multiple models.
    
    Args:
        models: Dictionary of models
        X: Features
        y: Labels
        cv: Number of folds
        scoring: Scoring metric
        
    Returns:
        Dictionary of CV results
    """
    print("=" * 70)
    print(f"CROSS-VALIDATION ({cv}-fold)")
    print("=" * 70)
    
    cv_results = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        
        scores = cross_val_score(
            model, X, y, cv=cv, scoring=scoring, n_jobs=-1
        )
        
        cv_results[model_name] = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
        
        print(f"   Scores: {scores}")
        print(f"   Mean:   {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return cv_results


def evaluate_on_test(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: List of class names for report
        
    Returns:
        Dictionary of test metrics and predictions
    """
    print("=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    
    # Predict
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred, average='weighted'),
        'precision': precision_score(y_test, y_test_pred, average='weighted'),
        'recall': recall_score(y_test, y_test_pred, average='weighted')
    }
    
    print(f"\n✅ Test Set Performance:")
    print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"   F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    
    # Classification report
    if class_names is not None:
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred, 
                                   target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    return {
        'predictions': y_test_pred,
        'metrics': test_metrics,
        'confusion_matrix': cm,
        'classification_report': classification_report(
            y_test, y_test_pred, 
            target_names=class_names, 
            output_dict=True
        ) if class_names is not None else None
    }


def save_model(
    model,
    model_name: str,
    metrics: Dict,
    save_dir: str = '../models',
    metadata: Optional[Dict] = None
) -> str:
    """
    Save trained model and metadata.
    
    Args:
        model: Trained model
        model_name: Name of the model
        metrics: Dictionary of metrics
        save_dir: Directory to save model
        metadata: Additional metadata to save
        
    Returns:
        Path to saved model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_filename = f"{model_name.lower().replace(' ', '_')}_{timestamp}.pkl"
    model_path = os.path.join(save_dir, model_filename)
    joblib.dump(model, model_path)
    
    # Save metadata
    model_metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'metrics': metrics,
        **(metadata or {})
    }
    
    metadata_filename = f"{model_name.lower().replace(' ', '_')}_metadata_{timestamp}.json"
    metadata_path = os.path.join(save_dir, metadata_filename)
    
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=4)
    
    print(f"\n✅ Model saved:")
    print(f"   Model:    {model_path}")
    print(f"   Metadata: {metadata_path}")
    
    return model_path


def load_model(model_path: str):
    """
    Load saved model.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model
    """
    model = joblib.load(model_path)
    print(f"✅ Model loaded from: {model_path}")
    return model


def get_best_model(results: Dict) -> Tuple[str, object, Dict]:
    """
    Get the best performing model from results.
    
    Args:
        results: Dictionary of model results
        
    Returns:
        Tuple of (model_name, model, metrics)
    """
    best_model_name = max(results.keys(), 
                         key=lambda k: results[k]['metrics']['accuracy'])
    
    best_result = results[best_model_name]
    
    print(f"Best Model: {best_model_name}")
    print(f"Accuracy: {best_result['metrics']['accuracy']:.4f}")
    
    return best_model_name, best_result['model'], best_result['metrics']


# Example usage
if __name__ == "__main__":
    print("Model Training Utilities Module")
    print("="*70)
    print("\nAvailable functions:")
    print("  - get_default_models()")
    print("  - train_single_model()")
    print("  - train_all_models()")
    print("  - compare_models()")
    print("  - cross_validate_models()")
    print("  - evaluate_on_test()")
    print("  - save_model()")
    print("  - load_model()")
    print("  - get_best_model()")
    print("\nImport this module in your notebooks for easy model training!")