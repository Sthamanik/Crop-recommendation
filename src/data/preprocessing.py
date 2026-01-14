"""
Data preprocessing utilities for Crop Recommendation project.

This module contains functions for:
- Loading raw data
- Encoding target variables
- Splitting datasets 
- Scaleing features 
- Saving/loading preprocessed data
"""

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Tuple, Optional

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def encode_target(y: pd.Series, encoder: Optional[LabelEncoder] = None) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode target variable using LabelEncoder

    Args:
        y: Target variable (crop labels)
        encoder: Pre-fitted LabelEncoder (optional, for transforming new data)

    Returns:
        Tuple of (encoded_labels, fitted_encoder)
    """
    if encoder is None:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        print(f"✅ Encoded labels: {encoder.classes_.size} unique classes")
    else:
        y_encoded = encoder.transform(y)
        print(f"✅ Transformed using existing encoder")
    return y_encoded, encoder

def split_data(
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Split data into train, validation, and test sets with stratification.

    Args:
        X: Input features
        y: Target variable
        test_size: Fraction of data to allocate for test set (default: 0.15)
        val_size: Fraction of data to allocate for validation set (default: 0.15)
        random_state: Seed for random number generator (default: 42)

    Returns:
        Dictionary containing X_train, X_test, X_val, y_train, y_test, y_val
    """

    #First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y,
        random_state=random_state
    )

    #Calculate validation size relative to temp set
    val_size_adjusted = val_size/ (1-test_size)

    #Second split: separate train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_size_adjusted, 
        stratify=y_temp,
        random_state=random_state
    )

    total = len(X)
    print(f"\n✅ Data split completed:")
    print(f"   Train:      {len(X_train):4d} samples ({len(X_train)/total:.1%})")
    print(f"   Validation: {len(X_val):4d} samples ({len(X_val)/total:.1%})")
    print(f"   Test:       {len(X_test):4d} samples ({len(X_test)/total:.1%})")
    

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }

def scale_features(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale features using StandardScaler

    IMPORTANT: Scaler is fitted only on training data to prevent data leakage

    Args:
        X_train: Training features
        X_val: Validation features
        X_test: New data to transform
        scaler: Pre-fitted StandardScaler (optional, for transforming new data)

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_scaled, fitted_scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        print(f"✅ Scaler fitted on training data")
    else: 
        X_train_scaled = scaler.transform(X_train)
        print("✅ Using existing scaler")

    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)   

    #Veriy the scaling
    mean = X_train_scaled.mean(axis=0)
    std = X_train_scaled.std(axis=0)

    print(f"    Training set - Mean: {mean.mean():.6f}, Std: {std.mean():.6f}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def save_processed_data(
    data_dict: Dict[str, np.ndarray],
    feature_names: list,
    processed_dir: str = '../../data/processed',
) -> None:
    """
    Save processed data to CSV files.

    Args:
        data_dict: Dictionary containing train/val/test splits
        feature_names: List of feature names
        processed_dir: Directory to save the processed data (default: '../../data/processed')
    """
    os.makedirs(processed_dir, exist_ok=True)

    #Save features (X)
    for key in ['X_train', 'X_val', 'X_test']:
        df = pd.DataFrame(data_dict[key], columns= feature_names)
        filepath = os.path.join(processed_dir, f"{key}.csv")
        df.to_csv(filepath, index=False)
        print(f"✅ Saved {key} to {filepath}")

    #Save targets (y)
    for key in ['y_train', 'y_val', 'y_test']:
        df = pd.DataFrame(data_dict[key], columns=['label'])
        filepath = os.path.join(processed_dir, f"{key}.csv")
        df.to_csv(filepath, index=False)
        print(f"✅ Saved {key} to {filepath}")

def save_transformers(
    scaler: StandardScaler,
    encoder: LabelEncoder,
    models_dir: str = '../../models'
) -> None: 
    """
    Save scaler and encoder objects.

    Args:
        scaler: Fitted StandarScaler
        encoder: Fitted LabelEncoder
        models_dir: Directory to save the transformers (default: '../models')
    """
    os.makedirs(models_dir, exist_ok=True)

    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(encoder, encoder_path)
    
    print(f"\n✅ Saved transformers:")
    print(f"   Scaler:  {scaler_path}")
    print(f"   Encoder: {encoder_path}")

def load_transformers(
        models_dir: str = '../../models'
) -> Tuple[StandardScaler, LabelEncoder]:
    """
    Load saved scaler and encoder objects.

    Args:
        models_dir: Directory to load the transformers (default: '../models')

    Returns:
        Tuple of (scaler, encoder)
    """
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
    
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    
    print(f"\n✅ Loaded transformers:")
    return scaler, encoder

def load_processed_data(
        processed_dir: str = '../../data/processed'
) -> Dict[str, pd.DataFrame]:
    """
    Load processed data from CSV files.

    Args:
        processed_dir: Directory to load the processed data (default: '../data/processed')

    Returns:
        Dictionary containing train/val/test splits
    """
    data = {}
    for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
        filepath = os.path.join(processed_dir, f"{key}.csv")
        data[key] = pd.read_csv(filepath)
        print(f"✅ Loaded {key}: {data[key].shape}")
    return data

def preprocess_pipeline(
    raw_data_path: str,
    target_col: str = 'label',
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    save_data: bool = True
) -> Dict:
    """
    Complete preprocessing pipeline.

    This function executes the entire preprocessing workflow:
    1. Load rae data
    2. Separate features and target
    3. Encode target variable 
    4. Split into train/val/test 
    5. Scale features
    6. Save preprocessed data and transformers

    Args: 
        raw_data_path: Path to raw CSV file
        target_col: Name of target column (default: 'label')
        test_size: Fraction of data to allocate for test set (default: 0.15)
        val_size: Fraction of data to allocate for validation set (default: 0.15)
        random_state: Seed for random number generator (default: 42)
        save_data: Flag to save preprocessed data and transformers (default: True)

    Returns:
        Dictionary containing all processed data and transformers
    """

    print("="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
    # 1. Load data
    print("\n[1/6] Loading data...")
    df = load_raw_data(raw_data_path)
    
    # 2. Separate features and target
    print("\n[2/6] Separating features and target...")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    feature_names = X.columns.tolist()
    print(f"✅ Features: {len(feature_names)} columns")
    print(f"✅ Target: {target_col}")
    
    # 3. Encode target
    print("\n[3/6] Encoding target variable...")
    y_encoded, encoder = encode_target(y)
    
    # 4. Split data
    print("\n[4/6] Splitting data...")
    splits = split_data(X, y_encoded, test_size, val_size, random_state)
    
    # 5. Scale features
    print("\n[5/6] Scaling features...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        splits['X_train'],
        splits['X_val'],
        splits['X_test']
    )
    
    # Update splits with scaled data
    splits['X_train_scaled'] = X_train_scaled
    splits['X_val_scaled'] = X_val_scaled
    splits['X_test_scaled'] = X_test_scaled

    # 6. Save data
    if save_data:
        print("\n[6/6] Saving processed data...")
        
        # Prepare data for saving
        save_dict = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': splits['y_train'],
            'y_val': splits['y_val'],
            'y_test': splits['y_test']
        }
        
        save_processed_data(save_dict, feature_names)
        save_transformers(scaler, encoder)

    print("\n" + "="*60)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*60)
    
    return {
        'data': splits,
        'scaler': scaler,
        'encoder': encoder,
        'feature_names': feature_names
    }

# Example usage
if __name__ == "__main__":
    # Run complete pipeline
    result = preprocess_pipeline(
        raw_data_path='../../data/raw/Crop_recommendation.csv',
        target_col='label',
        test_size=0.15,
        val_size=0.15,
        random_state=42,
        save_data=True
    )
    
    print("\n📊 Preprocessing Summary:")
    print(f"   Features: {len(result['feature_names'])}")
    print(f"   Classes: {len(result['encoder'].classes_)}")
    print(f"   Train samples: {len(result['data']['y_train'])}")