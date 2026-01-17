"""
Test modeling modules
"""
import sys
sys.path.append('../')

import numpy as np
from src.data.preprocessing import load_processed_data, load_transformers
from src.models.train import get_default_models, train_all_models, compare_models

print("="*70)
print("TESTING MODELING MODULES")
print("="*70)

# Load data
print("\n[1/3] Loading data...")
data = load_processed_data()
scaler, encoder = load_transformers()

X_train = data['X_train'].values
X_val = data['X_val'].values
y_train = data['y_train'].values.ravel()
y_val = data['y_val'].values.ravel()

print(f"✅ Data loaded: Train {X_train.shape}, Val {X_val.shape}")

# Get models
print("\n[2/3] Getting default models...")
models = get_default_models()
print(f"✅ Loaded {len(models)} models")

# Train (just one model for quick test)
print("\n[3/3] Training test model...")
from src.models.train import train_single_model

result = train_single_model(
    models['Random Forest'],
    X_train, y_train,
    X_val, y_val,
    'Random Forest'
)

print(f"\n✅ Test completed!")
print(f"   Accuracy: {result['metrics']['accuracy']:.4f}")

print("\n" + "="*70)
print("ALL TESTS PASSED! 🎉")
print("="*70)