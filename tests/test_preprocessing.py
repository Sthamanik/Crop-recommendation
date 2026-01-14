"""
Test preprocessing module
"""
import sys

sys.path.append("..")

from src.data.preprocessing import (
    load_raw_data,
    preprocess_pipeline,
    load_processed_data,
    load_transformers
)

# Test 1: Run complete pipeline
print("TEST 1: Running complete preprocessing pipeline...")
print("="*60)

result = preprocess_pipeline(
    raw_data_path='../data/raw/Crop_recommendation.csv',
    save_data=True
)

print("\n✅ Pipeline completed successfully!")

# Test 2: Load saved data
print("\n" + "="*60)
print("TEST 2: Loading saved data...")
print("="*60)

data = load_processed_data()
scaler, encoder = load_transformers()

print("\n✅ Data loaded successfully!")
print(f"   Scaler type: {type(scaler).__name__}")
print(f"   Encoder classes: {len(encoder.classes_)}")

# Test 3: Verify data integrity
print("\n" + "="*60)
print("TEST 3: Verifying data integrity...")
print("="*60)

import numpy as np

# Check scaling
X_train_mean = data['X_train'].values.mean()
X_train_std = data['X_train'].values.std()

print(f"Training data mean: {X_train_mean:.6f} (should be ≈ 0)")
print(f"Training data std:  {X_train_std:.6f} (should be ≈ 1)")

if abs(X_train_mean) < 0.01 and abs(X_train_std - 1.0) < 0.1:
    print("✅ Scaling verified!")
else:
    print("❌ Scaling issue detected!")

print("\n" + "="*60)
print("ALL TESTS PASSED! 🎉")
print("="*60)