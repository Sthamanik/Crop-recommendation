"""
Test inference system
"""
import sys
sys.path.append('../')

from src.models.predict import load_latest_model

print("="*70)
print("TESTING INFERENCE SYSTEM")
print("="*70)

# Load model
print("\n[1/4] Loading model...")
predictor = load_latest_model()

# Test single prediction
print("\n[2/4] Testing single prediction...")
sample_data = {
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 20.8,
    'humidity': 82.0,
    'ph': 6.5,
    'rainfall': 202.9
}

result = predictor.predict(sample_data, return_proba=True)
print(f"✅ Predicted crop: {result['crop']}")
print(f"   Confidence: {result['confidence']:.2%}")

# Test batch prediction
print("\n[3/4] Testing batch prediction...")
batch_data = [
    {'N': 90, 'P': 42, 'K': 43, 'temperature': 20.8, 'humidity': 82, 'ph': 6.5, 'rainfall': 203},
    {'N': 100, 'P': 82, 'K': 50, 'temperature': 27.4, 'humidity': 80, 'ph': 6.0, 'rainfall': 105},
    {'N': 20, 'P': 134, 'K': 200, 'temperature': 22.6, 'humidity': 92, 'ph': 5.9, 'rainfall': 113}
]

results = predictor.predict_batch(batch_data, return_proba=False)
print(f"✅ Batch predictions: {[r['crop'] for r in results]}")

# Test model info
print("\n[4/4] Testing model info...")
info = predictor.get_model_info()
print(f"✅ Model type: {info['model_type']}")
print(f"   Features: {len(info['feature_names'])}")
print(f"   Classes: {len(info['classes'])}")

print("\n" + "="*70)
print("ALL TESTS PASSED! 🎉")
print("="*70)