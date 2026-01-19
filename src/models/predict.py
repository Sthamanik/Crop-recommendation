"""
Inference module for Crop Recommendation model.

This module provides functionality to:
- Load trained model and preprocessors
- Make predictions on new data
- Return predictions with confidence scores
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, List, Union, Tuple, Optional


class CropPredictor:
    """
    Crop recommendation predictor class.
    
    Handles loading model, preprocessing, and making predictions.
    """
    
    def __init__(self, model_path: str, scaler_path: str, encoder_path: str):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model (.pkl)
            scaler_path: Path to scaler object (.pkl)
            encoder_path: Path to label encoder (.pkl)
        """
        self.model = self._load_model(model_path)
        self.scaler = self._load_scaler(scaler_path)
        self.encoder = self._load_encoder(encoder_path)
        
        # Store feature names (in order)
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        print("✅ CropPredictor initialized successfully!")
        print(f"   Model type: {type(self.model).__name__}")
        print(f"   Number of classes: {len(self.encoder.classes_)}")
    
    def _load_model(self, path: str):
        """Load trained model."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)
    
    def _load_scaler(self, path: str):
        """Load scaler object."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found: {path}")
        return joblib.load(path)
    
    def _load_encoder(self, path: str):
        """Load label encoder."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Encoder file not found: {path}")
        return joblib.load(path)
    
    def _validate_input(self, data: Dict) -> None:
        """
        Validate input data.
        
        Args:
            data: Dictionary with feature values
            
        Raises:
            ValueError: If input is invalid
        """
        # Check all features present
        missing = set(self.feature_names) - set(data.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        # Check for valid ranges
        ranges = {
            'N': (0, 150),
            'P': (0, 150),
            'K': (0, 210),
            'temperature': (0, 50),
            'humidity': (0, 100),
            'ph': (0, 14),
            'rainfall': (0, 350)
        }
        
        for feature, (min_val, max_val) in ranges.items():
            value = data[feature]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"{feature}={value} outside valid range [{min_val}, {max_val}]"
                )
    
    def _prepare_input(self, data: Dict) -> np.ndarray:
        """
        Prepare input data for prediction.
        
        Args:
            data: Dictionary with feature values
            
        Returns:
            Scaled numpy array ready for prediction
        """
        # Create array in correct order
        X = np.array([[data[feat] for feat in self.feature_names]])
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, data: Dict, return_proba: bool = False) -> Dict:
        """
        Make prediction for given soil and climate conditions.
        
        Args:
            data: Dictionary with features (N, P, K, temperature, humidity, ph, rainfall)
            return_proba: Whether to return probability scores
            
        Returns:
            Dictionary with prediction and optional probabilities
            
        Example:
            >>> predictor = CropPredictor(...)
            >>> result = predictor.predict({
            ...     'N': 90, 'P': 42, 'K': 43,
            ...     'temperature': 20.8, 'humidity': 82.0,
            ...     'ph': 6.5, 'rainfall': 202.9
            ... })
            >>> print(result['crop'])
            'rice'
        """
        # Validate
        self._validate_input(data)
        
        # Prepare
        X = self._prepare_input(data)
        
        # Predict
        prediction = self.model.predict(X)[0]
        crop_name = self.encoder.inverse_transform([prediction])[0]
        
        result = {
            'crop': crop_name,
            'crop_index': int(prediction)
        }
        
        # Add probabilities if requested
        if return_proba:
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(X)[0]
                
                # Get top 5 predictions
                top_indices = np.argsort(probas)[-5:][::-1]
                top_crops = self.encoder.inverse_transform(top_indices)
                top_probas = probas[top_indices]
                
                result['confidence'] = float(probas[prediction])
                result['top_5'] = [
                    {'crop': crop, 'probability': float(prob)}
                    for crop, prob in zip(top_crops, top_probas)
                ]
            else:
                result['confidence'] = 1.0  # SVM without probability
        
        return result
    
    def predict_batch(
        self, 
        data_list: List[Dict], 
        return_proba: bool = False
    ) -> List[Dict]:
        """
        Make predictions for multiple inputs.
        
        Args:
            data_list: List of input dictionaries
            return_proba: Whether to return probabilities
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(data, return_proba) for data in data_list]
    
    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        return_proba: bool = False
    ) -> pd.DataFrame:
        """
        Make predictions from a pandas DataFrame.
        
        Args:
            df: DataFrame with feature columns
            return_proba: Whether to return probabilities
            
        Returns:
            DataFrame with predictions added
        """
        predictions = []
        
        for _, row in df.iterrows():
            data = row.to_dict()
            result = self.predict(data, return_proba)
            predictions.append(result['crop'])
        
        df_result = df.copy()
        df_result['predicted_crop'] = predictions
        
        return df_result
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if available.
        
        Returns:
            DataFrame with feature importances or None
        """
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            print("⚠️  Model doesn't support feature importance")
            return None
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': type(self.model).__name__,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'num_classes': len(self.encoder.classes_),
            'classes': self.encoder.classes_.tolist(),
            'supports_probability': hasattr(self.model, 'predict_proba')
        }
        
        return info


def load_latest_model(models_dir: str = '../models') -> CropPredictor:
    """
    Load the most recently saved model.
    
    Args:
        models_dir: Directory containing saved models
        
    Returns:
        CropPredictor instance
    """
    import glob
    
    # Find latest model file
    model_files = glob.glob(os.path.join(models_dir, 'final_*.pkl'))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    
    # Get most recent
    latest_model = max(model_files, key=os.path.getctime)
    
    # Standard paths for scaler and encoder
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
    
    print(f"Loading model: {os.path.basename(latest_model)}")
    
    return CropPredictor(latest_model, scaler_path, encoder_path)


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("CROP PREDICTOR - INFERENCE MODULE")
    print("="*70)
    
    # Load model
    predictor = load_latest_model()
    
    # Example prediction
    sample_data = {
        'N': 90,
        'P': 42,
        'K': 43,
        'temperature': 20.8,
        'humidity': 82.0,
        'ph': 6.5,
        'rainfall': 202.9
    }
    
    print("\n📊 Sample Prediction:")
    print(f"Input: {sample_data}")
    
    result = predictor.predict(sample_data, return_proba=True)
    
    print(f"\n🌾 Recommended Crop: {result['crop']}")
    if 'confidence' in result:
        print(f"   Confidence: {result['confidence']:.2%}")
        
        if 'top_5' in result:
            print("\n📈 Top 5 Predictions:")
            for i, pred in enumerate(result['top_5'], 1):
                print(f"   {i}. {pred['crop']:15s}: {pred['probability']:.2%}")
    
    # Model info
    print("\n" + "="*70)
    info = predictor.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        if key != 'classes':
            print(f"   {key}: {value}")