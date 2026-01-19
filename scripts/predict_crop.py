#!/usr/bin/env python3
"""
Command-line tool for crop recommendation.

Usage:
    python scripts/predict_crop.py --N 90 --P 42 --K 43 --temperature 20.8 --humidity 82 --ph 6.5 --rainfall 203
    
    python scripts/predict_crop.py --interactive
    
    python scripts/predict_crop.py --csv data/sample_input.csv
"""

import sys
import os
sys.path.append('../')

import argparse
import pandas as pd
from src.models.predict import load_latest_model


def predict_single(predictor, args):
    """Make single prediction from command-line arguments."""
    data = {
        'N': args.N,
        'P': args.P,
        'K': args.K,
        'temperature': args.temperature,
        'humidity': args.humidity,
        'ph': args.ph,
        'rainfall': args.rainfall
    }
    
    print("="*70)
    print("CROP RECOMMENDATION")
    print("="*70)
    
    print("\n📊 Input Conditions:")
    print(f"   Nitrogen (N):        {data['N']} kg/ha")
    print(f"   Phosphorus (P):      {data['P']} kg/ha")
    print(f"   Potassium (K):       {data['K']} kg/ha")
    print(f"   Temperature:         {data['temperature']}°C")
    print(f"   Humidity:            {data['humidity']}%")
    print(f"   pH:                  {data['ph']}")
    print(f"   Rainfall:            {data['rainfall']} mm")
    
    # Predict
    result = predictor.predict(data, return_proba=True)
    
    print("\n🌾 RECOMMENDATION:")
    print("="*70)
    print(f"   Best Crop: {result['crop'].upper()}")
    
    if 'confidence' in result:
        print(f"   Confidence: {result['confidence']:.1%}")
        
        if 'top_5' in result:
            print("\n📈 Alternative Crops:")
            for i, pred in enumerate(result['top_5'][1:], 2):  # Skip first (already shown)
                print(f"   {i}. {pred['crop']:15s}: {pred['probability']:.1%}")
    
    print("="*70)


def predict_interactive(predictor):
    """Interactive mode - prompt user for inputs."""
    print("="*70)
    print("INTERACTIVE CROP RECOMMENDATION")
    print("="*70)
    print("\nPlease enter the following soil and climate conditions:\n")
    
    try:
        data = {
            'N': float(input("Nitrogen (N) in kg/ha [0-150]: ")),
            'P': float(input("Phosphorus (P) in kg/ha [0-150]: ")),
            'K': float(input("Potassium (K) in kg/ha [0-210]: ")),
            'temperature': float(input("Temperature in °C [0-50]: ")),
            'humidity': float(input("Humidity in % [0-100]: ")),
            'ph': float(input("Soil pH [0-14]: ")),
            'rainfall': float(input("Rainfall in mm [0-350]: "))
        }
        
        # Predict
        result = predictor.predict(data, return_proba=True)
        
        print("\n" + "="*70)
        print("🌾 RECOMMENDATION:")
        print("="*70)
        print(f"\n   ✅ Recommended Crop: {result['crop'].upper()}")
        
        if 'confidence' in result:
            print(f"   📊 Confidence: {result['confidence']:.1%}")
            
            if 'top_5' in result:
                print("\n   📈 Top 5 Suitable Crops:")
                for i, pred in enumerate(result['top_5'], 1):
                    print(f"      {i}. {pred['crop']:15s}: {pred['probability']:.1%}")
        
        print("\n" + "="*70)
        
    except ValueError as e:
        print(f"\n❌ Error: Invalid input - {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def predict_from_csv(predictor, csv_path):
    """Make predictions from CSV file."""
    print(f"📄 Reading data from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"   Loaded {len(df)} samples")
        
        # Predict
        df_result = predictor.predict_from_dataframe(df, return_proba=False)
        
        # Save results
        output_path = csv_path.replace('.csv', '_predictions.csv')
        df_result.to_csv(output_path, index=False)
        
        print(f"\n✅ Predictions saved to: {output_path}")
        print(f"\n📊 Summary:")
        print(df_result['predicted_crop'].value_counts())
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Crop Recommendation System - Make predictions'
    )
    
    # Single prediction mode
    parser.add_argument('--N', type=float, help='Nitrogen content (kg/ha)')
    parser.add_argument('--P', type=float, help='Phosphorus content (kg/ha)')
    parser.add_argument('--K', type=float, help='Potassium content (kg/ha)')
    parser.add_argument('--temperature', type=float, help='Temperature (°C)')
    parser.add_argument('--humidity', type=float, help='Humidity (%)')
    parser.add_argument('--ph', type=float, help='Soil pH')
    parser.add_argument('--rainfall', type=float, help='Rainfall (mm)')
    
    # Alternative modes
    parser.add_argument('--interactive', action='store_true', 
                       help='Interactive mode - prompt for inputs')
    parser.add_argument('--csv', type=str, 
                       help='Path to CSV file with input data')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    predictor = load_latest_model()
    print()
    
    # Determine mode
    if args.interactive:
        predict_interactive(predictor)
    elif args.csv:
        predict_from_csv(predictor, args.csv)
    elif all([args.N, args.P, args.K, args.temperature, 
              args.humidity, args.ph, args.rainfall]):
        predict_single(predictor, args)
    else:
        parser.print_help()
        print("\n❌ Error: Please provide all required arguments or use --interactive mode")


if __name__ == "__main__":
    main()