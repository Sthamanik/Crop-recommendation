# 🌾 Crop Recommendation System

A machine learning system that recommends the most suitable crop to grow based on soil and climate conditions.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Results & Insights](#results--insights)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project uses machine learning to recommend optimal crops for farmers based on:
- **Soil nutrients**: Nitrogen (N), Phosphorus (P), Potassium (K)
- **Climate conditions**: Temperature, Humidity, Rainfall
- **Soil properties**: pH level

The system achieves **99%+ accuracy** in recommending the right crop from 22 different options.

## ✨ Features

- ✅ **High Accuracy**: 99%+ test accuracy
- ✅ **22 Crop Types**: Covers major crops including rice, wheat, cotton, fruits, etc.
- ✅ **Real-time Predictions**: Fast inference (< 100ms per prediction)
- ✅ **Confidence Scores**: Provides top-5 predictions with probability scores
- ✅ **Multiple Interfaces**: 
  - Command-line tool
  - Python API
  - Interactive mode
  - Batch predictions from CSV
- ✅ **Production Ready**: Complete with preprocessing pipeline and model artifacts

## 📊 Dataset

- **Source**: [Kaggle - Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **Size**: 2,200 samples
- **Features**: 7 (N, P, K, temperature, humidity, pH, rainfall)
- **Target**: 22 crop types (perfectly balanced - 100 samples each)
- **Quality**: No missing values, no duplicates

### Crops Covered
Rice, Maize, Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- Conda (recommended) or pip

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/crop-recommendation.git
cd crop-recommendation
```

2. **Create conda environment**
```bash
conda env create -f environment.yml
conda activate crop-rec
```

**OR using pip:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python -c "import sklearn, xgboost; print('✅ Setup complete!')"
```

## 💻 Usage

### Command Line Interface

**Single Prediction:**
```bash
python scripts/predict_crop.py \
  --N 90 \
  --P 42 \
  --K 43 \
  --temperature 20.8 \
  --humidity 82 \
  --ph 6.5 \
  --rainfall 203
```

**Interactive Mode:**
```bash
python scripts/predict_crop.py --interactive
```

**Batch Predictions from CSV:**
```bash
python scripts/predict_crop.py --csv data/sample_input.csv
```

### Python API
```python
from src.models.predict import load_latest_model

# Load model
predictor = load_latest_model()

# Make prediction
result = predictor.predict({
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 20.8,
    'humidity': 82.0,
    'ph': 6.5,
    'rainfall': 202.9
}, return_proba=True)

print(f"Recommended Crop: {result['crop']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📈 Model Performance

### Final Model: XGBoost Classifier

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 99.09% |
| **Test F1-Score** | 99.08% |
| **Cross-Validation (5-fold)** | 98.97% ± 0.52% |
| **Inference Time** | < 100ms |

### Model Comparison

| Model | Test Accuracy | F1-Score |
|-------|--------------|----------|
| **XGBoost** ⭐ | **99.09%** | **99.08%** |
| Random Forest | 98.48% | 98.47% |
| KNN | 97.58% | 97.56% |
| SVM | 97.27% | 97.25% |
| Logistic Regression | 94.55% | 94.53% |

### Key Insights

1. **Potassium (K)** is the most important feature (F-score: 27,238)
2. All crops achieve >95% individual accuracy
3. Model generalizes well (CV ≈ Test performance)
4. No signs of overfitting

## 📁 Project Structure
```
crop-recommendation/
│
├── app/
│   └── streamlit_app.py        # application
|
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Preprocessed train/val/test splits
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Data preprocessing
│   ├── 03_modeling.ipynb       # Baseline model comparison
│   └── 04_hyperparameter_tuning.ipynb  # Model optimization
│
├── src/
│   ├── data/
│   │   └── preprocessing.py    # Data preprocessing utilities
│   └── models/
│       ├── train.py           # Model training utilities
│       ├── evaluate.py        # Evaluation & visualization
│       └── predict.py         # Inference engine
│
├── models/                     # Saved models and artifacts
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── final_model.pkl
│
├── scripts/
│   └── predict_crop.py        # Command-line prediction tool
│
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment
└── README.md                  # Project documentation
```

## 🛠️ Technologies Used

- **Python 3.9+**
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn
- **Utilities**: Jupyter, Joblib

## 🔍 Results & Insights

### Feature Importance
```
1. Potassium (K):     45.2%
2. Humidity:          18.7%
3. Phosphorus (P):    14.3%
4. Nitrogen (N):      10.8%
5. Rainfall:           7.2%
6. Temperature:        2.9%
7. pH:                 0.9%
```

### Crop-Specific Patterns

- **High Potassium**: Apple, Grapes (200 kg/ha)
- **High Nitrogen**: Cotton, Coffee, Banana (100+ kg/ha)
- **High Rainfall**: Rice (236mm), Jute, Coconut (175mm+)
- **Low Humidity**: Chickpea (17%), Kidney Beans (22%)
- **Alkaline pH**: Chickpea (7.3), Black Gram (7.1)

### Why PCA Was Not Used

After evaluation, dimensionality reduction (PCA) was deemed unnecessary because:
- Only 7 features (not high-dimensional)
- Model achieves 99%+ accuracy without it
- Feature interpretability crucial for agricultural applications
- Minimal multicollinearity (max correlation: 0.74)
- Feature importance provides actionable insights

## 🚀 Future Improvements

- [ ] Deploy as REST API (Flask/FastAPI)
- [ ] Add mobile app interface
- [ ] Incorporate weather forecast API
- [ ] Add fertilizer recommendation
- [ ] Support for more crop varieties
- [ ] Multi-language support
- [ ] Integration with IoT soil sensors

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: [Sthamanik](https://github.com/Sthamanik)
- LinkedIn: [Manik Kumar Shrestha](https://www.linkedin.com/in/manik-kumar-shrestha-b9b1b0276/)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Dataset from [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- Inspired by precision agriculture initiatives
- Thanks to the open-source ML community

## Deployed App
- Using streamlit: Link to app - https://stham-crop-recommendation.streamlit.app/
---

⭐ **Star this repo** if you find it helpful!

📫 **Questions?** Open an issue or reach out!