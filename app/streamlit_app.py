"""
Streamlit Web Interface for Crop Recommendation System

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
sys.path.append('..')

import streamlit as st
import pandas as pd
from src.models.predict import load_latest_model

# Page config
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        margin-top: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        background-color: #E8F5E9;
        border: 2px solid #4CAF50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem 2rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return load_latest_model()

try:
    predictor = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")

# Header
st.markdown('<h1 class="main-header">🌾 Crop Recommendation System</h1>', unsafe_allow_html=True)
st.markdown("### Get AI-powered crop recommendations based on your soil and climate conditions")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x200.png?text=Agriculture+AI", use_container_width=True)
    st.markdown("## About")
    st.info("""
    This system uses machine learning to recommend the most suitable crop 
    based on:
    - Soil nutrients (N, P, K)
    - Climate conditions
    - Soil pH
    
    **Accuracy**: 99%+
    
    **Crops**: 22 varieties
    """)

# Main content
if model_loaded:
    # Input section
    st.markdown('<h2 class="sub-header">📊 Enter Field Conditions</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Soil Nutrients")
        nitrogen = st.number_input(
            "Nitrogen (N) - kg/ha",
            min_value=0.0,
            max_value=150.0,
            value=90.0,
            help="Nitrogen content in soil (0-150 kg/ha)"
        )
        
        phosphorus = st.number_input(
            "Phosphorus (P) - kg/ha",
            min_value=0.0,
            max_value=150.0,
            value=42.0,
            help="Phosphorus content in soil (0-150 kg/ha)"
        )
        
        potassium = st.number_input(
            "Potassium (K) - kg/ha",
            min_value=0.0,
            max_value=210.0,
            value=43.0,
            help="Potassium content in soil (0-210 kg/ha)"
        )
    
    with col2:
        st.markdown("#### Climate Conditions")
        temperature = st.number_input(
            "Temperature - °C",
            min_value=0.0,
            max_value=50.0,
            value=20.8,
            help="Average temperature (0-50°C)"
        )
        
        humidity = st.number_input(
            "Humidity - %",
            min_value=0.0,
            max_value=100.0,
            value=82.0,
            help="Relative humidity (0-100%)"
        )
        
        rainfall = st.number_input(
            "Rainfall - mm",
            min_value=0.0,
            max_value=350.0,
            value=202.9,
            help="Average rainfall (0-350mm)"
        )
    
    with col3:
        st.markdown("#### Soil Properties")
        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=6.5,
            help="Soil pH level (0-14)"
        )
        
        st.markdown("#### ")
        st.markdown("#### ")
        predict_button = st.button("🔍 Get Recommendation", use_container_width=True)
    
    # Prediction
    if predict_button:
        try:
            # Prepare input
            input_data = {
                'N': nitrogen,
                'P': phosphorus,
                'K': potassium,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            }
            
            # Make prediction
            with st.spinner('Analyzing conditions...'):
                result = predictor.predict(input_data, return_proba=True)
            
            # Display results
            st.markdown('<h2 class="sub-header">🎯 Recommendation Results</h2>', unsafe_allow_html=True)
            
            # Main recommendation
            st.markdown(f"""
            <div class="result-box">
                <h2 style="color: #2E7D32; margin: 0;">Recommended Crop: {result['crop'].upper()}</h2>
                <p style="font-size: 1.2rem; margin-top: 1rem;">
                    Confidence: <strong>{result.get('confidence', 1.0):.1%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Top 5 predictions
            if 'top_5' in result:
                st.markdown("#### Alternative Crops (Top 5)")
                
                top_5_df = pd.DataFrame(result['top_5'])
                top_5_df['probability'] = top_5_df['probability'].apply(lambda x: f"{x:.1%}")
                top_5_df.columns = ['Crop', 'Suitability']
                
                st.table(top_5_df)
            
            # Input summary
            with st.expander("📋 View Input Summary"):
                input_df = pd.DataFrame([input_data])
                st.table(input_df.T.rename(columns={0: 'Value'}))
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Made with ❤️ using Streamlit | Data from Kaggle</p>
    <p>⭐ Star on <a href="https://github.com/Sthamanik/crop-recommendation">GitHub</a></p>
</div>
""", unsafe_allow_html=True)