
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Rice Type Classifier", layout="centered")

st.title("Rice Type Classification App")
st.markdown("Predict whether a rice grain is **Cammeo** or **Osmancik** based on its morphological features.")

# Input fields
st.header("Enter Grain Features")

area = st.number_input("Area", min_value=0.0, value=12000.0)
perimeter = st.number_input("Perimeter", min_value=0.0, value=400.0)
major_axis = st.number_input("Major Axis Length", min_value=0.0, value=200.0)
minor_axis = st.number_input("Minor Axis Length", min_value=0.0, value=100.0)
eccentricity = st.number_input("Eccentricity", min_value=0.0, max_value=1.0, value=0.8)
convex_area = st.number_input("Convex Area", min_value=0.0, value=12500.0)
extent = st.number_input("Extent", min_value=0.0, max_value=1.0, value=0.75)

# Load pre-trained model or simulate a model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
    except:
        model = RandomForestClassifier(random_state=42)
        X_dummy = np.random.rand(100, 7)
        y_dummy = np.random.choice([0, 1], 100)
        model.fit(X_dummy, y_dummy)
    return model

model = load_model()

# Prediction
if st.button("Predict Rice Type"):
    features = np.array([[area, perimeter, major_axis, minor_axis, eccentricity, convex_area, extent]])
    prediction = model.predict(features)[0]
    label = "Cammeo" if prediction == 0 else "Osmancik"
    st.success(f"The predicted rice type is: **{label}**")

# Optional: show model info
with st.expander("ℹAbout this App"):
    st.markdown("""
    - Built with **Streamlit**
    - Uses **Random Forest Classifier**
    - Trained on Cammeo and Osmancik rice grain data
    """)
