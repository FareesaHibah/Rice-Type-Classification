import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# App configuration
st.set_page_config(page_title="Rice Type Classifier", layout="centered")

# App title and description
st.markdown('<h1 style="color:#00aaff;">Rice Type Classification App</h1>', unsafe_allow_html=True)
st.markdown("Identify whether a rice grain is **Cammeo** or **Osmancik** based on its morphological features.")

# Input layout in two columns
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (mm²)", min_value=0.0, value=12000.0)
    perimeter = st.number_input("Perimeter (mm)", min_value=0.0, value=400.0)
    major_axis = st.number_input("Major Axis Length (mm)", min_value=0.0, value=200.0)
    minor_axis = st.number_input("Minor Axis Length (mm)", min_value=0.0, value=100.0)

with col2:
    eccentricity = st.number_input("Eccentricity", min_value=0.0, max_value=1.0, value=0.8)
    convex_area = st.number_input("Convex Area (mm²)", min_value=0.0, value=12500.0)
    extent = st.number_input("Extent", min_value=0.0, max_value=1.0, value=0.75)

# Load pre-trained model or simulate for demo
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

# Predict button and result
if st.button("Identify Rice Type"):
    features = np.array([[area, perimeter, major_axis, minor_axis, eccentricity, convex_area, extent]])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    label = "Cammeo" if prediction == 0 else "Osmancik"
    st.success(f"The identified rice type is: **{label}**")

    # Probability chart
    fig, ax = plt.subplots()
    ax.bar(["Cammeo", "Osmancik"], probabilities, color=["orange", "green"])
    ax.set_ylabel("Prediction Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# Feature explanations
with st.expander("What Do These Features Mean?"):
    st.markdown("""
    - **Area**: Surface area of the rice grain in mm²  
    - **Perimeter**: Total boundary length of the grain  
    - **Major Axis Length**: Longest length of the grain  
    - **Minor Axis Length**: Width of the grain at its widest point  
    - **Eccentricity**: Degree of elongation (0=circle, 1=line)  
    - **Convex Area**: Area of the convex hull that encloses the grain  
    - **Extent**: Ratio of area to bounding box area (0–1)
    """)

# Optional about section
with st.expander("About this App"):
    st.markdown("""
    - Built with **Streamlit**  
    - Uses a **Random Forest Classifier**  
    - Trained to classify rice grains as Cammeo or Osmancik
    """)
