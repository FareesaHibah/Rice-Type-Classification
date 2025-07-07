import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import base64

def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("bg.png")  # your image file here
st.markdown("""
    <style>
    .main > div {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

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

# Load pre-trained model or simulate a model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")
model = load_model()

# Prediction
if st.button("Predict Rice Type"):
    features = np.array([[area, perimeter, major_axis, minor_axis, eccentricity, convex_area, extent]])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    if probabilities[0] > 0.8:
        label = "Cammeo"
    elif probabilities[1] > 0.8:
        label = "Osmancik"
    else:
        label = "Uncertain"
    st.success(f"The predicted rice type is: **{label}**")

    st.markdown(f"""
        <h3 style='color:#006400; text-align:center;'>
            Predicted Rice Type: <b>{label}</b>
        </h3>
    """, unsafe_allow_html=True)

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
