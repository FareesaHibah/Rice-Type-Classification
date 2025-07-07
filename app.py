import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import base64
from sklearn.ensemble import RandomForestClassifier

# 1. Set custom background
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

# 2. Call background + apply frosted glass effect to main layout
set_background("bg.png")

st.markdown("""
    <style>
    .main > div {
        background-color: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# 3. App config and title
st.set_page_config(page_title="Rice Type Classifier", layout="centered")

st.markdown("""
    <h1 style='text-align:center; color:#00aaff;'>Rice Type Classification App</h1>
    <p style='text-align:center;'>Identify whether a rice grain is <strong>Cammeo</strong> or <strong>Osmancik</strong> based on its morphological features.</p>
""", unsafe_allow_html=True)

# 4. Input columns
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

# 5. Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# 6. Prediction
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

    # Display result with transparent box
    st.markdown(f"""
        <div style='background-color: rgba(255,255,255,0.7); padding: 1rem; border-radius: 10px; text-align: center;'>
            <h3>Predicted Rice Type: <b>{label}</b></h3>
        </div>
    """, unsafe_allow_html=True)

    # Probability bar chart
    fig, ax = plt.subplots()
    ax.bar(["Cammeo", "Osmancik"], probabilities, color=["orange", "green"])
    ax.set_ylabel("Prediction Probability")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# 7. Feature explanation
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

# 8. About section
with st.expander("About this App"):
    st.markdown("""
    - Built with **Streamlit**  
    - Uses a **Random Forest Classifier**  
    - Trained to classify rice grains as Cammeo or Osmancik  
    - Data sourced from UCI Machine Learning Repository
    """)

