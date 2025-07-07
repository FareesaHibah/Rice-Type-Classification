
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import base64

# Set background image
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)

# Transparent blurred card container
def apply_transparent_container():
    st.markdown("""
    <style>
    .app-container {
        background-color: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: auto;
        width: 90%;
        max-width: 1000px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

# --- Main App ---
set_background("bg.png")
apply_transparent_container()
st.markdown("<div class='app-container'>", unsafe_allow_html=True)

st.markdown('<h1 style="color:#2a5c2a;">Rice Type Classifier</h1>', unsafe_allow_html=True)
st.markdown("Enter the grain's morphological features to predict whether it's **Cammeo** or **Osmancik**.")

# Inputs
col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (mm²)", value=12000.0)
    perimeter = st.number_input("Perimeter (mm)", value=400.0)
    major_axis = st.number_input("Major Axis Length (mm)", value=200.0)
    minor_axis = st.number_input("Minor Axis Length (mm)", value=100.0)
with col2:
    eccentricity = st.number_input("Eccentricity", min_value=0.0, max_value=1.0, value=0.8)
    convex_area = st.number_input("Convex Area (mm²)", value=12500.0)
    extent = st.number_input("Extent", min_value=0.0, max_value=1.0, value=0.75)

model = load_model()

if st.button("Predict Rice Type"):
    features = pd.DataFrame([[area, perimeter, major_axis, minor_axis, eccentricity, convex_area, extent]],
                            columns=["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "Eccentricity", "ConvexArea", "Extent"])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    label = "Cammeo" if prediction == 0 else "Osmancik"
    st.success(f"The predicted rice type is: **{label}**")

    # Show probability bar
    fig, ax = plt.subplots()
    ax.bar(["Cammeo", "Osmancik"], proba, color=["#a2c957", "#5c8a8a"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Prediction Probability")
    st.pyplot(fig)

with st.expander("What Do These Features Mean?"):
    st.markdown("""
    - **Area**: Surface area of the rice grain  
    - **Perimeter**: Boundary length  
    - **Major Axis Length**: Longest dimension  
    - **Minor Axis Length**: Width of the grain  
    - **Eccentricity**: Oval-ness (0–1)  
    - **Convex Area**: Area of outer boundary  
    - **Extent**: Compactness in bounding box
    """)

st.markdown("</div>", unsafe_allow_html=True)
