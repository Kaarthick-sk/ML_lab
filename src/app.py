# src/member5_app.py

import streamlit as st
import numpy as np
import joblib

model = joblib.load("../models/ridge_model.pkl")

st.title("💎 Diamond Price Prediction System")

page = st.sidebar.selectbox("Select Page",
                           ["Prediction", "EDA", "Results", "Workflow"])

# ---------------- Prediction ----------------
if page == "Prediction":
    st.header("Predict Price")

    carat = st.number_input("Carat")
    cut = st.number_input("Cut (0-4)")
    color = st.number_input("Color (0-6)")
    clarity = st.number_input("Clarity (0-6)")
    depth = st.number_input("Depth")
    table = st.number_input("Table")
    x = st.number_input("Length")
    y = st.number_input("Width")
    z = st.number_input("Height")

    volume = x * y * z

    if st.button("Predict"):
        features = np.array([[carat, cut, color, clarity,
                              depth, table, x, y, z, volume]])
        prediction = model.predict(features)
        st.success(f"Predicted Price: ${prediction.item():.2f}")

# ---------------- EDA ----------------
elif page == "EDA":
    st.header("EDA Visualizations")
    st.image("../outputs/carat_price.png")
    st.image("../outputs/heatmap.png")

# ---------------- Results ----------------
elif page == "Results":
    st.header("Model Comparison")
    with open("../outputs/results.txt") as f:
        st.text(f.read())

# ---------------- Workflow ----------------
elif page == "Workflow":
    st.header("Project Workflow")
    st.write("""
    1. Data Preprocessing  
    2. Feature Engineering  
    3. Model Training  
    4. Model Evaluation  
    5. Deployment using Streamlit  
    """)