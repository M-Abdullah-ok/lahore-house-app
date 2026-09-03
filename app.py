import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import requests

MODEL_PATH = "lahore_house_price_model.pkl"
MODEL_URL = "https://github.com/M-Abdullah-ok/lahore-house-app/releases/download/v1/lahore_house_price_model.pkl"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first run only)..."):
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

@st.cache_resource
def load_model():
    download_model()
    data = joblib.load(MODEL_PATH)
    return data["model"], data["columns"]

model, model_columns = load_model()

location_cols = [c for c in model_columns if c.startswith("Location_")]
locations = sorted([c.replace("Location_", "") for c in location_cols])

type_cols = [c for c in model_columns if c.startswith("Type_")]
types = sorted([c.replace("Type_", "") for c in type_cols]) + ["House"]

purpose_cols = [c for c in model_columns if c.startswith("Purpose_")]
purposes = sorted([c.replace("Purpose_", "") for c in purpose_cols]) + ["For Sale"]

st.set_page_config(page_title="Lahore House Price Predictor", page_icon="🏠")
st.title("🏠 Lahore House Price Predictor")
st.write("Enter house details to get an estimated price.")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (Marla)", min_value=1.0, max_value=100.0, value=10.0, step=0.5)
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=15, value=4)
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=15, value=4)

with col2:
    built_year = st.number_input("Built Year", min_value=1950, max_value=2026, value=2020)
    location = st.selectbox("Location", locations)
    house_type = st.selectbox("Type", types)
    purpose = st.selectbox("Purpose", purposes)

if st.button("Predict Price", type="primary"):
    new_house = pd.DataFrame([{
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Built Year": built_year,
        "Location": location,
        "Type": house_type,
        "Purpose": purpose,
    }])

    new_house = pd.get_dummies(
        new_house,
        columns=["Location", "Type", "Purpose"],
        dtype=int
    )

    new_house = new_house.reindex(columns=model_columns, fill_value=0)

    prediction_log = model.predict(new_house)
    prediction = np.expm1(prediction_log)

    st.success(f"### Estimated Price: PKR {prediction[0]:,.0f}")

    crore = prediction[0] / 10_000_000
    st.write(f"(~{crore:.2f} Crore)")