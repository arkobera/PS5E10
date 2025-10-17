import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats
import mlflow
import dagshub
import os
import pickle
from dotenv import load_dotenv
load_dotenv()

# -----------------------------------------------------
# 🔹 DagsHub Initialization
# -----------------------------------------------------
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "arkobera"
repo_name = "PS5E10"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

import mlflow
import dagshub
import os
from dotenv import load_dotenv
load_dotenv()


dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "arkobera"
repo_name = "PS5E10"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

model_name = "my_model"
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

# -----------------------------------------------------
# 🔹 Mappings
# -----------------------------------------------------
rt = {
    'highway': 0,
    'rural': 1,
    'urban': 2
}

light = {
    'dim': 0,
    'daylight': 1,
    'night': 2
}

weather = {
    'foggy': 0,
    'clear': 1,
    'rainy': 2
}

tod = {
    'morning': 0,
    'evening': 1,
    'afternoon': 2
}

# -----------------------------------------------------
# 🔹 Helper Functions
# -----------------------------------------------------
def f(X):
    return (
        0.3 * X["curvature"] +
        0.2 * (X["lighting"] == light["night"]).astype(int) +
        0.1 * (X["weather"] != weather["clear"]).astype(int) +
        0.2 * (X["speed_limit"] >= 60).astype(int) +
        0.1 * (X["num_reported_accidents"] > 2).astype(int)
    )

def clip(f):
    """Apply clipping to keep y within 0–1."""
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu / sigma, (1 - mu) / sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu * (Phi_b - Phi_a) + sigma * (phi_a - phi_b) + 1 - Phi_b
    return clip_f

# -----------------------------------------------------
# 🔹 Streamlit UI
# -----------------------------------------------------
st.title("🚗 Accident Risk Prediction Interface")

st.markdown("Provide the following inputs to estimate **accident risk**:")

col1, col2 = st.columns(2)

with col1:
    road_type = st.selectbox("Road Type", list(rt.keys()))
    lighting = st.selectbox("Lighting Condition", list(light.keys()))
    weather_c = st.selectbox("Weather Condition", list(weather.keys()))
    time_of_day = st.selectbox("Time of Day", list(tod.keys()))
    speed_limit = st.number_input("Speed Limit (km/h)", min_value=0, max_value=200, value=45)
    curvature = st.number_input("Curvature", min_value=0.0, max_value=10.0, value=0.78, step=0.01)

with col2:
    num_lanes = st.number_input("Number of Lanes", min_value=1, max_value=10, value=1)
    road_signs_present = st.checkbox("Road Signs Present", value=True)
    public_road = st.checkbox("Public Road", value=False)
    holiday = st.checkbox("Holiday", value=True)
    school_season = st.checkbox("School Season", value=False)
    num_reported_accidents = st.number_input("Number of Reported Accidents", min_value=0, max_value=50, value=0)

# -----------------------------------------------------
# 🔹 Prepare Input
# -----------------------------------------------------
input_data = pd.DataFrame([{
    "road_type": road_type,
    "num_lanes": num_lanes,
    "curvature": curvature,
    "speed_limit": speed_limit,
    "lighting": lighting,
    "weather": weather_c,
    "road_signs_present": road_signs_present,
    "public_road": public_road,
    "time_of_day": time_of_day,
    "holiday": holiday,
    "school_season": school_season,
    "num_reported_accidents": num_reported_accidents
}])

# Map categorical columns
input_data["road_type"] = input_data["road_type"].map(rt)
input_data["lighting"] = input_data["lighting"].map(light)
input_data["weather"] = input_data["weather"].map(weather)
input_data["time_of_day"] = input_data["time_of_day"].map(tod)

# Compute y using your formula
z = clip(f)(input_data)
input_data["y"] = z.values

st.subheader("🧾 Processed Input Features")
st.dataframe(input_data)

# -----------------------------------------------------
# 🔹 Model Prediction (You’ll handle model loading)
# -----------------------------------------------------
st.markdown("---")
st.markdown("### 🔍 Model Prediction")

if st.button("Predict Accident Risk"):
    st.info("⚙️ Model loading ...")
    model_version = get_latest_model_version(model_name)
    model_uri = f'models:/{model_name}/{model_version}'
    # print(f"Fetching model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    prediction = model.predict(input_data)
    st.success(f"Predicted Accident Risk: {prediction[0]:.4f}")
