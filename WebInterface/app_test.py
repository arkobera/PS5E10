import streamlit as st
import mlflow
import dagshub
import os
from dotenv import load_dotenv

# Import page functions from the separate files
from home import home_page
from games import play_quiz_page

load_dotenv()

# -----------------------------------------------------
# 🔹 Global Setup & DagsHub Initialization
# -----------------------------------------------------
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    # This warning will show up regardless of page if the token is missing
    st.sidebar.warning("MLflow integration disabled: CAPSTONE_TEST environment variable is not set.")
    
# Only set MLflow environment variables if the token exists
if dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "arkobera"
    repo_name = "PS5E10"
    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


# -----------------------------------------------------
# 🔹 Main Application Logic
# -----------------------------------------------------
def main():
    # Set global page configuration
    st.set_page_config(layout="wide", page_title="Road Risk App")
    
    # Sidebar Navigation Control
    page = st.sidebar.radio("Navigation", ["Home", "Play Quiz"])

    # Route to the selected page function
    if page == "Home":
        home_page()
    elif page == "Play Quiz":
        play_quiz_page()

if __name__ == "__main__":
    main()