import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Get the absolute path to the directory where this script (games.py) is located
# This ensures reliable path finding regardless of where Streamlit is executed.
BASE_DIR = Path(__file__).resolve().parent

# Define the path to the CSV file
DATA_PATH = BASE_DIR / "Data" / "data.csv"

MAX_SAMPLES = 2000

# Features to display in the quiz (all columns except the target)
FEATURE_COLUMNS = [
    'road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting',
    'weather', 'road_signs_present', 'public_road', 'time_of_day', 
    'holiday', 'school_season', 'num_reported_accidents'
]

def load_data():
    """Loads and caches the data for efficient use."""
    # Check for existence using the Path object
    if not DATA_PATH.exists():
        st.error(f"Error: Data file not found at {DATA_PATH}. Please ensure the 'Data' folder and 'data.csv' exist.")
        return None
    try:
        # Pass the Path object directly to read_csv
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def start_quiz(df, num_questions):
    """Initializes the quiz state by sampling data."""
    
    # 1. Sample twice the required number of rows for N pairs of scenarios
    if len(df) < 2 * num_questions:
        st.error(f"Cannot sample {2 * num_questions} rows. Only {len(df)} available.")
        return
    
    # Randomly sample the required number of total rows (2 * num_questions)
    sampled_data = df.sample(n=2 * num_questions, random_state=np.random.randint(0, 10000))
    
    # Reset quiz state
    st.session_state.quiz_started = True
    st.session_state.current_question = 0
    st.session_state.score = 0
    
    # Reshape sampled data into pairs (N questions, each containing 2 scenarios)
    quiz_pairs = []
    for i in range(num_questions):
        scenario_A = sampled_data.iloc[2 * i]
        scenario_B = sampled_data.iloc[2 * i + 1]
        quiz_pairs.append((scenario_A, scenario_B))
        
    st.session_state.quiz_data = quiz_pairs

def display_question():
    """Renders the current question and handles user input."""
    
    q_index = st.session_state.current_question
    scenario_A, scenario_B = st.session_state.quiz_data[q_index]
    
    st.subheader(f"Question {q_index + 1}: Which road is safer?")
    
    # Display the two scenarios side-by-side
    col_A, col_B = st.columns(2)
    
    with col_A:
        st.markdown("#### 🅰️ Scenario A")
        # Display features, excluding 'accident_risk'
        st.dataframe(scenario_A[FEATURE_COLUMNS].to_frame().T, use_container_width=True, hide_index=True)
    
    with col_B:
        st.markdown("#### 🅱️ Scenario B")
        st.dataframe(scenario_B[FEATURE_COLUMNS].to_frame().T, use_container_width=True, hide_index=True)

    # User choice buttons
    st.markdown("---")
    choice = st.radio(
        "Select the safer road (one with the **lower** accident risk):",
        ["Scenario A", "Scenario B"],
        key=f"choice_{q_index}",
        horizontal=True
    )
    
    if st.button("Submit Answer", key=f"submit_{q_index}"):
        
        # Determine the correct answer
        risk_A = scenario_A['accident_risk']
        risk_B = scenario_B['accident_risk']
        
        # The safer road is the one with MINIMUM risk
        is_A_safer = risk_A < risk_B
        
        # Determine user's answer
        user_chose_A = (choice == "Scenario A")
        
        # Check if the user was correct
        is_correct = (user_chose_A and is_A_safer) or (not user_chose_A and not is_A_safer)

        if is_correct:
            st.session_state.score += 1
            st.success("✅ Correct! You chose the safer path.")
        else:
            correct_choice = "Scenario A" if is_A_safer else "Scenario B"
            st.error(f"❌ Incorrect. The safer choice was **{correct_choice}**.")
            
        # Display detailed risk for learning
        st.info(f"Actual Risk: **Scenario A** = {risk_A:.4f}, **Scenario B** = {risk_B:.4f}")

        # Move to the next question after a short delay (simulated by rerunning)
        st.session_state.current_question += 1
        st.rerun() # Rerun to display the next state or the final score


def play_quiz_page():
    st.title("🧠 Road Safety Quiz")
    st.markdown("""
        Test your judgment! Analyze two hypothetical road scenarios and select the one you believe has the **lower accident risk**. 
        This quiz uses real data points from the underlying dataset.
    """)

    # --- Load Data ---
    df = load_data()
    if df is None:
        return # Stop if data loading failed

    # --- Initialize Session State ---
    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
        st.session_state.current_question = 0
        st.session_state.score = 0
        st.session_state.quiz_data = []

    # --- Quiz Setup/Control ---
    if not st.session_state.quiz_started:
        
        st.subheader("Setup Quiz")
        num_questions = st.number_input(
            "Number of Questions", 
            min_value=1, 
            max_value=MAX_SAMPLES // 2, 
            value=5, 
            step=1,
            help=f"Enter the number of road-comparison pairs to answer (max {MAX_SAMPLES//2})."
        )
        
        if st.button("Start New Quiz"):
            start_quiz(df, num_questions)
            st.rerun()

    # --- Active Quiz ---
    elif st.session_state.quiz_started:
        
        total_questions = len(st.session_state.quiz_data)
        current_q_index = st.session_state.current_question
        
        # --- Progress Bar ---
        progress_value = (current_q_index) / total_questions
        st.progress(progress_value, text=f"Attempted: {current_q_index} / {total_questions}")
        
        # Check if the quiz is finished
        if current_q_index >= total_questions:
            # --- Quiz End ---
            st.balloons()
            st.markdown(f"## 🏆 Quiz Finished!")
            final_score = st.session_state.score
            st.success(f"Your final score is **{final_score}** out of **{total_questions}**.")
            st.markdown("---")
            if st.button("Restart Quiz"):
                # Clear state and rerun
                st.session_state.quiz_started = False
                st.rerun()
                
        else:
            # --- Continue Quiz ---
            display_question()