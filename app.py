import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
import pandas as pd
import pickle
import datetime
import numpy as np
import plotly.express as px
from cleaning import process_text
from prediction import get_predictions

# Load model and transformer
model_path = "models/lr_final_model.pkl"
transformer_path = "models/transformer.pkl"

loaded_model = pickle.load(open(model_path, 'rb'))
loaded_transformer = pickle.load(open(transformer_path, 'rb'))

# Path for Excel file to store prompts and responses
log_file_path = "logs/prediction_history.xlsx"

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Function to load existing Excel file or create a new one
def load_or_create_log_file():
    try:
        return pd.read_excel(log_file_path)
    except FileNotFoundError:
        # Create a new DataFrame with required columns
        df = pd.DataFrame(columns=[
            'Timestamp', 
            'Prompt', 
            'Prediction', 
            'Classification',
            'Fake_Probability'
        ])
        return df

# Function to save data to Excel
def save_prediction_to_log(prompt, prediction, fake_probability):
    df = load_or_create_log_file()
    
    # Create new row with current data
    new_row = {
        'Timestamp': datetime.datetime.now(),
        'Prompt': prompt,
        'Prediction': 'Real' if prediction == 1 else 'Fake',
        'Classification': prediction,
        'Fake_Probability': fake_probability
    }
    
    # Append new row to DataFrame
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save to Excel
    df.to_excel(log_file_path, index=False)
    return df

# Page configuration
st.set_page_config(page_title="Fake News Classifier", layout="centered")

# App Title
st.title("📰 Fake News Classifier App")
st.markdown("Using Machine Learning to detect **Fake News**.")

# Tabs for different views
tab1, tab2 = st.tabs(["Predict News", "View History"])

with tab1:
    # Input box
    namequery = st.text_input("Enter a news headline or statement")
    
    if st.button("Predict"):
        if namequery.strip() == "":
            st.warning("Please enter a news statement!")
        else:
            clean_data = process_text(str([namequery]))
            test_features = loaded_transformer.transform([" ".join(clean_data)])
            prediction = get_predictions(loaded_model, test_features)
            
            # Get prediction probability (fake news probability)
            # For logistic regression, we can get probability using predict_proba
            prediction_proba = loaded_model.predict_proba(test_features)
            fake_probability = prediction_proba[0][0]  # Probability of class 0 (Fake)
            
            # Save prediction to Excel
            updated_df = save_prediction_to_log(namequery, prediction, fake_probability)
            
            st.markdown("---")
            st.subheader("📊 Prediction Result")
            st.write(f"**Input:** {namequery}")
            
            if prediction == 0:
                st.error(f"🚫 This news is **Fake** (Confidence: {fake_probability:.2%})")
                
                # Display gauge chart for fake news probability
                fig = px.pie(
                    values=[fake_probability, 1-fake_probability],
                    names=['Fake', 'Real'],
                    hole=0.7,
                    color_discrete_sequence=['#FF5252', '#4CAF50'],
                    title="Fake News Probability"
                )
                fig.update_layout(
                    annotations=[dict(text=f"{fake_probability:.1%}", x=0.5, y=0.5, font_size=25, showarrow=False)]
                )
                st.plotly_chart(fig)
                
            elif prediction == 1:
                st.success(f"✅ This news is **Real** (Confidence: {1-fake_probability:.2%})")

with tab2:
    st.subheader("📋 Prediction History")
    
    try:
        history_df = pd.read_excel(log_file_path)
        if len(history_df) > 0:
            st.dataframe(history_df)
            
            # Create a histogram of predictions
            st.subheader("Prediction Distribution")
            fig = px.histogram(
                history_df, 
                x="Prediction",
                color="Prediction",
                color_discrete_map={"Fake": "#FF5252", "Real": "#4CAF50"},
                title="Distribution of Real vs Fake News Predictions"
            )
            st.plotly_chart(fig)
            
            # Timeline of predictions
            st.subheader("Prediction Timeline")
            timeline_fig = px.scatter(
                history_df,
                x="Timestamp",
                y="Fake_Probability",
                color="Prediction",
                color_discrete_map={"Fake": "#FF5252", "Real": "#4CAF50"},
                title="Fake News Probability Timeline",
                hover_data=["Prompt"]
            )
            st.plotly_chart(timeline_fig)
        else:
            st.info("No prediction history available yet. Make some predictions first!")
    except FileNotFoundError:
        st.info("No prediction history available yet. Make some predictions first!")