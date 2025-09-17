
import streamlit as st
import cloudpickle
import pandas as pd

# -------------------------------
# Streamlit app title and info
# -------------------------------
st.title("🎈 Fake News Detector")
st.info("This app uses ML models to classify news articles.")

# -------------------------------
# Load the trained models
# -------------------------------
with open("binary_model.pkl", "rb") as f:
    binary_model = cloudpickle.load(f)

with open("multi_model.pkl", "rb") as f:
    multi_model = cloudpickle.load(f)

# -------------------------------
# Define expected columns for the model
# Adjust this list according to your training data
# -------------------------------
expected_cols = ["title", "text", "domain_rank", "country", "full_text"]  
# 'full_text' is required by the model, combining title + text
# Add more columns if your model was trained with them

# -------------------------------
# User inputs
# -------------------------------
title = st.text_input("Enter news title:")
text = st.text_area("Enter news text:")
domain_rank = st.number_input("Enter domain rank:", min_value=0, step=1)
country = st.text_input("Enter country:")

# -------------------------------
# When Predict button is clicked
# -------------------------------
if st.button("Predict"):
    # -------------------------------
    # Create derived columns if needed
    # -------------------------------
    full_text = f"{title} {text}"  # combine title and text

    # -------------------------------
    # Create a DataFrame with all expected columns
    # Fill missing ones with empty string or 0
    # -------------------------------
    input_data = pd.DataFrame([{
        col: locals().get(col, "")  # get variable if exists, else empty string
        for col in expected_cols
    }])

    # -------------------------------
    # Show input DataFrame for debugging
    # -------------------------------
    st.write("Input DataFrame:", input_data)

    # -------------------------------
    # Step 1: Binary classifier (Fake or Not)
    # -------------------------------
    try:
        binary_pred = binary_model.predict(input_data)[0]
    except Exception as e:
        st.error(f"Binary model error: {e}")
    else:
        if binary_pred == "bs":
            st.error("⚠️ This looks like **BS News**")
        else:
            # -------------------------------
            # Step 2: Multi-class classifier (if not fake)
            # -------------------------------
            try:
                multi_pred = multi_model.predict(input_data)[0]
            except Exception as e:
                st.error(f"Multi-class model error: {e}")
            else:
                st.success(f"✅ Classified as: {multi_pred}")

