import streamlit as st
import cloudpickle
import pandas as pd

# -------------------------------
# App title and info
# -------------------------------
st.title("🎈 Fake News Detector")
st.info("This app detects if a news article is fake or not, and classifies it if real.")

# -------------------------------
# Load the trained ML models
# -------------------------------
with open("binary_model.pkl", "rb") as f:
    binary_model = cloudpickle.load(f)

with open("multi_model.pkl", "rb") as f:
    multi_model = cloudpickle.load(f)

# -------------------------------
# Define the columns expected by the model
# Make sure these match the columns used during training
# -------------------------------
expected_cols = ["title", "text", "domain_rank", "country", "full_text"]

# -------------------------------
# User inputs
# -------------------------------
title = st.text_input("Enter news title:")
text = st.text_area("Enter news text:")
domain_rank = st.number_input("Enter domain rank:", min_value=0, step=1)
country = st.text_input("Enter country:")

# -------------------------------
# When the Predict button is clicked
# -------------------------------
if st.button("Predict"):
    # -------------------------------
    # Create derived column full_text by combining title and text
    # -------------------------------
    full_text = f"{title} {text}"

    # -------------------------------
    # Create a DataFrame with all expected columns
    # Text columns are strings, numeric columns are numeric
    # -------------------------------
    input_data = pd.DataFrame([{
        "title": str(title or ""),
        "text": str(text or ""),
        "domain_rank": int(domain_rank),
        "country": str(country or ""),
        "full_text": str(full_text)
    }])

    # -------------------------------
    # Show input DataFrame for debugging (optional)
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
            # Step 2: Multi-class classifier if not fake
            # -------------------------------
            try:
                multi_pred = multi_model.predict(input_data)[0]
            except Exception as e:
                st.error(f"Multi-class model error: {e}")
            else:
                st.success(f"✅ Classified as: {multi_pred}")

