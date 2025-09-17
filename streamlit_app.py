
import streamlit as st
import cloudpickle
import pandas as pd

# -------------------------------
# App title and info
# -------------------------------
st.title("🎈 Fake News Detector")
st.info("This app detects if a news article is fake or not, and classifies it if real.")

# -------------------------------
# Load the trained ML pipelines
# -------------------------------
with open("pipeline_binary.pkl", "rb") as f:
    pipeline_binary = cloudpickle.load(f)

with open("pipeline_multi.pkl", "rb") as f:
    pipeline_multi = cloudpickle.load(f)

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
    # Prepare input DataFrame for pipeline
    # Must match the columns used during training
    # -------------------------------
    input_data = pd.DataFrame([{
        "full_text": f"{title} {text}",
        "domain_rank": int(domain_rank),
        "country": str(country or "")
    }])

    # -------------------------------
    # Step 1: Binary classifier (Fake or Not)
    # -------------------------------
    try:
        binary_pred = pipeline_binary.predict(input_data)[0]
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
                multi_pred = pipeline_multi.predict(input_data)[0]
            except Exception as e:
                st.error(f"Multi-class model error: {e}")
            else:
                st.success(f"✅ Classified as: {multi_pred}")
