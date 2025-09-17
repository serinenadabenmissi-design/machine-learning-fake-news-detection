
import streamlit as st
import pandas as pd
import cloudpickle

# -------------------------------
# App title and info
# -------------------------------
st.title("🎈 Fake News Detector")
st.info("Enter news details to classify if it's fake (bs) or real. If real, classify the type.")

# -------------------------------
# Load trained pipelines
# -------------------------------
try:
    with open("pipeline_binary.pkl", "rb") as f:
        pipeline_binary = cloudpickle.load(f)
    with open("pipeline_multi.pkl", "rb") as f:
        pipeline_multi = cloudpickle.load(f)
except FileNotFoundError:
    st.error("Pipeline files not found! Upload pipeline_binary.pkl and pipeline_multi.pkl in the same folder.")
    st.stop()

# -------------------------------
# User input
# -------------------------------
title = st.text_input("Enter news title:")
text = st.text_area("Enter news text:")
domain_rank = st.number_input("Enter domain rank:", min_value=0, step=1)
country = st.text_input("Enter country:")

# -------------------------------
# Predict button
# -------------------------------
if st.button("Predict"):
    if not title or not text or not country:
        st.warning("Please fill in all fields.")
    else:
        # Prepare DataFrame for pipeline (must match training columns)
        input_data = pd.DataFrame([{
            "full_text": f"{title} {text}",
            "domain_rank": int(domain_rank),
            "country": str(country)
        }])

        # -------------------------------
        # Step 1: Binary classification
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
                # Step 2: Multiclass prediction
                # -------------------------------
                try:
                    multi_pred = pipeline_multi.predict(input_data)[0]
                except Exception as e:
                    st.error(f"Multiclass model error: {e}")
                else:
                    st.success(f"✅ Classified as: {multi_pred}")

