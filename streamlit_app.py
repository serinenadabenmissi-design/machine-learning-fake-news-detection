
import streamlit as st
import cloudpickle
import pandas as pd

with open("binary_model.pkl", "rb") as f:
    binary_model = cloudpickle.load(f)

with open("multi_model.pkl", "rb") as f:
    multi_model = cloudpickle.load(f)

st.title("🎈 Fake News Detector")
st.info("This app uses ML models to classify news articles.")

# User input
title = st.text_input("Enter news title:")
text = st.text_area("Enter news text:")
domain_rank = st.number_input("Enter domain rank:", min_value=0, step=1)
country = st.text_input("Enter country:")

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "title": title,
        "text": text,
        "domain_rank": domain_rank,
        "country": country
    }])

    # Step 1: binary classifier
    binary_pred = binary_model.predict(input_data)[0]

    if binary_pred == "bs":
        st.error("⚠️ This looks like **BS News**")
    else:
        # Step 2: multi-class classifier
        multi_pred = multi_model.predict(input_data)[0]
        st.success(f"✅ Classified as: {multi_pred}")
