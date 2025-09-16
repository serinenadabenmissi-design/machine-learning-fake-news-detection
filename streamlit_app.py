import streamlit as st
import joblib

# Load models
binary_model = joblib.load("binary_model.pkl")
multi_model = joblib.load("multi_model.pkl")

st.title('🎈 Fake News Detector')
st.info('This app uses ML models to classify news articles.')

# User input
title = st.text_input("Enter news title:")
text = st.text_area("Enter news text:")
domain_rank = st.number_input("Enter domain rank:", min_value=0, step=1)
country = st.text_input("Enter country:")

if st.button("Predict"):
    # نبني DataFrame صغير باش يوافق pipeline
    import pandas as pd
    input_data = pd.DataFrame([{
        "title": title,
        "text": text,
        "domain_rank": domain_rank,
        "country": country
    }])

    # Step 1: predict bs vs others
    binary_pred = binary_model.predict(input_data)[0]

    if binary_pred == "bs":
        st.error("⚠️ This looks like **BS News**")
    else:
        # Step 2: predict specific class
        multi_pred = multi_model.predict(input_data)[0]
        st.success(f"✅ Classified as: {multi_pred}")
