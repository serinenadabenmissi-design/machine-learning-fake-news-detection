import streamlit as st
import cloudpickle
import pandas as pd

st.title("🎈 Fake News Detector")
st.info("This app detects if a news article is fake or not, and classifies it if real.")

# Load the preprocessor, normalizer, and models
with open("preprocessor.pkl", "rb") as f:
    preprocessor = cloudpickle.load(f)

with open("normalizer.pkl", "rb") as f:
    normalizer = cloudpickle.load(f)

with open("binary_model.pkl", "rb") as f:
    binary_model = cloudpickle.load(f)

with open("multi_model.pkl", "rb") as f:
    multi_model = cloudpickle.load(f)

# User inputs
title = st.text_input("Enter news title:")
text = st.text_area("Enter news text:")
domain_rank = st.number_input("Enter domain rank:", min_value=0, step=1)
country = st.text_input("Enter country:")

if st.button("Predict"):
    if not title or not text or not country:
        st.warning("Please fill in all fields.")
    else:
        # Prepare input_data
        input_data = pd.DataFrame([{
            "full_text": f"{title} {text}",
            "domain_rank": int(domain_rank),
            "country": str(country)
        }])

        # Transform and normalize the input
        input_processed = preprocessor.transform(input_data)
        input_normalized = normalizer.transform(input_processed)

        # Binary prediction
        try:
            binary_pred = binary_model.predict(input_normalized)[0]
        except Exception as e:
            st.error(f"Binary model error: {e}")
        else:
            if binary_pred == "bs":
                st.error("⚠️ This looks like **BS News**")
            else:
                # Multiclass prediction
                try:
                    multi_pred = multi_model.predict(input_normalized)[0]
                except Exception as e:
                    st.error(f"Multi-class model error: {e}")
                else:
                    st.success(f"✅ Classified as: {multi_pred}")

