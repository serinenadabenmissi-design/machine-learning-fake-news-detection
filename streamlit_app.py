import streamlit as st
import cloudpickle
import pandas as pd

st.title("🎈 Fake News Detector")
st.info("This app detects if a news article is fake or not, and classifies it if real.")

# Load your trained ML models (with pipeline inside)
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
        # Prepare input_data for the pipeline
        input_data = pd.DataFrame([{
            "full_text": f"{title} {text}",
            "domain_rank": int(domain_rank),
            "country": str(country)
        }])

        st.write("Input DataFrame:", input_data)

        # Binary prediction
        try:
            binary_pred = binary_model.predict(input_data)[0]
        except Exception as e:
            st.error(f"Binary model error: {e}")
        else:
            if binary_pred == "bs":
                st.error("⚠️ This looks like **BS News**")
            else:
                # Multiclass prediction
                try:
                    multi_pred = multi_model.predict(input_data)[0]
                except Exception as e:
                    st.error(f"Multi-class model error: {e}")
                else:
                    st.success(f"✅ Classified as: {multi_pred}")

