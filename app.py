
import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# ------------------------------
# Load trained model
# ------------------------------
with open("xgb_credit_model.pkl", "rb") as f:
    model = pickle.load(f)

explainer = shap.TreeExplainer(model)

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

st.title("💳 Credit Card Default Risk Prediction")
st.write("Predict default probability and explain decisions using SHAP")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Customer Details")

LIMIT_BAL = st.sidebar.number_input("Credit Limit", 10000, 1000000, 200000)
AGE = st.sidebar.slider("Age", 18, 80, 35)
PAY_0 = st.sidebar.selectbox("Last Payment Delay (PAY_0)", [-2, -1, 0, 1, 2, 3, 4])
BILL_AMT1 = st.sidebar.number_input("Last Bill Amount", 0, 1000000, 50000)
PAY_AMT1 = st.sidebar.number_input("Last Payment Amount", 0, 500000, 10000)

# ------------------------------
# Input DataFrame
# ------------------------------
input_df = pd.DataFrame({
    "LIMIT_BAL": [LIMIT_BAL],
    "AGE": [AGE],
    "PAY_0": [PAY_0],
    "BILL_AMT1": [BILL_AMT1],
    "PAY_AMT1": [PAY_AMT1]
})

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Default Risk"):
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction")
    st.metric("Default Probability", f"{prob*100:.2f}%")

    if prob > 0.5:
        st.error("High Risk Customer")
    elif prob > 0.3:
        st.warning("Medium Risk Customer")
    else:
        st.success("Low Risk Customer")

    # ------------------------------
    # SHAP Explanation
    # ------------------------------
    st.subheader("🔍 SHAP Explanation")

    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots()
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_df.iloc[0],
        matplotlib=True,
        show=False
    )
    st.pyplot(fig)
