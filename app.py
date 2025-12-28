
import streamlit as st
import pandas as pd
import xgboost as xgb

st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

@st.cache_resource
def load_model():
    booster = xgb.Booster()
    booster.load_model("xgb_credit_model.json")
    return booster

model = load_model()

st.title("💳 Credit Card Default Risk Prediction")
st.write("Predict default probability using XGBoost")

st.sidebar.header("Customer Details")

LIMIT_BAL = st.sidebar.number_input("Credit Limit", 10000, 1000000, 200000)
AGE = st.sidebar.slider("Age", 18, 80, 35)
PAY_0 = st.sidebar.selectbox("Last Payment Delay (PAY_0)", [-2, -1, 0, 1, 2, 3, 4])
BILL_AMT1 = st.sidebar.number_input("Last Bill Amount", 0, 1000000, 50000)
PAY_AMT1 = st.sidebar.number_input("Last Payment Amount", 0, 500000, 10000)

input_df = pd.DataFrame({
    "LIMIT_BAL": [LIMIT_BAL],
    "AGE": [AGE],
    "PAY_0": [PAY_0],
    "BILL_AMT1": [BILL_AMT1],
    "PAY_AMT1": [PAY_AMT1]
})

if st.button("Predict Default Risk"):
    dmatrix = xgb.DMatrix(input_df)
    prob = model.predict(dmatrix)[0]

    st.metric("Default Probability", f"{prob*100:.2f}%")

    if prob > 0.5:
        st.error("High Risk Customer")
    elif prob > 0.3:
        st.warning("Medium Risk Customer")
    else:
        st.success("Low Risk Customer")
