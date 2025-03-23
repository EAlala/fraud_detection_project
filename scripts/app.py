import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# To view in browerser type " streamlit run (full path location of file)"'

# Load the data
@st.cache_data  # Cache the data to avoid reloading on every interaction
def load_data():
    # Load the transaction and identity datasets
    transaction_data = pd.read_csv("data/train_transaction.csv")
    identity_data = pd.read_csv("data/train_identity.csv")

    # Merge transaction and identity datasets
    data = transaction_data.merge(identity_data, on="TransactionID", how="left")

    return data

data = load_data()

# Load saved model
model = joblib.load("fraud_detection_model.pkl")

# Title for app
st.title("Fraud Detection System")

# Main app functionality
st.header("Enter Transaction Details")
transaction_amt = st.number_input("Transaction Amount", min_value=0.0)
card1 = st.number_input("card1", min_value=0.0)
card2 = st.number_input("card2", min_value=0.0)
addr1 = st.number_input("address", min_value=0.0)
card4 = st.selectbox("Card Type", ["Visa", "MasterCard", "American Express", "Discover"])
card6 = st.selectbox("Card Category", ["Credit", "Debit"])

# Convert categorical inputs to numerical values
card4_mapping = {"Visa": 0, "MasterCard": 1, "American Express": 2, "Discover": 3}
card6_mapping = {"Credit": 0, "Debit": 1}

card4_encoded = card4_mapping[card4]
card6_encoded = card6_mapping[card6]

# Create a DataFrame from the input data
input_data = pd.DataFrame({
    "TransactionAmt": [transaction_amt],
    "card1": [card1],
    "card2": [card2],
    "addr1": [addr1],
    "card4": [card4_encoded],
    "card6": [card6_encoded]
})

if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Safe.")

# Visualizations using the loaded data
st.header("Data Visualizations")
plot_transaction_amounts_over_time(data)
plot_fraud_rates_by_card_type(data)
plot_high_risk_transaction_patterns(data)