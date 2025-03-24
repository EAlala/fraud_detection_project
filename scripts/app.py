import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from security_monitoring import log_event, log_performance

# To view in browerser type " streamlit run (full path location of file)"'

# --- User Authentication ---
if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    st.login()
else:
    st.title("Fraud Detection System")
    st.write(f"Welcome, {st.session_state['username']}!")

    # Log user access
    log_event(f"User {st.session_state['username']} accessed the app.")

    # --- Existing App Code ---
    # (Keep all your existing code for transaction input, predictions, and visualizations here)

    # Example: Log a prediction event
    if st.button("Predict"):
        # Your existing prediction logic
        log_event("Prediction made by user.", level="info")

    # Example: Log performance metrics (if available)
    if "model_metrics" in st.session_state:
        log_performance(st.session_state["model_metrics"])

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

# Plot 1: Transaction Amounts Over Time
st.subheader("Transaction Amounts Over Time")
if "TransactionDT" in data.columns:
    fig1, ax1 = plt.subplots()
    ax1.plot(data["TransactionDT"], data["TransactionAmt"], label="Transaction Amount")
    ax1.set_xlabel("Transaction Date")
    ax1.set_ylabel("Transaction Amount")
    ax1.legend()
    st.pyplot(fig1)
else:
    st.warning("⚠️ Column 'TransactionDT' not found. Skipping this visualization.")

# Plot 2: Fraud Rates by Card Type
st.subheader("Fraud Rates by Card Type")
if "card4" in data.columns and "isFraud" in data.columns:
    fraud_by_card = data.groupby("card4")["isFraud"].mean().reset_index()
    fig2, ax2 = plt.subplots()
    sns.barplot(x="card4", y="isFraud", data=fraud_by_card, ax=ax2)
    ax2.set_xlabel("Card Type")
    ax2.set_ylabel("Fraud Rate")
    st.pyplot(fig2)
else:
    st.warning("⚠️ Required columns not found. Skipping this visualization.")

# Plot 3: High-Risk Transaction Patterns
st.subheader("High-Risk Transaction Patterns")
if "card4" in data.columns and "card6" in data.columns and "isFraud" in data.columns:
    heatmap_data = data.pivot_table(index="card4", columns="card6", values="isFraud", aggfunc="mean")
    fig3, ax3 = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd", ax=ax3)
    ax3.set_xlabel("Card Category")
    ax3.set_ylabel("Card Type")
    st.pyplot(fig3)
else:
    st.warning("⚠️ Required columns not found. Skipping this visualization.")