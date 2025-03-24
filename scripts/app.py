import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from security_monitoring import log_event, log_performance, login
import logging

# To view in browerser type " streamlit run (streamlit run c:/Users/yeai2_6rsknlh/OneDrive/Visual/fraud_detection_project/scripts/app.py)"'

# Configure logging (if not already configured in security_monitoring.py)
import logging
logging.basicConfig(
    filename="app.log",  # Log file name
    level=logging.INFO,  # Logging level
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    filemode="a",  # Append mode
)

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

# Load model and data
@st.cache_data
def load_resources():
    model = joblib.load("fraud_detection_model.pkl")
    data = pd.read_csv("data/train_transaction.csv")  # Adjust path as needed
    return model, data

model, data = load_resources()

# --- New Login Layout ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Fraud Detection System Login")
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                if username == "admin" and password == "password123":
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")

    st.stop()  # Don't show rest of app until logged in

# Only show the main app content if the user is logged in
if st.session_state["logged_in"]:
    st.write(f"Welcome, {st.session_state['username']}!")

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
        "card6": [card6_encoded],
        "TransactionDT": [0] 
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

    # Log performance metrics (example)
    metrics = {
        "accuracy": 0.95,
        "precision": 0.90,
        "recall": 0.85,
        "f1_score": 0.87,
        "roc_auc": 0.94
    }
    log_performance(metrics)

else:
    # Show a message prompting the user to log in
    st.warning("Please log in to access the Fraud Detection System.")