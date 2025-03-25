from venv import logger
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
    try:
        logger.info("Loading data...")
        transaction_data = pd.read_csv("data/train_transaction.csv")
        identity_data = pd.read_csv("data/train_identity.csv")
        data = transaction_data.merge(identity_data, on="TransactionID", how="left")
        logger.info("Data loaded successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

# Load saved model
model = joblib.load("fraud_detection_model_v2.pkl")

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
    st.title(("Login"))
    
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

# Prediction section:
if st.button("Predict"):
    try:
        logger.info(f"Making prediction for transaction amount: {transaction_amt}")
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            log_event("Fraudulent transaction detected", level="warning")
            st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            logger.info("Transaction classified as safe")
            st.success("✅ Transaction is Safe.")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        st.error("Error processing transaction")
        
    # --- Visualizations in Sidebar ---
with st.sidebar:
    st.header("Analytics Dashboard")
    
    viz_option = st.selectbox("Choose Visualization", [
        "Transaction Amounts",
        "Fraud by Card Type",
        "Risk Patterns"
    ])
    
    if viz_option == "Transaction Amounts":
        st.subheader("Transaction Amounts Over Time")
        fig1, ax1 = plt.subplots()
        ax1.hist(data["TransactionAmt"], bins=50)
        ax1.set_xlabel("Amount")
        ax1.set_ylabel("Frequency")
        st.pyplot(fig1)
        
    elif viz_option == "Fraud by Card Type":
        st.subheader("Fraud Rates by Card Type")
        fig2, ax2 = plt.subplots()
        sns.barplot(x="card4", y="isFraud", data=data)
        ax2.set_xlabel("Card Type")
        ax2.set_ylabel("Fraud Rate")
        st.pyplot(fig2)
        
    elif viz_option == "Risk Patterns":
        st.subheader("High-Risk Patterns")
        fig3, ax3 = plt.subplots(figsize=(6,4))
        sns.heatmap(
            data.pivot_table(index="card4", columns="card6", values="isFraud", aggfunc="mean"),
            annot=True, cmap="Reds"
        )
        st.pyplot(fig3)