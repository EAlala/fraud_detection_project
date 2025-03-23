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

# Line Chart: Trends in Transaction Amounts Over Time
st.subheader("Transaction Amounts Over Time")
fig, ax = plt.subplots()
ax.plot(data["TransactionDT"], data["TransactionAmt"], label="Transaction Amount")
ax.set_xlabel("Transaction Date")
ax.set_ylabel("Transaction Amount")
ax.legend()
st.pyplot(fig)

# Bar Chart: Fraud Rates by Card Type
st.subheader("Fraud Rates by Card Type")
fraud_by_card = data.groupby("card4")["isFraud"].mean().reset_index()
fig, ax = plt.subplots()
sns.barplot(x="card4", y="isFraud", data=fraud_by_card, ax=ax)
ax.set_xlabel("Card Type")
ax.set_ylabel("Fraud Rate")
st.pyplot(fig)

# Heatmap: High-Risk Transaction Patterns
st.subheader("High-Risk Transaction Patterns")
heatmap_data = data.pivot_table(index="card4", columns="card6", values="isFraud", aggfunc="mean")
fig, ax = plt.subplots()
sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd", ax=ax)
ax.set_xlabel("Card Category")
ax.set_ylabel("Card Type")
st.pyplot(fig)