import pandas as pd

def load_data():
    # Display Loading
    print("🔄 Loading data...")

    # Load the transaction and identity datasets
    transaction_data = pd.read_csv("data/train_transaction.csv")
    identity_data = pd.read_csv("data/train_identity.csv")

    # Display tha data was loaded 
    print("✅ Data loaded successfully!")

    # Merge transaction and identity datasets
    data = transaction_data.merge(identity_data, on="TransactionID", how="left")

    # Display Preview
    print("📊 Data Preview:\n")
    print(data.head())

    return data