import pandas as pd
import os
from datetime import datetime

def load_data():
    print("🔄 Loading data...")
    
    # Check for cached processed data
    if os.path.exists("data/processed_data.parquet"):
        data = pd.read_parquet("data/processed_data.parquet")
        print("✅ Loaded cached processed data!")
        return data
    
    # Load raw data
    transaction_data = pd.read_csv("data/train_transaction.csv")
    identity_data = pd.read_csv("data/train_identity.csv")
    
    # Merge datasets
    data = transaction_data.merge(
        identity_data, 
        on="TransactionID", 
        how="left",
        suffixes=('', '_identity')
    )
    
    # Cache the raw merged data
    data.to_parquet("data/raw_merged.parquet")
    print("✅ Data loaded and merged successfully!")
    return data