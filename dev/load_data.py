import pandas as pd
import os
from datetime import datetime

def load_data():
    print("🔄 Loading data...")
    
    def load_data():
        print("🔄 Loading data...")
        if os.path.exists("data/processed_data.parquet"):
            data = pd.read_parquet("data/processed_data.parquet")
            # Validate target column exists and is numeric
            if 'isFraud' not in data.columns:
                raise ValueError("Target column 'isFraud' missing")
            if not pd.api.types.is_numeric_dtype(data['isFraud']):
                raise ValueError("isFraud must be numeric")
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