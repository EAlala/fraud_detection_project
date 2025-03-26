import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Missing data cleaning
def clean_data(data):
    if data is None:
        raise ValueError("Input data is None!")
    
    for col in data.columns:
        # Handle numerical columns (fill NaN with median)
        if data[col].dtype != "object":
            data.loc[:, col] = data[col].fillna(data[col].median())
        # Handle categorical columns (fill NaN with mode)
        else:
            data.loc[:, col] = data[col].fillna(data[col].mode()[0])
    
    return data

def encode_features(data):
    # Convert categorical features into numerical ones
    encoder = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = encoder.fit_transform(data[col])  
    return data

def scale_features(data):
    # Scale numerical features using StandardScaler
    scaler = StandardScaler()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data

def preprocess_data(data):
    # Apply full preprocessing pipeline
    print("\n🧹 Data cleaning process...")
    data = clean_data(data)
    
    # Ensure isFraud is properly formatted as binary integers
    if 'isFraud' in data.columns:
        # Convert to integer first in case it's boolean or string
        data['isFraud'] = data['isFraud'].astype(int)
        
        # Verify we only have 0s and 1s
        unique_values = data['isFraud'].unique()
        if not set(unique_values).issubset({0, 1}):
            print(f"⚠️ Unexpected values in isFraud: {unique_values}")
            # Force binary classification by thresholding if needed
            data['isFraud'] = (data['isFraud'] > 0).astype(int)
        
        print("✅ Converted isFraud to binary integers (0/1)")
        print(f"Final value counts:\n{data['isFraud'].value_counts()}")
    
    data = encode_features(data)
    data = scale_features(data)
    return data