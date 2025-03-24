import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import feature_engineering

def clean_data(data):
    # Convert datetime
    if 'TransactionDT' in data.columns:
        data['TransactionDT'] = pd.to_datetime(data['TransactionDT'], unit='s')
    
    # Handle missing values
    numeric_cols = data.select_dtypes(include=np.number).columns
    cat_cols = data.select_dtypes(exclude=np.number).columns
    
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])
    data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])
    
    return data

def preprocess_data(data):
    print("\n🧹 Preprocessing data...")
    
    pipeline = Pipeline([
        ('clean', FunctionTransformer(clean_data)),
        ('feature_engineering', FunctionTransformer(feature_engineering))
    ])
    
    processed_data = pipeline.fit_transform(data)
    print("✅ Data preprocessing complete!")
    return processed_data