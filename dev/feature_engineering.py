import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

def feature_engineering(data):
    # Selecting useful features
    features = ["TransactionAmt", "card1", "card2", "addr1", "isFraud", "card4", "card6"]
    data = data[features]

    # Encoding categorical variables
    if "card4" in data.columns:
        data.loc[:, "card4"] = LabelEncoder().fit_transform(data["card4"].astype(str))
    if "card6" in data.columns:
        data.loc[:, "card6"] = LabelEncoder().fit_transform(data["card6"].astype(str))

    # Scaling numerical data
    if {"TransactionAmt", "card1", "card2", "addr1"}.issubset(data.columns):
        scaler = MinMaxScaler()
        data.loc[:, ["TransactionAmt", "card1", "card2", "addr1"]] = scaler.fit_transform(data[["TransactionAmt", "card1", "card2", "addr1"]])

    return data