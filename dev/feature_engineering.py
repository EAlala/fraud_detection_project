import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFECV, SelectKBest, f_classif

# Configuration
FEATURE_CONFIG = {
    "numerical": ["TransactionAmt", "card1", "card2", "addr1", "dist1", "D1", "D2"],
    "categorical": ["card4", "card6", "ProductCD"],
    "datetime": ["TransactionDT"],
    "target": "isFraud"
}

# Feature selection
def select_features(X, y):
    selector = RFECV(
        estimator=LogisticRegression(max_iter=1000),
        min_features_to_select=20,
        cv=3
    )
    return selector.fit_transform(X, y)

# Feature engineering
def feature_engineering(data):
    # Select features
    features = FEATURE_CONFIG["numerical"] + FEATURE_CONFIG["categorical"] + [FEATURE_CONFIG["target"]]
    data = data[features].copy()
    
    # Handle missing values
    for col in FEATURE_CONFIG["numerical"]:
        data[col] = data[col].fillna(data[col].median())
    
    for col in FEATURE_CONFIG["categorical"]:
        data[col] = data[col].fillna("UNKNOWN")
    
    # Feature encoding
    for col in FEATURE_CONFIG["categorical"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
    
    # Feature scaling
    scaler = StandardScaler()
    data[FEATURE_CONFIG["numerical"]] = scaler.fit_transform(data[FEATURE_CONFIG["numerical"]])
    
    # Feature selection
    selector = SelectKBest(f_classif, k=10)
    X = data.drop(FEATURE_CONFIG["target"], axis=1)
    y = data[FEATURE_CONFIG["target"]]
    X_new = selector.fit_transform(X, y)
    
    selected_features = X.columns[selector.get_support()]
    data = pd.concat([pd.DataFrame(X_new, columns=selected_features), y], axis=1)
    
    return data