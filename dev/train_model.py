from datetime import datetime  
import uuid
import sqlite3
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score

# Config
CFG = {
    'models': {
        'rf': {'n_estimators': 200,          
            'max_depth': None,            
            'min_samples_split': 50,      
            'min_samples_leaf': 20,       
            'max_features': 'sqrt',      
            'class_weight': {0: 1, 1: 10},
            'bootstrap': True,
            'oob_score': True,            
            'n_jobs': -1,
            'random_state': 42},
        'lr': {'C': 0.1, 'max_iter': 500, 'class_weight': 'balanced', 'solver': 'lbfgs'}
    },
    'test_size': 0.2,
    'random_state': 42,
    'threshold': 0.50
}

# Load data
def load_data():
    from load_data import load_data as ld
    from preprocess import preprocess_data
    data = preprocess_data(ld())
    return (data[['TransactionAmt', 'card1', 'card2', 'addr1', 'dist1', 
                'D1', 'D2', 'card4', 'card6', 'ProductCD', 'isFraud']]
            .assign(isFraud=lambda x: x['isFraud'].gt(0).astype(int)))

# Threshold optimizer
def find_optimal_threshold(model, X_test, y_test):
    from sklearn.metrics import accuracy_score
    thresholds = np.linspace(0.1, 0.9, 50)
    best_acc = 0
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred = (model.predict_proba(X_test)[:,1] >= thresh).astype(int)
        acc = accuracy_score(y_test, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    
    return best_thresh, best_acc

# Train model
def train_model(X, y, model_type):
    model = (RandomForestClassifier(**CFG['models']['rf']) if model_type == 'rf' 
            else LogisticRegression(**CFG['models']['lr']))
    model.fit(X, y)
    return model

# Evaluate model
def evaluate(m, X, y):
    proba = m.predict_proba(X)[:,1]
    pred = (proba > CFG['threshold']).astype(int)
    return {
        'precision': precision_score(y, pred),
        'recall': recall_score(y, pred),
        'roc_auc': roc_auc_score(y, proba),
        'f1': f1_score(y, pred)
    }

# Save model
def save(m, name, X, metrics):
    # Ensure database and table exist
    from security_monitoring import init_metrics_db
    init_metrics_db()
    
    # Ensure models directory exists
    Path("models").mkdir(exist_ok=True)
    
    # Save model components
    model_path = f'models/{name}_model.pkl'
    joblib.dump(m, model_path)
    joblib.dump(list(X.columns), f'models/{name}_features.pkl')
    
    # Connect to database
    conn = sqlite3.connect('model_performance.db')
    cursor = conn.cursor()
    
    try:
        # Insert new record
        cursor.execute('''
            INSERT INTO model_versions (
                version_id, model_type, timestamp, 
                features_used, performance_metrics,
                training_data_size, path
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()),
            name,
            datetime.now().isoformat(),
            str(list(X.columns)),
            str(metrics),
            len(X),
            model_path
        ))
        conn.commit()
        print(f"Saved {name} model metrics to database")
    except Exception as e:
        print(f"Error saving to database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop('isFraud', axis=1), 
        data['isFraud'],
        test_size=CFG['test_size'],
        random_state=CFG['random_state'],
        stratify=data['isFraud']
    )
    
    for name in ['rf', 'lr']:
        print(f"\n{'='*50}")
        print(f"Training {name.upper()} model...")
        
        model = train_model(X_train, y_train, name)
        metrics = evaluate(model, X_test, y_test)
        
        # Print metrics with type checking
        print(f"\n{name.upper()} Metrics:")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")
        
        save(model, name, X_train, metrics)