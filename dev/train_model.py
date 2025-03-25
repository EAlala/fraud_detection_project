import joblib
import sqlite3
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, f1_score, precision_score, 
                           recall_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from dev.visualizations import plot_roc_curve

# --- Model Training ---
def train_model(x_train, y_train, model_type='logistic'):
    print(f"\n🏋️ Training {model_type} model...")
    
    if model_type == 'logistic':
        model = LogisticRegression(random_state=42, class_weight='balanced')
    elif model_type == 'xgboost':
        model = XGBClassifier(scale_pos_weight=100, random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(class_weight='balanced', random_state=42)
    
    model.fit(x_train, y_train)
    print(f"✅ {model_type.capitalize()} model trained successfully!")
    return model

# --- Evaluation ---
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Log to database
    log_metrics_to_db(metrics, str(model.__class__.__name__))
    
    print("\n📊 Model Evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    plot_roc_curve(y_test, y_pred_proba, metrics['roc_auc'])
    return metrics

def log_metrics_to_db(metrics, model_name, training_duration=None):
    conn = sqlite3.connect('model_performance.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        str(model_name),
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['roc_auc'],
        training_duration,
        "1.0"  # You can version your data
    ))
    
    conn.commit()
    conn.close()

# --- Data Splitting ---
def split_data(data, test_size=0.2, random_state=42):
    x = data.drop("isFraud", axis=1)
    y = data["isFraud"]
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

def save_model(model, filename="fraud_detection_model.pkl"):
    joblib.dump(model, filename)
    print(f"💾 Model saved to {filename}")