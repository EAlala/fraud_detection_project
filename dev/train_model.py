import joblib
import sqlite3
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, f1_score, precision_score, 
                           recall_score, roc_auc_score, average_precision_score)
from visualizations import plot_roc_curve, plot_pr_curve

# --- Data Splitting ---
def split_data(data, test_size=0.2, random_state=42):
    """Split data into training and test sets"""
    x = data.drop("isFraud", axis=1)
    y = data["isFraud"]
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

# --- Model Training ---
def train_model(x_train, y_train, model_type='logistic'):
    """Train specified model type with proper feature handling"""
    print(f"\n🏋️ Training {model_type} model...")
    
    if model_type == 'logistic':
        model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
    elif model_type == 'xgboost':
        model = XGBClassifier(
            scale_pos_weight=100,
            random_state=42,
            eval_metric='aucpr',
            use_label_encoder=False
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_estimators=100
        )
    
    model.fit(x_train, y_train)
    
    # Set feature names for XGBoost
    if model_type == 'xgboost':
        model.get_booster().feature_names = list(x_train.columns)
    
    print(f"✅ {model_type.capitalize()} model trained successfully!")
    return model

# --- Evaluation ---
def evaluate_model(model, x_test, y_test):
    """Evaluate model performance with comprehensive metrics"""
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print("\n📊 Model Evaluation:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot performance curves
    plot_roc_curve(y_test, y_pred_proba, metrics['roc_auc'])
    plot_pr_curve(y_test, y_pred_proba)
    
    return metrics

# --- Model Saving ---
def save_models(x_train, y_train):
    """Train and save all model types with proper feature handling"""
    os.makedirs("models", exist_ok=True)
    
    models = {
        'logistic': train_model(x_train, y_train, 'logistic'),
        'xgboost': train_model(x_train, y_train, 'xgboost'),
        'random_forest': train_model(x_train, y_train, 'random_forest')
    }
    
    for name, model in models.items():
        filename = f"models/{name}_model.pkl"
        joblib.dump(model, filename)
        print(f"💾 Saved {name} model to {filename}")
        
        # Evaluate and log metrics for each model
        x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )
        metrics = evaluate_model(model, x_test_split, y_test_split)
        log_metrics_to_db(metrics, name.capitalize())

# --- Metrics Logging ---
def log_metrics_to_db(metrics, model_name, training_duration=None):
    """Log model performance metrics to database"""
    conn = sqlite3.connect('model_performance.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        model_name,
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['roc_auc'],
        metrics['pr_auc'],
        training_duration,
        "1.0"  # Data version
    ))
    
    conn.commit()
    conn.close()

# --- Main Training Pipeline ---
if __name__ == "__main__":
    # Load and preprocess data
    from load_data import load_data
    from preprocess import preprocess_data
    
    print("🚀 Starting model training pipeline...")
    raw_data = load_data()
    processed_data = preprocess_data(raw_data)
    
    # Split data
    x_train, x_test, y_train, y_test = split_data(processed_data)
    
    # Train and save all models
    save_models(x_train, y_train)
    
    print("\n🎉 All models trained and saved successfully!")