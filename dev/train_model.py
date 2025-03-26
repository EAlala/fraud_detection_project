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
from visualizations import plot_roc_curve

# --- Data Splitting ---
def split_data(data, test_size=0.2, random_state=42):
    """Split data into training and test sets with target validation"""
    if 'isFraud' not in data.columns:
        raise ValueError("Target column 'isFraud' not found in data")
        
    x = data.drop("isFraud", axis=1)
    y = data["isFraud"]
    
    # Convert target to proper binary format
    y = y.astype(float)  # First convert to float to handle any numeric types
    
    # Diagnostic output
    print("\n🔍 Target Variable Analysis:")
    print(f"Original dtype: {y.dtype}")
    print(f"Unique values before conversion: {np.unique(y)}")
    print(f"Value counts:\n{y.value_counts()}")
    
    # Convert to binary (0/1) ensuring proper classification labels
    y_binary = np.where(y > 0.5, 1, 0).astype(int)
    
    print("\n✅ Converted target to binary:")
    print(f"New unique values: {np.unique(y_binary)}")
    print(f"New value counts:\n{pd.Series(y_binary).value_counts()}")
    
    return train_test_split(x, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary)

# --- Model Training ---
def train_model(x_train, y_train, model_type='logistic'):
    """Train specified model type with enhanced validation"""
    print(f"\n🏋️ Training {model_type} model...")
    
    # Final validation check
    unique_labels = np.unique(y_train)
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"Invalid labels found: {unique_labels}. Must be binary (0/1)")
    
    if model_type == 'logistic':
        model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000,
            solver='liblinear'  # More stable for binary classification
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
            class_weight='balanced_subsample',
            random_state=42,
            n_estimators=100
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    try:
        model.fit(x_train, y_train)
        print(f"✅ {model_type.capitalize()} model trained successfully!")
        
        # Set feature names for XGBoost
        if model_type == 'xgboost':
            model.get_booster().feature_names = list(x_train.columns)
        
        return model
    except Exception as e:
        print(f"❌ Error training {model_type} model: {str(e)}")
        raise

# --- Evaluation ---
def evaluate_model(model, x_test, y_test):
    """Evaluate model performance with comprehensive metrics"""
    # Verify test labels are binary
    if not set(np.unique(y_test)).issubset({0, 1}):
        y_test = np.where(y_test > 0.5, 1, 0).astype(int)
    
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
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric.capitalize()}: {value:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot performance curves
    plot_roc_curve(y_test, y_pred_proba, metrics['roc_auc'])
    
    return metrics

# --- Model Saving ---
def save_models(x_train, y_train):
    """Train and save all model types with versioning"""
    os.makedirs("models", exist_ok=True)
    model_version = datetime.now().strftime("%Y%m%d_%H%M")
    
    models = {
        'logistic': train_model(x_train, y_train, 'logistic'),
        'xgboost': train_model(x_train, y_train, 'xgboost'),
        'random_forest': train_model(x_train, y_train, 'random_forest')
    }
    
    for name, model in models.items():
        filename = f"models/{name}_model_v{model_version}.pkl"
        joblib.dump(model, filename)
        print(f"💾 Saved {name} model to {filename}")
        
        # Evaluate and log metrics
        x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )
        metrics = evaluate_model(model, x_test_split, y_test_split)
        log_metrics_to_db(metrics, f"{name.capitalize()} v{model_version}")

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
    print("🚀 Starting model training pipeline...")
    
    try:
        from load_data import load_data
        from preprocess import preprocess_data
        
        raw_data = load_data()
        processed_data = preprocess_data(raw_data)
        
        # Split data with automatic binary conversion
        x_train, x_test, y_train, y_test = split_data(processed_data)
        
        # Train and save all models
        save_models(x_train, y_train)
        
        print("\n🎉 All models trained and saved successfully!")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise