import joblib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from visualizations import plot_roc_curve

# Split data into training and testing sets
def split_data(data, test_size=.2, random_state=42):

    # Display splitting data
    print("\n✂️  Splitting data into training and testing sets...")
    # seprate features
    x = data.drop("isFraud", axis=1)
    y = data["isFraud"]

    # Split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    y_train = (y_train > .5).astype(int)

    # Display split data
    print("✅ Successfully split data!\n")
    print("Training set shape:", x_train.shape, y_train.shape)
    print("Testing set shape:", x_test.shape, y_test.shape)


    return x_train, x_test, y_train, y_test

# Train model
def train_model(x_train, y_train):
    # Display training model
    print("\n🏋️  Training the model...")

    # Trains a Logistic Regressing model
    model = LogisticRegression(random_state=42)
    model.fit(x_train, y_train)

    # Display sucsessful train
    print("✅ Model trained successfully!")

    return model

# Evaluate model
def evaluate_model(model, x_test, y_test):
    y_test = (y_test > 0.5).astype(int)

    # Make predictions
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5).astype(int)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Display accuracy and report
    print(f"📊 Model Accuracy: {accuracy:.2f}")
    print(f"📊 ROC-AUC: {roc_auc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # ROC Curve (using the visualization function)
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    plot_roc_curve(y_test, y_pred_proba, roc_auc)
    
# Save model to a file
def save_model(model, filename="fraud_detection_model.pkl"):
    # Use joblib to save model to file
    joblib.dump(model, filename)

    # Display that it saved to file
    print(f"💾 Now saving model to {filename}")
    print(f"✅ Model saved sucessfully!")
    