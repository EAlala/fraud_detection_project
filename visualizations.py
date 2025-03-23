from turtle import st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve


def plot_transaction_amounts_over_time(data):
    # Plots a line chart of transaction amounts over time.
    print("\n📈 Plotting Transaction Amounts Over Time...")
    plt.figure()
    plt.plot(data["TransactionDT"], data["TransactionAmt"], label="Transaction Amount")
    plt.xlabel("Transaction Date")
    plt.ylabel("Transaction Amount")
    plt.legend()
    plt.title("Transaction Amounts Over Time")
    plt.show()  # Display the plot for 2 seconds before moving to the next one

def plot_fraud_rates_by_card_type(data):
    # Plots a bar chart of fraud rates by card type.
    print("\n📊 Plotting Fraud Rates by Card Type...")
    fraud_by_card = data.groupby("card4")["isFraud"].mean().reset_index()
    plt.figure()
    sns.barplot(x="card4", y="isFraud", data=fraud_by_card)
    plt.xlabel("Card Type")
    plt.ylabel("Fraud Rate")
    plt.title("Fraud Rates by Card Type")
    plt.show() # Display the plot for 2 seconds before moving to the next one

def plot_high_risk_transaction_patterns(data):
    # Plots a heatmap of high-risk transaction patterns by card type and category.
    print("\n🔥 Plotting High-Risk Transaction Patterns...")
    heatmap_data = data.pivot_table(index="card4", columns="card6", values="isFraud", aggfunc="mean")
    plt.figure()
    sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd")
    plt.xlabel("Card Category")
    plt.ylabel("Card Type")
    plt.title("High-Risk Transaction Patterns")
    plt.show()

def plot_roc_curve(y_test, y_pred_proba, roc_auc):
    # Plots the ROC Curve for the model.
    print("\n📉 Plotting ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()