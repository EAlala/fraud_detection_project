import sqlite3
import joblib
import pandas as pd
import streamlit as st
from security_monitoring import login, log_event, log_performance
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- Configuration ---
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# --- Data Loading (Cached) ---
@st.cache_data
def load_data():
    transaction_data = pd.read_csv("data/train_transaction.csv")
    identity_data = pd.read_csv("data/train_identity.csv")
    return transaction_data.merge(identity_data, on="TransactionID", how="left")

@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model_v2.pkl")

# --- Authentication ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Fraud Detection System Login")
    if login():
        st.rerun()
    st.stop()

# --- Main Application ---
model = load_model()
data = load_data()

st.title(f"Fraud Detection Dashboard")
st.write(f"Welcome, {st.session_state.username}!")

# --- Sidebar Filters ---
st.sidebar.header("Filters")
min_amount = st.sidebar.number_input("Minimum Amount", value=0.0)
max_amount = st.sidebar.number_input("Maximum Amount", value=float(data["TransactionAmt"].max()))
card_types = st.sidebar.multiselect(
    "Card Types",
    options=data["card4"].unique(),
    default=data["card4"].unique()
)

# Apply filters
filtered_data = data[
    (data["TransactionAmt"] >= min_amount) &
    (data["TransactionAmt"] <= max_amount) &
    (data["card4"].isin(card_types))
]

# --- Main Dashboard ---
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Visualizations", "Predict Fraud", "Monitoring"])

with tab1:
    st.header("Data Overview")
    st.dataframe(filtered_data.head(1000))

with tab2:
    st.header("Data Visualizations")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fraud Distribution")
        fig = px.pie(filtered_data, names="isFraud", title="Fraud vs Non-Fraud")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Transaction Amounts")
        fig = px.histogram(filtered_data, x="TransactionAmt", nbins=50)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Fraud by Card Type")
    fig = px.bar(
        filtered_data.groupby("card4")["isFraud"].mean().reset_index(),
        x="card4", y="isFraud", title="Fraud Rate by Card Type"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Fraud Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Transaction Amount", min_value=0.0)
            card1 = st.number_input("Card1", min_value=0.0)
            card2 = st.number_input("Card2", min_value=0.0)
        with col2:
            addr1 = st.number_input("Address", min_value=0.0)
            card4 = st.selectbox("Card Type", ["Visa", "MasterCard", "American Express"])
            card6 = st.selectbox("Card Category", ["Credit", "Debit"])
        
        if st.form_submit_button("Predict"):
            input_data = pd.DataFrame({
                "TransactionAmt": [amount],
                "card1": [card1],
                "card2": [card2],
                "addr1": [addr1],
                "card4": [0 if card4 == "Visa" else 1 if card4 == "MasterCard" else 2],
                "card6": [0 if card6 == "Credit" else 1],
                "TransactionDT": [0]
            })
            
            prediction = model.predict(input_data)
            if prediction[0] == 1:
                st.error("⚠️ Fraud Detected!")
                log_event("Fraud predicted", "warning")
            else:
                st.success("✅ Legitimate Transaction")
                log_event("Legitimate transaction predicted")

with tab4:
    st.header("Model Performance Monitoring")
    
    # Connect to the metrics database
    conn = sqlite3.connect('model_performance.db')
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Plot metrics over time
        st.subheader("Performance Trends")
        metrics_df = pd.read_sql("SELECT * FROM metrics ORDER BY timestamp", conn)
        
        # Convert timestamp to datetime
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
        
        # Plot all metrics
        fig = px.line(metrics_df, x='timestamp', y=['accuracy', 'precision', 'recall', 'roc_auc'],
                     title='Model Metrics Over Time',
                     labels={'value': 'Score', 'timestamp': 'Date'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Show latest metrics
        st.subheader("Latest Metrics")
        latest = metrics_df.iloc[-1]
        
        st.metric("Accuracy", f"{latest['accuracy']:.2%}")
        st.metric("Precision", f"{latest['precision']:.2%}")
        st.metric("Recall", f"{latest['recall']:.2%}")
        st.metric("ROC-AUC", f"{latest['roc_auc']:.2f}")
        
        # Model information
        st.subheader("Model Info")
        st.write(f"**Type:** {latest['model']}")
        st.write(f"**Last Trained:** {latest['timestamp'].strftime('%Y-%m-%d %H:%M')}")
    
    # Add performance thresholds and alerts
    st.subheader("Performance Alerts")
    
    # Check if any metric is below threshold
    alert_thresholds = {
        'accuracy': 0.95,
        'precision': 0.90,
        'recall': 0.85,
        'roc_auc': 0.90
    }
    
    alerts = []
    for metric, threshold in alert_thresholds.items():
        if latest[metric] < threshold:
            alerts.append(f"⚠️ {metric.capitalize()} below threshold ({latest[metric]:.2%} < {threshold:.2%})")
    
    if alerts:
        for alert in alerts:
            st.error(alert)
    else:
        st.success("All metrics within expected ranges")
    
    conn.close()

    
# Log dashboard activity
log_performance({
    "user": st.session_state.username,
    "filtered_records": len(filtered_data),
    "fraud_rate": filtered_data["isFraud"].mean()
})