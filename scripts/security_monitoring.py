import logging
from logging.handlers import RotatingFileHandler
import streamlit as st
from cryptography.fernet import Fernet

# --- Security Features ---

key = Fernet.generate_key()
cipher_suite = Fernet(key)

def encrypt_data(data):
    # Encrypt data
    return cipher_suite.encrypt(data.encode())

def decrypt_data(encrypted_data):
    # Decrypt Data
    return cipher_suite.decypt(encrypt_data).decode()

def login():
    # Simulate a user database (store securely in production)
    users = {
        "admin": "password123",
        "user": "user123"
    }

    st.sidebar.header("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username in users and users[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.sidebar.success("Logged in successfully!")
        else:
            st.sidebar.error("Invalid username or password.")
# --- Monitoring Features ---

# Configure logging
logging.basicConfig(
    filename="app.log",  # Log file name
    level=logging.INFO,  # Logging level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    filemode="a",  # Append mode (use "w" to overwrite)
)

def log_event(message, level="info"):
    """Log system events."""
    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)
    elif level == "warning":
        logging.warning(message)

def log_performance(metrics):
    """Log model performance metrics."""
    log_event(f"Model Performance: {metrics}")