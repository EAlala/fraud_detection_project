import bcrypt
import os
from dotenv import load_dotenv
import logging
import streamlit as st
from cryptography.fernet import Fernet

# Initialize environment
load_dotenv()

# --- Encryption Setup ---
key = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
cipher_suite = Fernet(key.encode())

def encrypt_data(data):
    return cipher_suite.encrypt(data.encode())

def decrypt_data(encrypted_data):
    return cipher_suite.decrypt(encrypted_data).decode()

# --- Password Management ---
def get_hashed_password(plain_text_password):
    return bcrypt.hashpw(plain_text_password.encode(), bcrypt.gensalt()).decode()

def verify_password(plain_text_password, hashed_password):
    return bcrypt.checkpw(plain_text_password.encode(), hashed_password.encode())

# User database (in production, use a real database)
users = {
    "admin": get_hashed_password("admin123"),  # Will generate proper hash
    "analyst": get_hashed_password("analyst123")
}

def login():
    """Handle user login with proper password verification"""
    st.sidebar.header("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    if st.sidebar.button("Login"):
        if username in users and verify_password(password, users[username]):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.sidebar.success("Logged in successfully!")
            log_event(f"User {username} logged in", "info")
            return True
        else:
            st.sidebar.error("Invalid credentials")
            log_event("Failed login attempt", "warning")
    return False

# --- Logging Configuration ---
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_event(message, level="info"):
    if level == "info":
        logging.info(message)
    elif level == "warning":
        logging.warning(message)
    elif level == "error":
        logging.error(message)

def log_performance(metrics):
    log_event(f"Performance Metrics: {metrics}")