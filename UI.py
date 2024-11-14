import streamlit as st
import requests
import os
import json
from addon import MinioConnector  # Import the connector

# Set the API URL of the Flask app
API_URL = "http://localhost:5002/process"  # Change this if your API URL is different
CREDENTIALS_FILE = "memory/minio_credentials.json"

# Title of the app
st.title("QA Chat App with SUGIJANTO-V1")

# Load MinIO credentials if they exist
def load_minio_credentials():
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, 'r') as f:
            return json.load(f)
    return None

# Save MinIO credentials
def save_minio_credentials(credentials):
    os.makedirs("memory", exist_ok=True)
    with open(CREDENTIALS_FILE, 'w') as f:
        json.dump(credentials, f)

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "available_files" not in st.session_state:
    st.session_state["available_files"] = []
if "model_status" not in st.session_state:
    st.session_state["model_status"] = "Model not started learning yet."

# Load existing credentials
credentials = load_minio_credentials()

# Menu for setup
st.sidebar.header("Setup Menu")
if st.sidebar.checkbox("Connect to MinIO"):
    minio_url = st.sidebar.text_input("MinIO URL", value=credentials.get("url") if credentials else "")
    access_key = st.sidebar.text_input("Access Key", value=credentials.get("access_key") if credentials else "")
    secret_key = st.sidebar.text_input("Secret Key", value=credentials.get("secret_key") if credentials else "", type="password")
    bucket_name = st.sidebar.text_input("Bucket Name", value=credentials.get("access_key") if credentials else "")  # Add bucket name input

    if st.sidebar.button("Save Credentials"):
        save_minio_credentials({"url": minio_url, "access_key": access_key, "secret_key": secret_key, "bucket_name": bucket_name})
        st.sidebar.success("Credentials saved!")
        st.experimental_rerun()  # Reload the app to apply credentials

# Option to proceed without MinIO
st.sidebar.header("Chat Mode")
chat_mode = st.sidebar.radio("Choose mode:", ("Chat with MinIO context", "Chat without MinIO context"))

# Check if MinIO connection is available
if chat_mode == "Chat with MinIO context" and not credentials:
    st.warning("Please provide MinIO credentials in the setup menu.")
else:
    # Display the model status
    st.markdown(f"**Model Status:** {st.session_state['model_status']}")

    # Dropdown menu for selecting file context (only for MinIO chat mode)
    selected_file = None
    if chat_mode == "Chat with MinIO context":
        if credentials:
            minio_connector = MinioConnector(credentials["url"], credentials["access_key"], credentials["secret_key"])
            # Use "bucket_name" instead of "test-bucket"
            st.session_state["available_files"] = minio_connector.list_files(credentials["bucket_name"])

        selected_file = st.selectbox("Select a file for context:", st.session_state["available_files"])

    st.write(credentials["bucket_name"])
    # Function to send the user's question to the Flask API
    def ask_question(question, selected_file=None):
        try:
            payload = {"question": question, "bucket": credentials["bucket_name"]}
            if selected_file:
                payload["selected_file"] = selected_file
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                return response.json().get("answer", "Error: No answer received.")
            else:
                return f"Error: {response.json().get('error', 'Unknown error occurred.')}"
        except Exception as e:
            return f"Error: {str(e)}"

    # Chat interface to display the messages (bot/user)
    st.markdown("### Chat History")

    # Display chat messages from history
    for chat in st.session_state["chat_history"]:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    # Input form to get the user's question
    if user_input := st.chat_input("Ask a question:"):
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        # Update model status to learning
        st.session_state["model_status"] = "Model is learning, might take a few minutes."
        
        # Get the response from the API
        response = ask_question(user_input, selected_file if chat_mode == "Chat with MinIO context" else None)
        
        # Add the bot's response to the chat history
        st.session_state["chat_history"].append({"role": "assistant", "content": response})
        st.chat_message("assistant").markdown(response)

        # Update model status to indicate completion
        st.session_state["model_status"] = "Model learning completed, ready for questions."
