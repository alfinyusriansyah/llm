from pandasai.llm.local_llm import LocalLLM
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
import csv


# Initialize the local language model
model = LocalLLM(
    api_base="http://localhost:11434/v1",
    model="mistral:latest"
)

# Title of the Streamlit app
st.title("LLM with Tabular Data")

# File uploader to accept CSV, XLS, and XLSX files
uploaded_file = st.file_uploader("Upload file", type=["csv", "xls", "xlsx"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded file based on its type
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    # with open(data_csv, 'w', newline="") as file:
    #     csvwrite = csv.writer(file)

    # Display the first few rows of the uploaded data
    st.write("Data Preview:")
    st.write(data.head())

    # Initialize SmartDataframe with the uploaded data and LLM model
    df = SmartDataframe(data, config={"llm": model})
    
    # Text area for user input prompt
    prompt = st.text_area("Enter prompt")

    # Generate response on button click
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating..."):
                try:
                    response = df.chat(prompt)
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a prompt to generate a response.")
