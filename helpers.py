import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All
from minio import Minio
import urllib3
from io import BytesIO
from PIL import Image
from PyPDF2 import PdfReader
import hnswlib
from docx import Document
from dotenv import load_dotenv
import pytesseract
import torch
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import hnswlib
from minio import Minio
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO
import os
import json
import pandas as pd
import csv


# Load environment variables
load_dotenv()

# Folder to store memory
MEMORY_DIR = "memory"
CREDENTIALS_FILE = os.path.join(MEMORY_DIR, "minio_credentials.json")
HISTORY_FILE = os.path.join(MEMORY_DIR, "history.json")

embedding_model = None
gpt4all_qa_model = None
minio_client = None

def process_question(user_question, selected_file, embedding_model, gpt4all_qa_model, bucket_name):
    ensure_models_initialized()
    initialize_minio_client()
    
    print(12121212)
    print("buccket", bucket_name)
    if embedding_model is None or gpt4all_qa_model is None:
        return {"error": "Failed to load models. Please check the model paths and device compatibility."}

    predefined_responses = load_predefined_responses()
    
    # Check for a predefined response
    if user_question in predefined_responses:
        return {"answer": predefined_responses[user_question]}
    
    # Load history to check for previously answered questions
    history = load_history()
    if user_question in history:
        return {"answer": history[user_question]}
    
    # Determine if the question is related to coding
    coding_keywords = ['code', 'function', 'class', 'method', 'algorithm', 'python', 'javascript', 'java', 'c++']
    if any(keyword in user_question for keyword in coding_keywords):
        context = "Here is the code snippet based on your question:"
    else:
        context = "No context available, answering based on general knowledge."
    
    print("file : ",selected_file)
    print("embeding :",embedding_model)
    # If a file is selected, process it
    if selected_file is not None:
        index, chunks = preload_file(selected_file, embedding_model, bucket_name)
        print("embeding :",embedding_model)
        if index is None:
            return {"error": "Failed to learn from the file. Please check the file name or format."}

        user_question_embedding = embedding_model.encode([user_question])
        labels, distances = index.knn_query(user_question_embedding, k=3)
        context_chunks = [chunks[label] for label in labels[0]]
        context = " ".join(context_chunks)
        
        # Limit the token length
        max_token_length = 4096
        if len(context) > max_token_length:
            context = context[:max_token_length]
    
    # Generate answer using the GPT4All model
    try:
        prompt = f"Context: {context}\n\nQuestion: {user_question}\nAnswer:"
        response = gpt4all_qa_model.generate(prompt)

        # Format code response if applicable
        if any(keyword in user_question for keyword in coding_keywords):
            response = f"```python\n{response}\n```"  # Change 'python' to the relevant language if needed
        
        # Save to history.json
        save_to_history(user_question, response)
        return {"answer": response}
    except Exception as e:
        return {"error": str(e)}

# Modified process_question to ensure models are initialized
# def process_question(user_question, selected_file, embedding_model, gpt4all_qa_model):
#     ensure_models_initialized()
    
#     if embedding_model is None or gpt4all_qa_model is None:
#         return {"error": "Failed to load models. Please check the model paths and device compatibility."}

#     predefined_responses = load_predefined_responses()
    
#     # Check for a predefined response
#     if user_question in predefined_responses:
#         return {"answer": predefined_responses[user_question]}
    
#     # Load history to check for previously answered questions
#     history = load_history()
#     if user_question in history:
#         return {"answer": history[user_question]}
    
#     # If no file is selected, provide general context
#     if selected_file is None:
#         print("No file selected. Processing question directly without MinIO context.")
#         context = "No context available, answering based on general knowledge."
#     else:
#         index, chunks = preload_file(selected_file, embedding_model)
#         if index is None:
#             return {"error": "Failed to learn from the file. Please check the file name or format."}
        
#         user_question_embedding = embedding_model.encode([user_question])
#         labels, distances = index.knn_query(user_question_embedding, k=3)
#         context_chunks = [chunks[label] for label in labels[0]]
#         context = " ".join(context_chunks)
        
#         # Limit the token length
#         max_token_length = 4096
#         if len(context) > max_token_length:
#             context = context[:max_token_length]
    
#     # Generate answer using the GPT4All model
#     try:
#         prompt = f"Context: {context}\n\nQuestion: {user_question}\nAnswer:"
#         response = gpt4all_qa_model.generate(prompt)
        
#         # Save to history.json
#         save_to_history(user_question, response)
#         return {"answer": response}
#     except Exception as e:
#         return {"error": str(e)}
    
# Initialize Sentence Transformer and GPT4All models
# Initialize Sentence Transformer and GPT4All models
def initialize_embedding_model():
    global embedding_model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        print(f"Embedding model loaded on {device}!")
    except Exception as e:
        print("Failed to load embedding model:", str(e))
        embedding_model = None  # Ensure it's explicitly set to None on failure


def load_gpt4all_model():
    global gpt4all_qa_model
    model_path = "C:/Users/Asus/data-alfin/ngoding/LLM/sugijantoV1_LLM.gguf"
    try:
        print("Attempting to load GPT4All model with GPU...")
        # model_path = "/home/ubuntu/API/sugijantoV1_LLM.gguf"
        gpt4all_qa_model = GPT4All(model_name=model_path, device="cuda")
        print(gpt4all_qa_model)
        print(11111)
        print("GPT4All model loaded with GPU support.")
    except Exception as e:
        print("GPU load failed with error:", e)
        print("Falling back to CPU for GPT4All model...")
        gpt4all_qa_model = GPT4All(model_name=model_path)
        print(2222222)
    if gpt4all_qa_model is None:
        print("Failed to load GPT4All model.")

# Ensure models are initialized
def ensure_models_initialized():
    if embedding_model is None:
        print("Initializing embedding model...")
        initialize_embedding_model()
    if gpt4all_qa_model is None:
        print("Initializing GPT4All model...")
        load_gpt4all_model()

# MinIO initialization
def load_minio_credentials():
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, 'r') as f:
            return json.load(f)
    return None

def initialize_minio_client():
    global minio_client
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    credentials = load_minio_credentials()
    if credentials:
        minio_client = Minio(
            credentials["url"],
            access_key=credentials["access_key"],
            secret_key=credentials["secret_key"],
            secure=False,
            http_client=urllib3.PoolManager(cert_reqs='CERT_NONE')
        )

def list_files_in_bucket(bucket_name):
    if not minio_client:
        return "MinIO client not initialized."
    try:
        return [obj.object_name for obj in minio_client.list_objects(bucket_name)]
    except Exception as e:
        return str(e)

def read_from_minio(bucket_name, object_name):
    print(1212)
    if not minio_client:
        return "MinIO client not initialized."
    try:
        response = minio_client.get_object(bucket_name, object_name)
        return BytesIO(response.read())
    except Exception as e:
        return str(e)

# File processing functions (PDF, Image, DOCX)
def process_pdf_from_minio(data):
    extracted_text = ""
    try:
        reader = PdfReader(data)
        for page in reader.pages:
            extracted_text += page.extract_text() + "\n\n"
    except Exception as e:
        return str(e)
    return extracted_text

def process_image_from_minio(data):
    try:
        image = Image.open(data)
        return pytesseract.image_to_string(image)
    except Exception as e:
        return str(e)

def process_docx_from_minio(data):
    try:
        document = Document(data)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    except Exception as e:
        return str(e)


# Save/load embeddings and text chunks
def save_memory(file_name, embeddings, chunks, data_csv):
    file_dir = os.path.join(MEMORY_DIR, file_name)
    os.makedirs(file_dir, exist_ok=True)
    np.save(os.path.join(file_dir, 'embeddings.npy'), embeddings)
    with open(data_csv, 'w', newline="") as file:
        csvwrite = csv.writer(file)
        
    with open(os.path.join(file_dir, 'chunks.txt'), 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + "\n")

# Updated preload_file to check embedding model initialization
def preload_file(file_name, embedding_model, bucket_name):
    print("inini bucket",bucket_name)
    # bucket_name = "test"
    if embedding_model is None:
        return None, "Embedding model is not initialized."
    
    file_data = read_from_minio(bucket_name, file_name)
    print("====",file_data)
    print(11111)
    extracted_text = ""
    if file_name.endswith(".pdf"):
        extracted_text = process_pdf_from_minio(file_data)
        print(extracted_text)
    elif file_name.endswith((".png", ".jpg", ".jpeg")):
        extracted_text = process_image_from_minio(file_data)
    elif file_name.endswith(".docx"):
        extracted_text = process_docx_from_minio(file_data)
    elif file_name.endswith(".csv"):
        data_csv = pd.read_csv(file_data)

    if not extracted_text:
        return None, "No text extracted from the file."

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1500, chunk_overlap=0, length_function=len)
    chunks = text_splitter.split_text(extracted_text)
    embeddings = embedding_model.encode(chunks)


    dim = embeddings.shape[1]
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=len(chunks), ef_construction=200, M=16)
    index.add_items(embeddings)

    save_memory(file_name, embeddings, chunks)

    return index, chunks

# Load predefined responses and history functions
def load_predefined_responses():
    with open('memory/responses.json', 'r') as f:
        return json.load(f)

def load_history():
    if not os.path.exists('memory/history.json'):
        return {}
    with open('memory/history.json', 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print("Warning: history.json is empty or contains invalid JSON.")
            return {}

def save_to_history(question, answer):
    history = load_history()
    history[question] = answer
    with open('memory/history.json', 'w') as f:
        json.dump(history, f)
