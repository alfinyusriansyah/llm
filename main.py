from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import torch
from gpt4all import GPT4All
import helpers  # Import helper functions


# Initialize Flask app
app = Flask(__name__)

# Initialize Sentence Transformer and GPT4All
def initialize_embedding_model():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        print(f"Embedding model loaded on {device}!")
        return model
    except Exception as e:
        print("GPU initialization failed, falling back to CPU:", str(e))
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Use CPU fallback

def load_gpt4all_model():
    model_path = "C:/Users/Asus/data-alfin/ngoding/LLM/sugijantoV1_LLM.gguf"
    try:
        print("Attempting to load GPT4All model with GPU...")
        # model_path = "/home/ubuntu/API/sugijantoV1_LLM.gguf"
        # model_path = "sugijantoV1_LLM.gguf"
        model = GPT4All(model_name=model_path, device="cuda")  # Attempt GPU use
        print("GPT4All model loaded with GPU support.")
        return model
    except Exception as e:
        print("GPU load failed with errorrrrrrrrrrrrrr:", e)
        print("Falling back to CPU for GPT4All model...")
        return GPT4All(model_name=model_path)  # Fallback to CPU

# Initialize models
embedding_model = initialize_embedding_model()
gpt4all_qa_model = load_gpt4all_model()

# Endpoint to process the file and answer a question
@app.route('/process', methods=['POST'])
def process_file():
    data = request.json
    user_question = data['question'].lower()
    selected_file = data.get('selected_file')
    bucket_name = data.get('bucket')
    print("00000",bucket_name)
    print("Received data:", data)

    # Call helper function to process the file and answer the question
    response = helpers.process_question(
        user_question, selected_file, embedding_model, gpt4all_qa_model, bucket_name
    )
    return jsonify(response)

# Start the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5002, debug=True, use_reloader=False)