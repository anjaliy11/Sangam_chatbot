from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import os
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS
from dotenv import load_dotenv
import json

load_dotenv()

# Flask app setup
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Logging setup
logging.basicConfig(
    filename="flask_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Environment variables
host = os.getenv('HOST', '0.0.0.0')  
port = int(os.getenv('PORT', 5001))

# Model setup
MODEL_PATH = "ml/models/fine_tuned_qa_model"

try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForQuestionAnswering.from_pretrained(MODEL_PATH)
except Exception as e:
    logger.error(f"Error loading model/tokenizer: {str(e)}")
    raise e

# Move model to GPU or CPU based on availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to predict answers
def predict_answer(question, context):
    model.eval()
    try:
        # Prepare inputs
        inputs = tokenizer(
            question, 
            context, 
            return_tensors="pt", 
            truncation="only_second", 
            padding="max_length", 
            max_length=384
        ).to(device)

        # Perform prediction
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract logits
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Find start and end token indices
        start_idx = torch.argmax(start_logits)
        end_idx = torch.argmax(end_logits) + 1  # +1 to include the token

        # If start equals end, no confident answer
        if start_idx == end_idx:
            return "No answer found...this query is out of basic knowledge. You can directly reach to concerned departments/authorities customer support management."

        # Decode the answer
        if start_idx <= end_idx:
            answer = tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx], skip_special_tokens=True)
            
            # Post-process the answer (cleanup)
            answer = answer.replace(",", "")
            answer = re.sub(r'\s+', ' ', answer).strip()

            return answer
        else:
            return "No answer found...this query is out of basic knowledge. You can directly reach to concerned departments/authorities customer support management."
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return "Error: Could not process the input."

# Flask route for predictions
@app.route('/chatbot_predict', methods=['POST'])
def predict_answers():
    try:
        # Log the incoming request
        logger.info(f"Request received: {request.get_json()}")

        # Get JSON data from the request
        data = request.get_json()

        # Validate required keys
        if 'question' not in data or 'context' not in data:
            return jsonify({'error': "Both 'question' and 'context' fields are required"}), 400

        question = data['question']
        context = data['context']

        # Predict the answer
        answer = predict_answer(question, context)

        return jsonify({'answer': answer})

    except Exception as e:
        logger.error(f"Exception: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Running the app
if __name__ == "__main__":
    app.run(host=host, port=port, debug=True)
