from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
import faiss
import torch
from recognizer import Identificator, calculate_signature, image_transform_test, preprocessing
import skimage.io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize FAISS index
dimension = 512  # dimension of the signature embeddings
index = faiss.IndexFlatL2(dimension)

# Load celebrity signatures database
# TODO: Add your celebrity signatures database here
# For now, we'll use a placeholder
celebrity_signatures = []  # List of (name, embedding) tuples

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # try:
        # Process the signature
        embedding = calculate_signature(filepath, Identificator)
        
        # Search for similar signatures
        # TODO: Implement similarity search using FAISS
        # For now, return dummy results
        results = [
            {'name': 'Celebrity 1', 'similarity': 0.95},
            {'name': 'Celebrity 2', 'similarity': 0.85},
            {'name': 'Celebrity 3', 'similarity': 0.75}
        ]
        
        return jsonify({'results': results})
            
        # except Exception as e:
        #     return jsonify({'error': str(e)}), 500
        # finally:
        #     # Clean up uploaded file
        #     if os.path.exists(filepath):
        #         os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True) 