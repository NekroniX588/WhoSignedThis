from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import numpy as np
import faiss
import torch
from recognizer import Identificator, calculate_signature, image_transform_test, preprocessing
import skimage.io
import pickle
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logger.info("Initializing application...")

# Load FAISS index and names
try:
    logger.info("Loading FAISS index and celebrity names...")
    faiss_index = faiss.read_index("signature_index.faiss")
    with open("signature_names.pkl", "rb") as f:
        celebrity_names = pickle.load(f)
    logger.info(f"Successfully loaded index with {len(celebrity_names)} signatures")
except Exception as e:
    logger.error(f"Failed to load FAISS index or names: {str(e)}")
    raise

@app.route('/')
def index():
    logger.info("Serving index page")
    return render_template('index.html')

@app.route('/signature/<name>')
def get_signature(name):
    logger.info(f"Requesting signature for: {name}")
    signature_path = os.path.join('signature_base', f"{name}.png")
    if os.path.exists(signature_path):
        logger.info(f"Found signature for {name}")
        return send_file(signature_path, mimetype='image/png')
    logger.warning(f"Signature not found for: {name}")
    return jsonify({'error': 'Signature not found'}), 404

@app.route('/upload', methods=['POST'])
def upload_file():
    start_time = time.time()
    logger.info("Received file upload request")
    
    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving uploaded file: {filename}")
            file.save(filepath)
            
            # Process the signature
            logger.info("Calculating signature embedding")
            embedding = calculate_signature(filepath, Identificator)
            
            # Convert embedding to numpy array and reshape for FAISS
            query_vector = np.array([embedding], dtype=np.float32)
            logger.info("Searching for similar signatures using FAISS")
            k = 3  # number of nearest neighbors to return
            distances, indices = faiss_index.search(query_vector, k)
            
            # Prepare results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(celebrity_names):  # Ensure index is valid
                    similarity = 1 / (1 + distance)
                    results.append({
                        'name': celebrity_names[idx],
                        'similarity': float(similarity),
                        'image_url': f'/signature/{celebrity_names[idx]}'
                    })
            
            processing_time = time.time() - start_time
            logger.info(f"Successfully processed signature in {processing_time:.2f} seconds")
            logger.info(f"Found {len(results)} matches")
            return jsonify({'results': results})
                
        except Exception as e:
            logger.error(f"Error processing signature: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                logger.info(f"Cleaning up uploaded file: {filename}")
                os.remove(filepath)

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(host='0.0.0.0', debug=True) 