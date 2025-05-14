import os
import numpy as np
import faiss
import torch
from pathlib import Path
from recognizer import Identificator, calculate_signature
import pickle

def process_signatures():
    # Define source directory
    source_dir = Path('signature_base')
    
    # Initialize FAISS index
    dimension = 512  # dimension of the signature embeddings
    index = faiss.IndexFlatL2(dimension)
    
    # Store names and their corresponding indices
    names = []
    
    # Process each PNG file in the source directory
    for file in source_dir.glob('*.png'):
        # Get the filename without extension (this will be the name we want to use)
        name = file.stem
        
        try:
            # Calculate signature embedding
            embedding = calculate_signature(str(file), Identificator)
            
            # Add to FAISS index
            index.add(np.array([embedding], dtype=np.float32))
            
            # Store the name
            names.append(name)
            
            print(f"Processed signature for: {name}")
            
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")
    
    # Save the FAISS index
    faiss.write_index(index, "signature_index.faiss")
    
    # Save the names list
    with open("signature_names.pkl", "wb") as f:
        pickle.dump(names, f)
    
    print(f"\nProcessed {len(names)} signatures successfully")
    print("Saved FAISS index to signature_index.faiss")
    print("Saved names to signature_names.pkl")

if __name__ == "__main__":
    process_signatures() 