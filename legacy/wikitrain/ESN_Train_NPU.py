import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from openvino.runtime import Core
import torch
import datetime
from src.models.ESN_Model import EchoStateNetworkNPU, tprint


def read_text_files_batch(data_dir, start_idx, batch_size):
    texts = []
    filenames = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    end_idx = min(start_idx + batch_size, len(filenames))
    batch_files = filenames[start_idx:end_idx]
    
    for filename in batch_files:
        with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    
    return texts, end_idx < len(filenames)

def save_model(esn, vectorizer, model_path, vectorizer_path):
    """Save model and vectorizer with proper NPU state handling"""
    try:
        # Ensure model is in a serializable state
        esn.update_model = None  # Clear compiled model
        esn.update_infer = None  # Clear inference request
        esn.core = None  # Clear OpenVINO core

        # Save the model and vectorizer
        tprint(f"Saving model to {model_path}")
        joblib.dump(esn, model_path)
        tprint(f"Saving vectorizer to {vectorizer_path}")
        joblib.dump(vectorizer, vectorizer_path)

        # Recompile the model after saving
        esn.core = Core()
        esn.compile_model()
        
    except Exception as e:
        tprint(f"Error saving model: {str(e)}")
        raise

def verify_saved_model(model_path):
    try:
        loaded_model = joblib.load(model_path)
        tprint(f"Successfully loaded model from {model_path}")
        # Recompile the model for NPU
        loaded_model.core = Core()
        loaded_model.compile_model()
        return True
    except Exception as e:
        tprint(f"Error loading model {model_path}: {str(e)}")
        return False

def main():
    # Create models directory if it doesn't exist
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize OpenVINO and check NPU capabilities
    core = Core()
    devices = core.available_devices
    
    if "NPU" not in devices:
        tprint("Intel NPU not available. Available devices:", devices)
        user_input = input("Continue with CPU instead? (y/n): ")
        if user_input.lower() != 'y':
            return
    else:
        try:
            npu_info = core.get_property("NPU", "FULL_DEVICE_NAME")
            tprint(f"Using Intel NPU: {npu_info}")
        except Exception as e:
            tprint("Using Intel NPU with default configuration")
            tprint(f"Note: Could not get detailed NPU info: {e}")

        # Use known supported NPU configurations
        npu_config = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "FP16",
            "NUM_STREAMS": "1",
            "PERF_COUNT": "NO"
        }
        tprint("Using NPU configuration:", npu_config)

    data_dir = 'scraped_data'
    batch_size = 50
    start_idx = 0
    has_more_files = True
    batch_number = 1
    all_texts = []

    tprint("Reading all files in the data directory...")
    while has_more_files:
        texts, has_more_files = read_text_files_batch(data_dir, start_idx, batch_size)
        if not texts:
            break
        all_texts.extend(texts)
        start_idx += batch_size

    tprint(f"Found {len(all_texts)} files in total....")

    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_texts)
    tprint("Vectorizer initialized")

    start_idx = 0
    has_more_files = True
    batch_number = 1

    while has_more_files:
        try:
            tprint(f"Processing batch {batch_number}...")
            texts, has_more_files = read_text_files_batch(data_dir, start_idx, batch_size)
            
            if not texts:
                break
                
            tprint(f"Processing {len(texts)} files in this batch")
            X = vectorizer.transform(texts).toarray()
            y = np.zeros((X.shape[0], 1))
            
            input_size = X.shape[1]
            reservoir_size = 1000
            output_size = 1
            
            tprint("Training ESN model...")
            esn = EchoStateNetworkNPU(input_size, reservoir_size, output_size)
            esn.fit(X, y)
            
            # Save model and vectorizer with batch number in models directory
            model_path = os.path.join(models_dir, f'esn_model_npu_batch_{batch_number}.pkl')
            vectorizer_path = os.path.join(models_dir, f'vectorizer_npu_batch_{batch_number}.pkl')
            
            tprint(f"Saving batch {batch_number} model and vectorizer...")
            save_model(esn, vectorizer, model_path, vectorizer_path)
            verify_saved_model(model_path)
            
            start_idx += batch_size
            batch_number += 1
            tprint(f"Completed batch {batch_number-1}")
            
        except Exception as e:
            tprint(f"Error processing batch {batch_number}: {str(e)}")
            # Try to continue with next batch
            start_idx += batch_size
            batch_number += 1
            continue

if __name__ == "__main__":
    main()
