import os
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import datetime
import torch.cuda.nvtx as nvtx
from src.models.ESN_Model import EchoStateNetworkGPU, tprint

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
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(esn, model_path)
    joblib.dump(vectorizer, vectorizer_path)

def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        
        props = torch.cuda.get_device_properties(0)
        tprint("CUDA Device Properties:")
        tprint(f"\tDevice: {props.name}")
        tprint(f"\tMemory: {props.total_memory / 1024**3:.2f} GB")
        tprint(f"\tCUDA Capability: {props.major}.{props.minor}")
        tprint(f"\tCUDA Cores: {props.multi_processor_count * 64}")
        tprint(f"\tShared Memory: {props.max_threads_per_multi_processor / 1024:.0f} KB")
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True
        device = torch.device('cuda:0')
    else:
        tprint("CUDA is not available. Running on CPU.")
        device = torch.device('cpu')

    tprint(f"Using device: {device}")
    
    available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    batch_size = min(1000, int(available_memory * 100))
    tprint(f"Using batch size: {batch_size}")

    data_dir = 'E:/ESN Data/scraped_data'
    start_idx = 0
    has_more_files = True
    
    tprint("Initializing vectorizer...")
    all_texts = []
    while has_more_files:
        with nvtx.range("read_batch"):
            texts, has_more_files = read_text_files_batch(data_dir, start_idx, batch_size)
            all_texts.extend(texts)
            start_idx += batch_size
    
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_texts)
    tprint("Vectorizer initialized")
    
    reservoir_size = 1000  # Size of the reservoir
    output_size = 1       # Size of the output layer
    
    start_idx = 0
    has_more_files = True
    batch_number = 1
    total_files_processed = 0
    
    while has_more_files:
        with nvtx.range(f"process_batch_{batch_number}"):
            tprint(f"Processing batch {batch_number}...")
            
            torch.cuda.empty_cache()
            
            texts, has_more_files = read_text_files_batch(data_dir, start_idx, batch_size)
            if not texts:
                break
            
            X = vectorizer.transform(texts).toarray()
            X = torch.from_numpy(X).float()  # CPU tensor instead of CPU vector
            X = X.to(device)  # Move to GPU
            
            y = torch.zeros((X.shape[0], 1), device=device)
            
            with torch.cuda.amp.autocast(True):
                esn = EchoStateNetworkGPU(X.shape[1], reservoir_size, output_size, device=device)
                esn.fit(X, y)
            
            tprint(f"GPU Memory Usage:")
            tprint(f"\tAllocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
            tprint(f"\tReserved:  {torch.cuda.memory_reserved(0)/1e9:.2f} GB")
            
            model_path = os.path.join('models', f'esn_model_gpu_batch_{batch_number}.pkl')
            vectorizer_path = os.path.join('models', f'vectorizer_gpu_batch_{batch_number}.pkl')
            save_model(esn, vectorizer, model_path, vectorizer_path)
            tprint(f"Model saved.... to {model_path} and {vectorizer_path}")
            
            total_files_processed += len(texts)
            tprint(f"Total files processed: {total_files_processed}")
            
            del esn, X, y
            torch.cuda.empty_cache()
            
            start_idx += batch_size
            batch_number += 1


if __name__ == "__main__":
    main()
