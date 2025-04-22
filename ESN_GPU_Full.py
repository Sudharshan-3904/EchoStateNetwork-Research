import os
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import datetime
import torch.cuda.nvtx as nvtx  # For CUDA profiling

def tprint(*args, **kwargs):
    """Print with timestamp"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f"[{timestamp}]", *args, **kwargs)

class EchoStateNetworkGPU:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.95, 
                 sparsity=0.1, input_scaling=1.0, leaking_rate=1.0, device='cuda'):
        self.device = device
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        
        self.initialize_weights()

    def initialize_weights(self):
        self.W_in = torch.empty((self.reservoir_size, self.input_size + 1), 
                               device=self.device).uniform_(-self.input_scaling, self.input_scaling)
        
        self.W = torch.empty((self.reservoir_size, self.reservoir_size), 
                            device=self.device).uniform_(-0.5, 0.5)
        
        mask = (torch.rand(self.W.shape, device=self.device) > self.sparsity).float()
        self.W *= mask
        
        eigenvalues = torch.linalg.eigvals(self.W)
        max_abs_eigenvalue = torch.max(torch.abs(eigenvalues))
        self.W *= self.spectral_radius / max_abs_eigenvalue

        self.W_out = None
        self.reservoir_state = torch.zeros((self.reservoir_size, 1), device=self.device)

    def _update_reservoir(self, input_vector):
        input_vector = input_vector.reshape(-1, 1)
        augmented_input = torch.vstack((torch.ones(1, device=self.device), input_vector))
        pre_activation = torch.mm(self.W_in, augmented_input) + torch.mm(self.W, self.reservoir_state)
        self.reservoir_state = (1 - self.leaking_rate) * self.reservoir_state + \
                             self.leaking_rate * torch.tanh(pre_activation)

    def collect_states(self, inputs):
        states = []
        for input_vector in inputs:
            self._update_reservoir(input_vector)
            states.append(self.reservoir_state.flatten())
        states = torch.stack(states)
        return torch.hstack((torch.ones(states.shape[0], 1, device=self.device), states))

    def fit(self, inputs, targets, regularization=1e-8):
        states = self.collect_states(inputs)
        targets = torch.tensor(targets, device=self.device)
        identity = torch.eye(states.shape[1], device=self.device)
        temp = torch.mm(states.T, states) + regularization * identity
        self.W_out = torch.mm(torch.linalg.pinv(temp), torch.mm(states.T, targets))

    def predict(self, inputs):
        predictions = []
        for input_vector in inputs:
            self._update_reservoir(input_vector)
            augmented_state = torch.vstack((torch.ones(1, device=self.device), 
                                          self.reservoir_state))
            output = torch.mm(self.W_out.T, augmented_state)
            predictions.append(output.flatten())
        return torch.stack(predictions).cpu().numpy()

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
    # Ensure single GPU optimization
    if torch.cuda.is_available():
        # Force using only one GPU
        torch.cuda.set_device(0)
        # Clear GPU memory and cache
        torch.cuda.empty_cache()
        
        # Get GPU properties
        props = torch.cuda.get_device_properties(0)
        tprint("CUDA Device Properties:")
        tprint(f"\tDevice: {props.name}")
        tprint(f"\tMemory: {props.total_memory / 1024**3:.2f} GB")
        tprint(f"\tCUDA Capability: {props.major}.{props.minor}")
        tprint(f"\tCUDA Cores: {props.multi_processor_count * 64}")
        tprint(f"\tShared Memory: {props.max_threads_per_multi_processor / 1024:.0f} KB")
        
        # Optimize CUDA settings for single GPU
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32
        device = torch.device('cuda:0')  # Explicitly use first GPU
    else:
        tprint("CUDA is not available. Running on CPU.")
        device = torch.device('cpu')

    tprint(f"Using device: {device}")
    
    # Adjust batch size based on GPU memory
    available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    batch_size = min(1000, int(available_memory * 100))  # Scale batch size with GPU memory
    tprint(f"Using batch size: {batch_size}")

    data_dir = 'E:/ESN Data/scraped_data'
    start_idx = 0
    has_more_files = True
    
    # Initialize vectorizer
    tprint("Initializing vectorizer...")
    all_texts = []
    while has_more_files:
        with nvtx.range("read_batch"):  # CUDA profiling marker
            texts, has_more_files = read_text_files_batch(data_dir, start_idx, batch_size)
            all_texts.extend(texts)
            start_idx += batch_size
    
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_texts)
    tprint("Vectorizer initialized")
    
    # Define model parameters before processing
    reservoir_size = 1000  # Size of the reservoir
    output_size = 1       # Size of the output layer
    
    # Process in batches
    start_idx = 0
    has_more_files = True
    batch_number = 1
    total_files_processed = 0
    
    while has_more_files:
        with nvtx.range(f"process_batch_{batch_number}"):
            tprint(f"Processing batch {batch_number}...")
            
            # Clear cache before each batch
            torch.cuda.empty_cache()
            
            texts, has_more_files = read_text_files_batch(data_dir, start_idx, batch_size)
            if not texts:
                break
            
            # First create CPU tensor, then transfer to GPU
            X = vectorizer.transform(texts).toarray()
            X = torch.from_numpy(X).float()  # Create CPU tensor
            X = X.to(device)  # Move to GPU
            
            # Create zero tensor directly on GPU
            y = torch.zeros((X.shape[0], 1), device=device)
            
            # Enable automatic mixed precision for better performance
            # with torch.cuda.amp.autocast(device_type="cuda"):
            with torch.cuda.amp.autocast(True):
                esn = EchoStateNetworkGPU(X.shape[1], reservoir_size, output_size, device=device)
                esn.fit(X, y)
            
            # Monitor GPU memory usage
            tprint(f"GPU Memory Usage:")
            tprint(f"\tAllocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
            tprint(f"\tReserved:  {torch.cuda.memory_reserved(0)/1e9:.2f} GB")
            
            # Save model and clear GPU memory
            model_path = os.path.join('models', f'esn_model_gpu_batch_{batch_number}.pkl')
            vectorizer_path = os.path.join('models', f'vectorizer_gpu_batch_{batch_number}.pkl')
            save_model(esn, vectorizer, model_path, vectorizer_path)
            tprint(f"Model saved.... to {model_path} and {vectorizer_path}")
            
            total_files_processed += len(texts)
            tprint(f"Total files processed: {total_files_processed}")
            
            # To run a new batch without memory issues, clear the previous batch data
            del esn, X, y
            torch.cuda.empty_cache()
            
            start_idx += batch_size
            batch_number += 1


if __name__ == "__main__":
    main()
