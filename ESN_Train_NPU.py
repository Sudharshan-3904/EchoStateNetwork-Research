import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import openvino.runtime as ov
from openvino.runtime import Core
import torch  # Still needed for initial tensor operations
import datetime

def tprint(*args, **kwargs):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f"[{timestamp}]", *args, **kwargs, end='\n', sep=' ')

class EchoStateNetworkNPU:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.95, 
                 sparsity=0.1, input_scaling=1.0, leaking_rate=1.0):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate

        # Initialize weights
        self.W_in = np.random.uniform(-self.input_scaling, self.input_scaling, 
                                    (self.reservoir_size, self.input_size + 1))
        self.W = np.random.uniform(-0.5, 0.5, 
                                 (self.reservoir_size, self.reservoir_size))
        
        # Apply sparsity
        mask = (np.random.rand(*self.W.shape) > self.sparsity)
        self.W *= mask
        
        # Scale by spectral radius
        eigenvalues = np.linalg.eigvals(self.W)
        max_abs_eigenvalue = np.max(np.abs(eigenvalues))
        self.W *= self.spectral_radius / max_abs_eigenvalue

        self.W_out = None
        self.reservoir_state = np.zeros((self.reservoir_size, 1))

        # Initialize OpenVINO Core and check for NPU
        self.core = Core()
        self.devices = self.core.available_devices
        if "NPU" not in self.devices:
            tprint("Warning: NPU not found, available devices:", self.devices)
            self.device = "CPU"  # Fallback to CPU
        else:
            self.device = "NPU"
            try:
                device_info = self.core.get_property("NPU", "FULL_DEVICE_NAME")
                tprint(f"Using Intel NPU: {device_info}")
            except:
                tprint("Using Intel NPU (details unavailable)")

        # Set NPU configuration
        self.config = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "FP16",
            "NUM_STREAMS": "1",
            "PERF_COUNT": "NO"
        }
        
        self.compile_model()

    def compile_model(self):
        try:
            # Create OpenVINO model with optimized configuration
            self.update_model = self.core.compile_model(
                self._create_reservoir_update_model(), 
                device_name=self.device,
                config=self.config
            )
            self.update_infer = self.update_model.create_infer_request()
            tprint(f"Model compiled successfully for {self.device}")
        except Exception as e:
            tprint(f"Error compiling model for {self.device}: {str(e)}")
            tprint("Falling back to CPU...")
            self.device = "GPU"
            # Try again with GPU
            try:
                self.update_model = self.core.compile_model(
                    self._create_reservoir_update_model(), 
                    device_name=self.device
                )
                self.update_infer = self.update_model.create_infer_request()
                tprint("Model compiled successfully on CPU")
            except Exception as e2:
                tprint(f"Fatal error: Could not compile model on CPU: {str(e2)}")
                raise

    def _create_reservoir_update_model(self):
        # Create OpenVINO model for reservoir update computation
        # Fix: Use correct OpenVINO data type definition
        input_node = ov.opset8.parameter(
            [1, self.input_size], 
            dtype=ov.Type.f32
        )
        state_node = ov.opset8.parameter(
            [self.reservoir_size, 1], 
            dtype=ov.Type.f32
        )
        
        # Convert weights to constants with correct data type
        w_in_node = ov.opset8.constant(
            self.W_in.astype(np.float32), 
            dtype=ov.Type.f32
        )
        w_node = ov.opset8.constant(
            self.W.astype(np.float32), 
            dtype=ov.Type.f32
        )
        
        # Create computation graph
        new_state = ov.opset8.tanh(
            ov.opset8.add(
                ov.opset8.matmul(w_in_node, input_node),
                ov.opset8.matmul(w_node, state_node)
            )
        )
        
        return ov.Model([input_node, state_node], [new_state], "reservoir_update")

    def _update_reservoir(self, input_vector):
        # Update reservoir state using compiled OpenVINO model
        input_tensor = np.array(input_vector).reshape(1, -1)
        self.update_infer.set_input_tensors([input_tensor, self.reservoir_state])
        self.update_infer.infer()
        self.reservoir_state = self.update_infer.get_output_tensor().data

    def fit(self, inputs, targets, regularization=1e-8):
        states = []
        for input_vector in inputs:
            self._update_reservoir(input_vector)
            states.append(self.reservoir_state.flatten())
        states = np.array(states)

        # Add bias term
        states = np.hstack((np.ones((states.shape[0], 1)), states))
        
        # Compute output weights using Ridge Regression
        targets = np.array(targets)
        identity = np.eye(states.shape[1])
        temp = np.dot(states.T, states) + regularization * identity
        self.W_out = np.dot(np.linalg.pinv(temp), np.dot(states.T, targets))

    def predict(self, inputs):
        predictions = []
        for input_vector in inputs:
            self._update_reservoir(input_vector)
            augmented_state = np.vstack((1, self.reservoir_state))
            output = np.dot(self.W_out.T, augmented_state)
            predictions.append(output.flatten())
        return np.array(predictions)


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
