import os
import time
import pickle
import numpy as np
import torch
import glob
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def timestamp_print(*args, **kwargs):
    """Print with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]", *args, **kwargs)

class ESNModelGPU:
    """GPU-accelerated Echo State Network for next token prediction"""
    def __init__(self, input_size, reservoir_size, output_size, 
                 spectral_radius=0.99, sparsity=0.05, 
                 input_scaling=1.0, leaking_rate=0.3):
        # Check if CUDA is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            timestamp_print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            timestamp_print(f"GPU Memory: {total_memory:.2f} GB")
        else:
            timestamp_print("CUDA not available. Using CPU.")
        
        # Model parameters
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        
        # Initialize weights
        # Input weights: dense matrix with values between -input_scaling and input_scaling
        self.W_in = torch.rand(reservoir_size, input_size + 1, device=self.device) * 2 * input_scaling - input_scaling
        
        # Reservoir weights: sparse matrix with spectral radius
        W = torch.rand(reservoir_size, reservoir_size, device=self.device)
        # Apply sparsity
        mask = torch.rand(reservoir_size, reservoir_size, device=self.device) < sparsity
        W = W * mask.float()
        
        # Scale to desired spectral radius
        radius = torch.max(torch.abs(torch.linalg.eigvals(W)))
        self.W = W * (spectral_radius / radius)
        
        # Output weights will be learned
        self.W_out = None
        
        # Initial reservoir state
        self.reservoir_state = torch.zeros(reservoir_size, 1, device=self.device)
        
        # Enable CUDA optimizations
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True
    
    def _update_reservoir(self, input_vector):
        """Update the reservoir state with the given input vector."""
        # Convert input to tensor if it's not already
        if not isinstance(input_vector, torch.Tensor):
            input_vector = torch.tensor(input_vector, device=self.device, dtype=torch.float32)
        
        # Add bias term
        input_with_bias = torch.cat([torch.ones(1, device=self.device), input_vector])
        
        # Calculate new state
        new_state = torch.tanh(torch.matmul(self.W_in, input_with_bias.unsqueeze(1)) + 
                              torch.matmul(self.W, self.reservoir_state))
        
        # Apply leaking rate
        self.reservoir_state = (1 - self.leaking_rate) * self.reservoir_state + self.leaking_rate * new_state
    
    def fit(self, inputs, targets, regularization=1e-5, batch_size=128):
        """Train the model using ridge regression with batched processing."""
        timestamp_print(f"Training on {len(inputs)} samples with batch size {batch_size}...")
        
        # Convert inputs and targets to tensors if they're not already
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float32)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, device=self.device, dtype=torch.float32)
        
        # Initialize matrices for ridge regression
        X_T_X = torch.zeros((self.reservoir_size + 1, self.reservoir_size + 1), device=self.device)
        X_T_Y = torch.zeros((self.reservoir_size + 1, self.output_size), device=self.device)
        
        # Process in batches to avoid memory issues
        num_batches = (len(inputs) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(inputs))
            
            batch_inputs = inputs[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx]
            
            # Collect states for this batch
            batch_states = []
            for i in range(len(batch_inputs)):
                # Reset reservoir state for each sequence
                self.reservoir_state = torch.zeros(self.reservoir_size, 1, device=self.device)
                self._update_reservoir(batch_inputs[i])
                batch_states.append(self.reservoir_state.flatten())
            
            # Convert list to tensor
            batch_states = torch.stack(batch_states)
            
            # Add bias term
            batch_states_with_bias = torch.cat([torch.ones(batch_states.shape[0], 1, device=self.device), batch_states], dim=1)
            
            # Update matrices for ridge regression
            X_T_X += torch.matmul(batch_states_with_bias.t(), batch_states_with_bias)
            X_T_Y += torch.matmul(batch_states_with_bias.t(), batch_targets)
            
            # Progress update
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                timestamp_print(f"Processed batch {batch_idx + 1}/{num_batches} ({(batch_idx + 1) / num_batches * 100:.1f}%)")
        
        # Solve for W_out using ridge regression
        timestamp_print("Computing output weights...")
        regularization_matrix = regularization * torch.eye(self.reservoir_size + 1, device=self.device)
        self.W_out = torch.linalg.solve(X_T_X + regularization_matrix, X_T_Y)
        
        timestamp_print("Training complete!")
    
    def predict(self, inputs):
        """Predict outputs for the given inputs."""
        # Convert inputs to tensor if it's not already
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float32)
        
        predictions = []
        with torch.no_grad():  # Disable gradient calculation for inference
            for i in range(inputs.shape[0]):
                # Reset reservoir state for each sequence
                self.reservoir_state = torch.zeros(self.reservoir_size, 1, device=self.device)
                input_vector = inputs[i]
                self._update_reservoir(input_vector)
                augmented_state = torch.cat([torch.ones(1, device=self.device), self.reservoir_state.flatten()])
                output = torch.matmul(self.W_out.t(), augmented_state)
                predictions.append(output.cpu().numpy())
        
        return np.array(predictions)
    
    def save(self, path):
        """Save the model to a file."""
        # Convert tensors to numpy arrays for saving
        model_data = {
            'input_size': self.input_size,
            'reservoir_size': self.reservoir_size,
            'output_size': self.output_size,
            'spectral_radius': self.spectral_radius,
            'sparsity': self.sparsity,
            'input_scaling': self.input_scaling,
            'leaking_rate': self.leaking_rate,
            'W_in': self.W_in.cpu().numpy(),
            'W': self.W.cpu().numpy(),
            'W_out': self.W_out.cpu().numpy() if self.W_out is not None else None,
            'reservoir_state': self.reservoir_state.cpu().numpy()
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path):
        """Load a model from a file."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new model with the loaded parameters
        model = cls(
            input_size=model_data['input_size'],
            reservoir_size=model_data['reservoir_size'],
            output_size=model_data['output_size'],
            spectral_radius=model_data['spectral_radius'],
            sparsity=model_data['sparsity'],
            input_scaling=model_data['input_scaling'],
            leaking_rate=model_data['leaking_rate']
        )
        
        # Load weights
        model.W_in = torch.tensor(model_data['W_in'], device=model.device)
        model.W = torch.tensor(model_data['W'], device=model.device)
        if model_data['W_out'] is not None:
            model.W_out = torch.tensor(model_data['W_out'], device=model.device)
        model.reservoir_state = torch.tensor(model_data['reservoir_state'], device=model.device)
        
        return model

def load_and_preprocess_data(data_dir, max_files=None, max_len=20, num_words=10000):
    """Load and preprocess text data for next token prediction."""
    timestamp_print(f"Loading data from {data_dir}...")
    
    # Find all text files
    text_files = glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
    if max_files:
        text_files = text_files[:max_files]
    
    timestamp_print(f"Found {len(text_files)} text files")
    
    # Load text data
    all_texts = []
    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                all_texts.append(text)
        except Exception as e:
            timestamp_print(f"Error reading {file_path}: {e}")
    
    timestamp_print(f"Loaded {len(all_texts)} text files")
    
    # Tokenize text
    timestamp_print("Tokenizing text...")
    tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(all_texts)
    
    # Create input-target pairs for next token prediction
    timestamp_print("Creating input-target pairs...")
    input_sequences = []
    target_sequences = []
    
    for text in all_texts:
        # Convert text to sequence of token IDs
        token_list = tokenizer.texts_to_sequences([text])[0]
        
        # Create input-target pairs
        for i in range(1, len(token_list)):
            # Input: tokens up to position i
            input_seq = token_list[:i]
            # Target: one-hot encoding of token at position i
            target_token = token_list[i]
            
            # Only use sequences that are not too long and have valid targets
            if len(input_seq) <= max_len and target_token < num_words:
                input_sequences.append(input_seq)
                target_sequences.append(target_token)
    
    timestamp_print(f"Created {len(input_sequences)} input-target pairs")
    
    # Pad input sequences
    timestamp_print("Padding input sequences...")
    input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='post')
    
    # Normalize input
    input_sequences = input_sequences / float(num_words)
    
    # Convert target to one-hot encoding
    timestamp_print("Converting targets to one-hot encoding...")
    target_one_hot = np.zeros((len(target_sequences), num_words))
    for i, target in enumerate(target_sequences):
        target_one_hot[i, target] = 1
    
    # Split into training and validation sets
    timestamp_print("Splitting into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        input_sequences, target_one_hot, test_size=0.1, random_state=42
    )
    
    timestamp_print(f"Training set: {X_train.shape[0]} samples")
    timestamp_print(f"Validation set: {X_val.shape[0]} samples")
    
    return X_train, X_val, y_train, y_val, tokenizer

def train_esn_model(data_dir, output_dir="SavedModels", max_files=None):
    """Train an ESN model for next token prediction."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    max_len = 20
    num_words = 10000
    X_train, X_val, y_train, y_val, tokenizer = load_and_preprocess_data(
        data_dir, max_files=max_files, max_len=max_len, num_words=num_words
    )
    
    # Create and train ESN model
    timestamp_print("Creating ESN model...")
    reservoir_size = 2000  # Larger reservoir for better performance
    
    # Clear CUDA cache before creating model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    # Create model
    esn = ESNModelGPU(
        input_size=max_len,
        reservoir_size=reservoir_size,
        output_size=num_words,
        spectral_radius=0.99,
        sparsity=0.05,
        input_scaling=1.0,
        leaking_rate=0.3
    )
    
    # Train model
    timestamp_print("Training model...")
    batch_size = 128
    esn.fit(X_train_tensor, y_train_tensor, regularization=1e-5, batch_size=batch_size)
    
    # Evaluate model
    timestamp_print("Evaluating model...")
    # Sample a subset of validation data to evaluate (to save time)
    val_subset_size = min(1000, len(X_val))
    val_indices = np.random.choice(len(X_val), val_subset_size, replace=False)
    X_val_subset = X_val[val_indices]
    y_val_subset = y_val[val_indices]
    
    # Make predictions
    y_pred = esn.predict(X_val_subset)
    
    # Calculate accuracy (top-1 and top-5)
    top1_correct = 0
    top5_correct = 0
    for i in range(len(y_pred)):
        true_token = np.argmax(y_val_subset[i])
        pred_tokens = np.argsort(y_pred[i])[-5:][::-1]  # Top 5 predictions
        
        if pred_tokens[0] == true_token:
            top1_correct += 1
        if true_token in pred_tokens:
            top5_correct += 1
    
    top1_accuracy = top1_correct / len(y_pred)
    top5_accuracy = top5_correct / len(y_pred)
    
    timestamp_print(f"Top-1 accuracy: {top1_accuracy:.4f}")
    timestamp_print(f"Top-5 accuracy: {top5_accuracy:.4f}")
    
    # Save model and tokenizer
    model_path = os.path.join(output_dir, "next_token_esn_model.pkl")
    tokenizer_path = os.path.join(output_dir, "next_token_tokenizer.pkl")
    
    timestamp_print(f"Saving model to {model_path}...")
    esn.save(model_path)
    
    timestamp_print(f"Saving tokenizer to {tokenizer_path}...")
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    timestamp_print("Training complete!")
    return esn, tokenizer

def generate_text(esn, tokenizer, seed_text="Hello", max_length=50, temperature=0.7):
    """Generate text using the trained ESN model."""
    # Convert seed text to sequence
    max_len = 20  # Same as during training
    num_words = tokenizer.num_words or 10000
    
    # Tokenize seed text
    seed_seq = tokenizer.texts_to_sequences([seed_text])[0]
    
    # Pad sequence
    padded_seq = pad_sequences([seed_seq], maxlen=max_len, padding='post')[0]
    
    # Generate text
    generated_text = seed_text
    
    for _ in range(max_length):
        # Normalize input
        x = padded_seq / float(num_words)
        
        # Predict next token
        prediction = esn.predict(np.array([x]))[0]
        
        # Apply temperature scaling
        prediction = np.exp(np.log(prediction + 1e-10) / temperature)
        prediction = prediction / np.sum(prediction)
        
        # Sample from the distribution
        next_token = np.random.choice(len(prediction), p=prediction)
        
        # Convert token to word
        next_word = ""
        for word, index in tokenizer.word_index.items():
            if index == next_token:
                next_word = word
                break
        
        # Add to generated text
        if next_word:
            generated_text += " " + next_word
        
        # Update seed sequence
        seed_seq.append(next_token)
        if len(seed_seq) > max_len:
            seed_seq = seed_seq[1:]
        
        # Update padded sequence
        padded_seq = pad_sequences([seed_seq], maxlen=max_len, padding='post')[0]
    
    return generated_text

if __name__ == "__main__":
    try:
        # Set data directory
        data_dir = "crawled_data"  # Change this to your data directory
        
        # Train model
        esn, tokenizer = train_esn_model(data_dir, max_files=1000)  # Limit to 1000 files for faster training
        
        # Generate some sample text
        print("\nGenerating sample text:")
        for seed in ["Hello", "The", "Once upon a time"]:
            generated = generate_text(esn, tokenizer, seed_text=seed, max_length=30, temperature=0.7)
            print(f"\nSeed: '{seed}'")
            print(f"Generated: '{generated}'")
        
        print("\nTraining and text generation complete!")
        
    except Exception as e:
        timestamp_print(f"Error: {str(e)}")
        # Save error log
        with open("error_log.txt", "a") as f:
            f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}: {str(e)}")