import pickle
import numpy as np
import os
import random
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

def timestamp_print(*args, **kwargs):
    """Print with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}]", *args, **kwargs)

class ESNModelGPU:
    """GPU-accelerated wrapper for the ESN model"""
    def __init__(self, cpu_model):
        # Check if CUDA is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            timestamp_print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            timestamp_print(f"GPU Memory: {total_memory:.2f} GB")
        else:
            timestamp_print("CUDA not available. Using CPU.")
        
        # Copy model parameters to GPU
        self.input_size = cpu_model.input_size
        self.reservoir_size = cpu_model.reservoir_size
        self.output_size = cpu_model.output_size
        self.spectral_radius = cpu_model.spectral_radius
        self.sparsity = cpu_model.sparsity
        self.input_scaling = cpu_model.input_scaling
        self.leaking_rate = cpu_model.leaking_rate
        
        # Convert weights to PyTorch tensors on GPU
        self.W_in = torch.tensor(cpu_model.W_in, device=self.device, dtype=torch.float32)
        self.W = torch.tensor(cpu_model.W, device=self.device, dtype=torch.float32)
        self.W_out = torch.tensor(cpu_model.W_out, device=self.device, dtype=torch.float32) if cpu_model.W_out is not None else None
        self.reservoir_state = torch.tensor(cpu_model.reservoir_state, device=self.device, dtype=torch.float32)
        
        # Enable CUDA optimizations
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            # Enable TensorFloat-32 for faster computation on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
    
    def _update_reservoir(self, input_vector):
        """Update the reservoir state with the given input vector."""
        # Convert input to tensor if it's not already
        if not isinstance(input_vector, torch.Tensor):
            input_vector = torch.tensor(input_vector, device=self.device, dtype=torch.float32)
        
        # Add bias term
        input_with_bias = torch.cat([torch.ones(1, device=self.device), input_vector])
        
        # Calculate new state
        new_state = torch.tanh(torch.matmul(self.W_in, input_with_bias) + 
                              torch.matmul(self.W, self.reservoir_state))
        
        # Apply leaking rate
        self.reservoir_state = (1 - self.leaking_rate) * self.reservoir_state + self.leaking_rate * new_state
    
    def predict(self, inputs):
        """Predict outputs for the given inputs."""
        # Convert inputs to tensor if it's not already
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float32)
        
        predictions = []
        with torch.no_grad():  # Disable gradient calculation for inference
            for i in range(inputs.shape[0]):
                input_vector = inputs[i]
                self._update_reservoir(input_vector)
                augmented_state = torch.cat([torch.ones(1, device=self.device), self.reservoir_state.flatten()])
                output = torch.matmul(self.W_out.T, augmented_state)
                predictions.append(output.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_with_temperature(self, inputs, temperature=0.7):
        """Predict with temperature scaling for more diverse outputs."""
        # Get raw predictions
        predictions = self.predict(inputs)
        
        # Apply temperature scaling (done on CPU as it's just post-processing)
        scaled_predictions = predictions / temperature
        
        return scaled_predictions

def load_model_and_tokenizer():
    """Load the combined ESN model and tokenizer."""
    model_path = os.path.join("SavedModels", "combined_esn_model.pkl")
    tokenizer_path = os.path.join("SavedModels", "combined_tokenizer.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    
    timestamp_print("Loading model and tokenizer...")
    with open(model_path, "rb") as f:
        cpu_esn = pickle.load(f)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    
    # Convert the CPU model to GPU
    esn = ESNModelGPU(cpu_esn)
    
    timestamp_print(f"Model loaded and transferred to GPU. Reservoir size: {esn.reservoir_size}")
    timestamp_print(f"Tokenizer loaded. Vocabulary size: {len(tokenizer.word_index)}")
    
    return esn, tokenizer

def chat_with_esn(esn, tokenizer, max_len=20, temperature=0.7):
    """Interactive chat with the ESN model."""
    print("\n🤖 GPU-Accelerated ESN Chatbot is ready. Type 'exit' to quit.")
    
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    
    # Fallback responses for when the model doesn't generate good output
    fallback_responses = [
        "I'm still learning. Could you ask me something else?",
        "That's an interesting point. Let me think about that more.",
        "I don't have enough information to respond properly to that yet.",
        "I'm not sure how to respond to that. Could you try a different question?",
        "I'm learning from our conversation. Please continue!"
    ]
    
    # Track conversation history for context
    conversation_history = []
    
    # Warm up the GPU
    timestamp_print("Warming up GPU...")
    dummy_input = np.zeros((1, max_len))
    esn.predict(dummy_input)
    timestamp_print("GPU ready for inference")
    
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            print("Bot: Please say something!")
            continue
            
        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break
        
        # Add to conversation history
        conversation_history.append(user_input)
        if len(conversation_history) > 5:  # Keep only last 5 exchanges
            conversation_history.pop(0)
        
        try:
            # Preprocess input
            input_seq = tokenizer.texts_to_sequences([user_input])
            input_pad = pad_sequences(input_seq, maxlen=max_len, padding='post')
            input_pad = input_pad / float(tokenizer.num_words or 5000)

            # Measure inference time
            start_time = time.time()
            
            # Get prediction
            output = esn.predict_with_temperature(input_pad, temperature=temperature)
            
            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Safer probability calculation
            output_probs = output[0] * (tokenizer.num_words or 5000)
            output_probs = np.clip(output_probs, 1e-10, None)  # Clip to avoid negative/zero values
            
            # Apply temperature scaling for more diverse responses
            scaled_probs = np.power(output_probs, 1.0 / temperature)
            
            # Normalize probabilities
            sum_probs = np.sum(scaled_probs)
            if sum_probs > 0:
                output_probs = scaled_probs / sum_probs
            else:
                # If sum is zero, use uniform distribution
                output_probs = np.ones_like(output_probs) / len(output_probs)
            
            # Ensure no NaN values
            if np.any(np.isnan(output_probs)):
                output_probs = np.ones_like(output_probs) / len(output_probs)
            
            # Sample from probability distribution
            try:
                output_seq = np.random.choice(len(output_probs), size=max_len, p=output_probs)
            except ValueError as e:
                # Fallback to argmax
                output_seq = np.argsort(output_probs)[-max_len:]
            
            # Convert indices to words with improved filtering
            response_words = []
            prev_word = ""
            for idx in output_seq:
                if idx > 0 and idx < len(reverse_word_index) + 1:  # Check if index is valid
                    word = reverse_word_index.get(idx, '')
                    if (word and 
                        word != '<OOV>' and 
                        word != prev_word and  # Remove duplicates
                        len(word) > 1 and  # Filter single characters
                        word.isalnum()):  # Only allow alphanumeric words
                        response_words.append(word)
                        prev_word = word
            
            # Improve response formatting
            response_words = response_words[:15]  # Limit to 15 words for more complete responses
            response = ' '.join(response_words)
            
            # Check if response is empty or too short
            if not response or len(response_words) < 3:
                # Use fallback response
                response = random.choice(fallback_responses)
            else:
                # Format the response
                response = response.capitalize()
                if not response.endswith(('.', '!', '?')):
                    response += '.'
                    
            # Quality check - if response contains too many repeated words or looks nonsensical
            word_counts = {}
            for word in response_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # If any word appears more than twice or response has less than 3 unique words
            if max(word_counts.values(), default=0) > 2 or len(word_counts) < 3:
                response = random.choice(fallback_responses)
                
            print(f"Bot: {response}")
            print(f"(Inference time: {inference_time:.2f} ms)")
            
        except Exception as e:
            print(f"Error processing input: {str(e)}")
            print(f"Bot: {random.choice(fallback_responses)}")

if __name__ == "__main__":
    try:
        # Clear CUDA cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        esn, tokenizer = load_model_and_tokenizer()
        chat_with_esn(esn, tokenizer)
    except Exception as e:
        timestamp_print(f"Error: {str(e)}")
        # Save error log
        with open("error_log.txt", "a") as f:
            f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}: {str(e)}")