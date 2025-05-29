import pickle
import numpy as np
import os
import random
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
from src.models.ESN_Model import ESNModelGPU, tprint


def load_model_and_tokenizer():
    """Load the combined ESN model and tokenizer."""
    model_path = os.path.join("SavedModels", "combined_esn_model.pkl")
    tokenizer_path = os.path.join("SavedModels", "combined_tokenizer.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    
    tprint("Loading model and tokenizer...")
    with open(model_path, "rb") as f:
        cpu_esn = pickle.load(f)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    
    # Convert the CPU model to GPU
    esn = ESNModelGPU(cpu_esn)
    
    tprint(f"Model loaded and transferred to GPU. Reservoir size: {esn.reservoir_size}")
    tprint(f"Tokenizer loaded. Vocabulary size: {len(tokenizer.word_index)}")
    
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
    tprint("Warming up GPU...")
    dummy_input = np.zeros((1, max_len))
    esn.predict(dummy_input)
    tprint("GPU ready for inference")
    
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
        tprint(f"Error: {str(e)}")
        # Save error log
        with open("error_log.txt", "a") as f:
            f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}: {str(e)}")