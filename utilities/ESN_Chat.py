import numpy as np
import joblib
import torch
import json

# Load the trained model and vectorizer
def load_model(model_path, vectorizer_path):
    esn = joblib.load(model_path)
    print(f"Model loaded. W_out shape: {esn.W_out.shape if esn.W_out is not None else 'None'}")
    # Convert W_out to float32 if it exists
    if esn.W_out is not None:
        esn.W_out = esn.W_out.to(dtype=torch.float32)
    vectorizer = joblib.load(vectorizer_path)
    print(f"Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}")
    return esn, vectorizer

# Preprocess the user input
def preprocess_input(input_text, vectorizer):
    # Basic text cleaning
    input_text = input_text.lower().strip()
    # Remove multiple spaces
    input_text = ' '.join(input_text.split())
    # Convert to vector
    input_vector = vectorizer.transform([input_text]).toarray()
    return input_vector

# Generate a response from the model
def generate_response(esn, input_vector):
    prediction = esn.predict(input_vector)
    
    # Debug information
    print(f"Debug \t\t - Raw prediction shape: {prediction.shape}")
    print(f"Debug \t\t - Raw prediction values: {prediction[0][:5]}...")  # Show first 5 values
    
    # Increase sensitivity threshold
    threshold = 1e-3  # Changed from 1e-5
    
    if np.all(np.abs(prediction[0]) < threshold):
        # Add more contextual default responses
        default_responses = [
            "I'm not sure how to respond to that. Could you rephrase your question?",
            "I'm still learning about that topic. Could you try asking differently?",
            "I don't have enough information to provide a good response to that.",
            "Could you please provide more context or rephrase your question?"
        ]
        return np.random.choice(default_responses)
    
    response = format_response(prediction[0])
    return response

def format_response(prediction):
    if isinstance(prediction, (np.ndarray, list)):
        # Normalize predictions
        pred_norm = prediction / np.max(np.abs(prediction))
        # Filter out weak signals
        filtered_pred = [str(x) for x in pred_norm if abs(x) > 0.1]
        
        if not filtered_pred:
            return "I'm still learning how to respond to that type of input."
            
        # Limit response length
        response = ' '.join(filtered_pred[:10])  # Limit to first 10 significant values
    else:
        response = str(prediction)
    
    # Basic text cleanup
    response = response.strip()
    response = response.capitalize()
    if not response.endswith(('.', '!', '?')):
        response += '.'
    
    return response

# Load configuration from JSON file
def load_config(config_path='config.json'):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        return {
            'model_path': 'SavedModels/esn_model.pkl',
            'vectorizer_path': 'SavedModels/vectorizer.pkl',
            'max_response_length': 100,
            'temperature': 0.7
        }

# Main function for chatbot interaction
def chatbot():
    try:
        config = load_config()
        model_path = config['model_path']
        vectorizer_path = config['vectorizer_path']
        esn, vectorizer = load_model(model_path, vectorizer_path)
    except FileNotFoundError:
        print("Error: Model files not found. Please ensure the SavedModels directory exists.")
        return
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    print("Chatbot initialized. Type 'exit' or 'quit' to end conversation.")
    
    while True:
        try:
            user_input = input("You\t\t: ").strip()
            if not user_input:
                print("Chatbot: Please type something!")
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("Chatbot: Goodbye!")
                break

            input_vector = preprocess_input(user_input, vectorizer)
            response = generate_response(esn, input_vector)
            print(f"Chatbot\t\t: {response}")
            
        except KeyboardInterrupt:
            print("\nChatbot: Goodbye!")
            break
        except Exception as e:
            print(f"Chatbot: Sorry, I encountered an error: {str(e)}")

if __name__ == '__main__':
    chatbot()