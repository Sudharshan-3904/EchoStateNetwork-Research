import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class EchoStateNetworkModular:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.95, sparsity=0.1, input_scaling=1.0, leaking_rate=1.0):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate

        self.W_in = np.random.uniform(-self.input_scaling, self.input_scaling, (self.reservoir_size, self.input_size + 1))
        self.W = np.random.uniform(-0.5, 0.5, (self.reservoir_size, self.reservoir_size))
        
        mask = np.random.rand(*self.W.shape) > self.sparsity
        self.W[mask] = 0
        
        eigenvalues = np.max(np.abs(np.linalg.eigvals(self.W)))
        self.W *= self.spectral_radius / eigenvalues

        self.W_out = None
        self.reservoir_state = np.zeros((self.reservoir_size, 1))

    def _update_reservoir(self, input_vector):
        input_vector = np.reshape(input_vector, (-1, 1))
        augmented_input = np.vstack((1, input_vector))
        pre_activation = np.dot(self.W_in, augmented_input) + np.dot(self.W, self.reservoir_state)
        self.reservoir_state = (1 - self.leaking_rate) * self.reservoir_state + self.leaking_rate * np.tanh(pre_activation)

    def fit(self, inputs, targets, regularization=1e-8):
        states = []
        for input_vector in inputs:
            self._update_reservoir(input_vector)
            states.append(self.reservoir_state.flatten())
        states = np.array(states)

        states = np.hstack((np.ones((states.shape[0], 1)), states))

        targets = np.array(targets)
        self.W_out = np.dot(np.linalg.pinv(np.dot(states.T, states) + regularization * np.eye(states.shape[1])), np.dot(states.T, targets))

    def predict(self, inputs):
        predictions = []
        for input_vector in inputs:
            self._update_reservoir(input_vector)
            augmented_state = np.vstack((1, self.reservoir_state))
            output = np.dot(self.W_out.T, augmented_state)
            predictions.append(output.flatten())
        return np.array(predictions)

# Directory containing your text files
data_dir = 'scraped_data'

# Read all text files in the directory
def read_text_files(data_dir):
    texts = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

# Read text files in batches
def read_text_files_batch(data_dir, start_idx, batch_size):
    texts = []
    filenames = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    end_idx = min(start_idx + batch_size, len(filenames))
    batch_files = filenames[start_idx:end_idx]
    
    for filename in batch_files:
        with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    
    return texts, end_idx < len(filenames)

# Preprocess the text data using TF-IDF
def preprocess_text(texts):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts).toarray()
    return X, vectorizer

# Train the ESN model
def train_esn(X, y, input_size, reservoir_size, output_size):
    esn = EchoStateNetworkModular(input_size, reservoir_size, output_size)
    esn.fit(X, y)
    return esn

# Save the model and vectorizer
def save_model(esn, vectorizer, model_path, vectorizer_path):
    joblib.dump(esn, model_path)
    joblib.dump(vectorizer, vectorizer_path)

# Modified main function to handle batches
def main():
    batch_size = 1000
    start_idx = 0
    has_more_files = True
    
    # Initialize vectorizer once
    all_texts = []
    while has_more_files:
        texts, has_more_files = read_text_files_batch(data_dir, start_idx, batch_size)
        all_texts.extend(texts)
        start_idx += batch_size
    
    # Fit vectorizer on all texts
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_texts)
    
    # Process in batches
    start_idx = 0
    has_more_files = True
    batch_number = 1
    
    while has_more_files:
        print(f"Processing batch {batch_number}...")
        texts, has_more_files = read_text_files_batch(data_dir, start_idx, batch_size)
        
        if not texts:  # Skip if no texts in batch
            break
            
        X = vectorizer.transform(texts).toarray()
        y = np.zeros((X.shape[0], 1))  # Placeholder for target values
        
        input_size = X.shape[1]
        reservoir_size = 1000
        output_size = 1
        
        # Train on current batch
        esn = train_esn(X, y, input_size, reservoir_size, output_size)
        
        # Save model and vectorizer with batch number
        model_path = f'esn_model_batch_{batch_number}.pkl'
        vectorizer_path = f'vectorizer_batch_{batch_number}.pkl'
        save_model(esn, vectorizer, model_path, vectorizer_path)
        
        start_idx += batch_size
        batch_number += 1
        print(f"Completed batch {batch_number-1}")

if __name__ == "__main__":
    main()
