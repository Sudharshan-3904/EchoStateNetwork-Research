import os
import time
import pickle
import numpy as np
import torch
import glob
import requests
from bs4 import BeautifulSoup
import concurrent.futures
import urllib.parse
from collections import deque
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data if needed
try:
    nltk.download('punkt')
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

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

# Web scraping functions
def fetch_url(url):
    """Fetch the URL content."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        timestamp_print(f"Error fetching {url}: {e}")
        return None

def parse_html(html, url):
    """Parse HTML and extract text content."""
    if html is None:
        return None, []
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Extract article title
    title_tag = soup.find('h1', {'id': 'firstHeading'})
    article_title = title_tag.text if title_tag else 'Unknown_Article'
    
    # Extract text content
    # For Wikipedia, focus on the content div
    content_div = soup.find('div', {'id': 'mw-content-text'})
    if content_div:
        paragraphs = content_div.find_all('p')
    else:
        paragraphs = soup.find_all('p')
    
    text = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
    
    # Extract links to follow
    links = []
    if content_div:
        for a_tag in content_div.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith('/wiki/') and ':' not in href and 'Main_Page' not in href:
                full_url = urllib.parse.urljoin(url, href)
                links.append(full_url)
    
    return text, links

def crawl_wikipedia(seed_urls, max_urls=1000, output_dir='scraped_data'):
    """Crawl Wikipedia starting from seed URLs."""
    os.makedirs(output_dir, exist_ok=True)
    
    visited_urls = set()
    urls_to_process = deque(seed_urls)
    scraped_texts = []
    
    timestamp_print(f"Starting Wikipedia crawler with limit of {max_urls} URLs...")
    
    # Add more seed URLs using Wikipedia's random article feature
    timestamp_print("Adding additional seed URLs from Wikipedia's random article feature...")
    for _ in range(10):  # Add 10 random articles as seeds
        try:
            random_url = "https://en.wikipedia.org/wiki/Special:Random"
            response = requests.get(random_url, timeout=10)
            if response.status_code == 200:
                random_article_url = response.url
                if random_article_url not in visited_urls and random_article_url not in urls_to_process:
                    timestamp_print(f"Added random seed: {random_article_url}")
                    urls_to_process.append(random_article_url)
            time.sleep(0.5)  # Respect rate limits
        except Exception as e:
            timestamp_print(f"Error getting random article: {str(e)}")
    
    # Use ThreadPoolExecutor for parallel fetching
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        while urls_to_process and len(visited_urls) < max_urls:
            # Get next batch of URLs to process
            batch_size = min(20, len(urls_to_process))
            batch_urls = []
            for _ in range(batch_size):
                if urls_to_process:
                    url = urls_to_process.popleft()
                    if url not in visited_urls:
                        batch_urls.append(url)
                        visited_urls.add(url)
            
            if not batch_urls:
                continue
            
            # Fetch URLs in parallel
            future_to_url = {executor.submit(fetch_url, url): url for url in batch_urls}
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    html = future.result()
                    if html:
                        text, links = parse_html(html, url)
                        if text:
                            # Save text to file
                            filename = os.path.join(output_dir, f"{len(scraped_texts)}.txt")
                            with open(filename, 'w', encoding='utf-8') as f:
                                f.write(text)
                            scraped_texts.append(text)
                            
                            # Add new links to process
                            for link in links:
                                if link not in visited_urls and link not in urls_to_process:
                                    urls_to_process.append(link)
                            
                            timestamp_print(f"Scraped {url} - {len(text)} chars, {len(links)} links")
                            timestamp_print(f"Progress: {len(visited_urls)}/{max_urls} URLs, {len(scraped_texts)} texts")
                except Exception as e:
                    timestamp_print(f"Error processing {url}: {e}")
            
            # Respect Wikipedia's robots.txt
            time.sleep(1)
    
    timestamp_print(f"Crawling complete. Scraped {len(scraped_texts)} texts from {len(visited_urls)} URLs.")
    return scraped_texts

def preprocess_scraped_data(texts, max_len=20, num_words=10000):
    """Preprocess scraped text data for next token prediction."""
    timestamp_print("Tokenizing text...")
    tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    # Create input-target pairs for next token prediction
    timestamp_print("Creating input-target pairs...")
    input_sequences = []
    target_sequences = []
    
    for text in texts:
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            # Convert text to sequence of token IDs
            token_list = tokenizer.texts_to_sequences([sentence])[0]
            
            if len(token_list) < 2:  # Skip very short sequences
                continue
            
            # Create input-target pairs
            for i in range(1, len(token_list)):
                # Input: tokens up to position i
                input_seq = token_list[:i]
                # Target: token at position i
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

def train_esn_model_with_web_scraping(output_dir="SavedModels"):
    """Train an ESN model using web-scraped data."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Web scraping
    seed_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Deep_learning",
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://en.wikipedia.org/wiki/Computer_science",
        "https://en.wikipedia.org/wiki/Data_science",
        "https://en.wikipedia.org/wiki/Neural_network",
        "https://en.wikipedia.org/wiki/Robotics",
        "https://en.wikipedia.org/wiki/Computer_vision",
        "https://en.wikipedia.org/wiki/Expert_system",
        # Add more diverse topics
        "https://en.wikipedia.org/wiki/History",
        "https://en.wikipedia.org/wiki/Science",
        "https://en.wikipedia.org/wiki/Mathematics",
        "https://en.wikipedia.org/wiki/Physics",
        "https://en.wikipedia.org/wiki/Biology",
        "https://en.wikipedia.org/wiki/Chemistry",
        "https://en.wikipedia.org/wiki/Literature",
        "https://en.wikipedia.org/wiki/Art",
        "https://en.wikipedia.org/wiki/Music",
        "https://en.wikipedia.org/wiki/Film"
    ]
    
    scraped_data_dir = "scraped_data"
    
    # Check if we already have scraped data
    if os.path.exists(scraped_data_dir) and len(os.listdir(scraped_data_dir)) > 0:
        timestamp_print(f"Found existing scraped data in {scraped_data_dir}")
        # Load existing texts
        texts = []
        for filename in os.listdir(scraped_data_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(scraped_data_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())
        timestamp_print(f"Loaded {len(texts)} existing text files")
    else:
        # Crawl Wikipedia
        max_urls = 500  # Adjust based on your needs
        texts = crawl_w