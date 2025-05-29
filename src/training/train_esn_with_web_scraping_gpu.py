import os
import time
import pickle
import numpy as np
import torch
import requests
from bs4 import BeautifulSoup
import concurrent.futures
import urllib.parse
from collections import deque
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize
from src.models.ESN_Model import ESNModelGPU, tprint

# Download NLTK data if needed
try:
    nltk.download('punkt')
except Exception as e:
    print(f"Error downloading NLTK data: {e}")


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
        tprint(f"Error fetching {url}: {e}")
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
    
    tprint(f"Starting Wikipedia crawler with limit of {max_urls} URLs...")
    
    # Add more seed URLs using Wikipedia's random article feature
    tprint("Adding additional seed URLs from Wikipedia's random article feature...")
    for _ in range(10):  # Add 10 random articles as seeds
        try:
            random_url = "https://en.wikipedia.org/wiki/Special:Random"
            response = requests.get(random_url, timeout=10)
            if response.status_code == 200:
                random_article_url = response.url
                if random_article_url not in visited_urls and random_article_url not in urls_to_process:
                    tprint(f"Added random seed: {random_article_url}")
                    urls_to_process.append(random_article_url)
            time.sleep(0.5)  # Respect rate limits
        except Exception as e:
            tprint(f"Error getting random article: {str(e)}")
    
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
                            
                            tprint(f"Scraped {url} - {len(text)} chars, {len(links)} links")
                            tprint(f"Progress: {len(visited_urls)}/{max_urls} URLs, {len(scraped_texts)} texts")
                except Exception as e:
                    tprint(f"Error processing {url}: {e}")
            
            # Respect Wikipedia's robots.txt
            time.sleep(1)
    
    tprint(f"Crawling complete. Scraped {len(scraped_texts)} texts from {len(visited_urls)} URLs.")
    return scraped_texts

def preprocess_scraped_data(texts, max_len=20, num_words=10000):
    """Preprocess scraped text data for next token prediction."""
    tprint("Tokenizing text...")
    tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    # Create input-target pairs for next token prediction
    tprint("Creating input-target pairs...")
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
    
    tprint(f"Created {len(input_sequences)} input-target pairs")
    
    # Pad input sequences
    tprint("Padding input sequences...")
    input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='post')
    
    # Normalize input
    input_sequences = input_sequences / float(num_words)
    
    # Convert target to one-hot encoding
    tprint("Converting targets to one-hot encoding...")
    target_one_hot = np.zeros((len(target_sequences), num_words))
    for i, target in enumerate(target_sequences):
        target_one_hot[i, target] = 1
    
    # Split into training and validation sets
    tprint("Splitting into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        input_sequences, target_one_hot, test_size=0.1, random_state=42
    )
    
    tprint(f"Training set: {X_train.shape[0]} samples")
    tprint(f"Validation set: {X_val.shape[0]} samples")
    
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
        tprint(f"Found existing scraped data in {scraped_data_dir}")
        # Load existing texts
        texts = []
        for filename in os.listdir(scraped_data_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(scraped_data_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())
        tprint(f"Loaded {len(texts)} existing text files")
    else:
        # Crawl Wikipedia
        max_urls = 500  # Adjust based on your needs
        texts = crawl_wikipedia(seed_urls, max_urls=max_urls, output_dir=scraped_data_dir)
        tprint(f"Scraped {len(texts)} texts from Wikipedia")