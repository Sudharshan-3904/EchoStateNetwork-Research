import requests
from bs4 import BeautifulSoup
import numpy as np
import pickle
import time
import csv
import os
from collections import deque
import random

from src.models.ESN_Model import EchoStateNetworkModular, tprint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
nltk.data.path.append("E:/Tester/ESN Modelling/esn_scraping/nltk_data")

# Download both punkt and punkt_tab
try:
    nltk.download('punkt', download_dir="E:/Tester/ESN Modelling/esn_scraping/nltk_data")
    nltk.download('punkt_tab', download_dir="E:/Tester/ESN Modelling/esn_scraping/nltk_data")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Import after downloads
from nltk.tokenize import sent_tokenize

# Test tokenization
try:
    test_result = sent_tokenize("NLTK should now tokenize this. Without errors.")
    print("Test: ", test_result)
except Exception as e:
    print(f"Tokenization error: {e}")
    exit(1)

# ---------- Step 1: Scrape Paragraphs ----------
def scrape_paragraphs(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
        return paragraphs
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

def get_wiki_links(url, visited_urls, max_links=100):
    """Extract Wikipedia links from a page."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find the main content area to focus on relevant links
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if not content_div:
            content_div = soup  # Fallback to entire page if main content not found
        
        links = []
        for a_tag in content_div.find_all('a', href=True):
            href = a_tag['href']
            # Filter for Wikipedia article links
            if href.startswith('/wiki/') and ':' not in href and 'Main_Page' not in href:
                full_url = 'https://en.wikipedia.org' + href
                if full_url not in visited_urls and full_url not in links:
                    links.append(full_url)
                    if len(links) >= max_links:
                        break
        
        return links
    except Exception as e:
        print(f"Error getting links from {url}: {e}")
        return []

def crawl_wikipedia(seed_urls, max_urls=1000, urls_per_batch=500, links_per_page=500):
    """
    Continuous crawler that processes URLs until reaching the max_urls limit.
    Returns a list of batches, where each batch is a list of URLs.
    """
    all_batches = []
    visited_urls = set()
    urls_to_process = deque(seed_urls)
    total_urls_collected = 0
    
    # Track overall crawling time
    overall_start_time = time.time()
    
    tprint(f"🕸️ Starting Wikipedia crawler with limit of {max_urls} URLs in batches of {urls_per_batch}...")
    
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
    
    tprint(f"Starting with {len(urls_to_process)} seed URLs")
    
    # Process URLs until we reach the limit
    while total_urls_collected < max_urls and urls_to_process:
        # Collect URLs for this batch
        batch_urls = []
        batch_start_time = time.time()
        
        tprint(f"\n====== Collecting Batch {len(all_batches) + 1} ======\n")
        
        # If we don't have enough URLs to process, add more random articles
        if len(urls_to_process) < urls_per_batch:
            tprint("Adding more random articles to URL queue...")
            for _ in range(20):  # Try to add 20 random articles
                try:
                    random_url = "https://en.wikipedia.org/wiki/Special:Random"
                    response = requests.get(random_url, timeout=10)
                    if response.status_code == 200:
                        random_article_url = response.url
                        if random_article_url not in visited_urls and random_article_url not in urls_to_process:
                            urls_to_process.append(random_article_url)
                    time.sleep(0.5)  # Respect rate limits
                except Exception as e:
                    tprint(f"Error getting random article: {str(e)}")
        
        # Process URLs for this batch
        while len(batch_urls) < urls_per_batch and urls_to_process and total_urls_collected < max_urls:
            try:
                current_url = urls_to_process.popleft()
                
                if current_url in visited_urls:
                    continue
                
                # Verify URL is accessible
                response = requests.head(current_url, timeout=10)
                if response.status_code != 200:
                    continue
                
                batch_urls.append(current_url)
                visited_urls.add(current_url)
                total_urls_collected += 1
                
                # Get links from this URL for future batches
                links = get_wiki_links(current_url, visited_urls, links_per_page)
                for link in links:
                    if link not in visited_urls and link not in urls_to_process:
                        urls_to_process.append(link)
                
                # Progress update
                tprint(f"Added URL {len(batch_urls)}/{urls_per_batch}: {current_url}")
                tprint(f"Found {len(links)} new links, queue size: {len(urls_to_process)}")
                tprint(f"Total URLs collected: {total_urls_collected}/{max_urls}")
                
                # Respect Wikipedia's robots.txt
                time.sleep(0.5)
                
            except Exception as e:
                tprint(f"Error processing URL: {str(e)}")
                continue
        
        # Add batch if not empty
        if batch_urls:
            all_batches.append(batch_urls)
            
            # Calculate batch statistics
            batch_elapsed = time.time() - batch_start_time
            overall_elapsed = time.time() - overall_start_time
            
            tprint(
                f"✅ Batch {len(all_batches)} complete: {len(batch_urls)} URLs collected\n"
                f"   Batch time: {batch_elapsed:.1f}s\n"
                f"   Overall time: {overall_elapsed:.1f}s"
            )
            
            # Save this batch's URLs
            save_dir = "saved_pkls"
            os.makedirs(save_dir, exist_ok=True)
            urls_file = os.path.join(save_dir, f"crawled_urls_batch_{len(all_batches)}.txt")
            with open(urls_file, "w") as f:
                for url in batch_urls:
                    f.write(url + "\n")
        else:
            tprint(f"⚠️ Batch has no URLs, skipping...")
        
        # Check if we've reached the URL limit
        if total_urls_collected >= max_urls:
            tprint(f"🎯 Reached URL limit of {max_urls}. Stopping crawler.")
            break
    
    # Final statistics
    overall_elapsed = time.time() - overall_start_time
    total_urls = sum(len(batch) for batch in all_batches)
    
    tprint(
        f"\n🏁 Crawling complete!\n"
        f"   Total batches: {len(all_batches)}\n"
        f"   Total URLs: {total_urls}\n"
        f"   Total time: {overall_elapsed:.1f}s"
    )
    
    return all_batches

def main_batch_processing():
    # Create necessary directories
    save_dir = "saved_pkls"
    os.makedirs(save_dir, exist_ok=True)
    
    # Expanded seed URLs for better coverage
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
        "https://en.wikipedia.org/wiki/History",
        "https://en.wikipedia.org/wiki/Science",
        "https://en.wikipedia.org/wiki/Mathematics",
        "https://en.wikipedia.org/wiki/Physics",
    ]
    
    # Batch Settings
    max_urls = 10000  # Total URL limit
    urls_per_batch = 500  # URLs per batch
    
    # Update progress file path
    progress_file = os.path.join(save_dir, "batch_progress.txt")
    
    # Check if we have previous progress
    completed_batches = 0
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            completed_batches = len(f.readlines())
        tprint(f"Found previous progress: {completed_batches} batches completed")
    
    # Crawl all URLs until reaching the limit
    all_batches = crawl_wikipedia(
        seed_urls,
        max_urls=max_urls,
        urls_per_batch=urls_per_batch
    )
    
    # Collect all sentence pairs from all batches
    tprint("\n🔄 Collecting all sentence pairs from all batches...")
    all_input_sentences = []
    all_target_sentences = []
    
    # If we have no batches, use seed URLs directly
    if not all_batches and seed_urls:
        tprint("No batches collected, using seed URLs directly...")
        all_batches = [seed_urls[:urls_per_batch]]
    
    # Process each batch and train the model incrementally
    esn = None
    tokenizer = None
    
    for batch_idx, batch_urls in enumerate(all_batches):
        batch_num = completed_batches + batch_idx + 1
        tprint(f"\n====== Processing Batch {batch_num} ======\n")
        
        if not batch_urls:
            tprint(f"⚠️ No URLs collected for batch {batch_num}, skipping...")
            continue
        
        # Extract text from batch
        tprint(f"🔍 Extracting text from batch {batch_num} pages...")
        batch_inputs = []
        batch_targets = []
        
        for i, url in enumerate(batch_urls, 1):
            tprint(f"Processing page {i}/{len(batch_urls)}: {url}")
            paragraphs = scrape_paragraphs(url)
            for para in paragraphs:
                sentences = sent_tokenize(para)
                for i in range(len(sentences) - 1):
                    batch_inputs.append(sentences[i])
                    batch_targets.append(sentences[i + 1])
            time.sleep(0.3)  # Reduced from 0.5 to 0.3
        
        tprint(f"✅ Collected {len(batch_inputs)} sentence pairs from batch {batch_num}")
        
        # Save progress
        with open(progress_file, "a") as f:
            f.write(f"Batch {batch_num}: {len(batch_inputs)} sentence pairs collected\n")
        
        # Skip if insufficient data
        if len(batch_inputs) < 50:
            tprint(f"⚠️ Batch {batch_num} has insufficient data, skipping...")
            continue
            
        # Preprocess batch data
        tprint(f"🔄 Preprocessing batch {batch_num} data...")
        max_len = 20
        num_words = 5000
        
        if tokenizer is None:
            X, y, tokenizer = preprocess_text(batch_inputs, batch_targets, max_len=max_len, num_words=num_words)
        else:
            X, y, _ = preprocess_text(batch_inputs, batch_targets, tokenizer=tokenizer, max_len=max_len, num_words=num_words)
        
        # Initialize or update ESN
        tprint(f"🧠 Training on batch {batch_num}...")
        if esn is None:
            esn = EchoStateNetworkModular(
                input_size=X.shape[1],
                reservoir_size=2000,
                output_size=y.shape[1],
                spectral_radius=0.99,
                sparsity=0.05,
                input_scaling=1.2,
                leaking_rate=0.2
            )
        
        # Train on batch
        batch_size = 128
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch_X, batch_y = X[start:end], y[start:end]
            esn.fit(batch_X, batch_y, regularization=1e-9)
            tprint(f"Trained on mini-batch {start//batch_size+1}/{len(X)//batch_size+1}")
        
        # Save intermediate models
        tprint(f"💾 Saving model and tokenizer for batch {batch_num}...")
        model_path = os.path.join(save_dir, f"esn_model_batch_{batch_num}.pkl")
        tokenizer_path = os.path.join(save_dir, f"esn_tokenizer_batch_{batch_num}.pkl")
        
        with open(model_path, "wb") as f:
            pickle.dump(esn, f)
        with open(tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)
        
        # Add to overall collection for final training
        all_input_sentences.extend(batch_inputs)
        all_target_sentences.extend(batch_targets)
    
    # Final training on all collected data
    tprint("\n🚀 Final training on all collected data...")
    if len(all_input_sentences) > 0:
        X, y, tokenizer = preprocess_text(all_input_sentences, all_target_sentences, tokenizer=tokenizer, max_len=max_len, num_words=num_words)
        
        # Train on all data with multiple epochs
        num_epochs = 3
        batch_size = 128
        
        for epoch in range(num_epochs):
            tprint(f"\n==== Final Training Epoch {epoch+1}/{num_epochs} ====")
            
            # Shuffle data for each epoch
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for start in range(0, len(X_shuffled), batch_size):
                end = min(start + batch_size, len(X_shuffled))
                batch_X, batch_y = X_shuffled[start:end], y_shuffled[start:end]
                esn.fit(batch_X, batch_y, regularization=1e-9)
                
                # Progress update
                progress = min(100, (end / len(X_shuffled)) * 100)
                tprint(f"Epoch {epoch+1}/{num_epochs} - Batch {start//batch_size+1}/{len(X_shuffled)//batch_size+1} - {progress:.1f}%")
        
        # Save final model and tokenizer
        tprint("\n💾 Saving final model and tokenizer...")
        model_path = os.path.join(save_dir, "esn_model_final.pkl")
        tokenizer_path = os.path.join(save_dir, "esn_tokenizer_final.pkl")
        
        with open(model_path, "wb") as f:
            pickle.dump(esn, f)
        with open(tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)
        
        tprint(f"Model saved to {model_path}")
        tprint(f"Tokenizer saved to {tokenizer_path}")
    else:
        tprint("⚠️ No data collected. Cannot train model.")
        return None, None
    
    return esn, tokenizer

# ---------- Step 2: Preprocessing ----------
def preprocess_text(inputs, targets, tokenizer=None, max_len=100, num_words=5000):
    """
    Preprocess text data for ESN training.
    
    Args:
        inputs: List of input sentences
        targets: List of target sentences
        tokenizer: Optional pre-fitted tokenizer
        max_len: Maximum sequence length
        num_words: Maximum number of words in vocabulary
        
    Returns:
        X: Preprocessed input sequences
        y: Preprocessed target sequences
        tokenizer: Fitted tokenizer
    """
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
        # Fit on both inputs and targets to build vocabulary
        all_texts = inputs + targets
        tokenizer.fit_on_texts(all_texts)
        
        # Print vocabulary size for debugging
        print(f"Vocabulary size: {len(tokenizer.word_index)}")

    input_seq = tokenizer.texts_to_sequences(inputs)
    target_seq = tokenizer.texts_to_sequences(targets)

    X = pad_sequences(input_seq, maxlen=max_len, padding='post')
    y = pad_sequences(target_seq, maxlen=max_len, padding='post')

    # Normalize between 0 and 1
    X = X / float(num_words)
    y = y / float(num_words)

    return X, y, tokenizer

# ---------- Step 3: Train in Batches ----------
def train_in_batches(X, y, batch_size=128, csv_path=None):
    if csv_path is None:
        csv_path = os.path.join("saved_pkls", "batch_times.csv")
    
    input_size = X.shape[1]
    output_size = y.shape[1]

    esn = EchoStateNetworkModular(
        input_size=input_size,
        reservoir_size=5000,
        output_size=output_size,
        spectral_radius=0.95,
        sparsity=0.1,
        input_scaling=1.0,
        leaking_rate=0.3
    )

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        os.remove(csv_path)

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Batch Number', 'Time (s)'])

        for start in range(0, len(X), batch_size):
            end = start + batch_size
            batch_X, batch_y = X[start:end], y[start:end]

            t0 = time.time()
            esn.fit(batch_X, batch_y)
            t1 = time.time()

            batch_time = round(t1 - t0, 4)
            writer.writerow([start // batch_size + 1, batch_time])
            tprint(f"Trained on batch {start // batch_size + 1}, Time: {batch_time}s")

    return esn

# ---------- Step 4: Inference ----------
def chat_with_esn(esn, tokenizer, max_len=20):
    print("🤖 Chatbot is ready. Type 'exit' to quit.")
    
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    
    # Add minimal fallback responses for when the model doesn't generate good output
    fallback_responses = [
        "I'm still learning. Could you ask me something else?",
        "That's an interesting point. Let me think about that more.",
        "I don't have enough information to respond properly to that yet.",
        "I'm not sure how to respond to that. Could you try a different question?",
        "I'm learning from our conversation. Please continue!"
    ]
    
    # Track conversation history for context
    conversation_history = []
    
    while True:
        user_input = input("You: ").strip()
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

            # Get prediction
            output = esn.predict(input_pad)
            
            # Debug information (only in development)
            # print(f"Debug - Input shape: {input_pad.shape}, Output shape: {output.shape}")
            
            # Safer probability calculation
            output_probs = output[0] * (tokenizer.num_words or 5000)
            output_probs = np.clip(output_probs, 1e-10, None)  # Clip to avoid negative/zero values
            
            # Apply temperature scaling for more diverse responses
            temperature = 0.7
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
            response_words = response_words[:15]  # Increased from 10 to 15 for more complete responses
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
            
        except Exception as e:
            print(f"Error processing input: {str(e)}")
            print(f"Bot: {random.choice(fallback_responses)}")


# ---------- Main Pipeline ----------
if __name__ == "__main__":
    tprint("🚀 Starting batch processing of Wikipedia pages...")
    
    try:
        esn, tokenizer = main_batch_processing()
        
        if esn is not None and tokenizer is not None:
            tprint("\n✅ Processing complete. Starting chat interface...")
            chat_with_esn(esn, tokenizer)
        else:
            tprint("\n❌ Failed to train model due to insufficient data.")
            
            # Try to load a previously saved model if available
            save_dir = "saved_pkls"
            model_path = os.path.join(save_dir, "esn_model_final.pkl")
            tokenizer_path = os.path.join(save_dir, "esn_tokenizer_final.pkl")
            
            if os.path.exists(model_path) and os.path.exists(tokenizer_path):
                tprint("Attempting to load previously saved model...")
                try:
                    with open(model_path, "rb") as f:
                        esn = pickle.load(f)
                    with open(tokenizer_path, "rb") as f:
                        tokenizer = pickle.load(f)
                    tprint("Successfully loaded previous model. Starting chat interface...")
                    chat_with_esn(esn, tokenizer)
                except Exception as e:
                    tprint(f"Error loading previous model: {str(e)}")
            else:
                tprint("No previous model found. Cannot start chat interface.")
    except Exception as e:
        tprint(f"\n❌ Error during processing: {str(e)}")
        # Save error log
        with open("error_log.txt", "a") as f:
            f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')}: {str(e)}")
