import requests
from bs4 import BeautifulSoup
import numpy as np
import pickle
import nltk
from ESN_Model import EchoStateNetworkModular
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# -------- STEP 1: SCRAPE PARAGRAPHS --------
def scrape_paragraphs(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
        return paragraphs
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

def collect_and_chunk_sentences():
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Deep_learning"
    ]
    input_sentences = []
    target_sentences = []

    for url in urls:
        paragraphs = scrape_paragraphs(url)
        for para in paragraphs:
            sentences = sent_tokenize(para)
            for i in range(len(sentences) - 1):
                input_sentences.append(sentences[i])
                target_sentences.append(sentences[i + 1])

    return input_sentences, target_sentences

# -------- STEP 2: PREPROCESS TEXT --------
def preprocess_text(inputs, targets, tokenizer=None, max_len=20, num_words=4000):
    if tokenizer is None:
        tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(inputs + targets)

    input_seq = tokenizer.texts_to_sequences(inputs)
    target_seq = tokenizer.texts_to_sequences(targets)

    X = pad_sequences(input_seq, maxlen=max_len, padding='post')
    y = pad_sequences(target_seq, maxlen=max_len, padding='post')

    X = X / np.max(X)
    y = y / np.max(y)

    return X, y, tokenizer

# -------- STEP 3: TRAIN IN BATCHES --------
def train_in_batches(X, y, batch_size=32):
    input_size = X.shape[1]
    output_size = y.shape[1]

    esn = EchoStateNetworkModular(
        input_size=input_size,
        reservoir_size=100,
        output_size=output_size,
        spectral_radius=0.95,
        sparsity=0.1,
        input_scaling=1.0,
        leaking_rate=0.3
    )

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        esn.fit(X[start:end], y[start:end])
        print(f"Trained on batch {start // batch_size + 1}")

    return esn

# -------- MAIN PIPELINE --------
if __name__ == "__main__":
    print("Scraping and processing paragraph data...")
    inputs, targets = collect_and_chunk_sentences()

    if len(inputs) < 10:
        print("Insufficient data scraped. Try with more/different URLs.")
        exit()

    print(f"Generated {len(inputs)} sentence pairs.")
    X, y, tokenizer = preprocess_text(inputs, targets)

    print("Training Echo State Network...")
    esn = train_in_batches(X, y, batch_size=32)

    with open("esn_paragraph_chatbot.pkl", "wb") as f:
        pickle.dump(esn, f)
    with open("esn_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("Model and tokenizer saved.")
