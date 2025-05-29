import pickle
import joblib
import os
import glob
from src.models.ESN_Model import EnhancedESN
from tensorflow.keras.preprocessing.text import Tokenizer


models_dir = 'src/models_pkl'
output_dir = 'SavedModels'
os.makedirs(output_dir, exist_ok=True)

output_model_path = os.path.join(output_dir, 'combined_esn_model.pkl')
output_tokenizer_path = os.path.join(output_dir, 'combined_tokenizer.pkl')

model_files = glob.glob(os.path.join(models_dir, 'esn_model_*batch_*.pkl'))
print(f"Found {len(model_files)} model files to combine")

if len(model_files) == 0:
    print("No model files found!")
    exit(1)

with open(model_files[0], 'rb') as f:
    base_esn = pickle.load(f)

combined_W_in = base_esn.W_in.copy()
combined_W = base_esn.W.copy()
combined_W_out = base_esn.W_out.copy() if base_esn.W_out is not None else None
combined_reservoir_state = base_esn.reservoir_state.copy()

for file_path in model_files[1:]:
    print(f"Processing {file_path}...")
    with open(file_path, 'rb') as f:
        esn = pickle.load(f)
    
    combined_W_in += esn.W_in
    combined_W += esn.W
    if combined_W_out is not None and esn.W_out is not None:
        combined_W_out += esn.W_out
    combined_reservoir_state += esn.reservoir_state

n_models = len(model_files)
combined_W_in /= n_models
combined_W /= n_models
if combined_W_out is not None:
    combined_W_out /= n_models
combined_reservoir_state /= n_models

combined_esn = EnhancedESN(
    input_size=base_esn.input_size,
    reservoir_size=base_esn.reservoir_size,
    output_size=base_esn.output_size,
    spectral_radius=base_esn.spectral_radius,
    sparsity=base_esn.sparsity,
    input_scaling=base_esn.input_scaling,
    leaking_rate=base_esn.leaking_rate
)

combined_esn.W_in = combined_W_in
combined_esn.W = combined_W
combined_esn.W_out = combined_W_out
combined_esn.reservoir_state = combined_reservoir_state

with open(output_model_path, 'wb') as f:
    pickle.dump(combined_esn, f)

print(f"Combined ESN model saved to {output_model_path}")

vectorizer_files = glob.glob(os.path.join(models_dir, 'vectorizer_*batch_*.pkl'))
if vectorizer_files:
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    output_vectorizer_path = os.path.join(output_dir, 'combined_vectorizer.pkl')
    
    with open(vectorizer_files[0], 'rb') as f:
        base_vectorizer = joblib.load(f)
    
    combined_vocabulary = dict(base_vectorizer.vocabulary_)
    
    for file_path in vectorizer_files[1:]:
        print(f"Processing vectorizer {file_path}...")
        with open(file_path, 'rb') as f:
            vectorizer = joblib.load(f)
        combined_vocabulary.update(vectorizer.vocabulary_)
    
    combined_vectorizer = TfidfVectorizer()
    combined_vectorizer.vocabulary_ = combined_vocabulary
    
    with open(output_vectorizer_path, 'wb') as f:
        joblib.dump(combined_vectorizer, f)
    
    print(f"Combined vectorizer saved to {output_vectorizer_path}")

tokenizer_files = glob.glob(os.path.join(models_dir, 'esn_tokenizer_*batch_*.pkl'))
if tokenizer_files:
    print(f"Found {len(tokenizer_files)} tokenizer files to combine")
    
    with open(tokenizer_files[0], 'rb') as f:
        base_tokenizer = pickle.load(f)
    
    combined_tokenizer = Tokenizer(num_words=base_tokenizer.num_words, oov_token=base_tokenizer.oov_token)
    
    combined_word_index = dict(base_tokenizer.word_index)
    combined_word_counts = dict(base_tokenizer.word_counts)
    
    for file_path in tokenizer_files[1:]:
        print(f"Processing tokenizer {file_path}...")
        with open(file_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        max_index = max(combined_word_index.values())
        for word, index in tokenizer.word_index.items():
            if word not in combined_word_index:
                combined_word_index[word] = max_index + 1
                max_index += 1
        
        for word, count in tokenizer.word_counts.items():
            if word in combined_word_counts:
                combined_word_counts[word] += count
            else:
                combined_word_counts[word] = count
    
    combined_tokenizer.word_index = combined_word_index
    combined_tokenizer.word_counts = combined_word_counts
    combined_tokenizer.index_word = {v: k for k, v in combined_word_index.items()}
    combined_tokenizer.document_count = sum(1 for t in tokenizer_files for _ in [1])
    
    with open(output_tokenizer_path, 'wb') as f:
        pickle.dump(combined_tokenizer, f)
    
    print(f"Combined tokenizer saved to {output_tokenizer_path}")
    
