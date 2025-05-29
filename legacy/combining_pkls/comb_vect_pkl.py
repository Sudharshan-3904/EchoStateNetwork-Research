import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

file1_path = 'models/vectorizer_gpu_batch_1.pkl'
file2_path = 'models/vectorizer_gpu_batch_2.pkl'

output_file_path = 'SavedModels/combined_vectorizer_model.pkl'

with open(file1_path, 'rb') as file1:
    vectorizer1 = joblib.load(file1)

with open(file2_path, 'rb') as file2:
    vectorizer2 = joblib.load(file2)

combined_vocabulary = {**vectorizer1.vocabulary_, **vectorizer2.vocabulary_}

combined_vectorizer = TfidfVectorizer()
combined_vectorizer.vocabulary_ = combined_vocabulary

with open(output_file_path, 'wb') as output_file:
    joblib.dump(combined_vectorizer, output_file)

print(f"Combined TfidfVectorizer saved to {output_file_path}")