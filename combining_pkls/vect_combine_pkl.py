import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# File paths for the .pkl files to combine
file1_path = 'C:/Users/cks/Desktop/FILES/Projects/Code/Echo-Sate-Networks-Chatbot/models/vectorizer_gpu_batch_1.pkl'
file2_path = 'C:/Users/cks/Desktop/FILES/Projects/Code/Echo-Sate-Networks-Chatbot/models/vectorizer_gpu_batch_2.pkl'

# Output file path for the combined .pkl file
output_file_path = 'c://Users/cks/Desktop/FILES/Projects/Code/Echo-Sate-Networks-Chatbot/SavedModels/combined_vectorizer_model.pkl'

# Load the TfidfVectorizer objects
with open(file1_path, 'rb') as file1:
    vectorizer1 = joblib.load(file1)

# Load the second TfidfVectorizer object
with open(file2_path, 'rb') as file2:
    vectorizer2 = joblib.load(file2)

# Combine the vocabularies of the two vectorizers
combined_vocabulary = {**vectorizer1.vocabulary_, **vectorizer2.vocabulary_}

# Create a new TfidfVectorizer with the combined vocabulary
combined_vectorizer = TfidfVectorizer()
combined_vectorizer.vocabulary_ = combined_vocabulary


# Save the combined TfidfVectorizer to a new .pkl file
with open(output_file_path, 'wb') as output_file:
    joblib.dump(combined_vectorizer, output_file)

print(f"Combined TfidfVectorizer saved to {output_file_path}")