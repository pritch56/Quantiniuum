from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define sentences
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "In the middle of difficulty lies opportunity.",
    "The only limit to our realization of tomorrow is our doubts of today."
]

# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode sentences to get their embeddings
embeddings = model.encode(sentences)

# Calculate cosine similarity between the first sentence and the rest
similarity_scores = cosine_similarity([embeddings[0]], embeddings[1:])

# Print the similarity scores
for i, score in enumerate(similarity_scores[0]):
    print(f"Cosine similarity between sentence 1 and sentence {i + 2}: {score:.4f}")
