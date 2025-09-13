from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Example dataset (3 short messages)
docs = [
    "free money now",
    "call now for free prize",
    "let's meet for lunch tomorrow"
]

# Step 1: Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Step 2: Fit and transform the documents
X = vectorizer.fit_transform(docs)

# Step 3: Get the vocabulary (list of words)
words = vectorizer.get_feature_names_out()

# Step 4: Convert TF-IDF matrix to array for easier handling
tfidf_values = X.toarray()

# Step 5: Show TF-IDF matrix
print("Vocabulary:", words)
print("\nTF-IDF Matrix (rows=documents, cols=words):")
print(tfidf_values.round(2))

# Step 6: For each document, show high and low TF-IDF words
for i, doc in enumerate(docs):
    row = tfidf_values[i]
    max_val = row.max()
    min_val = row[row > 0].min()  # smallest nonzero (ignore absent words)

    # Find the words for these values
    high_words = [words[j] for j, val in enumerate(row) if val == max_val]
    low_words  = [words[j] for j, val in enumerate(row) if val == min_val]

    print(f"\nDocument {i+1}: \"{doc}\"")
    print("  Highest TF-IDF word(s):", high_words, "value:", round(max_val, 2))
    print("  Lowest TF-IDF word(s):", low_words, "value:", round(min_val, 2))


"""
Real-world use of TF-IDF
Search engines: find the most relevant documents for a query.
Spam filters: detect spammy words (“free,” “prize”) that stand out in spam messages.
Text mining: summarize documents or find keywords.
"""