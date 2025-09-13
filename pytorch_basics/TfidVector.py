from sklearn.feature_extraction.text import TfidfVectorizer

# Example dataset (3 documents)
docs = [
    "free money now",
    "call now for free prize",
    "let's meet for lunch tomorrow"
]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
X = vectorizer.fit_transform(docs)

# Show words in vocabulary
print("Vocabulary:", vectorizer.get_feature_names_out())

# Show TF-IDF matrix
print("\nTF-IDF values (rows=documents, cols=words):")
print(X.toarray().round(2))

"""
Real-world use of TF-IDF
Search engines: find the most relevant documents for a query.
Spam filters: detect spammy words (“free,” “prize”) that stand out in spam messages.
Text mining: summarize documents or find keywords.
"""