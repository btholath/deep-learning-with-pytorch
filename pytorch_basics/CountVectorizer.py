"""
CountVectorizer is often the first magical step in NLP pipelines.

PURPOSE:
Show how CountVectorizer turns text messages into numbers
so a computer can understand them.

REAL-WORLD:
This is how spam filters, sentiment analyzers, and chatbots
start processing text: by converting words into numeric features.
"""

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# -------------------------------
# 1) Example messages (pretend dataset)
# -------------------------------
data = {
    "message": [
        "Free money now!!!",        # looks like spam
        "Hi, are we still meeting?",# normal message
        "Win a FREE prize today!",  # spammy again
        "Let's have lunch tomorrow" # normal message
    ],
    "spam": [1, 0, 1, 0]  # labels: 1 = spam, 0 = not spam
}
df = pd.DataFrame(data)

# -------------------------------
# 2) CountVectorizer
# -------------------------------
# max_features=5 means: only keep the 5 most common words
cv = CountVectorizer(max_features=5)

# Learn the vocabulary from ALL messages and transform them
X = cv.fit_transform(df["message"])

# -------------------------------
# 3) Show results
# -------------------------------
print("Vocabulary (word → column index):")
print(cv.vocabulary_)  # dictionary of chosen words

print("\nBag-of-Words representation:")
print(X.toarray())     # matrix of word counts

print("\nOriginal messages:")
print(df["message"].tolist())

"""
Bag-of-Words matrix (rows = messages, columns = word counts):
[[0 1 0 1 1]   # "Free money now!!!"
 [0 0 0 0 0]   # "Hi, are we still meeting?" (no matching top-5 words)
 [1 1 1 0 0]   # "Win a FREE prize today!"
 [0 0 0 0 0]]  # "Let's have lunch tomorrow"

 
Purpose (easy explanation)
Computers don’t understand words → they understand numbers.
CountVectorizer builds a dictionary of common words (up to max_features).
Each message is turned into a row of word counts (Bag of Words).
Example: “Free money now!!!” → [free=1, money=1, now=1, prize=0, today=0].
Once messages are numbers, we can feed them into ML models (like logistic regression, neurons, etc.). 
"""
