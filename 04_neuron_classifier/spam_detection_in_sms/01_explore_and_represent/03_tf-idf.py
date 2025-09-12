"""
GOAL:
Turn SMS text messages into numbers using TF-IDF,
so a computer can learn which words are important for detecting spam.

BIG IDEA:
Not all words are equally useful. Words like "the", "is", "and"
appear everywhere, so they don’t help much.
But words like "prize", "win", or "free" appear mostly in spam.
TF-IDF gives "important" words bigger numbers.

REAL-WORLD USEFULNESS:
TF-IDF is used in:
- Search engines (Google ranks results based on important words),
- Document matching (finding similar articles),
- Spam filters (spotting spammy words).
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------
# 1) Load the dataset
# -------------------------------
# File has two parts per row:
#   - "ham" (not spam) or "spam"
#   - the SMS text
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/04_neuron_classifier/spam_detection_in_sms/data/SMSSpamCollection", 
                 sep="\t", 
                 names=["type", "message"])

# Add a new column "spam" (True for spam, False for ham)
df["spam"] = df["type"] == "spam"

# Drop the original "type" column since we now have spam=True/False
df.drop("type", axis=1, inplace=True)

# -------------------------------
# 2) Turn text into TF-IDF numbers
# -------------------------------
# TfidfVectorizer does two things:
#   1. Counts how often words appear (Term Frequency = TF)
#   2. Reduces the weight of common words (Inverse Document Frequency = IDF)
# RESULT: rare but important words get higher scores
vectorizer = TfidfVectorizer(max_features=1000)

# Learn vocabulary + calculate TF-IDF scores for each message
messages = vectorizer.fit_transform(df["message"])

# -------------------------------
# 3) Peek at the transformed data
# -------------------------------
# Print the TF-IDF numbers for the first message
print(messages[0, :])

# Print the 888th word in the learned vocabulary
print(vectorizer.get_feature_names_out()[888])
