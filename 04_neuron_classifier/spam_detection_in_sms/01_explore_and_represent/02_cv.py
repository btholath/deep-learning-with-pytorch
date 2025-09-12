"""
GOAL:
Turn text messages (SMS) into numbers so a computer can work with them.

BIG IDEA:
Computers don’t understand words like "free" or "win".
We must turn each word into a number. One simple way is
the "Bag of Words" method: count how often each word appears.

REAL-WORLD USEFULNESS:
This is how spam filters, search engines, and even chatbots
start understanding text—by turning words into numeric features.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------------
# 1) Load the dataset
# -------------------------------
# The file has two parts in each row:
#   - "ham" (not spam) or "spam"
#   - the actual SMS text
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/04_neuron_classifier/spam_detection_in_sms/data/SMSSpamCollection",
                 sep="\t",           # columns separated by tab
                 names=["type", "message"])  # give column names

# Create a new column called "spam" with True/False values:
#   True if it's spam, False if it's ham
df["spam"] = df["type"] == "spam"

# We don’t need the original "type" column anymore (we replaced it with spam=True/False)
df.drop("type", axis=1, inplace=True)

# -------------------------------
# 2) Turn text into numbers
# -------------------------------
# CountVectorizer = tool that builds a "dictionary" of words,
# then counts how often each word appears in every message.
# max_features=1000 means: keep only the 1000 most common words.
cv = CountVectorizer(max_features=1000)

# Fit: learn the dictionary of words from all messages
# Transform: turn every message into a row of numbers (word counts)
messages = cv.fit_transform(df["message"])

# -------------------------------
# 3) Peek at the transformed data
# -------------------------------
# Print the word counts for the first message
print(messages[0, :])

# Print the 888th word from the dictionary, just for fun
print(cv.get_feature_names_out()[888])

# -------------------------------
# 4) (Optional mini-demo)
# -------------------------------
# Let’s test CountVectorizer with tiny made-up sentences.
# Uncomment the section below to run it.

# cv = CountVectorizer(max_features=6)
# documents = [
#     "Hello world. Today is amazing. Hello hello",
#     "Hello mars, today is perfect"
# ]
# cv.fit(documents)                 # learn the dictionary
# print(cv.get_feature_names_out()) # show the learned words
# out = cv.transform(documents)     # turn sentences into numbers
# print(out.todense())              # show word counts in table form
