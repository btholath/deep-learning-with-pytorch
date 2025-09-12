"""
GOAL (in simple words):
Teach a small program to look at an SMS and say:
  -> 'spam' (bad) or 'ham' (not spam).

BIG IDEA:
We can't feed words directly to a computer brain.
We first turn text into numbers with TF-IDF,
then train a simple model (Logistic Regression) to spot patterns.

REAL-WORLD:
This is how spam filters help keep your inbox clean.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # turns text into useful numbers
from sklearn.linear_model import LogisticRegression           # a simple, fast classifier
from sklearn.metrics import classification_report, confusion_matrix

# 1) Load the dataset (same as explore step)
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/04_neuron_classifier/spam_detection_in_sms/data/SMSSpamCollection", sep="\t", names=["label", "message"])

# Turn labels into 0/1 so the model can learn:
#   ham -> 0 (not spam)
#   spam -> 1 (is spam)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# 2) Split into TRAIN (teach the model) and TEST (see if it really learned)
# test_size=0.2 means 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label_num"], test_size=0.2, random_state=42, stratify=df["label_num"]
)

# 3) Turn text into numbers using TF-IDF
# TF = term frequency (how often a word appears in a message)
# IDF = inverse document frequency (down-weights common words like "the" or "and")
# RESULT: important words get higher scores; boring common words get lower scores
vectorizer = TfidfVectorizer(
    lowercase=True,        # make everything lowercase so 'Free' and 'free' are the same
    stop_words="english",  # drop common filler words (you can remove this if you prefer)
    max_features=10000     # keep things small & fast; plenty for this dataset
)
X_train_vec = vectorizer.fit_transform(X_train)  # learn vocabulary + transform train texts
X_test_vec  = vectorizer.transform(X_test)       # transform test texts using the same rules

# 4) Choose a simple model: Logistic Regression
# WHY: fast, interpretable, surprisingly strong for text
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)  # LEARN from training data

# 5) Test on messages the model has never seen (the test set)
y_pred = model.predict(X_test_vec)

# 6) Print results that humans can understand
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# HOW TO READ THIS:
# - Precision for 'spam': of the messages we called 'spam', how many were truly spam?
# - Recall for 'spam': of all the real spam messages, how many did we catch?
# - F1: balance of precision & recall (bigger = better)
# - Confusion matrix:
#     [[ham correctly, ham called spam],
#      [spam called ham, spam correctly]]

# 7) Try a few custom messages to make it feel real
samples = [
    "WIN a brand new iPhone! Click here to claim your prize",
    "Hey, are we still meeting at 6pm?",
    "URGENT! Your account is at risk. Verify now to avoid fees.",
]
samples_vec = vectorizer.transform(samples)
preds = model.predict(samples_vec)

print("\n=== Custom Message Predictions ===")
for msg, p in zip(samples, preds):
    label = "spam" if p == 1 else "ham"
    print(f"[{label}] {msg}")
