"""
GOAL:
Teach a simple "robot brain" (a neuron) to decide if an SMS is spam or not spam.

BIG IDEA:
1. Convert text messages into numbers (Bag of Words).
2. Train a tiny model (just one layer) to spot spammy words.
3. Show predictions as numbers (later we’ll squish them with sigmoid into probabilities).

REAL-WORLD USEFULNESS:
This is the basic idea behind spam filters in Gmail, Yahoo, WhatsApp, etc.
It’s also the same foundation used in more complex systems like
recommendation engines (Netflix, YouTube) and fraud detection.
"""

import torch
from torch import nn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------------
# 1) Load and clean the dataset
# -------------------------------
# File format: two columns, separated by a tab
#   - "ham" or "spam"
#   - the SMS text
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/04_neuron_classifier/spam_detection_in_sms/data/SMSSpamCollection", 
                 sep="\t", 
                 names=["type", "message"])

# Add a new column: spam = True if type is "spam", else False
df["spam"] = df["type"] == "spam"

# Drop the original 'type' column (we don’t need it anymore)
df.drop("type", axis=1, inplace=True)

# -------------------------------
# 2) Bag of Words: turn text into numbers
# -------------------------------
# CountVectorizer makes a "dictionary" of up to 1000 most common words
# Then it counts how often each word appears in each message
cv = CountVectorizer(max_features=1000)
messages = cv.fit_transform(df["message"])

# Convert to a PyTorch tensor (dense matrix: each row = message, each column = word count)
X = torch.tensor(messages.todense(), dtype=torch.float32)

# Labels: 1 if spam, 0 if ham
y = torch.tensor(df["spam"], dtype=torch.float32).reshape((-1, 1))

# -------------------------------
# 3) Define the model
# -------------------------------
# nn.Linear(1000, 1):
#   - 1000 inputs (one for each word in the dictionary)
#   - 1 output (prediction: spam or not spam)
model = nn.Linear(1000, 1)

# Loss function: Mean Squared Error (how wrong we are)
# NOTE: Later we’ll use Binary Cross-Entropy (better for classification),
# but MSE works for now as a teaching step.
loss_fn = torch.nn.MSELoss()

# Optimizer: helps the model adjust its "weights" (word importance)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# -------------------------------
# 4) Train the model
# -------------------------------
# Training loop = guess → check → adjust → repeat
for i in range(0, 10000):
    optimizer.zero_grad()       # clear old learning signals
    outputs = model(X)          # model makes predictions
    loss = loss_fn(outputs, y)  # compare guesses vs actual labels
    loss.backward()             # compute how to fix weights
    optimizer.step()            # apply the fix (small step)

    if i % 100 == 0: 
        print(f"Step {i}, Loss: {loss.item()}")

# -------------------------------
# 5) Evaluate the model
# -------------------------------
# Turn off gradient tracking (we don’t need it for testing)
model.eval()
with torch.no_grad():
    y_pred = model(X)
    print("Sample predictions:", y_pred[:5])  # show first 5 predictions
    print("Lowest prediction:", y_pred.min()) # close to 0 = ham
    print("Highest prediction:", y_pred.max())# close to 1 = spam
