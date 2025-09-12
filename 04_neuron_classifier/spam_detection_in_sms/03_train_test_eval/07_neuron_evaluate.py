"""
GOAL:
Teach a simple "robot brain" (neural network) to decide if a text message
is spam (bad) or ham (good).

BIG IDEA:
1. Turn words into numbers (Bag of Words).
2. Feed those numbers into a neuron model.
3. Train the neuron to spot spammy patterns.
4. Test how well it learned.

REAL-WORLD:
This is how email services (like Gmail, Yahoo) and chat apps
filter out unwanted spam messages.
"""

import torch
from torch import nn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------------
# 1) Load and prepare the dataset
# -------------------------------
# File format: each row has:
#   - "ham" (not spam) or "spam"
#   - the actual SMS message
df = pd.read_csv(
    "/workspaces/deep-learning-with-pytorch/04_neuron_classifier/spam_detection_in_sms/data/SMSSpamCollection",
    sep="\t",
    names=["type", "message"]
)

# Add a new column: True if spam, False if ham
df["spam"] = df["type"] == "spam"

# Drop the old 'type' column since we now have spam=True/False
df.drop("type", axis=1, inplace=True)

# -------------------------------
# 2) Turn text into numbers
# -------------------------------
# CountVectorizer makes a dictionary of the most common 1000 words.
# Each message becomes a "bag of words" = how many times each word appears.
cv = CountVectorizer(max_features=1000)
messages = cv.fit_transform(df["message"])

# Convert messages to PyTorch tensors (rows = messages, columns = word counts)
X = torch.tensor(messages.todense(), dtype=torch.float32)

# Labels (targets): 1 for spam, 0 for ham
y = torch.tensor(df["spam"], dtype=torch.float32).reshape((-1, 1))

# -------------------------------
# 3) Build the model
# -------------------------------
# nn.Linear(1000, 1):
# - 1000 inputs (word counts for each message)
# - 1 output (spam score)
model = nn.Linear(1000, 1)

# Loss function: Binary Cross Entropy with Logits
# Measures how wrong our predictions are (perfect for spam vs ham)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Optimizer: Stochastic Gradient Descent (SGD)
# Updates the neuron's weights step by step to reduce loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

# -------------------------------
# 4) Train the model
# -------------------------------
# Repeat many times:
#   - Guess → Check error → Adjust weights
for i in range(0, 10000):
    optimizer.zero_grad()     # clear old corrections
    outputs = model(X)        # make predictions (raw scores)
    loss = loss_fn(outputs, y)# calculate how wrong we are
    loss.backward()           # figure out how to adjust weights
    optimizer.step()          # update the weights a little

    if i % 1000 == 0:         # print progress every 1000 steps
        print("Step", i, "Loss:", loss.item())

# -------------------------------
# 5) Evaluate the model
# -------------------------------
model.eval()  # switch to "testing" mode
with torch.no_grad():  # no training here, just testing
    # Sigmoid squishes raw scores into probabilities (0–1).
    # > 0.25 means classify as spam (threshold chosen by us).
    y_pred = nn.functional.sigmoid(model(X)) > 0.25

    # Accuracy: of all messages, how many did we classify correctly?
    print("accuracy:", (y_pred == y).type(torch.float32).mean().item())

    # Sensitivity (Recall for spam):
    # of all spam messages, how many did we catch?
    print("sensitivity:", (y_pred[y == 1] == y[y == 1]).type(torch.float32).mean().item())

    # Specificity (Recall for ham):
    # of all ham messages, how many did we correctly keep as ham?
    print("specificity:", (y_pred[y == 0] == y[y == 0]).type(torch.float32).mean().item())

    # Precision:
    # of all the messages we called "spam," how many really were spam?
    print("precision:", (y_pred[y_pred == 1] == y[y_pred == 1]).type(torch.float32).mean().item())
