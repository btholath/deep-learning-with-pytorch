"""
GOAL:
Train a simple "robot brain" (a single neuron) to classify SMS messages
as spam (1) or ham (0). This time, we use a sigmoid function so the output
is a PROBABILITY between 0 and 1.

BIG IDEA:
1. Turn messages into numbers (Bag of Words).
2. Train a single neuron to learn which words are spammy.
3. Use sigmoid to squish predictions into probabilities.
   Example: 0.95 = 95% chance spam, 0.05 = 5% chance ham.

REAL-WORLD USEFULNESS:
This is the foundation of spam filters in Gmail, WhatsApp, and Outlook.
It’s also how binary classification works in medicine (e.g. "does this X-ray show pneumonia?").
"""

import torch
from torch import nn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------------
# 1) Load and prepare the dataset
# -------------------------------
# File format: two columns, separated by tab
#   - "ham" or "spam"
#   - the SMS text
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/04_neuron_classifier/spam_detection_in_sms/data/SMSSpamCollection", 
                 sep="\t", 
                 names=["type", "message"])

# Add a new column spam=True if it's spam, False if it's ham
df["spam"] = df["type"] == "spam"

# We no longer need the "type" column
df.drop("type", axis=1, inplace=True)

# -------------------------------
# 2) Convert text to numbers (Bag of Words)
# -------------------------------
# CountVectorizer builds a "dictionary" of the 1000 most common words,
# then represents each SMS as a row of word counts.
cv = CountVectorizer(max_features=1000)
messages = cv.fit_transform(df["message"])

# Convert to PyTorch tensors
X = torch.tensor(messages.todense(), dtype=torch.float32)  # inputs
y = torch.tensor(df["spam"], dtype=torch.float32).reshape((-1, 1))  # labels (1=spam, 0=ham)

# -------------------------------
# 3) Build the model
# -------------------------------
# nn.Linear(1000, 1) = a neuron with:
#   - 1000 inputs (word counts)
#   - 1 output (spam score)
model = nn.Linear(1000, 1)

# Loss function: Binary Cross-Entropy with Logits
# WHY: Perfect for binary classification
loss_fn = torch.nn.BCEWithLogitsLoss()

# Optimizer: Stochastic Gradient Descent (SGD)
# It tweaks the neuron's weights step by step to reduce loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

# -------------------------------
# 4) Train the model
# -------------------------------
for i in range(0, 10000):
    optimizer.zero_grad()        # clear old gradients
    outputs = model(X)           # raw predictions (logits, can be -∞ to +∞)
    loss = loss_fn(outputs, y)   # calculate how wrong we are
    loss.backward()              # compute corrections
    optimizer.step()             # apply corrections

    if i % 1000 == 0: 
        print(f"Step {i}, Loss: {loss.item()}")

# -------------------------------
# 5) Evaluate with sigmoid
# -------------------------------
model.eval()  # switch to evaluation mode
with torch.no_grad():
    # Apply sigmoid to turn logits into probabilities (0 to 1)
    y_pred = nn.functional.sigmoid(model(X))
    print("Sample probabilities:", y_pred[:5])  # first 5 predictions
    print("Lowest probability (ham-ish):", y_pred.min())
    print("Highest probability (spam-ish):", y_pred.max())


"""
Bag of Words = count words.
Neuron = learns which words increase spam probability (like “prize”, “win”).
Sigmoid = squishes outputs into 0–1 probabilities.
Training = thousands of guess → check → adjust steps.
Output = “95% sure this is spam” instead of just “spam”.
"""