"""
GOAL:
Teach a tiny "robot brain" to spot SPAM text messages.

BIG IDEA (what we're doing):
1) Turn words into numbers (so computers can understand them).
2) Split messages into a training group (to learn) and a validation group (to test fairly).
3) Train a simple neuron model to predict if a message is spam (1) or not spam (0).
4) Measure how good it is with accuracy, sensitivity (recall for spam), specificity (recall for ham), and precision.

REAL-WORLD:
This is basically what email/SMS apps do to filter spam from your inbox.
"""

import sys
import torch
from torch import nn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# -----------------------------
# 1) LOAD and PREP the data
# -----------------------------
# The file has two columns per row:
#   - "ham" or "spam"
#   - the SMS text
df = pd.read_csv(
    "/workspaces/deep-learning-with-pytorch/04_neuron_classifier/spam_detection_in_sms/data/SMSSpamCollection",
    sep="\t",
    names=["type", "message"]
)

# Make a new column: spam=True if type == "spam", else False (ham)
df["spam"] = df["type"] == "spam"

# We no longer need the original text label column
df.drop("type", axis=1, inplace=True)

# -----------------------------
# 2) TRAIN / VALIDATION SPLIT
# -----------------------------
# IDEA: We teach the model with 80% of the data (train),
# then we test it on 20% it has NEVER seen (validation).
# This tells us if it learned real patterns, not just memorized.
df_train = df.sample(frac=0.8, random_state=0)  # 80% train (random pick)
df_val   = df.drop(index=df_train.index)        # the rest (20%) for validation

# -----------------------------
# 3) TEXT → NUMBERS (Bag of Words)
# -----------------------------
# CountVectorizer builds a dictionary (up to 5000 most common words)
# and counts how often each appears in each message.
# Each message becomes a row of numbers (word counts).
cv = CountVectorizer(max_features=5000)

# Learn vocabulary from TRAIN texts, transform both TRAIN and VAL the same way
messages_train = cv.fit_transform(df_train["message"])
messages_val   = cv.transform(df_val["message"])

# Turn the sparse matrices into PyTorch tensors (float32 numbers)
X_train = torch.tensor(messages_train.todense(), dtype=torch.float32)
y_train = torch.tensor(df_train["spam"].values, dtype=torch.float32).reshape((-1, 1))

X_val = torch.tensor(messages_val.todense(), dtype=torch.float32)
y_val = torch.tensor(df_val["spam"].values, dtype=torch.float32).reshape((-1, 1))

# -----------------------------
# 4) BUILD the model
# -----------------------------
# nn.Linear(5000, 1):
# - 5000 inputs (one number for each word in the dictionary)
# - 1 output (a score: higher = more likely spam)
model = nn.Linear(5000, 1)

# Loss function: BCEWithLogitsLoss = Binary Cross-Entropy on raw scores ("logits")
# WHY: Perfect for yes/no problems (spam vs ham).
loss_fn = torch.nn.BCEWithLogitsLoss()

# Optimizer (the coach): adjusts model weights to reduce loss
# lr (learning rate) = how big each step of learning is.
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

# -----------------------------
# 5) TRAIN the model (practice!)
# -----------------------------
# Loop many times:
#   - Guess → Check error (loss) → Learn a little → Repeat
for i in range(0, 10000):
    optimizer.zero_grad()          # clear old correction notes
    outputs = model(X_train)       # model makes raw predictions (logits)
    loss = loss_fn(outputs, y_train)# how wrong are we?
    loss.backward()                # figure out how to nudge weights
    optimizer.step()               # apply the nudge (learn a bit)

    if i % 1000 == 0:
        print(f"Step {i}, loss = {loss.item():.4f}")

# -----------------------------
# 6) EVALUATION helper
# -----------------------------
def evaluate_model(X, y):
    """
    Turn model scores into probabilities with sigmoid, then into yes/no (spam/ham)
    using a threshold. Compute and print simple metrics students can read.
    """
    model.eval()  # switch to "testing" mode (no dropout, etc.)
    with torch.no_grad():  # no training/gradients here
        # Sigmoid squishes raw scores into 0..1 probabilities.
        # THRESHOLD: if prob > 0.25 → call it spam (you can try 0.5 too).
        probs = torch.sigmoid(model(X))
        y_pred = probs > 0.25

        # Accuracy: out of all messages, % we got right
        accuracy = (y_pred == y).type(torch.float32).mean().item()

        # Sensitivity (Recall for spam): out of real spam, % we caught
        sensitivity = (y_pred[y == 1] == y[y == 1]).type(torch.float32).mean().item()

        # Specificity (Recall for ham): out of real ham, % we correctly kept as ham
        specificity = (y_pred[y == 0] == y[y == 0]).type(torch.float32).mean().item()

        # Precision: of the messages we called "spam", % that were truly spam
        # (If we predicted no spam at all, the slice is empty—handle gently)
        if (y_pred == 1).any():
            precision = (y_pred[y_pred == 1] == y[y_pred == 1]).type(torch.float32).mean().item()
        else:
            precision = float("nan")

        print(f"accuracy   : {accuracy:.3f}")
        print(f"sensitivity: {sensitivity:.3f}")
        print(f"specificity: {specificity:.3f}")
        print(f"precision  : {precision:.3f}")

# -----------------------------
# 7) CHECK how we did
# -----------------------------
print("\nEvaluating on the TRAINING data (what we learned from):")
evaluate_model(X_train, y_train)

print("\nEvaluating on the VALIDATION data (brand-new to the model):")
evaluate_model(X_val, y_val)

"""
Why split the data? To be fair! We test on messages the robot never saw, to check real learning (not memorization).

Why sigmoid + threshold? Sigmoid gives a probability (0–1). The threshold (here 0.25) turns it into a yes/no decision. Try 0.5 and compare results.

What do the metrics mean?

Accuracy: overall percent correct.

Sensitivity (recall for spam): how many spams we actually caught.

Specificity (recall for ham): how many good messages we kept as good.

Precision: of the messages we called spam, how many were truly spam.

Real world: Spam filters aim for high recall (catch spam) while keeping precision high (don’t flag normal messages by mistake). Thresholds help tune that trade-off.
"""