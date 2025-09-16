"""
GOAL:
Teach a neural network to predict if a loan will be approved (1) or not (0)
using facts about a person (income, credit score, and loan intent).

REAL LIFE:
Banks and lenders use models like this to quickly screen loan applications.
"""

import torch
from torch import nn
import pandas as pd

# 1) READ THE DATA -----------------------------------------------------------
# We load the CSV into a table (DataFrame). Each row is one loan application.
df = pd.read_csv("data/loan_data.csv")  # adjust path if needed

# 2) PICK USEFUL COLUMNS (FEATURES + LABEL) ---------------------------------
# Keep:
#  - loan_status (our target: 1 = approved, 0 = not approved)
#  - person_income, loan_intent, loan_percent_income, credit_score (inputs)
df = df[["loan_status", "person_income", "loan_intent", "loan_percent_income", "credit_score"]]

# 3) TURN WORDS INTO SWITCHES (ONE-HOT ENCODING) -----------------------------
# The column 'loan_intent' is text (e.g., 'EDUCATION', 'DEBTCONSOLIDATION').
# Models need numbers, so we create extra 0/1 columns for each intent.
df = pd.get_dummies(df, columns=["loan_intent"])

# 4) BUILD THE TARGET y (WHAT WE WANT TO PREDICT) ---------------------------
# y must be a column vector of float numbers (0.0 or 1.0).
y = torch.tensor(df["loan_status"].values, dtype=torch.float32).reshape((-1, 1))

# 5) BUILD THE INPUT MATRIX X (WHAT THE MODEL READS) ------------------------
# Drop the answer column and keep the features as float32.
X_data = df.drop("loan_status", axis=1).astype("float32").values
X = torch.tensor(X_data, dtype=torch.float32)

# 6) NORMALIZE FEATURES (FAIR COMPARISON) -----------------------------------
# Different columns have different scales (income vs. percent vs. score).
# We subtract the mean and divide by the standard deviation so each
# feature feels "equally important" at the start.
X_mean = X.mean(dim=0)
X_std = X.std(dim=0)
X = (X - X_mean) / (X_std + 1e-8)  # add tiny number to avoid division by zero

print("X shape (rows, features):", X.shape)  # e.g., (N, 9)

# 7) BUILD A SMALL NEURAL NETWORK -------------------------------------------
# A stack of layers:
#   Input(=number of columns) -> 32 -> ReLU -> 16 -> ReLU -> 1 (final score)
# ReLU helps the model learn non-linear patterns (more than just straight lines).
model = nn.Sequential(
    nn.Linear(X.shape[1], 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)  # output = one raw score (called a "logit")
)

# 8) CHOOSE LOSS + OPTIMIZER ------------------------------------------------
# Loss function: "How wrong are we?" for yes/no problems that use logits.
loss_fn = nn.BCEWithLogitsLoss()

# Optimizer: Adam is like a smart coach that adapts the step size for each weight.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 9) TRAIN WITH MINI-BATCHES (LEARN IN CHUNKS) ------------------------------
# Homework analogy: do a little at a time, many times.
num_entries = X.size(0)
batch_size = 32
epochs = 100

for epoch in range(epochs):
    loss_sum = 0.0
    # Walk through the data 32 rows at a time
    for start in range(0, num_entries, batch_size):
        end = min(num_entries, start + batch_size)
        X_b, y_b = X[start:end], y[start:end]

        optimizer.zero_grad()        # clear old training notes
        logits = model(X_b)          # forward pass: raw scores (any real numbers)
        loss = loss_fn(logits, y_b)  # how wrong are we on this batch?
        loss.backward()              # figure out how to adjust weights
        optimizer.step()             # nudge weights a tiny bit

        loss_sum += loss.item()

    # Print progress every 10 epochs so we can SEE learning happen
    if epoch % 10 == 0:
        print(f"epoch {epoch:3d} | total_loss_this_epoch = {loss_sum:.4f}")

# 10) EVALUATE (NO TRAINING, JUST CHECKING) ---------------------------------
model.eval()
with torch.no_grad():
    # Turn final raw scores (logits) into probabilities with sigmoid (0..1).
    probs = torch.sigmoid(model(X))

    # If probability > 0.5, predict "approved" (1), else "not approved" (0).
    preds = probs > 0.5

    # Compare predictions to truth and compute accuracy.
    accuracy = (preds.float() == y).float().mean().item()
    print("accuracy:", accuracy)
