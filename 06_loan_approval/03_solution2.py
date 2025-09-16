"""
Understanding the Loan Prediction Neural Network
This Python script is a great example of a complete machine learning workflow. 
Its goal is to teach a computer how to predict if a loan application should be approved or not, based on a person's financial information.

Here is a step-by-step breakdown of how the code works and what each part does:

GOAL:
The main objective is to build a "smart program" (a neural network) that can look at a new loan application and make a decision: approved (1) or not approved (0).
Teach a neural network to predict if a loan will be approved (1) or not (0) using facts about a person (income, credit score, and loan intent).

REAL LIFE:
Banks and lenders use models like this to quickly screen loan applications.
The main objective is to build a "smart program" (a neural network) that can look at a new loan application and make a decision: approved (1) or not approved (0).

"""

import torch
from torch import nn
import pandas as pd

# 1) READ THE DATA -----------------------------------------------------------
# We load the CSV into a table (DataFrame). Each row is one loan application.
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/06_loan_approval/data/loan_data.csv")  # adjust path if needed

# 2) PICK USEFUL COLUMNS (FEATURES + LABEL) ---------------------------------
# Keep:
#  - loan_status (our target: 1 = approved, 0 = not approved)
#  - person_income, loan_intent, loan_percent_income, credit_score (inputs)
df = df[["loan_status", "person_income", "loan_intent", "loan_percent_income", "credit_score"]]

# 3) TURN WORDS INTO SWITCHES (ONE-HOT ENCODING) -----------------------------
# The column 'loan_intent' is text (e.g., 'EDUCATION', 'DEBTCONSOLIDATION').
# Models need numbers, so we create extra 0/1 columns for each intent.
# Computers and neural networks can only understand numbers, but some of our data, like loan_intent, is text (e.g., "EDUCATION").
# The code uses a clever trick called one-hot encoding to fix this. 
# It creates a new column for each possible loan intent and puts a 1 in the column that matches the intent for that row, and a 0 everywhere else.
df = pd.get_dummies(df, columns=["loan_intent"])


"""
Separating Inputs and Outputs
To train the model, we need to separate the "answers" from the "facts."
The inputs (X) are all the facts the model will use (income, credit score, and the new one-hot encoded columns).
The target (y) is the single answer we want to predict (loan_status)
"""
# 4) BUILD THE TARGET y (WHAT WE WANT TO PREDICT) ---------------------------
# y must be a column vector of float numbers (0.0 or 1.0).
y = torch.tensor(df["loan_status"].values, dtype=torch.float32).reshape((-1, 1))

# 5) BUILD THE INPUT MATRIX X (WHAT THE MODEL READS) ------------------------
# Drop the answer column and keep the features as float32.
X_data = df.drop("loan_status", axis=1).astype("float32").values
X = torch.tensor(X_data, dtype=torch.float32)


"""
Normalizing the Features (Making a Fair Comparison)
Imagine trying to compare "person's income" (a big number, like $50,000) with "loan percentage of income" (a small number, like 0.1). 
The big number might seem more important to the model just because it's bigger.
To prevent this, the code normalizes all the input features. 
It subtracts the average value from each column and divides by a scaling factor. 
This makes all the numbers in our input matrix feel equally important to the model.
"""
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
