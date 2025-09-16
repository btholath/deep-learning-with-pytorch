import sys
import torch
from torch import nn
import pandas as pd

# Step 1: Loading and Cleaning the Data
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/06_loan_approval/data/loan_data.csv")
df = df[["loan_status", "person_income", "loan_intent", "loan_percent_income", "credit_score"]]
print(df.head())

# Step 2: Turning Words into Numbers
# performs one-hot encoding. 
# It's a key data preparation task that converts the text in the loan_intent column into a series of numerical columns (one for each unique intent). 
# This makes the data usable by the model.
df = pd.get_dummies(df, columns=["loan_intent"])
print(df.columns)

# Step 3: Separating Inputs and Outputs
# Purpose: This final section separates the "answer" (y) from the "facts" (X). This is a crucial step to prepare the data for the training process.
y = torch.tensor(df["loan_status"], dtype=torch.float32)\
    .reshape((-1, 1))

print(df.drop("loan_status", axis=1))
X_data = df.drop("loan_status", axis=1).astype('float32').values
print(X_data.dtype)
X = torch.tensor(X_data, dtype=torch.float32)
print(X)
