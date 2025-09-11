# We are using two main tools here:
# 1. pandas (pd) - helps us read and clean up data from a file (like Excel for Python).
# 2. torch - a library for building and training "neurons" (like a robot brain).

import pandas as pd
import torch
from torch import nn   # nn = "neural network" tools

# ------------------------------------------------------
# Step 1: Read the car data from a CSV file (like a spreadsheet).
# ------------------------------------------------------
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/data/used_cars.csv")

# ------------------------------------------------------
# Step 2: Prepare the features (inputs) we want the neuron to learn from.
# ------------------------------------------------------

# Age of the car = newest car year (max) minus the car's year
# Example: if the newest car is 2020 and this one is 2015 → age = 5
age = df["model_year"].max() - df["model_year"]

# Clean up the "milage" column:
milage = df["milage"]
milage = milage.str.replace(",", "")     # remove commas like "45,000" → "45000"
milage = milage.str.replace(" mi.", "")  # remove the text " mi."
milage = milage.astype(int)              # turn it into whole numbers (integers)

# Clean up the "price" column:
price = df["price"]
price = price.str.replace("$", "")       # remove dollar signs
price = price.str.replace(",", "")       # remove commas
price = price.astype(int)                # turn it into whole numbers (integers)

# ------------------------------------------------------
# Step 3: Create the input (X) and output (y) for training.
# ------------------------------------------------------

# X = the features (inputs) we will give the neuron
# Here: two columns → [age, mileage]
X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])

# y = the target (output) we want the neuron to learn
# In this case: the car's price
y = torch.tensor(price, dtype=torch.float32).view(-1, 1)

# .view() in PyTorch reshapes a tensor (like changing the shape of a Lego block without changing how many pieces you have).
# The arguments inside .view(rows, cols) tell PyTorch the new shape
# means:
# -1 → “figure out this dimension automatically so that the total number of elements stays the same.”
# 1 → “make sure there’s exactly 1 column.”


# ------------------------------------------------------
# Step 4: Build a simple neuron model.
# ------------------------------------------------------

# nn.Linear(2, 1) means:
# - 2 inputs (age and mileage)
# - 1 output (predicted price)
model = nn.Linear(2, 1)

# Loss function: tells us how wrong the neuron is
loss_fn = torch.nn.MSELoss()

# Optimizer: the "teacher" that helps the neuron adjust its numbers (weights & bias)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
# lr = learning rate → how big each correction step is

# ------------------------------------------------------
# Step 5: Try the model once (before training).
# ------------------------------------------------------

# Ask the neuron to make predictions for all cars in X
prediction = model(X)

# Print the neuron's first guesses for car prices
print(prediction)
