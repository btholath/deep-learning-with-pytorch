# GOAL: Teach a simple "robot brain" (model) to estimate car prices.
# IDEA: The robot will use 3 clues about each car:
#   1) Is it accident-free? (0 = had accident, 1 = none reported)
#   2) How old the car is (age in years)
#   3) How many miles it has (mileage)
# Then the robot will practice with real data to learn the pattern.

import sys
import pandas as pd      # pandas: helps us read and clean up data tables
import torch             # torch: library for math and machine learning
from torch import nn     # nn: neural network tools for building models

# -----------------------------
# 1) Read the data
# -----------------------------
# IDEA: Load a spreadsheet of used cars with details like model year, mileage, price, accident history.
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/data/used_cars.csv")

# -----------------------------
# 2) Prepare the features (inputs) and target (output)
# -----------------------------
# AGE of each car = newest car year - this car's year
age = df["model_year"].max() - df["model_year"]

# MILEAGE: remove commas and " mi." so it becomes a clean number
milage = df["milage"]
milage = milage.str.replace(",", "", regex=False)
milage = milage.str.replace(" mi.", "", regex=False)
milage = milage.astype(int)

# ACCIDENT-FREE: make it 1 if no accidents, 0 if there were accidents
accident_free = df["accident"] == "None reported"
accident_free = accident_free.astype(int)

# PRICE: remove "$" and commas so it's a number
price = df["price"]
price = price.str.replace("$", "", regex=False)
price = price.str.replace(",", "", regex=False)
price = price.astype(int)

# -----------------------------
# 3) Turn into tensors for PyTorch
# -----------------------------
# IDEA: Models work best when inputs are in a "matrix" (table of numbers).
# - X = input features (accident-free, age, mileage)
# - y = target output (price we want to predict)
X = torch.column_stack([
    torch.tensor(accident_free, dtype=torch.float32),  # 1st column: accident-free flag
    torch.tensor(age, dtype=torch.float32),            # 2nd column: age
    torch.tensor(milage, dtype=torch.float32)          # 3rd column: mileage
])

# NORMALIZE X → scale so each column has mean 0 and standard deviation 1
# IDEA: Without this, mileage (big numbers like 70,000) could "overpower" smaller numbers like age (5 years).
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# y = price (output we want to learn)
y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))

# NORMALIZE y → same reason, prices are very large numbers ($10k–$70k).
y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std

# -----------------------------
# 4) Build the model
# -----------------------------
# nn.Linear(3, 1) = a neuron that takes 3 inputs (accident-free, age, mileage)
# and outputs 1 value (predicted price).
model = nn.Linear(3, 1)

# Loss function = "how wrong was the guess?" (Mean Squared Error)
loss_fn = torch.nn.MSELoss()

# Optimizer = the "coach" that updates the robot’s internal numbers (weights & bias) after each mistake.
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # lr = learning rate = step size

# -----------------------------
# 5) Training loop (practice many times)
# -----------------------------
# IDEA: Each loop:
#   1. Robot guesses car prices.
#   2. Compare guess with actual prices (loss).
#   3. Calculate how to improve (backpropagation).
#   4. Update model slightly (optimizer step).
for i in range(0, 10000):
    optimizer.zero_grad()         # reset old learning signals
    outputs = model(X)            # robot guesses prices
    loss = loss_fn(outputs, y)    # measure how wrong
    loss.backward()               # calculate corrections
    optimizer.step()              # apply corrections

    if i % 100 == 0: 
        print(loss)               # print progress every 100 steps

# -----------------------------
# 6) Predictions on NEW cars
# -----------------------------
# IDEA: Test the trained robot with new examples.
# Format: [accident_free, age, mileage]

# Example: accident-free cars
X_data = torch.tensor([
    [1, 5, 10000],   # accident-free, 5 yrs old, 10k miles
    [1, 2, 10000],   # accident-free, 2 yrs old, 10k miles
    [1, 5, 20000]    # accident-free, 5 yrs old, 20k miles
], dtype=torch.float32)

# Normalize using same stats (X_mean, X_std) as training
prediction = model((X_data - X_mean) / X_std)

# Convert back to real dollars (undo normalization)
print(prediction * y_std + y_mean)

# Example: cars with accidents
X_data_accident = torch.tensor([
    [0, 5, 10000],   # had accident, 5 yrs old, 10k miles
    [0, 2, 10000],   # had accident, 2 yrs old, 10k miles
    [0, 5, 20000]    # had accident, 5 yrs old, 20k miles
], dtype=torch.float32)

prediction_accident = model((X_data_accident - X_mean) / X_std)
print(prediction_accident * y_std + y_mean)
# The output shows first the car prices of without accident and then with car prices that had accident


"""
We feed in car details (accident history, age, mileage).
The robot practices thousands of times to learn the pattern between details and price.
After training, the robot can predict car prices for new examples.
We even compare accident-free vs accident cars to see how accidents affect price.
"""