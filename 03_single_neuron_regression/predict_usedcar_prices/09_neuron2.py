# We are going to teach a computer "neuron" (a simple brain cell model)
# to guess the price of a used car based on two things:
# 1. How old the car is (age)
# 2. How many miles it has been driven (mileage)

import pandas as pd      # pandas = tool for reading and cleaning data (like Excel for Python)
import torch             # torch = library for building "neurons" and training them
from torch import nn     # nn = neural network tools from torch

# ------------------------------------------------------------
# STEP 1: Read the data from a CSV file (like a spreadsheet)
# ------------------------------------------------------------
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/data/used_cars.csv")

# ------------------------------------------------------------
# STEP 2: Prepare the input data (features)
# ------------------------------------------------------------

# Age of the car = (newest car's year) - (this car's year)
# Example: if the newest car is from 2020 and this one is 2015 → age = 5
age = df["model_year"].max() - df["model_year"]

# Clean up the mileage column
milage = df["milage"]
milage = milage.str.replace(",", "")      # remove commas → "45,000" → "45000"
milage = milage.str.replace(" mi.", "")   # remove the " mi." text
milage = milage.astype(int)               # turn it into whole numbers

# Clean up the price column
price = df["price"]
price = price.str.replace("$", "")        # remove the $ sign
price = price.str.replace(",", "")        # remove commas
price = price.astype(int)                 # turn it into whole numbers

# ------------------------------------------------------------
# STEP 3: Convert the data into tensors (the format Torch uses)
# ------------------------------------------------------------

# X = input features (age and mileage of each car)
X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])

# y = target output (the car's price, which we want the neuron to learn)
# .reshape((-1, 1)) means: make it into a column vector
# Example: instead of [12000, 18000], make it [[12000], [18000]]
y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))

# ------------------------------------------------------------
# STEP 4: Create the neuron model
# ------------------------------------------------------------

# nn.Linear(2, 1) = a neuron with:
# - 2 inputs (age and mileage)
# - 1 output (predicted price)
model = nn.Linear(2, 1)

# Loss function: measures how wrong the neuron is
loss_fn = torch.nn.MSELoss()

# Optimizer: the "teacher" that helps the neuron adjust itself
# lr (learning rate) = how big the correction steps are
optimizer = torch.optim.SGD(model.parameters(), lr=0.00000000001)

# ------------------------------------------------------------
# STEP 5: Train the neuron
# ------------------------------------------------------------

# Do this 1000 times → like practicing 1000 math problems
for i in range(0, 1000):
    optimizer.zero_grad()       # clear old "memory of mistakes"
    outputs = model(X)          # neuron guesses car prices
    loss = loss_fn(outputs, y)  # compare guesses vs real prices
    loss.backward()             # figure out how to adjust itself
    optimizer.step()            # make small corrections (update weights & bias)

    print(loss)  # show how wrong the neuron still is (should get smaller)

    # If you want to peek at the neuron's "numbers" (bias and weight), uncomment below:
    # if i % 100 == 0: 
    #     print(model.bias)
    #     print(model.weight)

# ------------------------------------------------------------
# STEP 6: Test the trained neuron
# ------------------------------------------------------------

# Ask the neuron to predict the price of a car that is:
# - 5 years old
# - has 20,000 miles
prediction = model(torch.tensor([
    [5, 20000]
], dtype=torch.float32))

print(prediction)  # the neuron's predicted price


"""
Read the car data (age, mileage, price).
Clean it up (remove $, commas, "mi.").
Give it to a neuron (like a little robot brain).
Neuron practices (training loop) → each time, it guesses, gets corrected, and improves.
Neuron predicts the price of a car it hasn’t seen before.
"""