# We’re going to teach a tiny "robot brain" to guess used-car prices.
# The robot will look at two clues for each car:
#   1) how old the car is (age)
#   2) how many miles it has (mileage)
# Then it will practice many times to get better at guessing.

import sys
import pandas as pd          # pandas: helps us read and clean table data (like Excel)
import torch                 # torch: tools for building and training tiny robot brains
from torch import nn         # nn: "neural network" building blocks from torch


# -----------------------------
# 1) READ the data from a file
# -----------------------------
# Think of this as opening a spreadsheet of cars.
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/data/used_cars.csv")


# -------------------------------------------
# 2) CLEAN and PREPARE the columns we need
# -------------------------------------------

# AGE of each car:
# newest_car_year - this_car_year
# Example: newest is 2020, this car is 2015 → age = 5
age = df["model_year"].max() - df["model_year"]

# MILEAGE:
# The file has text like "45,000 mi.".
# Remove commas and the " mi." text, then turn it into a number.
milage = df["milage"]
milage = milage.str.replace(",", "", regex=False)     # "45,000" → "45000"
milage = milage.str.replace(" mi.", "", regex=False)  # remove the " mi." part
milage = milage.astype(int)                           # now it's an integer (whole number)

# PRICE:
# The file has things like "$12,500".
# Remove "$" and commas so it becomes a plain number, like 12500.
price = df["price"]
price = price.str.replace("$", "", regex=False)
price = price.str.replace(",", "", regex=False)
price = price.astype(int)


# -------------------------------------------------------------
# 3) Turn the data into TENSORS (the format PyTorch likes)
# -------------------------------------------------------------
# X is our INPUT table with two columns:
#   column 1 = age, column 2 = mileage
# Each row is one car.
X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])

# STANDARDIZE X (make numbers easier to learn from):
# We shift and scale each column so:
#   - the average becomes 0
#   - the size/spread becomes about 1
# This helps the robot learn faster and not get confused by huge numbers.
X_mean = X.mean(axis=0)      # mean (average) of each column: [mean_age, mean_mileage]
X_std  = X.std(axis=0)       # std (spread) of each column:  [std_age, std_mileage]
X = (X - X_mean) / X_std     # z-score: (value - mean) / std

# y is our TARGET (the correct answer we want the robot to guess) = price
# We reshape to a COLUMN so it lines up with rows in X.
y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))

# STANDARDIZE y as well (same idea as X):
y_mean = y.mean()            # average price
y_std  = y.std()             # how much prices vary
y = (y - y_mean) / y_std     # scaled price (z-score)
# If you ever want to pause and inspect values, you could do:
# sys.exit()


# -----------------------------------------
# 4) Build the robot brain (the model)
# -----------------------------------------
# nn.Linear(2, 1) means:
#   - it takes 2 inputs (age and mileage)
#   - it produces 1 output (predicted price)
model = nn.Linear(2, 1)

# Loss function = "how wrong are we?"
# Smaller loss means our guesses are closer to the real prices.
loss_fn = torch.nn.MSELoss()

# Optimizer = the "coach" that updates the model to do better.
# lr (learning rate) = how big each correction step is.
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


# -----------------------------------------
# 5) TRAIN the model (lots of practice!)
# -----------------------------------------
# We repeat 10,000 times:
#   guess → check how wrong → learn a little → repeat
for i in range(0, 10000):
    optimizer.zero_grad()     # clear old learning signals
    outputs = model(X)        # model makes guesses for every car in our table
    loss = loss_fn(outputs, y)  # compare guesses vs true (standardized) prices
    loss.backward()           # figure out how to adjust the model's numbers
    optimizer.step()          # make a small improvement (one learning step)

    if i % 100 == 0: 
        print(loss)           # show progress every 100 steps (should go down over time)
    # If you want to peek under the hood:
    # if i % 100 == 0: 
    #     print(model.bias)    # the "bonus" value the model adds (b)
    #     print(model.weight)  # the "multipliers" for age and mileage (weights w1, w2)


# -----------------------------------------
# 6) USE the trained model to PREDICT
# -----------------------------------------
# Let's estimate prices for three cars we made up:
#   [age, mileage]
X_data = torch.tensor([
    [5, 10000],   # car A: 5 years old, 10k miles
    [2, 10000],   # car B: 2 years old, 10k miles
    [5, 20000]    # car C: 5 years old, 20k miles
], dtype=torch.float32)

# We must standardize new inputs the SAME WAY we did for training,
# using the same X_mean and X_std we computed earlier.
X_data_std = (X_data - X_mean) / X_std

# Get predicted prices (these are standardized, not in dollars yet)
prediction_std = model(X_data_std)

# Convert predictions back to real dollars (undo standardization)
# The prediction_dollars numbers are in dollars because you “de-standardized” them at the end:
prediction_dollars = prediction_std * y_std + y_mean
print(prediction_dollars)   # final price guesses in dollars

"""
What this output means
tensor([[69617.7266],
        [70284.3438],
        [65178.5117]], grad_fn=<AddBackward0>)

Each row corresponds to one of the cars you asked about:
    Car A → 5 years old, 10,000 miles → Predicted price ≈ $69,618
    Car B → 2 years old, 10,000 miles → Predicted price ≈ $70,284
    Car C → 5 years old, 20,000 miles → Predicted price ≈ $65,179

    You trained a robot to guess car prices from age and mileage.

When you gave it three new cars, it predicted their prices:
    The newer car (2 years old) with low mileage is worth the most.
    The older cars (5 years old) are worth less, especially the one with more mileage.
This shows the robot learned a pattern:
    Cars lose value as they get older or get more miles.

So your output is the predicted selling price for each car, according to what your model has learned.

    
Read & clean the car table.
Build a feature table X = [age, mileage] and answers y = price.
Standardize numbers so learning is easier.
A tiny model learns by guess → check → correct thousands of times.
Use the trained model to predict prices for new cars, then convert back to dollars.
"""

"""
Why is it important to denormalize model predictions before interpreting them?
Denormalizing converts predictions back to the original scale, making them interpretable in real-world terms.
Predictions are made on a normalized scale for model stability. To understand them in their original context (e.g. dollars or age), they must 
be scaled back to the original data's units.

"""