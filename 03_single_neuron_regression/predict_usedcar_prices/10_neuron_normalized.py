"""
So far, the problem with our model
Issue summary
    - Data was difficult for the neuron to learn
    - Large changes in output made learning unstable
Result: Gradient explosion:
    - High difference between prediction “y-bar” and the actual value (y)    
    - Large gradients caused drastic weight updates
Impact: Unstable learning
    - Weights became too large to compute
    - Gradients became too large
    - This led to invalid numbers (NaN)


Solution:
Normalizing the data
Why to do?
    Stabilizes learning
        - Puts predictions into a smaller range
        - Results in smaller, controlled gradients
        - Leads to smoother weight updates and more stable learning
    How to normalize (Z-score)            :
    Step#1
        - Center data around zero 0
    Step#2
        - Divide by the standard deviation σ (sigma) of the dataset
    This ensures most data is (mostly) between -2 and 2


Purpose of Normalizing the output data in our model:
    to stabilize learning and allow smoother weight updates
Normalizing output data brings values into a smaller, controlled range, making learning stable and weight updates smoother. 
This helps the model avoid large weight adjustments and prevents gradient issues, enabling the neuron to learn effectively.

Given that the price column ranges from 0$ to 300,000$, after normalization, what is the smallest interval in which most values will fall?
Between -2 and +2, with some outliers possible.
Z-score normalization centers data around zero and scales it by its standard deviatio, so most values fall between -2 and +2, though some
outliers may extend beyond this range.

Purpose of Normalizing the input data for a neuron model:
    to bring input features to a similar scale, allowing uniform learning rates
Normalizing input data ensures that each feature (like age and mileage) has a similar scale,
helping the model adjust weights uniformly without needing different learning rates.

When normalizing input data with values like car age (1-20 years) and mileage (1,000-300,000 miles), what is the expected result?
Both age and mileage are transformed to a similar scale, generally between -2 and 2

"""

# We are teaching a tiny "robot brain" (a neuron) to guess car prices.
# It will use two clues (features): how old the car is (age) and how many miles it has (mileage).

import sys
import pandas as pd           # pandas: helps read and clean spreadsheet-like data
import torch                  # torch: tools for building and training tiny robot brains (neurons)
from torch import nn          # nn: neural network building blocks

# -----------------------------
# 1) READ the data from a CSV
# -----------------------------
# Think of this like opening a spreadsheet of cars.
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/data/used_cars.csv")

# ---------------------------------------
# 2) PREPARE features (inputs) and price
# ---------------------------------------

# AGE of each car = (newest year in the dataset) - (this car's year)
# Example: newest is 2020 and this car is 2015 → age = 5
age = df["model_year"].max() - df["model_year"]

# Clean up the "milage" column:
# The file has text like "45,000 mi." — we remove commas and the " mi." part
milage = df["milage"]
milage = milage.str.replace(",", "", regex=False)    # "45,000" → "45000"
milage = milage.str.replace(" mi.", "", regex=False) # remove text
milage = milage.astype(int)                          # turn text into whole numbers

# Clean up the "price" column:
# The file has "$12,500" — we remove "$" and commas so it's just 12500
price = df["price"]
price = price.str.replace("$", "", regex=False)
price = price.str.replace(",", "", regex=False)
price = price.astype(int)

# ----------------------------------------------------------
# 3) Convert to TENSORS (the format Torch models understand)
# ----------------------------------------------------------

# X = inputs (features) we give to the neuron: [age, mileage]
# Column stack builds a table:
#   row = one car
#   column 1 = age
#   column 2 = mileage
X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])

# y = the correct answers (target) = the car prices
# .reshape((-1, 1)) makes it a COLUMN (one price per car)
y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))

# ----------------------------------------------------------
# 4) Optional: STANDARDIZE y (make learning easier)
# ----------------------------------------------------------
# Prices can be very big numbers (like 25,000). We scale them so they are
# centered around 0 with a size of about 1. This helps the robot learn better.
y_mean = y.mean()   # average price
y_std  = y.std()    # how much prices vary | μ (mu) = the mean (average) of the dataset
y = (y - y_mean) / y_std    # now prices are "standardized" (z-scores) | 𝜎 (sigma) = the standard deviation of the dataset

# If you ever want to stop the script here to inspect values, you could uncomment:
# sys.exit()

# ----------------------------------------------
# 5) BUILD the model (our tiny robot brain)
# ----------------------------------------------
# nn.Linear(2, 1) means:
# - it takes 2 input numbers (age, mileage)
# - it outputs 1 number (predicted price)
model = nn.Linear(2, 1)

# Loss function: tells us "how wrong" the prediction is (smaller is better)
loss_fn = torch.nn.MSELoss()

# Optimizer: the "coach" that nudges the model to improve each time
# lr = learning rate (how big each nudge is). Very small here → tiny steps.
optimizer = torch.optim.SGD(model.parameters(), lr=0.0000000001)

# ----------------------------------------------
# 6) TRAIN the model (practice many times)
# ----------------------------------------------
for i in range(0, 1000):
    optimizer.zero_grad()   # clear old training signals
    outputs = model(X)      # model makes guesses for all cars
    loss = loss_fn(outputs, y)  # compare guesses to the true (standardized) prices
    loss.backward()         # figure out how to adjust the model
    optimizer.step()        # apply a small correction (learn a bit)

    # You can print the loss to watch it get smaller:
    # print(loss)

    # You can peek at the model’s internal numbers every 100 steps:
    # if i % 100 == 0:
    #     print(model.bias)
    #     print(model.weight)

# ----------------------------------------------
# 7) USE the model to PREDICT a new car’s price
# ----------------------------------------------
# Predict the price for a car that is:
# - age = 5 years
# - mileage = 10,000 miles
prediction_standardized = model(torch.tensor([[5, 10000]], dtype=torch.float32))

# Our model was trained to predict STANDARDIZED prices,
# so we convert back to real dollars:
prediction_dollars = prediction_standardized * y_std + y_mean

print(prediction_dollars)   # this prints the predicted price in dollars


"""
Read a table of cars (age, mileage, price).
Clean the numbers so the computer can understand them.
Give clues (age, mileage) to a tiny robot brain and show the answers (price).
The robot practices 1,000 times: guess → check → get corrected → improve.
Then we ask the robot to guess the price of a new car (age 5, 10k miles).
Because we “standardized” prices during training, we convert the robot’s answer back to real dollars to print it out.
Note: Your learning rate (lr) is extremely tiny, so learning may be very slow. If the predictions don’t change much, try a more reasonable lr, like 1e-6 or 1e-5, and watch the loss go down faster.
"""