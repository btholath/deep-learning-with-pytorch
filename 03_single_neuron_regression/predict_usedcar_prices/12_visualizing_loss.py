# GOAL: Teach a tiny "robot brain" (a simple model) to estimate used-car prices.
# IDEA: The robot will look at two clues (features) for each car:
#       1) how old the car is (age)
#       2) how many miles it has been driven (mileage)
# Then it practices (training) to make better and better price guesses.

import sys
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

# -----------------------------
# 1) READ the data table
# -----------------------------
# IDEA: We need real examples so the robot can learn.
# ACTION: Load the spreadsheet of cars (age-like info and prices).
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/data/used_cars.csv")

# -----------------------------
# 2) CLEAN and PREPARE columns
# -----------------------------
# IDEA: The CSV has extra symbols/words (like "$", ",", " mi.") that computers don't understand as numbers.
# ACTION: Remove those and turn the columns into clean numbers.

# AGE: "how old is the car?" = newest_model_year - this_car_model_year
age = df["model_year"].max() - df["model_year"]

# MILEAGE: remove commas and the " mi." text, then make it an integer
milage = df["milage"]
milage = milage.str.replace(",", "", regex=False)
milage = milage.str.replace(" mi.", "", regex=False)
milage = milage.astype(int)

# PRICE: remove "$" and commas, then make it an integer
price = df["price"]
price = price.str.replace("$", "", regex=False)
price = price.str.replace(",", "", regex=False)
price = price.astype(int)

# -----------------------------
# 3) BUILD INPUTS (X) and TARGET (y)
# -----------------------------
# IDEA: Models expect a "table" of inputs (features) and a matching list of correct answers (targets).
# - X = what the robot SEES (age, mileage)
# - y = what we WANT it to SAY (price)
X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),     # column 1: age
    torch.tensor(milage, dtype=torch.float32)   # column 2: mileage
])

# -----------------------------
# 4) NORMALIZE (z-score) X and y
# -----------------------------
# IDEA: Age and mileage (and price) are on very different scales (e.g., 5 years vs 70,000 miles vs $30,000).
#       That can confuse the robot. We "normalize" so each column is centered around 0 and has a similar scale.
# RESULT: Learning becomes smoother and faster.
X_mean = X.mean(axis=0)       # average age, average mileage
X_std  = X.std(axis=0)        # spread of age, spread of mileage
X = (X - X_mean) / X_std      # standardized X

y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))  # make prices a column vector
y_mean = y.mean()            # average price
y_std  = y.std()             # spread of prices
y = (y - y_mean) / y_std     # standardized y

# -----------------------------
# 5) DEFINE the model and training tools
# -----------------------------
# IDEA: Use the simplest possible model (a single linear layer) that tries to draw a best-fit plane
#       through the points [age, mileage] → price.
model = nn.Linear(2, 1)                 # 2 inputs (age, mileage) → 1 output (price)
loss_fn = torch.nn.MSELoss()            # "How wrong are we?" (smaller is better)
optimizer = torch.optim.SGD(            # The "coach" that nudges the model to improve
    model.parameters(), lr=0.01         # lr = learning rate (how big each nudge is)
)

# -----------------------------
# 6) TRAIN: guess → check → correct (repeat)
# -----------------------------
# IDEA: Practice many times. Each loop:
#   - model guesses prices,
#   - we measure error (loss),
#   - we adjust the model a tiny bit to reduce that error.
losses = []
for i in range(0, 250):
    optimizer.zero_grad()       # clear old gradients (old correction notes)
    outputs = model(X)          # robot guesses prices for all rows
    loss = loss_fn(outputs, y)  # compare to true (standardized) prices
    loss.backward()             # compute how to change weights/bias to reduce loss
    optimizer.step()            # apply a small update (learning step)

    losses.append(loss.item())  # remember the loss so we can plot learning progress
    # if i % 100 == 0:
    #     print(loss.item())     # peek at the loss occasionally

# -----------------------------
# 7) SEE learning progress (plot)
# -----------------------------
# IDEA: A picture helps us see learning. The line should go DOWN as the robot improves.
plt.plot(losses)
plt.title("Training Loss Going Down (Learning Progress)")
plt.xlabel("Training Step")
plt.ylabel("Loss (How Wrong the Model Is)")
plt.show()
plt.savefig("/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/12_visualizing_loss.png")

# -----------------------------
# 8) PREDICT on NEW cars
# -----------------------------
# IDEA: Now that the robot has learned, ask it to estimate prices for new examples.
# IMPORTANT: We must normalize new inputs with the SAME mean/std used during training.
X_data = torch.tensor([
    [5, 10000],   # car A: 5 years old, 10k miles
    [2, 10000],   # car B: 2 years old, 10k miles
    [5, 20000]    # car C: 5 years old, 20k miles
], dtype=torch.float32)

# Normalize new data using training stats (X_mean, X_std)
X_data_norm = (X_data - X_mean) / X_std

# Model predicts STANDARDIZED prices → convert back to DOLLARS using y_mean/y_std
prediction_std = model(X_data_norm)
prediction_dollars = prediction_std * y_std + y_mean

# RESULT: Estimated prices in real money
print(prediction_dollars)

"""
We cleaned the data so the computer can read it.
We normalized numbers so nothing is unfairly “huge” or “tiny.”
We trained a simple model that learns a pattern: older/more-miles → lower price (usually).
We plotted the loss to see learning improve.
We predicted prices for new cars and printed them in dollars.
"""

"""
What role does data visualization play before training a machine learning model?
Visaulization helps identify patterns, relationships and potential issues in the data.
By visualizing data, you can spot trends, correlations, and anomalies, which informs data preparation and model choices.

"""