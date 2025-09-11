import sys
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt   # NEW: for charts

# -----------------------------
# 1) READ the data
# -----------------------------
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/data/used_cars.csv")

# -----------------------------
# 2) CLEAN and PREPARE columns
# -----------------------------
age = df["model_year"].max() - df["model_year"]

milage = df["milage"]
milage = milage.str.replace(",", "", regex=False)
milage = milage.str.replace(" mi.", "", regex=False)
milage = milage.astype(int)

price = df["price"]
price = price.str.replace("$", "", regex=False)
price = price.str.replace(",", "", regex=False)
price = price.astype(int)

# -----------------------------
# 3) Convert to tensors (before normalization)
# -----------------------------
X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])
y = torch.tensor(price, dtype=torch.float32).reshape((-1, 1))

# Save copies before normalization for plotting
X_before = X.clone()
y_before = y.clone()

# -----------------------------
# 4) Normalize (z-score scaling)
# -----------------------------
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std

# -----------------------------
# 5) Plot BEFORE vs AFTER normalization
# -----------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Left: before normalization
ax[0].scatter(X_before[:, 0], y_before, alpha=0.6, c="blue")
ax[0].set_title("Before Normalization")
ax[0].set_xlabel("Age (years)")
ax[0].set_ylabel("Price ($)")

# Right: after normalization
ax[1].scatter(X[:, 0], y, alpha=0.6, c="green")
ax[1].set_title("After Normalization (z-score)")
ax[1].set_xlabel("Age (standardized)")
ax[1].set_ylabel("Price (standardized)")

plt.tight_layout()
plt.show()
plt.savefig("/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/11_1_neuron_normalized2.png")

# -----------------------------
# 6) Build and train the model
# -----------------------------
model = nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for i in range(0, 10000):
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    if i % 1000 == 0:
        print("Step", i, "Loss:", loss.item())

# -----------------------------
# 7) Make predictions
# -----------------------------
X_data = torch.tensor([
    [5, 10000],
    [2, 10000],
    [5, 20000]
], dtype=torch.float32)

# Apply same normalization to new data
X_data_std = (X_data - X_mean) / X_std

# Predict (standardized → convert back to $)
prediction = model(X_data_std)
print(prediction * y_std + y_mean)

"""
Left scatter plot (Before Normalization):
    Age is small numbers (like 2, 5, 10).
    Price is huge numbers (like 20,000–70,000).
    The scales look very different.

Right scatter plot (After Normalization):
    Both age and price are rescaled so they’re centered around 0.
    Most points fall between -2 and +2.
    Much easier for the model to learn.
"""