import sys
import os
import pandas as pd
import torch
from torch import nn

# Pandas: Reading the data
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/data/used_cars.csv")

# Pandas: Preparing the data
age = df["model_year"].max() - df["model_year"]

milage = df["milage"]
milage = milage.str.replace(",", "")
milage = milage.str.replace(" mi.", "")
milage = milage.astype(int)

price = df["price"]
price = price.str.replace("$", "")
price = price.str.replace(",", "")
price = price.astype(int)

if not os.path.isdir("/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/model"):
    os.mkdir("/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/model")

# Torch: Creating X and y data (as tensors)
X = torch.column_stack([
    torch.tensor(age, dtype=torch.float32),
    torch.tensor(milage, dtype=torch.float32)
])
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
torch.save(X_mean, "/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/model/X_mean.pt")
torch.save(X_std, "/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/model/X_std.pt")
X = (X - X_mean) / X_std

y = torch.tensor(price, dtype=torch.float32)\
    .reshape((-1, 1))
y_mean = y.mean()
y_std = y.std()
torch.save(y_mean, "/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/model/y_mean.pt")
torch.save(y_std, "/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/model/y_std.pt")
y = (y - y_mean) / y_std
# sys.exit()


model = nn.Linear(2, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(0, 2500):
    # Training pass
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    #if i % 100 == 0: 
    #    print(loss)

torch.save(model.state_dict(), "/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/model/model.pt")


"""
What is one advantage of storing a trained model, rather than retraining it each time a prediction is needed?
Storing the moel saves time, as predictions can be made immediately without retraining.
Once a model is stored, it can be loaded and used for predictions directly, avoiding the computational cost of training again.

"""