import sys
import pandas as pd
import torch
from torch import nn

X_mean = torch.load("/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/model/X_mean.pt", weights_only=True)
X_std = torch.load("/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/model/X_std.pt", weights_only=True)
y_mean = torch.load("/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/model/y_mean.pt", weights_only=True)
y_std = torch.load("/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/model/y_std.pt", weights_only=True)

model = nn.Linear(2, 1)
model.load_state_dict(
    torch.load("/workspaces/deep-learning-with-pytorch/03_single_neuron_regression/predict_usedcar_prices/model/model.pt", weights_only=True)
)
model.eval()

X_data = torch.tensor([
    [5, 10000],
    [2, 10000],
    [5, 20000]
], dtype=torch.float32)

with torch.no_grad():
    prediction = model((X_data - X_mean) / X_std)
    print(prediction * y_std + y_mean)