"""
This script introduces a single training step for a neuron using one data point.

Data: 
We use one example: 10°C should convert to 50°F.

Model: 
nn.Linear(1, 1) creates a neuron that does $ output = w_1 . input + b $.

Loss Function: 
MSELoss measures how far the prediction is from the actual value (squared difference).

Optimizer: 
SGD (Stochastic Gradient Descent) adjusts the weight and bias to reduce the error.

Training Step:
    Compute the prediction for 10°C.
    Calculate the error (loss).
    Use backward() to figure out how to tweak the weight and bias.
    Update them with step().

Output: 
Shows the bias before and after one update and the prediction for 10°C.

Why It’s Useful: 
Introduces the core idea of training a neuron with one step, showing how weights and biases change slightly.    
"""

import torch
from torch import nn

# Input: Temperature in Celsius (one example)
X1 = torch.tensor([[10]], dtype=torch.float32)  # 10°C
# Actual value: Temperature in Fahrenheit
y1 = torch.tensor([[50]], dtype=torch.float32)  # 50°F (10 * 1.8 + 32 = 50)

# Create a simple neuron (linear model: output = weight * input + bias)
model = nn.Linear(1, 1)  # 1 input, 1 output
loss_fn = torch.nn.MSELoss()  # Mean Squared Error loss to measure error
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # Optimizer adjusts weights

# Print initial bias (randomly set by PyTorch)
print("Initial bias:", model.bias.item())

# Training step
optimizer.zero_grad()  # Clear previous gradients
outputs = model(X1)    # Compute prediction: weight * X1 + bias
loss = loss_fn(outputs, y1)  # Calculate error (difference between prediction and actual)
loss.backward()        # Compute how to adjust weight and bias
optimizer.step()       # Update weight and bias

# Print updated bias after one step
print("Updated bias:", model.bias.item())

# Make a prediction for 10°C
y1_pred = model(X1)
print("Prediction for 10°C:", y1_pred.item(), "°F")