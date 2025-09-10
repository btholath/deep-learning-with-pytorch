"""
Step-by-Step Explanation:

Similarities: Like training_neuron2.py, it trains with two data points but runs for fewer iterations (50,000).
Differences: Slightly different print order (weight before bias) and no final prediction output.
Why It’s Useful: Reinforces the concept of training with multiple data points, showing consistent results with training_neuron2.py.
"""
import torch
from torch import nn

# Input: Temperatures in Celsius
X1 = torch.tensor([[10.0]])    # 10°C
y1 = torch.tensor([[50.0]])    # 50°F
X2 = torch.tensor([[37.78]])   # 37.78°C
y2 = torch.tensor([[100.0]])   # 100°F

# Create a neuron
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
loss_fn = torch.nn.MSELoss()

# Train for 50,000 iterations
for i in range(0, 50000):
    # Train with first data point
    optimizer.zero_grad()
    outputs = model(X1)
    loss = loss_fn(outputs, y1)
    loss.backward()
    optimizer.step()

    # Train with second data point
    optimizer.zero_grad()
    outputs = model(X2)
    loss = loss_fn(outputs, y2)
    loss.backward()
    optimizer.step()

    # Print progress
    if i % 100 == 0:
        print("Iteration", i)
        print("Weight:", model.weight.item())
        print("Bias:", model.bias.item())