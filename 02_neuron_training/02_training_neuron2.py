"""
This extends training_neuron.py by training with two data points over many iterations.

Step-by-Step Explanation:

Data: Two examples: (10°C, 50°F) and (37.78°C, 100°F).
Training Loop: Runs 100,000 times, updating the model with each data point separately.
Process:

For each data point, compute the prediction, calculate the loss, and update the weight and bias.
The model learns to make predictions closer to the actual Fahrenheit values.


Output: Prints the weight and bias every 100 iterations and the final prediction for 10°C.
Why It’s Useful: Shows how repeated training with multiple data points helps the neuron learn better weights and biases (closer to 1.8 and 32).
"""
import torch
from torch import nn

# Input: Temperatures in Celsius
X1 = torch.tensor([[10]], dtype=torch.float32)    # 10°C
y1 = torch.tensor([[50]], dtype=torch.float32)    # 50°F
X2 = torch.tensor([[37.78]], dtype=torch.float32) # 37.78°C
y2 = torch.tensor([[100.0]], dtype=torch.float32) # 100°F

# Create a neuron
model = nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)  # Small learning rate

# Train for 100,000 iterations
for i in range(0, 100000):
    # Train with first data point (10°C, 50°F)
    optimizer.zero_grad()
    outputs = model(X1)
    loss = loss_fn(outputs, y1)
    loss.backward()
    optimizer.step()

    # Train with second data point (37.78°C, 100°F)
    optimizer.zero_grad()
    outputs = model(X2)
    loss = loss_fn(outputs, y2)
    loss.backward()
    optimizer.step()

    # Print progress every 100 iterations
    if i % 100 == 0:
        print("Iteration", i)
        print("Bias:", model.bias.item())
        print("Weight:", model.weight.item())

# Make a prediction for 10°C
y1_pred = model(X1)
print("Prediction for 10°C:", y1_pred.item(), "°F")