"""
This introduces batch training, processing both data points together.

Step-by-Step Explanation:

Batch Input: Both inputs (10°C, 37.78°C) are processed together in a single tensor.
Training: The model computes predictions for both inputs, calculates the combined loss, and updates the weight and bias once per iteration.
Evaluation: Tests the model on a new input (37.5°C) to predict Fahrenheit.
Why It’s Useful: Introduces batch training, which is more efficient and common in deep learning, and shows how to test a trained model.

What It Does: Trains the neuron with both data points together (batch training) for 150,000 iterations and tests the model on a new input (37.5°C).
Key Idea: Batch training processes multiple data points at once, which is faster and more common in deep learning, and testing shows how the model generalizes.
Analogy: Like cooking for a group instead of one person at a time—it’s more efficient.
Why Last: Introduces advanced batch training and model evaluation, building on prior understanding.
"""
import torch
from torch import nn

# Input: Temperatures in Celsius (as a batch)
X = torch.tensor([
    [10],
    [37.78]
], dtype=torch.float32)

# Actual values: Temperatures in Fahrenheit
y = torch.tensor([
    [50],
    [100.0]
], dtype=torch.float32)

# Create a neuron
model = nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# Train for 150,000 iterations
for i in range(0, 150000):
    optimizer.zero_grad()
    outputs = model(X)  # Process both inputs at once
    loss = loss_fn(outputs, y)  # Compute loss for both predictions
    loss.backward()
    optimizer.step()

    # Print progress
    if i % 100 == 0:
        print("Iteration", i)
        print("Bias:", model.bias.item())
        print("Weight:", model.weight.item())

print("----")

# Test the model with a new input
measurements = torch.tensor([[37.5]], dtype=torch.float32)
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # No gradient computation for testing
    prediction = model(measurements)
    print("Prediction for 37.5°C:", prediction.item(), "°F")