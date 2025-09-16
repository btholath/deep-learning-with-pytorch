# %%
# adam_optimizer.py - Illustrating the Adaptive Moment Estimation (Adam) optimizer.
#
# Concept: This script demonstrates how the Adam optimizer works in a practical
# deep learning training scenario. Adam is an adaptive learning rate optimization
# algorithm that combines the benefits of two other extensions of stochastic gradient
# descent: AdaGrad and RMSProp.
#
# Purpose: To provide a clear, runnable example of how to use Adam in PyTorch
# and to show its effectiveness in training a simple neural network.
#
# Code Focus:
# 1. Defining a simple neural network.
# 2. Loading and preparing a sample dataset.
# 3. Setting up the Adam optimizer and comparing it to SGD.
# 4. Running a training loop to observe the loss reduction with Adam.

import sys
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt

# Check if CUDA (GPU support) is available and set the device.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device for training.")

# Load the student exam data from the CSV file.
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/05_neural_network/data/student_exam_data.csv")

# Prepare the data tensors for the network and move them to the selected device.
X = torch.tensor(
    df[["Study Hours", "Previous Exam Score"]].values, 
    dtype=torch.float32
).to(device)
y = torch.tensor(df["Pass/Fail"], dtype=torch.float32)\
    .reshape((-1, 1)).to(device)

# %%
# Define a simple neural network model.
# This is a two-layer network with a ReLU activation function.
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
).to(device)

# %%
# Illustrating the Adam Optimizer.
# Adam is a popular choice for many deep learning tasks due to its efficiency and
# low memory requirements. It uses estimates of the first and second moments of
# the gradients to adapt the learning rate for each parameter individually.

# Define the loss function.
loss_fn = torch.nn.BCEWithLogitsLoss()

# Define the optimizer.
# Here, we use Adam. You can easily switch this to a different optimizer,
# e.g., `torch.optim.SGD(model.parameters(), lr=0.01)` to see the difference.
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# A list to store loss values for plotting later.
loss_history = []

# %%
# Training Loop with Adam
print("Training the network with Adam...")
for i in range(0, 50000):
    # Zero the gradients.
    optimizer.zero_grad()
    
    # Forward pass.
    logits = model(X)
    
    # Calculate the loss.
    loss = loss_fn(logits, y)
    
    # Store the loss value.
    loss_history.append(loss.item())
    
    # Backpropagation.
    loss.backward()
    
    # Update the parameters using Adam.
    optimizer.step()
    
    # Print the loss periodically.
    if i % 5000 == 0:
        print(f"Iteration {i}: Loss = {loss.item():.4f}")

# %%
# Evaluation
# Switch the model to evaluation mode and disable gradient calculations.
model.eval()
with torch.no_grad():
    preds = (torch.sigmoid(model(X)) > 0.5)
    accuracy = (preds.float() == y).float().mean().item()
    print(f"\nFinal Accuracy: {accuracy:.4f}")

# %%
# Plotting the loss history to visualize the training process.
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title("Training Loss with Adam Optimizer")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# %%
