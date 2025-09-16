import torch
from torch import nn
import pandas as pd

# Load dataset
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/05_neural_network/data/student_exam_data.csv")

# Prepare input and target tensors
X = torch.tensor(df[["Study Hours", "Previous Exam Score"]].values, dtype=torch.float32)
y = torch.tensor(df["Pass/Fail"], dtype=torch.float32).reshape(-1, 1)

# Define a two-layer neural network: 2 input features -> 10 hidden neurons -> 1 output
model = nn.Sequential(
    nn.Linear(2, 10),  # Hidden layer: linear transformation (weights: (10, 2), bias: (10,))
    nn.ReLU(),         # ReLU activation for non-linearity
    nn.Linear(10, 1)   # Output layer: linear transformation (weights: (1, 10), bias: (1,))
)

# Define loss function for binary classification
loss_fn = nn.BCEWithLogitsLoss()

# Initialize Adam optimizer (Adaptive Moment Estimation)
# - Uses adaptive learning rates by computing first moment (mean) and second moment (uncentered variance) of gradients
# - Default parameters: lr=0.001, betas=(0.9, 0.999), eps=1e-8
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for i in range(50_000):
    # Zero gradients to prevent accumulation
    optimizer.zero_grad()
    
    # Forward pass: compute logits
    logits = model(X)
    
    # Compute loss
    loss = loss_fn(logits, y)
    
    # Backpropagation: compute gradients
    loss.backward()
    
    # Update parameters using Adam
    # - Adam combines momentum (first moment) and RMSProp (second moment) for faster convergence
    optimizer.step()
    
    # Print loss every 10,000 iterations
    if i % 10_000 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    # Compute predictions: apply sigmoid to logits and threshold at 0.5
    preds = torch.sigmoid(model(X)) > 0.5
    # Compute accuracy
    accuracy = (preds.float() == y).float().mean().item()
    print(f"Accuracy: {accuracy:.4f}")