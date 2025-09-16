# network_batch.py — Training a neural network with mini-batches.
#
# Concept: This script shows how to train a neural network using small chunks of
# data called "mini-batches." This is the standard way to train large models
# on real-world datasets.
#
# Teaching Analogy:
# Instead of doing all your homework at once (which is hard and slow), you
# do a small chunk of problems each day. Each chunk helps you learn a little
# bit, and by the end of the week, you've learned everything.
#
# GOAL: Train in chunks (mini-batches) and use the Adam optimizer to learn faster.
#
# REAL LIFE: When datasets have millions of entries, they won't fit into a
# computer's memory all at once. Training with mini-batches allows us to
# work with manageable chunks of data at a time.

import torch
from torch import nn
import pandas as pd

# Load the student exam data from the CSV file.
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/05_neural_network/data/student_exam_data.csv")

# Convert the data into PyTorch tensors.
X = torch.tensor(df[["Study Hours", "Previous Exam Score"]].values, dtype=torch.float32)
y = torch.tensor(df["Pass/Fail"], dtype=torch.float32).reshape((-1, 1))

# Define the neural network model: 2 inputs -> 10 hidden units -> 1 output.
model = nn.Sequential(
    nn.Linear(2, 10),  # First layer to get 10 "hidden ideas" from our 2 features.
    nn.ReLU(),         # A "keep positives" filter to help the network learn.
    nn.Linear(10, 1)   # Second layer to turn the ideas into a final score.
)

# Define the "wrongness meter" and the "smart teacher" (optimizer).
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # Adam is a smart teacher that learns quickly.

# The total number of data entries we have.
num_entries = X.size(0)
# The size of each homework "chunk" (mini-batch).
batch_size = 32

print(f"Starting training with a batch size of {batch_size}...")

# This is the outer loop, called an "epoch." It means we're going through
# the entire dataset one time.
for epoch in range(0, 100):
    # This is the inner loop for mini-batches. We create chunks of data
    # from the full dataset.
    # The 'start' variable goes from 0 to the end, in steps of 'batch_size'.
    for start in range(0, num_entries, batch_size):
        # We find the end of the current batch.
        end = min(num_entries, start + batch_size)
        
        # We select a small batch of data to work with.
        Xb, yb = X[start:end], y[start:end]

        # Reset the gradients (the "fix-it" notes for our robot brain).
        optimizer.zero_grad()
        
        # Get the network's prediction for this small batch.
        loss = loss_fn(model(Xb), yb)
        
        # Figure out how to fix the brain based on this small batch.
        loss.backward()
        
        # Apply the fixes.
        optimizer.step()
        
    # Every 10 epochs, we print the loss to see how well we're doing.
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss = {loss.item():.6f}")

# After training, we evaluate the final performance on all the data.
print("\nTraining complete. Evaluating the final model...")
model.eval()
with torch.no_grad():
    # Make predictions and get the accuracy.
    preds = (torch.sigmoid(model(X)) > 0.5)
    accuracy = (preds.float() == y).float().mean().item()
    print(f"Final Accuracy: {accuracy:.4f}")