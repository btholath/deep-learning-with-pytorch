"""
Concept: Combines nn.Linear with explicit data type specification, reinforcing both neural networks and data types.
Purpose: Shows best practices for real-world deep learning (using float32 for neural networks).
Teaching Approach:

Explain: Emphasize that neuron_dtype.py is like neuron.py but explicitly sets dtype=torch.float32 for compatibility with neural networks.
Activity: Run neuron_dtype.py and compare it to neuron.py. Discuss why float32` is standard in deep learning.
Code Focus: Highlight dtype=torch.float32 in the tensor and parameter definitions.
Engagement: Ask students to experiment with different data types (e.g., torch.int64) and see if the model runs (it may fail, reinforcing the importance of float32).

Why this order? It’s the most advanced script, combining neural networks and data types. It wraps up the lesson by showing a complete, realistic setup.
"""
import torch 
from torch import nn

X = torch.tensor([
    [10], 
    [38], 
    [100], 
    [150]
], dtype=torch.float32)

model = nn.Linear(1, 1)

model.bias = nn.Parameter(
    torch.tensor([32], dtype=torch.float32)
)
model.weight = nn.Parameter(
    torch.tensor([[1.8]], dtype=torch.float32)
)

print(model.bias)
print(model.weight)

y_pred = model(X)
print(y_pred)