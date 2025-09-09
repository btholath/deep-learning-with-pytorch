"""
Concept:
Reinforces tensor operations and introduces extracting a single prediction using .item().

Purpose: 
Shows students how to work with specific outputs and reinforces tensor shapes, preparing them for more complex operations.

Teaching Approach:
Explain: 
Remind students that this script does the same math as tensor1.py. Explain .item() as a way to get a single number from a tensor (e.g., y_pred[1].item() gets 100.4).

Activity: 
Run tensor2.py and compare its output to tensor1.py. Uncomment the print(b.shape) and print(X1.shape) lines to show tensor shapes (e.g., b is a scalar, X1 is a vector of size 4).

Code Focus: 
Highlight y_pred[1].item() and discuss why we might want a single prediction.

Engagement: 
Ask students to extract a different prediction (e.g., y_pred[2].item()) and verify it matches their manual calculation.


Why this order? It’s a small step from tensor1.py, adding one new concept (.item()) while reinforcing tensors. It keeps the focus on familiar math.
"""

import torch 

b = torch.tensor(32)
w1 = torch.tensor(1.8)

X1 = torch.tensor([10, 38, 100, 150])

y_pred = 1 * b + X1 * w1

print("b.shape =", b.shape)
print("X1.shape =",X1.shape)
print("b.size() = ",b.size())
print("X1.size() =", X1.size())
print("y_pred[1].item() =", y_pred[1].item())