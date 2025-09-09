"""
Concept:
Introduces PyTorch and tensors, showing how they simplify the same calculation from manual.py for multiple inputs at once.

Purpose: 
Transitions students from manual math to PyTorch’s way of handling data, preparing them for deep learning tools.

Teaching Approach:
Explain: 
Introduce tensors as “fancy lists” that let computers do math faster. Show that torch.tensor([10, 38, 100, 150]) is like a Python list but optimized for machine learning.

Activity: 
Run tensor1.py and compare its output ([50.0, 100.4, 212.0, 302.0]) to manual.py. Emphasize that y_pred = b + w1 * X1 now works on all inputs simultaneously.

Code Focus: 
Highlight torch.tensor for b, w1, and X1, and the equation y_pred = b + w1 * X1.

Engagement: 
Ask students to add a new input to X1 (e.g., 200) and predict the output before running the code.


Why this order? 
It builds on manual.py by introducing PyTorch and tensors but keeps the math familiar. Students see the power of vectorized operations (doing math on all inputs at once).
"""

import torch 

b = torch.tensor(32)
w1 = torch.tensor(1.8)

X1 = torch.tensor([10, 38, 100, 150])

y_pred = 1 * b + X1 * w1
print(y_pred)
