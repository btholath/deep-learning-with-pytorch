"""
Concept: 
Introduces 2D tensors (matrices) and indexing, which are critical for neural networks that process data in batches.

Purpose: 
Prepares students for nn.Linear in later scripts, which expects 2D input tensors.

Teaching Approach:
Explain: 
Show that X = torch.tensor([[10], [38], [100], [150]]) is a 2D tensor (4 rows, 1 column). Explain X[:, 0] as “get all rows from the first column,” yielding [10, 38, 100, 150].

Activity: 
Run tensor_matrix.py and print X.size() to show (4, 1). Have students predict what X[:, 0] will output before running it.

Code Focus: 
Highlight the 2D tensor syntax and indexing (X[:, 0]).

Engagement: 
Ask students to create a 2D tensor with two columns (e.g., [[10, 1], [38, 2], [100, 3], [150, 4]]) and extract the second column.

Why this order? 
It introduces 2D tensors, which are needed for nn.Linear in the next scripts. It’s a natural progression from 1D tensors in tensor1.py and tensor2.py.
"""

import torch 

X = torch.tensor([
    [10], 
    [38], 
    [100], 
    [150]
])

print("X.size(1) =", X.size(1))
print("X[:, 0] =", X[:, 0])
