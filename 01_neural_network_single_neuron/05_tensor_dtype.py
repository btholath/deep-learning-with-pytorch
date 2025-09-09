"""
Concept: 
Introduces tensor data types (float32 vs. int64), which are important for neural network compatibility.

Purpose: 
Teaches students why data types matter in deep learning (e.g., neural networks need decimals for precision).

Teaching Approach:
Explain: 
Describe float32 (decimals) vs. int64 (whole numbers). Show how X_dtype.type(torch.int64) removes decimals.

Activity: 
Run tensor_dtype.py and print the tensor before and after changing the data type. Uncomment the result = X * 0.5 part to show that int64 tensors lose precision (e.g., 10 * 0.5 = 5 instead of 5.0).

Code Focus: 
Highlight dtype=torch.float32 and X.type(torch.int64).

Engagement: 
Ask students to multiply X by 0.3 with both float32 and int64 types and compare results.

Why this order? 
Data types are a technical detail that makes sense after students understand tensors. It prepares them for neuron_dtype.py, which uses float32.
"""
import torch 
from torch import nn

X = torch.tensor([
    [10], 
    [38], 
    [100], 
    [150]
], dtype=torch.float32)

print("X = ", X)
print("X.dtype =",X.dtype)

X = X.type(torch.int64)
print("X = ", X)
print("X.dtype =",X.dtype)

result = X * 0.5
print("result =",result)
print("result.dtype =", result.dtype)