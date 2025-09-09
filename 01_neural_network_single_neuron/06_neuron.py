"""
Concept: Introduces PyTorch’s nn.Linear, a real neural network component that automates y = w * x + b.
Purpose: Shows how deep learning frameworks simplify the math students saw earlier, bridging to real neural networks.

Teaching Approach:
Explain: Describe nn.Linear(1, 1) as a “neuron” that does the same math as before but is ready to learn w and b from data (though here we set them manually).
Activity: Run neuron.py and compare its output ([[50.0], [100.4], [212.0], [302.0]]) to tensor1.py. Show that model.weight and model.bias match w1 and b.
Code Focus: Highlight nn.Linear, nn.Parameter, and the 2D input tensor X.
Engagement: Ask students to change model.weight to [[2.0]] and predict the new outputs.

Why this order? It builds on 2D tensors from tensor_matrix.py and introduces neural networks after students are comfortable with tensors.
"""
import torch 
from torch import nn

X = torch.tensor([
    [10.0], 
    [38.0], 
    [100.0], 
    [150.0]
])

model = nn.Linear(1, 1)

model.bias = nn.Parameter(
    torch.tensor([32.0])
)
model.weight = nn.Parameter(
    torch.tensor([[1.8]])
)

print("model.bias =",model.bias)
print("model.weight =",model.weight)

y_pred = model(X)
print("y_pred =",y_pred)