"""
Both nn.functional.sigmoid() and nn.Sigmoid() do the same math (the logistic squish → turns any number into 0–1).
The difference is how they’re used inside PyTorch code.
nn.functional.sigmoid(x) → function call.
    Good for quick one-off use.
    Doesn’t store any learnable parameters (because sigmoid has none).
    Doesn’t “remember” itself as part of a module (like a LEGO block).

nn.Sigmoid() → layer object (module).
    Good for putting inside a model (e.g., in nn.Sequential).
    Plays nicely when saving/loading models (state_dict).
    Slightly more typing, but clearer in structured models.
"""

import torch
from torch import nn

# Example input: raw scores (logits) from a model
x = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0])

# ----------------------------
# 1) Using nn.functional.sigmoid (function style)
# ----------------------------
out_func = nn.functional.sigmoid(x)
print("Functional sigmoid output:", out_func)

# ----------------------------
# 2) Using nn.Sigmoid (module style)
# ----------------------------
sigmoid_layer = nn.Sigmoid()     # create a "layer" version of sigmoid
out_layer = sigmoid_layer(x)
print("Module sigmoid output   :", out_layer)

# ----------------------------
# 3) Showing they’re the same
# ----------------------------
print("Are they equal? ->", torch.allclose(out_func, out_layer))

# ----------------------------
# 4) Example in a model
# ----------------------------
# Functional version inside forward
class ModelFunctional(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
    def forward(self, x):
        return nn.functional.sigmoid(self.fc(x))

# Module version in Sequential
model_module = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

print("Functional model:", ModelFunctional())
print("Module model    :", model_module)

