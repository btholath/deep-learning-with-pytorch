"""
what loss.backward() does? 
It uses just 1 input, 1 weight, and 1 target so students can see the math in action.
"""

"""
DEMO: What does loss.backward() do?

We build the simplest model:
    y_pred = w * x

We compare prediction to the true label (y),
compute the loss, and then call loss.backward().

PyTorch will calculate the gradient (slope) of the loss
with respect to w, which tells us how to adjust w
to make the prediction better.
"""

import torch

# -------------------------------
# 1) Setup
# -------------------------------
# Input (x) and true output (y)
x = torch.tensor([[2.0]])   # input value
y = torch.tensor([[10.0]])  # true label

# Weight (w), start with a guess (requires_grad=True so PyTorch tracks it)
w = torch.tensor([[1.0]], requires_grad=True)

# -------------------------------
# 2) Forward pass
# -------------------------------
y_pred = x * w  # simple model: y_pred = w * x

# Loss = (prediction - truth)^2
loss = (y_pred - y) ** 2
print("Prediction:", y_pred.item())
print("Loss:", loss.item())

# -------------------------------
# 3) Backward pass
# -------------------------------
loss.backward()   # compute gradient

# -------------------------------
# 4) Show gradient
# -------------------------------
print("Gradient (dLoss/dw):", w.grad.item())

# -------------------------------
# 5) Manually update weight (1 step)
# -------------------------------
lr = 0.1  # learning rate
with torch.no_grad():       # turn off gradient tracking for update
    w -= lr * w.grad        # update rule
    w.grad.zero_()          # reset gradient to 0 (important!)

# -------------------------------
# 6) Check new prediction
# -------------------------------
new_pred = x * w
new_loss = (new_pred - y) ** 2
print("Updated weight:", w.item())
print("New prediction:", new_pred.item())
print("New loss:", new_loss.item())
