import torch
import torch.nn as nn

# -------------------------------
# 1) Example setup
# -------------------------------
# True labels (spam=1, ham=0)
y_true = torch.tensor([[1.0], [0.0], [1.0]])

# Raw model outputs (logits) - not probabilities yet
# Higher = more "spammy", Lower = more "hammy"
y_logits = torch.tensor([[2.0], [-1.0], [0.0]])

# -------------------------------
# 2) BCEWithLogitsLoss
# -------------------------------
loss_fn = nn.BCEWithLogitsLoss()

loss = loss_fn(y_logits, y_true)
print("Loss with logits:", loss.item())

# -------------------------------
# 3) Compare: Manual sigmoid + BCELoss
# -------------------------------
sigmoid = torch.sigmoid(y_logits)         # convert logits -> probabilities
print("Sigmoid probabilities:", sigmoid)

bce = nn.BCELoss()                        # plain BCE needs probabilities
loss_manual = bce(sigmoid, y_true)
print("Loss with sigmoid + BCE:", loss_manual.item())


"""
Why use BCEWithLogitsLoss?
For binary classification (yes/no, spam/ham, disease/no disease).
It measures the difference between predicted probability and the true label.

Example:
True = 1 (spam), predicted = 0.9 → good (small loss).
True = 1 (spam), predicted = 0.1 → bad (large loss).


What does “WithLogits” mean?
Models often output raw scores (logits) (can be any number, -∞ … +∞).
To get probabilities (0…1), we apply sigmoid.
BCEWithLogitsLoss combines sigmoid + binary cross entropy in one function:
This is more stable and numerically safer than applying sigmoid separately.

When to use it
Always use BCEWithLogitsLoss when:
    You have a binary classification problem.
    Your model outputs raw scores (logits).

Why not just BCE?
    If you use BCELoss, you must apply torch.sigmoid() manually first.
    If you forget, results are wrong.
    BCEWithLogitsLoss is safer, more efficient, and recommended.
"""