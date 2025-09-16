"""
ADAM IN PYTORCH: fit y ≈ 2x + 1

We’ll train the SAME tiny model twice:
- once with SGD
- once with Adam
and print the loss every so often.
"""

import torch
from torch import nn

# Make a tiny dataset: y = 2x + 1 with small noise
torch.manual_seed(0)
X = torch.linspace(-1, 1, 50).reshape(-1, 1)
y = 2*X + 1 + 0.1*torch.randn_like(X)

# Same model architecture for both runs
def make_model():
    return nn.Sequential(nn.Linear(1, 1))  # 1→1 linear

loss_fn = nn.MSELoss()

def train(model, optimizer, tag):
    for step in range(201):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"[{tag}] step {step:3d} | loss = {loss.item():.6f}")
    # Show learned weight and bias
    W = model[0].weight.item()
    b = model[0].bias.item()
    print(f"[{tag}] learned: y ≈ {W:.3f}x + {b:.3f}\n")

# Train with plain SGD
model_sgd = make_model()
opt_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.1)
train(model_sgd, opt_sgd, "SGD")

# Train with Adam
model_adam = make_model()
opt_adam = torch.optim.Adam(model_adam.parameters(), lr=0.1)  # Adam often tolerates higher lr
train(model_adam, opt_adam, "Adam")


"""
What you’ll see
    Both should learn weights near 2 and bias near 1.
    Adam typically reaches low loss faster and is less sensitive to the exact learning rate than plain SGD.


When to use Adam (pros & cons)
    Merits
    - Adaptive steps per parameter → faster, often easier to tune.
    - Handles noisy/imbalanced gradients well.
    - Great default for many deep learning tasks.

    Demerits
    - Can overfit or generalize slightly worse than well-tuned SGD on some vision tasks.
    - A tiny bit more memory (stores m and v for each parameter).
    - Sometimes benefits from lowering the learning rate mid-training (e.g., 0.001 → 0.0003).

Practical default: torch.optim.Adam(model.parameters(), lr=1e-3)
Then adjust if loss plateaus (try smaller lr) or oscillates (smaller lr / weight decay).    
"""