# Re-run after reset: Need to re-import everything and re-create chart

import matplotlib.pyplot as plt
import torch
from torch import nn

# Input: Temperature in Celsius (one example)
X1 = torch.tensor([[10.0]])  # 10°C
y1 = torch.tensor([[50.0]])  # 50°F

# Create a simple neuron
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # bigger lr for visible change

# Store initial values
initial_w = model.weight.item()
initial_b = model.bias.item()
initial_pred = model(X1).item()
initial_loss = loss_fn(model(X1), y1).item()

# Training step
optimizer.zero_grad()
outputs = model(X1)
loss = loss_fn(outputs, y1)
loss.backward()
optimizer.step()

# Store updated values
updated_w = model.weight.item()
updated_b = model.bias.item()
updated_pred = model(X1).item()
updated_loss = loss_fn(model(X1), y1).item()

# --- Visualization ---
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Before training
ax[0].axis("off")
ax[0].set_title("Before Training Step")
ax[0].text(0.5, 0.8, f"w1 = {initial_w:.3f}\nb = {initial_b:.3f}", ha="center", fontsize=12,
           bbox=dict(boxstyle="round,pad=0.4", fc="lightcoral", ec="black"))
ax[0].text(0.5, 0.4, f"Prediction for 10°C = {initial_pred:.2f}°F", ha="center", fontsize=11)
ax[0].text(0.5, 0.2, f"Loss (error) = {initial_loss:.2f}", ha="center", fontsize=11)

# Panel 2: After training
ax[1].axis("off")
ax[1].set_title("After One Training Step")
ax[1].text(0.5, 0.8, f"w1 = {updated_w:.3f}\nb = {updated_b:.3f}", ha="center", fontsize=12,
           bbox=dict(boxstyle="round,pad=0.4", fc="lightgreen", ec="black"))
ax[1].text(0.5, 0.4, f"Prediction for 10°C = {updated_pred:.2f}°F", ha="center", fontsize=11)
ax[1].text(0.5, 0.2, f"Loss (error) = {updated_loss:.2f}", ha="center", fontsize=11)

plt.suptitle("Neuron Learning Fahrenheit from Celsius (10°C → 50°F)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])

chart_path = "/workspaces/deep-learning-with-pytorch/02_neuron_training/output/neuron_training_step.png"
plt.savefig(chart_path, dpi=200)
plt.show()

chart_path
