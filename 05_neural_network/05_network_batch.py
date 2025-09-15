"""
network_batch.py — Mini-batches

Concept: Train on small chunks (batches) instead of all data at once.

Purpose: Faster, more stable training; uses less memory.

Teaching Approach: “Do homework in chunks each day, not all at once.”

Explain: “Each batch gives a nudge; many nudges per epoch.”

Activity: Try batch sizes 8, 32, 128; watch loss smoothness.

Code Focus: the for start in range(0, num_entries, batch_size): loop.

Engagement: “Find the batch size that makes loss fall fastest for our data.”

Why this order: After optimizers, batching is the next practical lever.

GOAL: Train in chunks (mini-batches) and use Adam to learn faster.

REAL LIFE: Big datasets don't fit in memory; batching is the norm.
"""

import torch
from torch import nn
import pandas as pd

df = pd.read_csv("./data/student_exam_data.csv")
X = torch.tensor(df[["Study Hours","Previous Exam Score"]].values, dtype=torch.float32)
y = torch.tensor(df["Pass/Fail"], dtype=torch.float32).reshape((-1,1))

model = nn.Sequential(nn.Linear(2,10), nn.ReLU(), nn.Linear(10,1))
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # faster steps

num_entries = X.size(0)
batch_size = 32

for epoch in range(0, 100):
    for start in range(0, num_entries, batch_size):
        end = min(num_entries, start + batch_size)
        Xb, yb = X[start:end], y[start:end]

        optimizer.zero_grad()
        loss = loss_fn(model(Xb), yb)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print("epoch", epoch, "loss", loss.item())

model.eval()
with torch.no_grad():
    preds = (torch.sigmoid(model(X)) > 0.5)
    print("accuracy:", (preds.float()==y).float().mean().item())
