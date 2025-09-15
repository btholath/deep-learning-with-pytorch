"""
network1.py & 9) network2.py — Build two-layer net “by hand”

Concept: What Sequential does under the hood.

Purpose: Demystify layer-by-layer forward pass and parameter lists.

Teaching Approach: Unplug the LEGO and wire each block yourself.

Explain: “Compute hidden = Linear → Sigmoid → Linear → Logits.”

Activity: Print hidden_model.weight.shape, etc.

Code Focus: Two separate modules, manual forward pass, shared optimizer.

Engagement: Comment out the activation and see what happens.

Why this order: Great capstone—after convenience tools, see the gears.
"""
import sys
import torch
from torch import nn
import pandas as pd

df = pd.read_csv("./data/student_exam_data.csv")

X = torch.tensor(
    df[["Study Hours", "Previous Exam Score"]].values, 
    dtype=torch.float32
)
y = torch.tensor(df["Pass/Fail"], dtype=torch.float32)\
    .reshape((-1, 1))

hidden_model = nn.Linear(2, 10)
output_model = nn.Linear(10, 1)
loss_fn = torch.nn.BCEWithLogitsLoss()
parameters = list(hidden_model.parameters()) + list(output_model.parameters())
optimizer = torch.optim.SGD(parameters, lr=0.005)

for i in range(0, 500000):
    optimizer.zero_grad()
    outputs = hidden_model(X)
    outputs = nn.functional.sigmoid(outputs)
    outputs = output_model(outputs)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()

    if i % 10000 == 0:
        print(loss)

hidden_model.eval()
output_model.eval()
with torch.no_grad():
    outputs = hidden_model(X)
    outputs = nn.functional.sigmoid(outputs)
    outputs = output_model(outputs)
    y_pred = nn.functional.sigmoid(outputs) > 0.5
    y_pred_correct = y_pred.type(torch.float32) == y
    print(y_pred_correct.type(torch.float32).mean())
