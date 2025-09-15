"""
network_sequential.py — Build a small network with nn.Sequential
Concept: Stack layers into a pipeline.
Purpose: Show how to “snap together” layers cleanly.
Teaching Approach: LEGO analogy: input → hidden → output.
Explain: “Hidden neurons let the model learn ‘combos’ of features.”
Activity: Print the model, count parameters, change hidden size.
Code Focus: nn.Sequential(nn.Linear(2,10), nn.ReLU(), nn.Linear(10,1)).
Engagement: Ask students to guess what a hidden neuron might learn (“if hours high and prior score high”).

Why this order: From single rule → stack of rules. 

GOAL: Stack layers so the model can learn combos of inputs.
REAL LIFE: Hidden layers let models catch patterns like
"high hours AND improving scores → likely pass".
"""

import torch
from torch import nn
import pandas as pd

df = pd.read_csv("/workspaces/deep-learning-with-pytorch/05_neural_network/data/student_exam_data.csv")


X = torch.tensor(df[["Study Hours","Previous Exam Score"]].values, dtype=torch.float32)
y = torch.tensor(df["Pass/Fail"], dtype=torch.float32).reshape((-1,1))

# A small network: 2 → 10 → 1
model = nn.Sequential(
    nn.Linear(2, 10),  # first mix features into 10 hidden "ideas"
    nn.ReLU(),         # keep positives, drop negatives (speeds learning)
    nn.Linear(10, 1)   # turn hidden ideas into a final score
)
print(model)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

for i in range(0, 50_000):
    optimizer.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()
    if i % 10_000 == 0:
        print("loss:", loss.item())

model.eval()
with torch.no_grad():
    preds = (torch.sigmoid(model(X)) > 0.5)
    print("accuracy:", (preds.float()==y).float().mean().item())
