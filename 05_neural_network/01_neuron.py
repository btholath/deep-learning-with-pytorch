"""
neuron.py — Single neuron (logistic regression)

Concept: One “robot rule” that learns to say Pass/Fail (or Spam/Ham).
Purpose: Show the smallest learnable model and the train → loss → improve loop.
Teaching Approach: Paper example first: 2 features → 1 score → sigmoid → yes/no.
Explain: “Each input (study hours, previous score) gets a weight. Add them up, then squish to 0–1 with sigmoid.”
Activity: Change the learning rate and watch loss go up/down.
Code Focus: nn.Linear(2, 1), BCEWithLogitsLoss, optimizer.step().
Engagement: Predict classmates’ pass/fail from made-up hours.

Why this order: It’s the cleanest intro before hidden layers.
GOAL: Teach one tiny "robot rule" to predict Pass(1)/Fail(0).
REAL LIFE: Same idea powers spam filters (spam=1 / ham=0).
"""

import torch
from torch import nn
import pandas as pd

# 1) Load a tiny table of student data
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/05_neural_network/data/student_exam_data.csv")

# 2) Inputs (X): two facts about each student
X = torch.tensor(
    df[["Study Hours", "Previous Exam Score"]].values,
    dtype=torch.float32
)

# 3) Target (y): did they pass? (1 yes, 0 no)
y = torch.tensor(df["Pass/Fail"], dtype=torch.float32).reshape((-1, 1))

# 4) Model: 2 numbers in → 1 score out (bigger score = more likely pass)
model = nn.Linear(2, 1)

# 5) Loss: compares our score to the truth (0/1) in a safe way for logits
loss_fn = torch.nn.BCEWithLogitsLoss()

# 6) Coach (optimizer): slowly nudges weights to reduce the loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

# 7) Practice loop: guess → check → learn → repeat
for i in range(0, 50_000):
    optimizer.zero_grad()          # clear old notes
    logits = model(X)              # raw scores (can be any real number)
    loss = loss_fn(logits, y)      # how wrong are we?
    loss.backward()                # figure out how to fix weights
    optimizer.step()               # take a small step to improve

    if i % 10_000 == 0:
        print("loss:", loss.item())

# 8) Test time: squish scores to 0..1 and set a pass threshold at 0.5
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X))    # probabilities
    passed = probs > 0.5               # yes/no decision
    accuracy = (passed.float() == y).float().mean().item()
    print("accuracy:", accuracy)
