"""
GOAL (big picture):
Teach a tiny neural network to predict if a student will PASS (1) or FAIL (0)
based on two facts: Study Hours and Previous Exam Score.

REAL LIFE:
The same recipe (inputs → hidden layer → output → train) is used for
spam filters, recommendation systems, and medical risk prediction.
"""

import torch
from torch import nn
import pandas as pd

# 1) LOAD THE DATA -----------------------------------------------------------
# We read a table where each row = one student.
# Columns we care about:
#   - "Study Hours" (how long they studied)
#   - "Previous Exam Score" (last score)
#   - "Pass/Fail" (our target: 1 = pass, 0 = fail)
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/05_neural_network/data/student_exam_data.csv")

# 2) TURN DATA INTO TENSORS (numbers PyTorch understands) -------------------
# X = inputs the model will look at (two numbers per student)
#    Shape is [number_of_students, 2]
X = torch.tensor(
    df[["Study Hours", "Previous Exam Score"]].values,
    dtype=torch.float32
)

# y = the correct answers (labels) the model should learn to predict
#    We reshape to a column vector so each row lines up with each student
y = torch.tensor(df["Pass/Fail"].values, dtype=torch.float32).reshape((-1, 1))

# 3) BUILD A TINY NEURAL NETWORK --------------------------------------------
# nn.Sequential lets us stack layers like LEGO blocks:
#   - nn.Linear(2, 10): mixes the two inputs into 10 hidden "ideas"
#   - nn.ReLU(): squishes negatives to 0 → helps learn faster (nonlinear power)
#   - nn.Linear(10, 1): turns those ideas into ONE final score (called a "logit")
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)
print("Our model:", model)

# 4) CHOOSE HOW TO MEASURE WRONGNESS (LOSS) ----------------------------------
# BCEWithLogitsLoss is perfect for yes/no problems:
# - It expects raw scores (logits) from the model (no sigmoid yet).
# - It compares them to the true answers (0 or 1).
loss_fn = torch.nn.BCEWithLogitsLoss()

# 5) CHOOSE A "COACH" TO UPDATE WEIGHTS (OPTIMIZER) --------------------------
# The optimizer looks at the loss and nudges the model's weights to improve.
# lr (learning rate) = how big each nudge is (small = careful, big = bold).
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

# 6) TRAINING LOOP: GUESS → CHECK → LEARN → REPEAT ---------------------------
# We repeat many times so the model gets better step by step.
for i in range(0, 50_000):
    optimizer.zero_grad()      # clear old gradient notes from last step
    logits = model(X)          # forward pass: raw scores (can be any real number)
    loss = loss_fn(logits, y)  # how wrong are we right now?
    loss.backward()            # compute how each weight should change (backprop)
    optimizer.step()           # take a small step to improve the weights

    # Print progress every so often so we can SEE learning happening.
    if i % 10_000 == 0:
        print("loss:", loss.item())

# 7) EVALUATE (NO LEARNING, JUST CHECKING) -----------------------------------
# model.eval() = good habit; turns off training-only layers (like dropout).
model.eval()
with torch.no_grad():  # we’re not learning now; this makes it faster/safer
    # Turn final raw scores (logits) into probabilities with sigmoid (0..1).
    probs = torch.sigmoid(model(X))

    # Decide pass (True) if probability > 0.5; otherwise fail (False).
    preds = probs > 0.5

    # Compare predictions to the real answers and compute accuracy.
    accuracy = (preds.float() == y).float().mean().item()
    print("accuracy:", accuracy)


"""
What each part teaches (talking points)
    Inputs → Hidden → Output: hidden neurons learn useful combos like “studied a lot AND had a good prior score.”
    ReLU: speeds up learning by passing positive signals and zeroing negatives.
    Logits & Sigmoid: model works in raw scores (logits) for math stability; at the end we use sigmoid to get a probability (0..1).
    Loss & Backprop: loss = “how wrong”; backprop = “how to fix weights.”
    Optimizer: the “coach” that nudges weights a tiny bit each step.
    Accuracy: a simple score students understand immediately.

Easy experiments for students
    Change the learning rate (e.g., 0.001, 0.02) and see if loss goes down faster or slower.
    Change the hidden size (10 → 3 or 20) and see how accuracy changes.
    Try a different threshold (0.4 or 0.6) to see effects on predictions.
    Shuffle/split into train/validation to be fair (train on some rows, test on unseen rows).
"""
