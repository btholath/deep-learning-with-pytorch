"""
torch.optim.SGD is like a teacher correcting the neuron, and lr is how big the correction steps are.

What does torch.optim.SGD mean?
    SGD stands for Stochastic Gradient Descent.
    In plain English: It’s a teacher that helps the neuron fix its mistakes step by step.

Every time the neuron makes a guess:
    We measure how wrong it was (loss).
    SGD tells the neuron how to adjust its numbers (weights and bias).
    The neuron takes a little correction step.

What does lr (learning rate) mean?
    Learning rate (lr) is like the step size of corrections.
    Small lr = the neuron learns very slowly (tiny baby steps 🐢).
    Big lr = the neuron jumps too far and may miss the right answer (wild leaps 🐇).
We want a balance: not too small, not too big.


Imagine you’re throwing darts 🎯, trying to hit the bullseye:
    You throw a dart, but it lands too far left.
    A coach tells you:
        “Move a little to the right next time.”
        That’s SGD (the teacher correcting you).

Now:
    If your lr (learning rate) = 0.0001 → you move just a tiny bit to the right each time (slow learner).
    If your lr = 1.0 → you jump too far and overshoot the bullseye (wild learner).
    If your lr is just right → you slowly adjust and hit the bullseye after a few tries.

What was the observed effect when the "learning rate" was set too low during training?    
The model stopped learning, as the adjustments were tooooo small to improve results.
A low learning rate causes the model to make very miminal adjustments to the weights, resulting in slow
or stalled learning progress. This means the model struggles to reduce the loss effectively, as each
adjustment is too minor o bring about significant change.


What happened when the "learning rate" was set to too high during training?
Values became extremely large, resulting in infinities and NaNs, which dirupted the training process.
A high learning rate causes the model to make excessively large adjustments to weights, leading to values that rapidly
grow out of bounds. This numerical instability often proceduces infinites (inf) and "Not a Number" (NaN) values, which disrupt
the learning process and prevent the model from coverging.
"""
import torch
from torch import nn

# A simple model: y = weight * x + bias
model = nn.Linear(1, 1)

# Loss function (how wrong it is)
loss_fn = nn.MSELoss()

# Teacher (SGD) with a learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Input and correct output
x = torch.tensor([[1.0]])
y_true = torch.tensor([[2.0]])  # we want y = 2 when x = 1

for step in range(5):
    optimizer.zero_grad()
    y_pred = model(x)                # guess
    loss = loss_fn(y_pred, y_true)   # check error
    loss.backward()                  # find corrections
    optimizer.step()                 # make correction
    
    print(f"Step {step}: prediction={y_pred.item():.3f}, loss={loss.item():.3f}")

"""
Notice how the prediction gets closer to 2 and the loss gets smaller step by step.

"""