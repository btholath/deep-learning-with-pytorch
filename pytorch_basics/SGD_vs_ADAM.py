"""
Adaptive Moment Estimation (ADAM) Optimizer

ADAM IN PYTORCH: fit y ≈ 2x + 1

We’ll train the SAME tiny model twice:
- once with SGD
- once with Adam
and print the loss every so often.
"""

# ADAM IN PYTORCH: Teaching a robot to find a simple rule.
#
# Imagine you have a robot that doesn't know how to add and multiply.
# You want to teach it a simple rule, like "take a number, multiply it by 2, and add 1."
# But all you can do is give it some examples, like this:
#
# If the number is 0, the answer is about 1.
# If the number is 1, the answer is about 3.
# If the number is -1, the answer is about -1.
#
# The robot will try to guess the rule, and we'll tell it how close it is.
# The `optimizer` is like our teaching style—it helps the robot learn faster.
# We'll try two different teaching styles: SGD and Adam.

import torch
from torch import nn

# Make a tiny dataset: y = 2x + 1 with small noise
# This is our list of examples for the robot to learn from.
torch.manual_seed(0)
X = torch.linspace(-1, 1, 50).reshape(-1, 1) # The "numbers" we give the robot
y = 2*X + 1 + 0.1*torch.randn_like(X)      # The "correct answers" with a little wiggle

# Same model architecture for both runs
def make_model():
    # The brain is super simple: it has one input and gives one output.
    return nn.Sequential(nn.Linear(1, 1))  # 1→1 linear

# The "wrongness meter"
loss_fn = nn.MSELoss()


def train(model, optimizer, tag):
    # This is the "teaching session" for the robot.
    print(f"[{tag}] The robot is trying to learn...")
    for step in range(201):
        optimizer.zero_grad() # Step 1: Tell the robot to forget its last "wrongness" score.
        y_pred = model(X)     # Step 2: The robot makes a guess.
        loss = loss_fn(y_pred, y) # Step 3: We measure how "wrong" its guess was.
        loss.backward()       # Step 4: We figure out which parts of the brain need fixing.
        optimizer.step()      # Step 5: We use our teaching style to fix the brain.
        if step % 50 == 0:
            print(f"[{tag}] step {step:3d} | wrongness score = {loss.item():.6f}")
    # Show learned weight and bias
    # Show the rule the robot learned
    W = model[0].weight.item()
    b = model[0].bias.item()
    print(f"[{tag}] The robot learned: y ≈ {W:.3f}x + {b:.3f}\n")

# Train with the "old" teaching style (SGD)
# SGD is like giving the robot the same size "shove" every time it's wrong.
model_sgd = make_model()
opt_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.1)
train(model_sgd, opt_sgd, "SGD")

# Train with the "new and smarter" teaching style (Adam)
# Adam is like a smart teacher that gives bigger or smaller shoves depending
# on how "wrong" the robot is on different parts of the problem.
model_adam = make_model()
opt_adam = torch.optim.Adam(model_adam.parameters(), lr=0.1)
train(model_adam, opt_adam, "Adam")




# What you'll see:
# Both teaching styles should help the robot learn the rule pretty well.
# But you'll notice that the "Adam" teaching style often gets the robot to the right
# answer much faster. It's like a smarter, more efficient way to learn.
# Both should learn weights near 2 and bias near 1.
# Adam typically reaches low loss faster and is less sensitive to the exact learning rate than plain SGD.


# How is this useful in the real world?
# This simple idea of teaching a robot a rule from examples is used everywhere!
#
# - A phone's face recognition: A model learns the rule "if these pixels look like this, it's a smiling face."
# - A self-driving car: A model learns the rule "if the picture looks like this, the car should turn right."
# - A video game character: A model learns the rule "if the enemy is at this location, I should jump and attack."

# The Adam optimizer helps make all these things learn faster and better. 
# It's a key part of the "magic" that makes smart apps work.

"""
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