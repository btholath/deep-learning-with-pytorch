"""
- Imagine the model gives a raw score:
    Big negative numbers → very sure it’s ham (close to 0).
    Big positive numbers → very sure it’s spam (close to 1).
- Sigmoid is like a squishy S-shaped slide that pushes all values into the safe range 0 to 1.
- This way, the model can say: “I’m 80% sure this is spam.”

- Show how the sigmoid function works.
- Turn any number (from –∞ to +∞) into a probability between 0 and 1.
- Teach why models use sigmoid to decide: “Is this spam or not spam?”


GOAL:
Understand the sigmoid function, which is a math "squish" that
turns any number into something between 0 and 1.

WHY IT MATTERS:
When our model guesses spam/ham, it outputs raw scores (logits).
We use sigmoid to turn those scores into probabilities, like:
   - 0.95 = 95% sure it's spam
   - 0.05 = 5% sure it's ham

REAL-WORLD USEFULNESS:
- Spam filters use it to decide spam vs ham.
- Logistic regression uses it for binary decisions.
- Even in medicine, sigmoid outputs probabilities like
  "there’s a 70% chance this X-ray shows pneumonia".
"""

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1) Define the sigmoid function
# -------------------------------
def sigmoid(x):
    # Formula: 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

# -------------------------------
# 2) Test the sigmoid function
# -------------------------------
# Numbers going from -10 to 10
x = np.linspace(-10, 10, 200)

# Apply sigmoid to every number in x
y = sigmoid(x)

# -------------------------------
# 3) Plot sigmoid curve
# -------------------------------
plt.plot(x, y, color="blue")
plt.title("Sigmoid Function")
plt.xlabel("Input (raw score from model)")
plt.ylabel("Output (probability)")
plt.grid(True)
plt.show()

# -------------------------------
# 4) Demo: how sigmoid squishes numbers
# -------------------------------
test_values = [-5, -2, 0, 2, 5]
for val in test_values:
    print(f"sigmoid({val}) = {sigmoid(val):.3f}")
