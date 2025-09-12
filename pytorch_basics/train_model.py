"""
DEMO: Training a simple model WITHOUT PyTorch.

We want to learn:
    y_pred = w * x

Example:
    Input (x) = 2
    True output (y) = 10

We start with a random guess for w (like w = 1),
then repeatedly adjust w using gradient descent.
"""

# -------------------------------
# 1) Setup
# -------------------------------
x = 2       # input
y = 10      # true label
w = 1.0     # initial guess for weight

lr = 0.1    # learning rate (step size)

# -------------------------------
# 2) Training loop
# -------------------------------
for step in range(10):  # just 10 steps to keep it short
    # Forward pass: make a prediction
    y_pred = w * x

    # Calculate loss (squared error)
    loss = (y_pred - y) ** 2

    # Backward pass: compute gradient (derivative of loss wrt w)
    # Formula: dLoss/dw = 2 * (y_pred - y) * x
    grad = 2 * (y_pred - y) * x

    # Update weight using gradient
    w = w - lr * grad

    # Print progress
    print(f"Step {step}: w = {w:.4f}, prediction = {y_pred:.4f}, loss = {loss:.4f}, gradient = {grad:.4f}")

# -------------------------------
# 3) Final result
# -------------------------------
print("\nFinal weight:", w)
print("Final prediction:", w * x)
