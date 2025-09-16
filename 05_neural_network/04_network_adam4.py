"""
ADAM FROM SCRATCH: minimize f(w) = (w - 3)^2

Big idea:
- We keep moving w to make f(w) smaller.
- Adam remembers:
  m_t = moving average of gradient (like momentum)
  v_t = moving average of gradient^2 (how "noisy" the slope is)
- Then it corrects bias and adapts the step size for each parameter.

You will see w race toward 3.
"""

import math

# The function and its derivative
def f(w):
    return (w - 3)**2

def grad(w):
    return 2*(w - 3)

# Initialize
w = 0.0            # bad starting guess; the optimum is 3
alpha = 0.1        # base learning rate
beta1 = 0.9        # m_t decay rate (first moment: mean)
beta2 = 0.999      # v_t decay rate (second moment: variance)
eps = 1e-8         # small number to avoid divide-by-zero

m = 0.0            # first moment (mean of gradients)
v = 0.0            # second moment (mean of squared gradients)

print("step | w        | f(w)")
for t in range(1, 51):  # 50 steps
    g = grad(w)                     # gradient at current w
    m = beta1*m + (1-beta1)*g       # update first moment estimate
    v = beta2*v + (1-beta2)*(g*g)   # update second moment estimate

    # Bias-corrected moments (important at early steps)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    # Adam parameter update
    w = w - alpha * m_hat / (math.sqrt(v_hat) + eps)

    if t % 5 == 0 or t == 1:
        print(f"{t:>4} | {w:8.5f} | {f(w):8.5f}")

"""
The best 𝑤 is 3. Adam should steer 𝑤 toward 3 quickly and smoothly.

What to point out
    m (momentum) smooths the direction (less jitter).
    v (variance) scales the step so we don’t jump too far on steep or noisy slopes.
    Bias correction makes early estimates fair (since m and v start at 0).
    Adam adapts the step size automatically—great when features/gradients have different scales.
    
"""