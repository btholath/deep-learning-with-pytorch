"""
Concept: 
Introduces the linear equation y = w * x + b using plain Python, which feels like a math problem students already know (e.g., y = mx + b from algebra).

Purpose: 
Builds intuition for how a neural network’s math works without overwhelming students with new libraries or terminology.

Teaching Approach:
Explain: 
Start with the line equation. Show how w1 = 1.8 (slope) and b = 32 (y-intercept) work with an input like x = 100 to get y = 32 + 1.8 * 100 = 212.0.

Activity: 
Have students calculate y for a few inputs manually (e.g., on paper or a calculator) before running the script. Then run manual.py to show Python does the same thing.

Code Focus: 
Highlight the simple variables (b, w1, X1) and the equation y_pred = 1 * b + X1 * w1.

Engagement: 
Ask students to change w1 or b and predict how y_pred will change (e.g., “What if the slope is steeper?”).

Why this order? It’s the simplest script, using only basic Python. Students see the math behind neural networks without any library complexity.
"""


b = 32
w1 = 1.8

X1 = 100

y_pred = 1 * b + X1 * w1
print("y_pred =",y_pred)
