"""
This script introduces a single training step for a neuron using one data point.

Data: 
We use one example: 10°C should convert to 50°F.

Model: 
nn.Linear(1, 1) creates a neuron that does $ output = w_1 . input + b $.

Loss Function: 
MSELoss measures how far the prediction is from the actual value (squared difference).

Optimizer: 
SGD (Stochastic Gradient Descent) adjusts the weight and bias to reduce the error.

Training Step:
    Compute the prediction for 10°C.
    Calculate the error (loss).
    Use backward() to figure out how to tweak the weight and bias.
    Update them with step().

Output: 
Shows the bias before and after one update and the prediction for 10°C.

Why It’s Useful: 
Introduces the core idea of training a neuron with one step, showing how weights and biases change slightly.    
"""

import torch
from torch import nn

# Input: Temperature in Celsius (one example)
X1 = torch.tensor([[10]], dtype=torch.float32)  # 10°C
# Actual value: Temperature in Fahrenheit
y1 = torch.tensor([[50]], dtype=torch.float32)  # 50°F (10 * 1.8 + 32 = 50)

# Create a simple neuron (linear model: output = weight * input + bias)
model = nn.Linear(1, 1)  # 1 input, 1 output
loss_fn = torch.nn.MSELoss()  # Mean Squared Error loss to measure error
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # Optimizer adjusts weights

# Print initial bias (randomly set by PyTorch)
print("Initial bias:", model.bias.item())

# Training step
optimizer.zero_grad()  # Clear previous gradients
outputs = model(X1)    # Compute prediction: weight * X1 + bias
loss = loss_fn(outputs, y1)  # Calculate error (difference between prediction and actual)
loss.backward()        # Compute how to adjust weight and bias
optimizer.step()       # Update weight and bias

# Print updated bias after one step
print("Updated bias:", model.bias.item())

# Make a prediction for 10°C
y1_pred = model(X1)
print("Prediction for 10°C:", y1_pred.item(), "°F")


"""
---

# 🧑‍🏫 Explaining the Neuron Training Step

Imagine we want to teach a robot how to convert Celsius to Fahrenheit.
The robot doesn’t know the formula yet, so we show it an example:

👉 **10°C should equal 50°F.**

---

### 1. **Data (The Homework Problem)**

```python
X1 = torch.tensor([[10]], dtype=torch.float32)  # 10°C
y1 = torch.tensor([[50]], dtype=torch.float32)  # Correct answer: 50°F
```

This is like giving the robot a homework problem:
💡 *"When the input is 10, the correct output is 50."*

---

### 2. **Model (The Robot’s Guessing Formula)**

```python
model = nn.Linear(1, 1)
```

The robot starts with a simple formula:

$$
\text{output} = w_1 \times \text{input} + b
$$

* **$w_1$** is like the “multiplier” the robot chooses.
* **$b$** is like the “bonus” it adds on.

At first, these are **random numbers** (the robot is guessing).

---

### 3. **Loss Function (How Wrong Was the Robot?)**

```python
loss_fn = torch.nn.MSELoss()
```

This measures how far the robot’s guess is from the correct answer.
If the robot predicts 60°F instead of 50°F, the error is:

$$
(60 - 50)^2 = 100
$$

The bigger the error, the more wrong the robot is.

---

### 4. **Optimizer (The Teacher’s Correction Tool)**

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
```

Think of this as the teacher’s red pen ✏️.
It tells the robot:

* “Change your multiplier a little.”
* “Change your bonus a little.”

The learning rate (**0.001**) means the teacher corrects the robot *just a tiny bit at a time*.

---

### 5. **Training Step (The Robot Learns)**

```python
optimizer.zero_grad()
outputs = model(X1)
loss = loss_fn(outputs, y1)
loss.backward()
optimizer.step()
```

Here’s what happens:

1. **Guess:** The robot predicts Fahrenheit for 10°C using its random formula.
2. **Check:** We see how far it was from the correct 50°F.
3. **Feedback:** The robot figures out which direction to adjust its multiplier (w1) and bonus (b).
4. **Update:** The robot fixes its formula slightly.

---

### 6. **Output (Before and After Learning)**

```python
print("Initial bias:", model.bias.item())
print("Updated bias:", model.bias.item())
print("Prediction for 10°C:", y1_pred.item(), "°F")
```

* At first, the robot’s **bias** (bonus) is random.
* After one correction step, the bias changes a little.
* The robot’s prediction for 10°C gets **closer to 50°F**.

---

# 🎯 Why This Is Cool

This tiny example shows the *heart of machine learning*:

* Start with random guesses.
* Compare guesses to correct answers.
* Make small corrections.
* Over time, the robot learns the correct rule:

$$
F = 1.8 \times C + 32
$$

---

✅ **Analogy:**
It’s like practicing math problems: the first time you guess, you might be wrong. But after checking with the teacher and correcting yourself step by step, you get closer and closer to the right formula.

---


"""