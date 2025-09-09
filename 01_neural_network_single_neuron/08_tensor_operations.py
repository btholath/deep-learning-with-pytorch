"""
The code snippets demonstrate a linear equation of the form y = w * x + b, where w is the weight (slope), b is the bias (intercept), and x is the input.

Imagine we’re building a simple machine that predicts an output number (y) based on an input number (x) using a line equation: y = w * x + b. 
This is the core of a basic neural network “neuron.”

PyTorch Tensors


"""
# Import PyTorch library for deep learning and tensor operations
import torch
from torch import nn  # Neural network module

# --- Manual Calculation (without PyTorch's neural network tools) ---
# Define bias (b) and weight (w1) as simple numbers
b = 32  # Bias (like the y-intercept in a line equation)
w1 = 1.8  # Weight (like the slope in a line equation)

# Input data (X1) as a list of numbers
X1 = [10, 38, 100, 150]

# Calculate predictions manually: y = b + w1 * x for each input
print("Manual Calculation Results:")
for x in X1:
    y_pred = b + w1 * x  # Linear equation: y = b + w1 * x
    print(f"Input x = {x}, Predicted y = {y_pred}")

# --- PyTorch Tensor Calculation (without neural network module) ---
# Convert bias, weight, and inputs to PyTorch tensors
b_tensor = torch.tensor(32.0)  # Bias as a tensor (float)
w1_tensor = torch.tensor(1.8)  # Weight as a tensor (float)
X1_tensor = torch.tensor([10.0, 38.0, 100.0, 150.0])  # Inputs as a tensor

# Calculate predictions using tensor operations: y = b + w1 * x
y_pred_tensor = b_tensor + w1_tensor * X1_tensor
print("\nPyTorch Tensor Results:")
print(y_pred_tensor)  # Prints predictions for all inputs
print(f"Second prediction (index 1): {y_pred_tensor[1].item()}")  # Extract single value

# --- PyTorch Neural Network (using nn.Linear) ---
# Define input data as a 2D tensor (4 rows, 1 column) for nn.Linear
X = torch.tensor([[10.0], [38.0], [100.0], [150.0]], dtype=torch.float32)

# Create a linear model (y = w * x + b) with 1 input and 1 output
model = nn.Linear(1, 1)

# Set specific weight and bias values
model.weight = nn.Parameter(torch.tensor([[1.8]], dtype=torch.float32))
model.bias = nn.Parameter(torch.tensor([32.0], dtype=torch.float32))

# Print model parameters
print("\nPyTorch Neural Network Parameters:")
print(f"Weight: {model.weight}")
print(f"Bias: {model.bias}")

# Make predictions using the model
y_pred_model = model(X)
print("PyTorch Neural Network Results:")
print(y_pred_model)

# --- Exploring Tensor Data Types ---
# Define input tensor with explicit float32 data type
X_dtype = torch.tensor([[10], [38], [100], [150]], dtype=torch.float32)
print("\nTensor with float32 Data Type:")
print(X_dtype)
print(f"Data type: {X_dtype.dtype}")

# Change data type to int64
X_dtype = X_dtype.type(torch.int64)
print("Tensor after changing to int64:")
print(X_dtype)
print(f"Data type: {X_dtype.dtype}")