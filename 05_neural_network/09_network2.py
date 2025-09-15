"""
network1.py & 9) network2.py — Build two-layer net “by hand”

BIG PICTURE:
We will build a small brain with TWO steps (layers):
  1) Hidden layer: mixes the inputs into 10 hidden "ideas"
  2) Output layer: turns those ideas into a single score

WHY DO THIS BY HAND?
- Usually we use nn.Sequential (like snapping LEGO blocks together).
- Here we plug each piece ourselves so we can SEE the gears turning:
    hidden = Linear(2 -> 10) -> Sigmoid
    logits = Linear(10 -> 1)
- Then we compute loss, backprop, and update weights.

REAL-WORLD ANALOGY:
- Inputs could be "Study Hours" + "Previous Exam Score".
- Output is "Will pass the exam?" (1 = yes, 0 = no).
- The same recipe powers spam filters, recommendations, and more.

Summary of Neural Network Architecture in the Code
Overview:
The code implements a two-layer neural network to predict a binary outcome (Pass/Fail) based on two input features (Study Hours, Previous Exam Score).
It manually constructs the forward pass (Linear → Sigmoid → Linear → Logits) without using nn.Sequential, to demonstrate the underlying mechanics of a neural network.
The network is trained using stochastic gradient descent (SGD) and binary cross-entropy loss with logits.

Input Layer:
Data: Loaded from student_exam_data.csv using pandas, containing Study Hours, Previous Exam Score, and Pass/Fail columns.
Features: Two input features (Study Hours, Previous Exam Score) are extracted into a tensor X with shape (n_samples, 2), where n_samples is the number of records.
Target: The Pass/Fail column is converted to a tensor y with shape (n_samples, 1), representing binary labels (0 or 1).
Pre-processing: The input data is converted to torch.float32 tensors, but no explicit normalization or scaling is applied in the code (assumed to be pre-processed in the dataset).

Hidden Layers:
Structure: One hidden layer implemented using nn.Linear(2, 10) (named hidden_model).
Takes 2 input features and produces 10 outputs (neurons in the hidden layer).
Forward Pass:
    Computes outputs = hidden_model(X), applying a linear transformation: X @ W1 + b1, where W1 is the weight matrix and b1 is the bias vector.
    Applies the sigmoid activation function (nn.functional.sigmoid(outputs)) to introduce non-linearity, producing values in the range [0, 1] for each of the 10 neurons.


Weights and Biases:
Hidden Layer:
    hidden_model.weight: Weight matrix of shape (10, 2) (10 neurons, each connected to 2 input features).
    hidden_model.bias: Bias vector of shape (10,), one bias per neuron.

Output Layer:
    output_model.weight: Weight matrix of shape (1, 10) (1 output neuron connected to 10 hidden neurons).
    output_model.bias: Bias scalar of shape (1,).

Parameters: Both layers’ parameters (weights and biases) are combined into a single list (parameters) for optimization using list(hidden_model.parameters()) + list(output_model.parameters()).

Optimization: The SGD optimizer (torch.optim.SGD) updates these parameters with a learning rate of 0.005.    


Activation Functions:
Sigmoid in Hidden Layer: Applied after the hidden layer’s linear transformation (nn.functional.sigmoid(outputs)), mapping outputs to [0, 1] for non-linearity.
Sigmoid in Output (Evaluation): Applied during evaluation (nn.functional.sigmoid(outputs) > 0.5) to convert logits to probabilities for binary classification.
Note: The output layer’s linear transformation produces logits, which are combined with the binary cross-entropy loss (BCEWithLogitsLoss), so no explicit activation is applied during training.

Output Layer:
Structure: Implemented using nn.Linear(10, 1) (named output_model).
Takes 10 hidden layer outputs and produces 1 output (logit) for binary classification.
Forward Pass: Computes outputs = output_model(outputs), applying a linear transformation: hidden_outputs @ W2 + b2, where W2 is the weight matrix and b2 is the bias.
Loss Function: Uses BCEWithLogitsLoss, which combines a sigmoid activation and binary cross-entropy loss, computing the loss between logits and true labels (y).
Prediction: During evaluation, logits are passed through nn.functional.sigmoid to obtain probabilities, and a threshold of 0.5 is used to predict Pass/Fail (y_pred = nn.functional.sigmoid(outputs) > 0.5).

Training Process:
Epochs: The network is trained for 500,000 iterations.
Steps:
    Zeroes gradients (optimizer.zero_grad()).
    Computes forward pass: hidden_model(X) → sigmoid → output_model → logits.
    Calculates loss using BCEWithLogitsLoss.
    Backpropagates gradients (loss.backward()).
    Updates weights and biases (optimizer.step()).
Monitoring: Prints loss every 10,000 iterations to track training progress

Evaluation:
    Sets models to evaluation mode (hidden_model.eval(), output_model.eval()).
    Performs forward pass with torch.no_grad() to disable gradient computation.
    Computes predictions (y_pred) by applying sigmoid and thresholding at 0.5.
    Calculates accuracy as the mean of correct predictions (y_pred_correct.type(torch.float32).mean()).

Additional Notes:
Activity: The code suggests printing hidden_model.weight.shape (would output (10, 2)) to inspect parameter shapes, aiding understanding of the network’s structure.
Engagement: Commenting out the sigmoid activation in the hidden layer would remove non-linearity, potentially reducing the model’s ability to learn complex patterns, leading to poorer performance.
Purpose: By manually defining layers and the forward pass, the code demystifies how neural networks process data, showing the "gears" behind higher-level abstractions like nn.Sequential.    
    

"""
import sys
import torch
from torch import nn
import pandas as pd

# 1) LOAD DATA ---------------------------------------------------------------
# Load the data from a CSV file.
# We read a small table where each row is a student.
# Columns:
#   - "Study Hours" (how long they studied)
#   - "Previous Exam Score" (last score)
#   - "Pass/Fail" (1 = pass, 0 = fail) → this is what we want to predict
df = pd.read_csv("/workspaces/deep-learning-with-pytorch/05_neural_network/data/student_exam_data.csv")


# 2) MAKE TENSORS ------------------------------------------------------------
# Prepare the data tensors for the network.
# X is the input data (Study Hours, Previous Exam Score).
# X (inputs) must be floating-point numbers for the model.
# Shape: [num_students, 2] because we have 2 input features.
X = torch.tensor(
    df[["Study Hours", "Previous Exam Score"]].values, 
    dtype=torch.float32
)
# y is the target data (Pass/Fail).
# y (targets) is also a float tensor with shape [num_students, 1].
# We reshape to a COLUMN vector so each row matches one student.
y = torch.tensor(df["Pass/Fail"], dtype=torch.float32)\
    .reshape((-1, 1))


# 3) BUILD LAYERS BY HAND ----------------------------------------------------
# Define the network layers as separate modules.
# hidden_model: A linear layer that takes 2 inputs and outputs 10 hidden features out.
# Think of these as 10 tiny detectors learning useful combos like
# "studied a lot AND last score was high".
hidden_model = nn.Linear(2, 10)

# output_model: A linear layer that takes the 10 hidden features and outputs 1 result.
# Output layer: takes those 10 hidden features → makes 1 final score (logit).
# "Logit" means a raw score that we will later squash to 0..1 with sigmoid
# ONLY when we want a probability for evaluation.
output_model = nn.Linear(10, 1)


# 4) CHOOSE LOSS (HOW WRONG ARE WE?) ----------------------------------------
# BCEWithLogitsLoss is perfect for yes/no questions (binary classification).
# It expects raw scores (logits) from the final layer (no sigmoid before it).
# Define the loss function. `BCEWithLogitsLoss` combines a sigmoid activation
# and binary cross-entropy, which is numerically more stable.
loss_fn = torch.nn.BCEWithLogitsLoss()


# 5) TELL THE OPTIMIZER WHICH KNOBS (PARAMETERS) TO TURN --------------------
# We combine BOTH layers' parameters so one optimizer updates everything.
# Manually create a single list of all parameters from both layers.
# This is crucial for the optimizer to be able to update all weights and biases.
parameters = list(hidden_model.parameters()) + list(output_model.parameters())

# Optimizer = the "coach" that tweaks weights after each learning step.
# lr = learning rate (how big each tweak is). Small = careful; big = bold.
# Define the optimizer. It will update the parameters in the list we created.
optimizer = torch.optim.SGD(parameters, lr=0.005)

# Peek at shapes of weights and biases:
print("Hidden layer weight shape  (10 x 2):", hidden_model.weight.shape)
print("Hidden layer bias shape    (10,)   :", hidden_model.bias.shape)
print("Output layer weight shape  (1 x 10):", output_model.weight.shape)
print("Output layer bias shape    (1,)    :", output_model.bias.shape)



# 6) TRAINING LOOP (PRACTICE OVER AND OVER) ---------------------------------
# We repeat many times:
#   a) Forward pass: make a guess from X
#   b) Compute loss: compare guess vs truth y
#   c) Backward pass: figure out how to fix weights (gradients)
#   d) Step: apply a tiny fix with the optimizer
#
# NOTE: 500,000 is a LOT; it’s fine for a demo but you can lower it to run faster.
# Training Loop
# The forward pass is computed step-by-step.
print("Training the network...")
for i in range(0, 500000):
    # 1. Zero the gradients before the forward pass.
    optimizer.zero_grad()
    
    # 2. Manual Forward Pass:
    # Compute the output of the first linear layer.
    outputs = hidden_model(X)

    # Apply the non-linear activation function.
    outputs = nn.functional.sigmoid(outputs)

    # Compute the output of the second linear layer.
    outputs = output_model(outputs)

    # 3. Calculate the loss.
    loss = loss_fn(outputs, y)

    # 4. Perform backpropagation to compute gradients.
    loss.backward()

    # 5. Update the parameters using the optimizer.
    optimizer.step()

    # Print the loss periodically to monitor training progress.
    if i % 10000 == 0:
        print(loss)


# Evaluation
# Set the models to evaluation mode to disable layers like dropout.
hidden_model.eval()
output_model.eval()

# Disable gradient calculation for inference.
with torch.no_grad():
    # Manual forward pass for evaluation.
    outputs = hidden_model(X)
    outputs = nn.functional.sigmoid(outputs)
    outputs = output_model(outputs)

    # Apply a final sigmoid and threshold to get binary predictions.
    y_pred = nn.functional.sigmoid(outputs) > 0.5

    # Compare predictions to the actual labels.
    y_pred_correct = y_pred.type(torch.float32) == y

    # Calculate and print the final accuracy.
    print(y_pred_correct.type(torch.float32).mean())
