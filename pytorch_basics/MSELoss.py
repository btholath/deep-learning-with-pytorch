"""
MSELoss (Mean Squared Error Loss)
    MSELoss tells us how far off the guesses are from the truth.
    Small loss → guesses are close.
    Big loss → guesses are far.
    Squaring makes big mistakes count a lot more than small ones.

    
    
"""
import torch
import torch.nn as nn

# Real answer (teacher's age)
y_true = torch.tensor([30.0])

# Student's guesses
y_pred = torch.tensor([25.0, 40.0, 32.0])

# Loss function
loss_fn = nn.MSELoss()

# Compute loss
loss = loss_fn(y_pred, y_true.expand_as(y_pred))
print("MSE Loss =", loss.item())


"""
Imagine you are a teacher, and a student is trying to guess your age.
Your real age = 30 years

Student’s guesses:
    First try: 25
    Second try: 40
    Third try: 32
You want to measure how wrong the student is, on average.

Let us calculate this measure in manual steps instead of using MSELoss()
"""
# Step 1: Find the errors
# Subtract each guess from the real age (30):

real_age = 30
guess_1 = 25
guess_1_error = real_age - guess_1

guess_2 = 40
guess_2_error = real_age - guess_2

guess_3 = 32
guess_3_error = real_age - guess_3


# Step 2: Square the errors
# Why square? Because we don’t want negatives, and bigger mistakes should “hurt” more.
squared_error_1 = guess_1_error **2
squared_error_2 = guess_2_error **2
squared_error_3 = guess_3_error **2
print("squared_error_1 =", squared_error_1 )
print("squared_error_2 =", squared_error_2 )
print("squared_error_3 =", squared_error_3 )

# Step 3: Take the average
mean_squared_error = (squared_error_1 + squared_error_2 + squared_error_3)/3
print("mean_squared_error (MSE) =", mean_squared_error ) # That number means: on average, the student’s guesses are off by quite a lot.