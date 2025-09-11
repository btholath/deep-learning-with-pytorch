"""
In PyTorch, tensors (like little boxes of numbers) need to have the same shape if you want to compare them (like subtraction in MSE).
y_true = torch.tensor([30.0]) → shape [1] (just one number).
y_pred = torch.tensor([25.0, 40.0, 32.0]) → shape [3] (three numbers).
So if we try: 
loss_fn(y_pred, y_true)
PyTorch will complain, because one tensor has 3 numbers and the other has only 1.

So the Solution is
    y_true.expand_as(y_pred)
This tells PyTorch:
 “Take this single number (30.0) and pretend it is repeated so that it looks like the same shape as y_pred.”

So it becomes:
[30.0, 30.0, 30.0]

Now both tensors have shape [3]:
y_pred = [25.0, 40.0, 32.0]
y_true = [30.0, 30.0, 30.0]
And the loss can be calculated.


Think of expand_as() like making photocopies:
In below, example, you only had one answer key (30).
The student gave three guesses.
To grade fairly, you make 3 copies of the answer (30, 30, 30) so you can check each guess.


"""
import torch

y_true = torch.tensor([30.0])            # shape [1]
y_pred = torch.tensor([25.0, 40.0, 32.0]) # shape [3]

print("y_true shape:", y_true.shape)
print("y_pred shape:", y_pred.shape)

# Expand y_true to match y_pred
y_true_expanded = y_true.expand_as(y_pred)

print("Expanded y_true:", y_true_expanded)
print("Expanded shape:", y_true_expanded.shape)

