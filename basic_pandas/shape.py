import torch

# Say price has 5 numbers
price = [12000, 18000, 25000, 8500, 21000]


# When we make it a tensor
y = torch.tensor(price, dtype=torch.float32)
print(y.shape)   # → torch.Size([5])  (a 1D vector)


# Now if we reshape:
y = y.view(-1, 1)
print(y.shape)   # → torch.Size([5, 1])  (a column vector)
print(y)
# .view(-1, 1) turns your prices list into a column of prices, one per car, so the neuron knows how to line them up with its predictions.

"""
Why do we do this?
Machine learning models in PyTorch expect 2D input/output tensors:

shape [number_of_examples, number_of_features].

If we kept it as [5] (just one dimension), the neuron wouldn’t match shapes correctly.

By reshaping to [5, 1], we’re saying:
    5 rows (examples of cars),
    1 column (the target price we want the model to learn).
"""
