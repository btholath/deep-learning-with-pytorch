"""
Why we use torch.column_stack() when preparing data for a model.?

It takes separate 1D lists (columns) and puts them together side by side into one 2D table (matrix).
In machine learning, models expect the input to be in a table form:
    Each row = one example (like one car).
    Each column = one feature (like age, mileage).
    | Age (years) | Mileage (miles) |
    | ----------- | --------------- |
    | 5           | 20000           |
    | 10          | 85000           |
    | 2           | 15000           |

    Here:
    Column 1 = age of the car.
    Column 2 = mileage of the car.
    Your model needs this whole table, not just separate lists.

    
"""

import torch

age = torch.tensor([5, 10, 2])
mileage = torch.tensor([20000, 85000, 15000])

# These are just two separate lists. The model can’t use them directly.
print("Age:", age)
print("Mileage:", mileage)

X = torch.column_stack([age, mileage])
print(X)
"""
Now we have a table:
    Row 1 = Car 1 → [5, 20000]
    Row 2 = Car 2 → [10, 85000]
    Row 3 = Car 3 → [2, 15000]
    This is exactly the format the model expects:
    Each row = one example, each column = one feature.
"""


"""
Think of torch.column_stack() like putting two lists of information side by side to make a spreadsheet:
One list = ages of your classmates.
Another list = heights of your classmates.
Column stack puts them together into a single class table:
| Age | Height |
| --- | ------ |
| 12  | 150cm  |
| 13  | 155cm  |
| 12  | 148cm  |

We use torch.column_stack() to combine separate feature lists (like age and mileage) into a single table, because the model can only learn from data when it’s in this row-and-column format.
"""
