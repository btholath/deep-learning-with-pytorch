"""
GOAL:
Show what a dense matrix is.

BIG IDEA:
- A dense matrix stores ALL values (including zeros).
- This is fine when most entries are non-zero.
- But if most entries are zeros, it's wasteful → we prefer sparse matrices.

REAL-WORLD:
Dense matrices are common in:
    - Images (each pixel has a value, no zeros)
    - Sensor data
    - Small datasets with no wasted space
"""

import numpy as np

# -------------------------------
# 1) Create a dense matrix
# -------------------------------
dense_matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("Dense matrix:")
print(dense_matrix)

# -------------------------------
# 2) Access elements
# -------------------------------
print("\nElement at row 1, col 2:", dense_matrix[1, 2])  # should be 6

# -------------------------------
# 3) Memory usage
# -------------------------------
print("\nMemory used by dense matrix:", dense_matrix.nbytes, "bytes")


"""
Dense matrix = all values are stored explicitly.
Example above: every number is stored (no shortcuts).
Good for: images, numeric data with few zeros.
Bad for: text data or user–item ratings (where most entries are 0).
"""