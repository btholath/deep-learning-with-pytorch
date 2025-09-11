# %%
# This line, "# %%", designates the start of a new "cell".
# In the VS Code Python extension, you can run this cell individually.

# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# %%
# This second cell creates the data and the plot.
# The code is intentionally separated into two cells to show the "notebook-like"
# behavior you described. You can run the first cell to import libraries, and then
# run this cell to generate the plot.

# Generate some sample data
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x)

# Create the plot
plt.figure(figsize=(8, 6)) # Set the figure size for better visibility
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
plt.title('Sine Wave Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.legend()

# Display the plot
# When you run this cell in the Python Interactive Window, the plot will
# be displayed directly below this cell.
plt.show()

# %%
# You can add more cells for further analysis or plotting.
# For example, let's create a cosine wave plot.

y_cos = np.cos(x)
plt.figure(figsize=(8, 6))
plt.plot(x, y_cos, label='cos(x)', color='red', linestyle='--', linewidth=2)
plt.title('Cosine Wave Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.legend()
plt.show()
