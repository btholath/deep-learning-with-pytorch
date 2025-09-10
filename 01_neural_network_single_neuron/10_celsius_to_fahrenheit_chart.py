# Fahrenheit = 1.8 * Celsius + 32
# We'll implement a single "neuron" (a linear unit) with weight w1=1.8 and bias b=32.
# Then we'll generate a neural network style diagram labeling w1 and b, and save it as a PNG.

import numpy as np
import matplotlib.pyplot as plt

class SingleNeuron:
    """
    A single linear neuron: y = w1 * x + b
    For Celsius to Fahrenheit, w1 = 1.8, b = 32.
    """
    def __init__(self, w1: float = 1.8, b: float = 32.0):
        self.w1 = float(w1)
        self.b = float(b)

    def forward(self, x):
        """
        Forward pass: compute Fahrenheit given Celsius.
        x can be a scalar or a NumPy array/list.
        """
        return self.w1 * np.asarray(x) + self.b


# --- Use the neuron ---
neuron = SingleNeuron(w1=1.8, b=32.0)

# Example inputs (in Celsius)
celsius_values = np.array([-40, -10, 0, 10, 20, 37, 100])
fahrenheit_values = neuron.forward(celsius_values)

# Print a small table
print("Celsius -> Fahrenheit using a single neuron (w1=1.8, b=32):")
for c, f in zip(celsius_values, fahrenheit_values):
    print(f"{c:>7.2f} °C  ->  {f:>7.2f} °F")

# --- Plot 1: A simple neuron diagram with legends for w1 and b ---
fig, ax = plt.subplots(figsize=(8, 5))

ax.axis("off")

# Positions
x_in, y_in = 0.1, 0.5      # input node
x_sum, y_sum = 0.5, 0.5    # summation (neuron) node
x_out, y_out = 0.9, 0.5    # output node

# Draw nodes
input_circle = plt.Circle((x_in, y_in), 0.04, fill=False, linewidth=2)
sum_circle = plt.Circle((x_sum, y_sum), 0.06, fill=False, linewidth=2)
output_circle = plt.Circle((x_out, y_out), 0.04, fill=False, linewidth=2)

ax.add_patch(input_circle)
ax.add_patch(sum_circle)
ax.add_patch(output_circle)

# Labels
ax.text(x_in, y_in - 0.1, "Input\nCelsius (x)", ha="center", va="center")
ax.text(x_sum, y_sum + 0.12, "Neuron\nΣ = w1·x + b", ha="center", va="center", fontsize=11)
ax.text(x_out, y_out - 0.1, "Output\nFahrenheit (y)", ha="center", va="center")

# Arrows: input to neuron
ax.annotate("", xy=(x_sum - 0.06, y_sum), xytext=(x_in + 0.04, y_in),
            arrowprops=dict(arrowstyle="->", lw=2))
ax.text((x_in + x_sum)/2, y_in + 0.06, "w1 = 1.8", ha="center", va="bottom")

# Bias arrow (drawn from top into the neuron)
ax.annotate("", xy=(x_sum, y_sum + 0.06), xytext=(x_sum, y_sum + 0.28),
            arrowprops=dict(arrowstyle="->", lw=2))
ax.text(x_sum + 0.05, y_sum + 0.22, "bias b = 32", ha="left", va="center")

# Neuron to output
ax.annotate("", xy=(x_out - 0.04, y_out), xytext=(x_sum + 0.06, y_sum),
            arrowprops=dict(arrowstyle="->", lw=2))

# Legends box
legend_text = "Linear neuron: y = w1·x + b\nw1 = 1.8 (scale)\nb  = 32 (offset)"
ax.text(0.05, 0.1, legend_text, fontsize=11, va="center",
        bbox=dict(boxstyle="round,pad=0.4", ec="black", fc="white"))

# Save the diagram
diagram_path = "/workspaces/deep-learning-with-pytorch/01_neural_network_single_neuron/output/celsius_to_fahrenheit_neuron.png"
plt.tight_layout()
plt.savefig(diagram_path, dpi=200, bbox_inches="tight")
plt.show()

# --- Plot 2: Line plot of the mapping for reference ---
fig2, ax2 = plt.subplots(figsize=(6, 4))
x_lin = np.linspace(-50, 110, 100)
y_lin = neuron.forward(x_lin)
ax2.plot(x_lin, y_lin, linewidth=2)
ax2.set_xlabel("Celsius (°C)")
ax2.set_ylabel("Fahrenheit (°F)")
ax2.set_title("Celsius → Fahrenheit via y = 1.8x + 32")
ax2.grid(True)

line_plot_path = "/workspaces/deep-learning-with-pytorch/01_neural_network_single_neuron/output/c_to_f_line_plot.png"
plt.tight_layout()
plt.savefig(line_plot_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"\nSaved neuron diagram: {diagram_path}")
print(f"Saved mapping line plot: {line_plot_path}")
