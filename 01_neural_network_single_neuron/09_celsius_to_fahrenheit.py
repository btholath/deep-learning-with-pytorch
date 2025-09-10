"""
What’s Happening: This script mimics a neuron’s calculation without using deep learning. It multiplies the input (Celsius) by a weight (1.8) and adds a bias (32) to get Fahrenheit.
Key Concepts:

A neuron takes an input, applies a weight (like a multiplier), and adds a bias (a constant) to produce an output.
The formula $ F = 1.8 \cdot C + 32 $ is what we want our deep learning model to learn.

Why It’s Useful: It shows students the exact math we’re trying to achieve with a neural network, setting the stage for learning how to train a model to find these values (1.8 and 32) automatically.
"""
def celsius_to_fahrenheit(celsius):
    w1 = 1.8  # Weight for the Celsius input
    b = 32    # Bias
    fahrenheit = w1 * celsius + b  # Neuron computation: w1 * input + b
    return fahrenheit

# Get user input
try:
    celsius = float(input("Enter temperature in Celsius: "))
    fahrenheit = celsius_to_fahrenheit(celsius)
    print(f"{celsius}°C is equal to {fahrenheit}°F")
except ValueError:
    print("Please enter a valid number for the temperature.")