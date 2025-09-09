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