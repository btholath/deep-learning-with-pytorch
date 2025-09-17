import sys
import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt

import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

mnist_train = datasets.FashionMNIST(root='./data', download=True, train=True, transform=ToTensor())
mnist_test = datasets.FashionMNIST(root='./data', download=True, train=False, transform=ToTensor())

train_dataloader = DataLoader(mnist_train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Conv2d( in_channels=1, out_channels=3, kernel_size=(3, 3), padding=1, padding_mode="reflect"),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(2352, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

image = mnist_train[0][0].reshape(1, 1, 28, 28)
output = model(image)
print(output.shape)
print(output)

"""
Builds a basic Convolutional Neural Network (CNN) and runs a single image through it. 
The final output shape is torch.Size([1, 10]). This means the model took one input image and produced a list of 10 numbers as its output.

How the Code Works 🧠
Setting up the Model: The code defines a neural network using nn.Sequential. Think of nn.Sequential as a list of instructions that the computer will follow in order.

nn.Conv2d(1, 3, ...): This is the convolutional layer, the "feature detector." It looks for simple patterns like edges and textures in the image. It takes a single-channel image (like a grayscale image of clothing) and creates 3 new "feature maps" as its output.

nn.ReLU(): This is a Rectified Linear Unit. It is a simple function that changes all negative numbers in the data to zero. This helps the network learn complex patterns and is a common step after a convolutional layer.

nn.Flatten(): The convolutional layer outputs a 3D grid of numbers. The Flatten layer reshapes this 3D grid into a single, long line of numbers. This is like unrolling a poster so it can fit on a one-dimensional clothesline.

nn.Linear(2352, 100): This is a fully connected layer. It connects all the numbers from the flattened layer to a new layer with 100 numbers. This step helps the network learn more complex combinations of features. The number 2352 comes from the output of the convolutional layer (3 feature maps * 28 pixels * 28 pixels = 2352).

nn.ReLU(): Another ReLU layer is used here for the same reason as before, to help the network learn more complex patterns.

nn.Linear(100, 10): This is the final fully connected layer. It takes the 100 numbers from the previous layer and converts them into 10 final numbers. These 10 numbers are the most important part of the output, as they represent the model's "confidence score" for each of the 10 clothing categories (e.g., T-shirt, pants, dress, etc.). The final output torch.Size([1, 10]) means there is one output for the one image and 10 numbers in that output.

Running the Model: The code then takes the first image from the dataset, which is a tensor of size (28, 28). It adds two extra dimensions to it to get a size of (1, 1, 28, 28) so that the model can understand the input. Then, it passes the image through the model and prints the size of the final output, which is torch.Size([1, 10]).

Real-World Importance 🌍
This process is the core of how a CNN makes predictions. This simple example has many real-world applications:
    E-commerce: Classifying clothing items in an online store to recommend similar products to customers.
    Quality control: A manufacturing company could use a similar model to inspect products on a conveyor belt and identify defective items.
    Search engines: Using an image to search for similar images online by finding patterns in the images.
"""
