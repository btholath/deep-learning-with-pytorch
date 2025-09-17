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

"""
"What's in the box? The dataset."
The code you provided is a recipe for getting a dataset ready to train a machine learning model. It uses the Fashion-MNIST dataset, which is a collection of 70,000 images of clothing items. This is a very popular dataset for teaching and learning because the images are small and the task is simple.  Think of it as a beginner's puzzle for a computer.

How the code works
import: This is like gathering your tools. You're importing torch (the main library for building models), DataLoader (a tool for organizing your data), nn (for building neural networks), and datasets and ToTensor (for downloading and preparing the images).

datasets.FashionMNIST(...): This part is the "recipe" for getting the data.

root='./data': This tells the computer where to save the images on your disk.

download=True: This command tells the program to go online and download the dataset if it doesn't already have it.

train=True: This is for the training set. This is the larger part of the dataset (60,000 images) that the model will "study" to learn how to identify different clothing items.

train=False: This is for the test set. This is the smaller part (10,000 images) that the model will use to check its work. The model has never seen these images before, so it's a great way to see if it really learned or just memorized the training set.

transform=ToTensor(): This is a very important step. It converts the image data into a format that the computer can understand: a tensor. A tensor is a multi-dimensional grid of numbers. For an image, it represents the color or brightness of each pixel.

DataLoader(...): This is like a smart conveyor belt for your data.

It takes the entire dataset and organizes it into smaller groups called batches. In your code, each batch has 32 images.

shuffle=True: This is like shuffling a deck of cards. It makes sure the batches are in a random order. This is important because it prevents the model from learning to recognize images in a specific sequence, helping it to generalize better.



Real-world importance
Using a well-structured dataset like this is a core part of building any AI model. 
In the real world, this is how companies train their models to do things like:
    Retail: Classifying new products by type (e.g., shirts, pants, shoes) to automatically organize them on a website.
    Self-driving cars: Learning to identify different objects on the road (e.g., cars, pedestrians, traffic lights).
    Medical imaging: Helping doctors identify diseases by finding patterns in X-rays or MRI scans.
"""