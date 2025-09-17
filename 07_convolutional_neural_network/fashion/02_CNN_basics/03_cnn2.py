import sys
# Purpose: Gives access to system tools, but we don't use it in this code.
# Goal: Not actively used here, but it could help with system-related tasks.
# Real-world: Used in programs to check system settings, like memory or OS.

import torch
# Purpose: Imports PyTorch, a library for building and training neural networks.
# Goal: Lets us create and train a model to recognize images.
# Real-world: Used in apps like photo filters or self-driving cars to process data.

from torch.utils.data import DataLoader
# Purpose: Helps load data in small batches for training the model.
# Goal: Makes training faster by processing data in chunks, not all at once.
# Real-world: Like sorting clothes into small piles to wash them faster.

from torch import nn
# Purpose: Provides tools to build neural network layers, like puzzle pieces.
# Goal: Allows us to create a model with layers to process images.
# Real-world: Used in apps to recognize faces or objects in photos.

import torch.nn.functional as F
# Purpose: Adds functions for math operations in neural networks.
# Goal: Helps the model make predictions and learn from mistakes.
# Real-world: Like a calculator for the model to decide what an image shows.

import matplotlib.pyplot as plt
# Purpose: Lets us draw pictures and graphs to see images and results.
# Goal: Shows us the images and how well the model is learning.
# Real-world: Used in apps to display photos or charts, like in science projects.

import torchvision.datasets as datasets
# Purpose: Gives access to ready-made datasets, like Fashion MNIST.
# Goal: Loads images of clothes to train our model to recognize them.
# Real-world: Like downloading a library of pictures to teach a computer what clothes look like.

from torchvision.transforms import ToTensor

# Purpose: Converts images into a format the model can understand (numbers).
# Goal: Turns pictures into data the computer can process.
# Real-world: Like turning a photo into a code so a computer can analyze it.

# Load Fashion MNIST dataset
# Purpose: Gets 60,000 training and 10,000 test images of clothes (28x28 pixels).
# Goal: Gives the model examples to learn from and test its skills.
# Real-world: Like giving a robot examples of shirts and shoes to learn what they are.
mnist_train = datasets.FashionMNIST(root='./data', download=True, train=True, transform=ToTensor())
mnist_test = datasets.FashionMNIST(root='./data', download=True, train=False, transform=ToTensor())

# Create data loaders
# Purpose: Splits data into batches of 32 images for efficient training.
# Goal: Makes it easier for the model to learn without overloading the computer.
# Real-world: Like teaching a robot in small groups instead of a huge crowd.
train_dataloader = DataLoader(mnist_train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=32, shuffle=True)

# Define class names for Fashion MNIST
# Purpose: Names the 10 types of clothes in the dataset (e.g., T-shirt, Shoe).
# Goal: Helps us understand what the model predicts (human-readable labels).
# Real-world: Like labeling photos in a phone app to say "Sneaker" or "Dress."
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualize sample images
# Purpose: Shows 9 example images from the training set with their labels.
# Goal: Helps us see what the data looks like before training.
# Real-world: Like previewing clothes in an online store to check their style.
plt.figure(figsize=(10, 10))
for i in range(9):
    image, label = mnist_train[i]
    plt.subplot(3, 3, i + 1)
    plt.imshow(image.squeeze(), cmap='Blues')  # Squeeze removes extra dimension
    plt.title(class_names[label])
    plt.axis('off')
plt.show()

# Set device to CPU
# Purpose: Tells the model to use the computer's CPU (not GPU) for calculations.
# Goal: Ensures the code runs on your computer, which doesn't have a GPU.
# Real-world: Like choosing to do math with a calculator instead of a supercomputer.
device = torch.device('cpu')

# Define the CNN model
# Purpose: Creates a neural network to recognize clothes in images.
# Goal: Builds a model with layers to process images and predict what they show.
# Real-world: Like teaching a robot to look at a photo and say "That's a T-shirt!"
model = nn.Sequential(
    nn.Conv2d(1, 3, kernel_size=(3, 3), padding=1, padding_mode="reflect"),
    # Purpose: Looks at small parts of the image to find patterns (like edges).
    # Goal: Helps the model understand shapes in clothes, like a sleeve or collar.
    # Real-world: Like a robot scanning a shirt to find its buttons or seams.

    nn.ReLU(),
    # Purpose: Adds "excitement" to patterns, ignoring boring parts.
    # Goal: Makes the model focus on important features, like a shoe's laces.
    # Real-world: Like highlighting the cool parts of a picture to focus on.

    nn.Flatten(),
    # Purpose: Turns the image's patterns into a single list of numbers.
    # Goal: Prepares the data for the final decision-making layers.
    # Real-world: Like summarizing a photo into a checklist for the robot.

    nn.Linear(2352, 100),
    # Purpose: Connects patterns to a smaller set of ideas (100 features).
    # Goal: Simplifies the data to make predictions easier.
    # Real-world: Like grouping clothes into categories (e.g., tops, shoes).

    nn.ReLU(),
    # Purpose: Again, focuses on important patterns in the simplified data.
    # Goal: Keeps the model excited about key features for prediction.
    # Real-world: Like picking out the most important details in a description.

    nn.Linear(100, 10)
    # Purpose: Makes the final guess for which of the 10 clothing types it is.
    # Goal: Outputs a score for each clothing type (e.g., 0.9 for T-shirt).
    # Real-world: Like a robot saying, "I'm 90% sure this is a T-shirt!"
).to(device)  # Move model to CPU

# Define the loss function
# Purpose: Measures how wrong the model's guesses are.
# Goal: Helps the model learn by showing it what it got wrong.
# Real-world: Like a teacher grading a test to help a student improve.
loss_fn = torch.nn.CrossEntropyLoss()

# Define the optimizer
# Purpose: Adjusts the model to make better guesses over time.
# Goal: Helps the model learn from its mistakes faster.
# Real-world: Like a coach giving tips to a player to improve their game.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
# Purpose: Teaches the model by showing it 60,000 images 10 times (epochs).
# Goal: Improves the model's ability to recognize clothes correctly.
# Real-world: Like practicing a skill, like drawing, to get better at it.
for i in range(0, 10):  # 10 rounds of training (epochs)
    model.train()  # Set model to training mode
    # Purpose: Tells the model it's time to learn, not guess.
    # Goal: Prepares the model to update its knowledge.
    # Real-world: Like a student opening a textbook to study.

    loss_sum = 0  # Track total mistakes in this round
    for X, y in train_dataloader:  # Loop through batches of images
        # Purpose: Processes 32 images at a time with their correct labels.
        # Goal: Trains the model on small groups for efficiency.
        # Real-world: Like teaching a robot in small classes, not all at once.

        X, y = X.to(device), y.to(device)  # Move data to CPU
        # Purpose: Ensures data is on the same device as the model (CPU).
        # Goal: Makes sure the model can process the images.
        # Real-world: Like making sure a robot and its tools are in the same room.

        y = F.one_hot(y, num_classes=10).type(torch.float32)
        # Purpose: Converts labels (e.g., 0 for T-shirt) into a format the model understands.
        # Goal: Makes it easier for the model to compare its guesses to the truth.
        # Real-world: Like turning a label "T-shirt" into a checklist for the robot.

        optimizer.zero_grad()
        # Purpose: Clears old learning steps to start fresh for this batch.
        # Goal: Prevents mistakes from previous images affecting new ones.
        # Real-world: Like erasing a whiteboard before a new lesson.

        outputs = model(X)
        # Purpose: Makes predictions for the batch of images.
        # Goal: Gets the model's guesses for what each image shows.
        # Real-world: Like a robot looking at a photo and guessing "Shoe!"

        loss = loss_fn(outputs, y)
        # Purpose: Calculates how wrong the model's guesses were.
        # Goal: Measures the model's mistakes to help it improve.
        # Real-world: Like checking how many answers a student got wrong on a quiz.

        loss.backward()
        # Purpose: Figures out how to fix the model's mistakes.
        # Goal: Tells the model which parts need to change to be more accurate.
        # Real-world: Like a teacher explaining why an answer was wrong.

        optimizer.step()
        # Purpose: Updates the model with what it learned from this batch.
        # Goal: Makes the model smarter for the next batch.
        # Real-world: Like a student practicing to get better at a skill.

        loss_sum += loss.item()
        # Purpose: Adds up the mistakes for this round of training.
        # Goal: Tracks how much the model is improving overall.
        # Real-world: Like keeping a score of how well a student is doing in class.

    print(f"Epoch {i + 1} Loss: {loss_sum:.4f}")
    # Purpose: Shows how many mistakes the model made in this round.
    # Goal: Helps us see if the model is getting better.
    # Real-world: Like a report card showing progress after each practice.

# Test the model
# Purpose: Checks how well the model recognizes new images it hasn't seen.
# Goal: Measures the model's accuracy on the test dataset (10,000 images).
# Real-world: Like giving a robot a final exam to see if it can identify clothes.
model.eval()  # Set model to testing mode
# Purpose: Tells the model to make