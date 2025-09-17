import sys
# Purpose: Gives access to system tools, like checking computer settings.
# Goal: Not used here, but could help manage the program’s environment.
# Real-world: Like checking if a robot has enough battery to run.

import torch
# Purpose: Imports PyTorch, a tool for building smart computer models.
# Goal: Lets us create a model to recognize clothes in pictures.
# Real-world: Used in apps like photo editors or games to understand images.

from torch.utils.data import DataLoader
# Purpose: Loads data in small groups (batches) to train the model.
# Goal: Makes training faster by not loading all pictures at once.
# Real-world: Like sorting a big pile of clothes into small baskets to wash.

from torch import nn
# Purpose: Provides building blocks (layers) for our neural network model.
# Goal: Helps us stack layers to process images and make guesses.
# Real-world: Like giving a robot tools to look at and understand pictures.

import torch.nn.functional as F
# Purpose: Adds math functions to help the model learn and predict.
# Goal: Makes it easier for the model to decide what’s in an image.
# Real-world: Like a calculator helping a robot pick the right answer.

import matplotlib.pyplot as plt
# Purpose: Draws pictures and graphs to show images and results.
# Goal: Lets us see the clothes and how well the model learns.
# Real-world: Like showing photos or scores in an app or school project.

import torchvision.datasets as datasets
# Purpose: Gives access to the Fashion MNIST dataset (pictures of clothes).
# Goal: Loads images to teach the model what clothes look like.
# Real-world: Like downloading a photo album to teach a robot about shirts.

from torchvision.transforms import ToTensor

# Purpose: Turns images into numbers the computer can understand.
# Goal: Prepares pictures for the model to process.
# Real-world: Like turning a drawing into a code for a computer to read.

# Load Fashion MNIST dataset
# Purpose: Gets 60,000 training and 10,000 testing images of clothes (28x28 pixels).
# Goal: Gives the model examples to learn from and test its guesses.
# Real-world: Like showing a robot lots of clothes to learn what they are.
mnist_train = datasets.FashionMNIST(root='./data', download=True, train=True, transform=ToTensor())
mnist_test = datasets.FashionMNIST(root='./data', download=True, train=False, transform=ToTensor())

# Create data loaders
# Purpose: Splits data into batches of 32 images for easier training.
# Goal: Helps the model learn in small steps without crashing the computer.
# Real-world: Like teaching a robot in small groups instead of a huge class.
train_dataloader = DataLoader(mnist_train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=32, shuffle=True)

# Define class names for Fashion MNIST
# Purpose: Names the 10 types of clothes (e.g., T-shirt, Shoe).
# Goal: Makes it easy to understand what the model predicts.
# Real-world: Like labeling photos in a shopping app to say “Sneaker.”
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualize sample images
# Purpose: Shows 9 example pictures from the training set with their names.
# Goal: Helps us check what the clothes look like before training.
# Real-world: Like previewing items in an online store to see their styles.
plt.figure(figsize=(10, 10))
for i in range(9):
    image, label = mnist_train[i]
    plt.subplot(3, 3, i + 1)
    plt.imshow(image.squeeze(), cmap='gray')  # Squeeze removes extra dimension
    plt.title(class_names[label])
    plt.axis('off')
plt.show()

# Set device to CPU
# Purpose: Tells the model to use the computer’s CPU for calculations.
# Goal: Ensures the code runs on your computer, which doesn’t have a GPU.
# Real-world: Like using a regular calculator instead of a supercomputer.
device = torch.device('cpu')

# Define the CNN model
# Purpose: Creates a neural network to recognize types of clothes in pictures.
# Goal: Builds a model to look at images and guess what they show.
# Real-world: Like teaching a robot to say “That’s a T-shirt!” from a photo.
model = nn.Sequential(
    nn.Conv2d(1, 3, kernel_size=(3, 3), padding=1, padding_mode="reflect"),
    # Purpose: Finds patterns in images, like edges or shapes of clothes.
    # Goal: Helps the model notice details, like a shirt’s collar.
    # Real-world: Like a robot scanning a photo to find buttons or sleeves.

    nn.MaxPool2d(kernel_size=2),
    # Purpose: Shrinks the image to focus on the most important parts.
    # Goal: Makes the model faster by reducing the amount of data.
    # Real-world: Like summarizing a picture to save time for the robot.

    nn.ReLU(),
    # Purpose: Highlights important patterns and ignores boring ones.
    # Goal: Helps the model focus on key features, like a shoe’s laces.
    # Real-world: Like circling the cool parts of a picture to pay attention to.

    nn.Flatten(),
    # Purpose: Turns the image’s patterns into a single list of numbers.
    # Goal: Prepares the data for the model to make a final guess.
    # Real-world: Like making a checklist from a photo for the robot.

    nn.Linear(588, 100),
    # Purpose: Simplifies the patterns into 100 key ideas.
    # Goal: Makes it easier for the model to decide what the image is.
    # Real-world: Like grouping clothes into categories (e.g., tops, shoes).

    nn.ReLU(),
    # Purpose: Again, focuses on the most important simplified patterns.
    # Goal: Keeps the model excited about key details for guessing.
    # Real-world: Like picking out the best parts of a description.

    nn.Linear(100, 10)
    # Purpose: Makes a final guess for which of the 10 clothing types it is.
    # Goal: Outputs a score for each type (e.g., 0.9 for T-shirt).
    # Real-world: Like a robot saying, “I’m 90% sure this is a T-shirt!”
).to(device)  # Move model to CPU

# Define the loss function
# Purpose: Measures how wrong the model’s guesses are.
# Goal: Helps the model learn by showing it its mistakes.
# Real-world: Like a teacher grading a quiz to help a student get better.
loss_fn = torch.nn.CrossEntropyLoss()

# Define the optimizer
# Purpose: Updates the model to make better guesses over time.
# Goal: Helps the model improve by learning from mistakes.
# Real-world: Like a coach giving tips to a player to play better.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
# Purpose: Teaches the model by showing it 60,000 images 10 times (epochs).
# Goal: Makes the model better at recognizing clothes.
# Real-world: Like practicing a game to get really good at it.
for i in range(0, 10):  # 10 rounds of training (epochs)
    model.train()  # Set model to training mode
    # Purpose: Tells the model it’s time to learn, not guess.
    # Goal: Prepares the model to update its knowledge.
    # Real-world: Like a student opening a book to study.

    loss_sum = 0  # Track total mistakes in this round
    for X, y in train_dataloader:  # Loop through batches of images
        # Purpose: Processes 32 images at a time with their correct labels.
        # Goal: Trains the model on small groups for efficiency.
        # Real-world: Like teaching a robot in small classes, not all at once.

        X, y = X.to(device), y.to(device)  # Move data to CPU
        # Purpose: Ensures images and labels are on the same device as the model.
        # Goal: Makes sure the model can process the data.
        # Real-world: Like making sure a robot and its tools are in the same place.

        y = F.one_hot(y, num_classes=10).type(torch.float32)
        # Purpose: Turns labels (e.g., 0 for T-shirt) into a format the model understands.
        # Goal: Helps the model compare its guesses to the correct answers.
        # Real-world: Like turning “T-shirt” into a checklist for the robot.

        optimizer.zero_grad()
        # Purpose: Clears old learning steps to start fresh for this batch.
        # Goal: Prevents old mistakes from affecting new images.
        # Real-world: Like erasing a whiteboard before a new lesson.

        outputs = model(X)
        # Purpose: Makes predictions for the batch of images.
        # Goal: Gets the model’s guesses for what each image shows.
        # Real-world: Like a robot looking at a photo and guessing “Shoe!”

        loss = loss_fn(outputs, y)
        # Purpose: Calculates how wrong the model’s guesses were.
        # Goal: Measures mistakes to help the model improve.
        # Real-world: Like checking how many quiz answers a student got wrong.

        loss.backward()
        # Purpose: Figures out how to fix the model’s mistakes.
        # Goal: Tells the model which parts need to change to be better.
        # Real-world: Like a teacher explaining why an answer was wrong.

        optimizer.step()
        # Purpose: Updates the model with what it learned from this batch.
        # Goal: Makes the model smarter for the next batch.
        # Real-world: Like a student practicing to get better at a skill.

        loss_sum += loss.item()
        # Purpose: Adds up the mistakes for this round of training.
        # Goal: Tracks how much the model is improving overall.
        # Real-world: Like keeping a score of how well a student is doing.

    print(f"Epoch {i + 1} Loss: {loss_sum:.4f}")
    # Purpose: Shows how many mistakes the model made in this round.
    # Goal: Helps us see if the model is getting better.
    # Real-world: Like a report card showing progress after practice.

# Test the model
# Purpose: Checks how well the model recognizes new images it hasn’t seen.
# Goal: Measures the model’s accuracy on 10,000 test images.
# Real-world: Like giving a robot a final test to see if it knows clothes.
model.eval()  # Set model to testing mode
# Purpose: Tells the model to make guesses, not learn.
# Goal: Prepares the model to predict without changing.
# Real-world: Like a student taking a test without studying during it.

with torch.no_grad():  # Don’t update the model during testing
    # Purpose: Stops the model from learning while testing.
    # Goal: Ensures we only measure how good the model is.
    # Real-world: Like a teacher grading a test without helping the student.

    accurate = 0  # Count correct predictions
    total = 0  # Count total images tested
    for X, y in test_dataloader:  # Loop through test images
        # Purpose: Tests the model on batches of new images.
        # Goal: Checks how many images the model gets right.
        # Real-world: Like showing a robot new clothes to identify.

        X, y = X.to(device), y.to(device)  # Move