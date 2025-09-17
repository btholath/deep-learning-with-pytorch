import sys
# Purpose: Gives access to system tools, like checking computer settings.
# Goal: Not used here, but could help manage errors or system resources.
# Real-world: Like checking if a robot’s battery is charged before it starts.

import torch
# Purpose: Imports PyTorch, a library for building smart computer models.
# Goal: Lets us create a model to recognize clothes in pictures.
# Real-world: Used in apps like photo editors or games to understand images.

from torch.utils.data import DataLoader
# Purpose: Loads data in small groups (batches) to train the model efficiently.
# Goal: Makes training faster by processing a few images at a time.
# Real-world: Like sorting a big pile of clothes into small baskets to wash.

from torch import nn
# Purpose: Provides building blocks (layers) for our neural network model.
# Goal: Helps us stack layers to process images and make guesses.
# Real-world: Like giving a robot tools to look at and understand photos.

import torch.nn.functional as F
# Purpose: Adds math functions to help the model learn and predict.
# Goal: Makes it easier for the model to decide what’s in an image.
# Real-world: Like a calculator helping a robot choose the right answer.

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

# Set up the device (CPU or GPU)
# Purpose: Chooses whether to use the CPU (regular computer processor) or GPU (faster graphics processor).
# Goal: Makes sure the model runs on the best available hardware.
# Real-world: Like picking a fast calculator (GPU) or regular one (CPU) for math.
device = torch.device("cpu")  # Start with CPU
if torch.cuda.is_available():  # Check if NVIDIA GPU is available
    device = torch.device("cuda")
elif torch.mps.is_available():  # Check if Apple M1/M2 GPU is available
    device = torch.device("mps")
print("Running on device:", device)
# Purpose: Shows which device (CPU or GPU) the model will use.
# Goal: Helps us confirm the model is running on the right hardware.
# Real-world: Like checking if a robot is using its fastest brain.

# Load Fashion MNIST dataset
# Purpose: Gets 60,000 training and 10,000 testing images of clothes (28x28 pixels).
# Goal: Gives the model examples to learn from and test its guesses.
# Real-world: Like showing a robot lots of clothes to learn what they are.
try:
    mnist_train = datasets.FashionMNIST(root='./data', download=True, train=True, transform=ToTensor())
    mnist_test = datasets.FashionMNIST(root='./data', download=True, train=False, transform=ToTensor())
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)
# Purpose: Tries to load the dataset and stops if there’s a problem.
# Goal: Makes sure the data is ready before training.
# Real-world: Like making sure a robot has its photo album before learning.

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

# Define the CNN model
# Purpose: Creates a neural network to recognize types of clothes in pictures.
# Goal: Builds a model to look at images and guess what they show.
# Real-world: Like teaching a robot to say “That’s a T-shirt!” from a photo.
model = nn.Sequential(
    nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1, padding_mode="reflect"),
        # Purpose: Finds patterns in images, like edges or shapes, using 32 filters.
        # Goal: Helps the model notice details, like a shirt’s collar or shoe laces.
        # Real-world: Like a robot scanning a photo to find buttons or patterns.

        nn.MaxPool2d(kernel_size=2),
        # Purpose: Shrinks the image to focus on the most important parts.
        # Goal: Makes the model faster by reducing data size (from 28x28 to 14x14).
        # Real-world: Like summarizing a picture to save time for the robot.

        nn.BatchNorm2d(32),
        # Purpose: Balances the data to make learning smoother and faster.
        # Goal: Helps the model learn better by keeping numbers stable.
        # Real-world: Like adjusting a robot’s settings to make it work smoothly.

        nn.ReLU(),
        # Purpose: Highlights important patterns and ignores boring ones.
        # Goal: Focuses on key features, like a sleeve’s shape.
        # Real-world: Like circling the cool parts of a picture to pay attention to.

        nn.Dropout(0.1)
        # Purpose: Randomly ignores 10% of patterns to prevent over-learning.
        # Goal: Helps the model generalize to new images, not just memorize.
        # Real-world: Like a student practicing different problems to avoid cheating.
    ),
    nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, padding_mode="reflect"),
        # Purpose: Finds more complex patterns using 64 filters.
        # Goal: Notices finer details, like fabric texture or zipper shapes.
        # Real-world: Like a robot looking closer at a photo for tiny details.

        nn.MaxPool2d(kernel_size=2),
        # Purpose: Shrinks the image again (from 14x14 to 7x7).
        # Goal: Speeds up the model by focusing on key features.
        # Real-world: Like summarizing a summary to make it even shorter.

        nn.BatchNorm2d(64),
        # Purpose: Balances the data again to improve learning.
        # Goal: Keeps the model stable for better accuracy.
        # Real-world: Like fine-tuning a robot to work consistently.

        nn.ReLU(),
        # Purpose: Highlights important patterns from the second layer.
        # Goal: Keeps the model focused on useful details.
        # Real-world: Like underlining the best parts of a description.

        nn.Dropout(0.1)
        # Purpose: Ignores 10% of patterns to make the model more flexible.
        # Goal: Prevents the model from memorizing only training images.
        # Real-world: Like practicing with different examples to be ready for anything.
    ),
    nn.Flatten(),
    # Purpose: Turns the image’s patterns into a single list of numbers.
    # Goal: Prepares the data for the final guessing layers (64 * 7 * 7 = 3136).
    # Real-world: Like making a checklist from a photo for the robot.

    nn.Sequential(
        nn.Linear(64 * 7 * 7, 1000),
        # Purpose: Simplifies patterns into 1000 key ideas.
        # Goal: Makes it easier for the model to process patterns.
        # Real-world: Like grouping clothes into big categories.

        nn.BatchNorm1d(1000),
        # Purpose: Balances the simplified data for smoother learning.
        # Goal: Helps the model make better guesses by stabilizing numbers.
        # Real-world: Like adjusting a robot’s controls for consistent performance.

        nn.ReLU(),
        # Purpose: Focuses on the most important simplified patterns.
        # Goal: Keeps the model excited about key details.
        # Real-world: Like picking out the best parts of a description.

        nn.Dropout(0.3),
        # Purpose: Ignores 30% of patterns to prevent over-learning.
        # Goal: Helps the model work well on new images.
        # Real-world: Like studying varied problems to be ready for a test.

        nn.Linear(1000, 100),
        # Purpose: Simplifies further into 100 key ideas.
        # Goal: Prepares the model for the final guess.
        # Real-world: Like narrowing down categories to specific types.

        nn.BatchNorm1d(100),
        # Purpose: Balances the data again for better learning.
        # Goal: Keeps the model stable for accurate predictions.
        # Real-world: Like fine-tuning a robot to stay consistent.

        nn.ReLU(),
        # Purpose: Highlights the best patterns again.
        # Goal: Keeps the model focused on important details.
        # Real-world: Like circling the most important parts of a list.

        nn.Dropout(0.5),
        # Purpose: Ignores 50% of patterns to make the model very flexible.
        # Goal: Ensures the model doesn’t memorize, but learns generally.
        # Real-world: Like practicing with many examples to handle surprises.

        nn.Linear(100, 10)
        # Purpose: Makes a final guess for which of the 10 clothing types it is.
        # Goal: Outputs a score for each type (e.g., 0.9 for T-shirt).
        # Real-world: Like a robot saying, “I’m 90% sure this is a T-shirt!”
    )
).to(device)  # Move model to CPU or GPU
print(model)
# Purpose: Shows the model’s structure (layers and their order).
# Goal: Helps us understand how the model is built.
# Real-world: Like a blueprint showing how a robot’s brain is designed.

# Define the loss function
# Purpose: Measures how wrong the model’s guesses are.
# Goal: Helps the model learn by showing it its mistakes.
# Real-world: Like a teacher grading a quiz to help a student get better.
loss_fn = nn.CrossEntropyLoss()

# Define the optimizer
# Purpose: Updates the model to make better guesses over time.
# Goal: Helps the model improve by learning from mistakes.
# Real-world: Like a coach giving tips to a player to play better.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
# Purpose: Teaches the model by showing it 60,000 images 10 times (epochs).
# Goal: Makes the model better at recognizing clothes.
# Real-world: Like practicing a game to get really good at it.
try:
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

            X = X.to(device)
            y = F.one_hot(y, num_classes=10).type(torch.float32).to(device)
            # Purpose: Moves images and labels to CPU/GPU and formats labels.
            # Goal: Ensures data is ready for the model to process.
            # Real-world: Like making sure a robot and its tools are in the same place.

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

except Exception as e:
    print(f"Error during training: {e}")
    sys.exit(1)
# Purpose: Catches problems during training and stops the program safely.
# Goal: Prevents the program from crashing unexpectedly.
# Real-world: Like a robot stopping if it gets confused to avoid mistakes.

# Test the model
# Purpose: Checks how well the model recognizes new images it hasn’t seen.
# Goal: Measures the model’s accuracy on 10,000 test images.
# Real-world: Like giving a robot a final test to see if it knows clothes.
try:
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

            X, y = X.to(device), y.to(device)  # Move data to CPU/GPU
            outputs = nn.functional.softmax(model(X), dim=1)
            # Purpose: Makes predictions and turns them into probabilities.
            # Goal: Gets the model’s confidence for each clothing type.
            # Real-world: Like a robot saying, “I’m 90% sure this is a T-shirt.”

            correct_pred = (y == outputs.max(dim=1).indices)
            # Purpose: Checks if the model’s top guess matches the true label.
            # Goal: Counts how many images the model guessed correctly.
            # Real-world: Like checking if a robot’s guess matches the real clothing item.

            total += correct_pred.size(0)  # Add number of images in this batch
            accurate += correct_pred.type(torch.int).sum().item()  # Add correct guesses
            # Purpose: Keeps track of total images and correct predictions.
            # Goal: Calculates the overall accuracy of the model.
            # Real-world: Like counting correct answers on a test.

        accuracy = accurate / total
        print(f"Accuracy on validation data: {accuracy:.4f}")
        # Purpose: Shows the percentage of correct predictions.
        # Goal: Tells us how good the model is at recognizing clothes.
        # Real-world: Like a score showing how well a robot identifies clothes in a store.

except Exception as e:
    print(f"Error during testing: {e}")
    sys.exit(1)
# Purpose: Catches problems during testing and stops the program safely.
# Goal: Ensures the program doesn’t crash if something goes wrong.
# Real-world: Like a robot pausing if it can’t understand a test question.

# Visualize sample predictions
# Purpose: Shows 9 test images with the model’s guesses and true labels.
# Goal: Helps us see if the model is guessing correctly.
# Real-world: Like a robot showing you a photo and saying, “I think this is a Shoe!”
try:
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_dataloader))  # Get one batch
        images, labels = images.to(device), labels.to(device)
        outputs = nn.functional.softmax(model(images), dim=1)
        predicted_labels = outputs.max(dim=1).indices
        plt.figure(figsize=(10, 10))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].squeeze(), cmap='gray')
            plt.title(f"Pred: {class_names[predicted_labels[i]]}\nTrue: {class_names[labels[i]]}")
            plt.axis('off')
        plt.show()
except Exception as e:
    print(f"Error during visualization: {e}")
# Purpose: Shows test images and predictions, catching errors if they happen.
# Goal: Lets us visually check the model’s performance.
# Real-world: Like a robot showing its guesses to see if they make sense.