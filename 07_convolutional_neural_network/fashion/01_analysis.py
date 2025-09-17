import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# Link: https://keras.io/datasets/#/fashion-mnist-database-of-fashion-articles
# ~/aws_apps/convolutional_neural_network$ source /home/bijut/.virtualenvs/aws_apps/bin/activate
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

try:
    # Load Fashion MNIST dataset
    logger.info("Loading Fashion MNIST dataset...")
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Define class names for Fashion MNIST
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Print dataset shapes
    logger.info("Training data shape: %s", train_images.shape)
    logger.info("Training labels shape: %s", train_labels.shape)
    logger.info("Test data shape: %s", test_images.shape)
    logger.info("Test labels shape: %s", test_labels.shape)

    # Normalize pixel values to be between 0 and 1
    logger.info("Normalizing pixel values...")
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Visualize a sample image
    logger.info("Generating sample image visualization...")
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(train_images[i], cmap='gray')
        plt.title(class_names[train_labels[i]])
        plt.axis('off')
    plt.show()

    # Reshape images for CNN (add channel dimension)
    logger.info("Reshaping images for CNN...")
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))

    # Define data augmentation
    logger.info("Setting up data augmentation...")
    data_augmentation = keras.Sequential([
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1)
    ])

    # Build CNN model
    logger.info("Building CNN model...")
    model = keras.Sequential([
        data_augmentation,
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compile model
    logger.info("Compiling model...")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('fashion_mnist_model.h5', save_best_only=True)
    ]

    # Train model
    logger.info("Training model...")
    history = model.fit(train_images, train_labels, epochs=15,
                        validation_split=0.2, callbacks=callbacks, verbose=1)

    # Evaluate model
    logger.info("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    logger.info("Test accuracy: %.4f", test_accuracy)

    # Plot training history
    logger.info("Generating training history plot...")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='#1f77b4')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='#1f77b4')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='#ff7f0e')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Visualize predictions on test set
    logger.info("Generating prediction visualization...")
    predictions = model.predict(test_images[:9], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {class_names[predicted_labels[i]]}\nTrue: {class_names[test_labels[i]]}")
        plt.axis('off')
    plt.show()

except Exception as e:
    logger.error("An error occurred: %s", str(e))
    raise