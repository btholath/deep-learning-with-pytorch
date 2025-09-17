# Convolutional Neural Network (CNN)


# The Motivation Behind Convolutional Neural Networks
The core idea behind Convolutional Neural Networks (CNNs) is to solve a fundamental problem that a standard neural network has when dealing with images: they don't understand that pixels are connected.

# The Limitation of Traditional Neural Networks
Imagine an image as a simple grid of pixels. A normal neural network sees this grid as a long, flat list of numbers. If you have a 16x16 pixel image, the network sees 256 individual numbers with no knowledge of which pixels are next to each other.

As you pointed out, this is a big problem. If we show you an image of Mario, but only give you a handful of randomly picked pixels from that image , it's nearly impossible to recognize the character. The image loses all its structure and patterns. A traditional neural network has this same difficulty because it treats every pixel as a separate, unrelated piece of information.

# The Human Brain's Approach to Vision
Our own brains don't work that way. When we see something, our visual system processes it in stages. First, our eyes and lower-level neurons pick up on simple patterns like horizontal or vertical lines, edges, and colors. Then, higher-level parts of our brain take these simple patterns and combine them to recognize more complex shapes, and finally, whole objects. We recognize a nose, an eye, and a mustache, and then we put them together to identify the face of Mario

# The CNN Solution: Looking Through a Window
This is exactly what CNNs were designed to do. Instead of looking at every pixel individually, a CNN looks at small groups of neighboring pixels at a time, using a receptive field, which is like a small window that slides over the entire image.

By looking at a small window, the network can start to identify simple features. For example, a neuron might activate when it sees a horizontal line, a curve, or a corner within its window. As this window moves across the image, it builds a map of where these basic features are.

These maps of simple features are then passed to the next layer of the network, which can combine them to find more complex features like an eye, an ear, or a shoulder. This process continues, with each layer building on the last, until the network has enough information to identify the full object in the image.

This is why CNNs are so effective for image recognition. They are designed to understand the spatial relationships and patterns within an image, just like the human brain. This allows them to identify objects with remarkable accuracy, a task that is nearly impossible for a traditional neural network

# Understanding Stride in CNNs
The stride is a simple but powerful idea that controls how the network scans an image. Think of it as the size of the "step" the convolutional window takes as it slides across the image.

What is Stride?
Imagine your convolutional layer's receptive field—the small window that looks at a group of pixels—is a camera. Stride is the number of pixels this camera moves before it takes its next "picture."

If the stride is 1, the window moves just one pixel at a time, creating a lot of overlap between its views.
If the stride is 2, the window skips a pixel with each move.  This means less overlap and a smaller number of "pictures" taken.

The stride value you choose directly affects two important things:
Overlap: A smaller stride leads to more overlap between the receptive fields of neighboring neurons. This means more information is shared and reused. A larger stride means less overlap, and the neurons are looking at more distinct, non-overlapping regions of the image.

Output Size: A larger stride makes the output layer of the network smaller. Since each step is bigger, fewer steps are needed to cross the image. This reduces the number of neurons required in the next layer, making the network more efficient.

In essence, stride is a balance between capturing every detail (small stride) and compressing the information to save computation (large stride). Choosing the right stride depends on the specific problem you're trying to solve and the features you want the network to focus on.


# Neural Network Classification Hyperparameters
Hyperparameter          Typical value
input neurons           One per input feature
hidden layers           Depends on the problem, but typically 1 to 5
hidden activation       ReLU


Hyperparameter      Binary classification   Multilabel binary classification        Multiclass classification
output neurons      1                       1 per label                             1 per class
output layer 
activation          Logistic                Logistic                                Softmax
Loss function       Cross entropy           Cross entropy                           Cross entropy
