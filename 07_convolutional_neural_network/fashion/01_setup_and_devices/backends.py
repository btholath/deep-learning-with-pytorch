"""
“Where does the model run? Fast vs slow computers.”
"""
import torch
# Introduce CPU/GPU/MPS selection (shows computers can have different “brains”).
print("Cuda:", torch.cuda.is_available())
print("MPS:", torch.mps.is_available())

"""
A machine learning model, like a convolutional neural network, can run on different types of computer processors. Your code checks to see what kind of "brain" your computer has available for this task. It's looking for a CPU, GPU, or MPS processor.

CPU (Central Processing Unit) 🧠
The CPU is the primary brain of a computer. It's great at handling many different tasks one after another. However, when it comes to the massive number of calculations needed for a deep learning model, it's relatively slow. Think of it as a very smart chef who can prepare a whole meal by himself, but has to do each step one at a time.

GPU (Graphics Processing Unit) 🚀
The GPU is a specialized processor originally designed for rendering graphics in video games. Its strength is performing thousands of simple calculations at the same time. This is perfect for the way neural networks work, as they involve repeating the same mathematical operations over and over. Using a GPU is like having a whole team of chefs who can each chop vegetables at the same time, making the process much faster. The torch.cuda.is_available() command checks if you have a powerful NVIDIA GPU ready for this kind of work.

MPS (Apple's Metal Performance Shaders) 🍎
MPS is Apple's version of a high-performance engine for tasks like machine learning, available on newer Apple computers. It acts like a GPU but is specifically designed to work with Apple's hardware. The torch.mps.is_available() command checks if this special Apple "brain" is available.

In summary, the model runs on whichever processor you choose. For deep learning, running it on a GPU or MPS is much faster than using a CPU because these processors are built for the kind of parallel calculations the model needs. Your code is a simple way to figure out which "fast computer" brain is available to use.
"""