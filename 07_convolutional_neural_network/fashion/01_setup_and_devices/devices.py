# First, let's bring in the tools we need from the PyTorch library.
# We'll need `torch` for our tensors (which are like super-powered lists of numbers)
# and `nn` for building neural networks.
import torch
from torch import nn

# What is a device? Think of a device as the "engine" that does all the heavy math.
# We first set our default engine to the CPU, which is the brain of a computer.
# It's good, but not the fastest for this kind of work.
device = torch.device("cpu")

# Now, we check if there's a more powerful engine available.
# We look for a CUDA-compatible GPU (NVIDIA graphics card).
# A GPU is like a race car engine, designed to do thousands of simple calculations
# at the same time, which is perfect for AI.
if torch.cuda.is_available():
    device = torch.device("cuda")
# Apple computers have their own version of a super-fast engine called MPS.
# We check for that as our third option.
elif torch.mps.is_available():
    device = torch.device("mps")

# Let's tell the user which engine we're using.
# This shows the magic is working!
print("Running on device:", device)

# Now, we create our "super-powered lists" of numbers, called tensors.
# We create `a` and directly tell it to use our fastest available engine.
a = torch.tensor([5, 7], device=device)
# We create `b` on the default CPU, just to show the difference.
b = torch.tensor([3, 4])

# This is the most important part! We move `b` from the CPU engine
# to our super-fast engine (`device`). This is like transferring a project
# from your regular laptop to a supercomputer for faster results.
b = b.to(device)

# Let's check where each tensor is stored now.
# This proves that both `a` and `b` are on the same, fast device.
print("Tensor 'a' is on:", a.device)
print("Tensor 'b' is on:", b.device)

# Finally, we perform a simple addition.
# Because both tensors are on the GPU, this calculation happens incredibly fast,
# which is crucial when you have millions of numbers to add in real-world AI problems!
print(a + b)