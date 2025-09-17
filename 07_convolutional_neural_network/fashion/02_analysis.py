import torch
from torch import nn

import torchvision.datasets as datasets

mnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=None)
print(mnist_train)

print(mnist_train[5][0])
