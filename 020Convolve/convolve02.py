import torch
import torchvision as tv

source = tv.datasets.MNIST("../MNIST", train = False, download = True)
DATA = source.data.unsqueeze(1).float() / 255.
print(DATA.shape)
