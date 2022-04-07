import torch
import torchvision as tv

source = tv.datasets.MNIST("../MNIST", train = False, download = True)
DATA = source.data.float() / 255.
print(DATA.shape)
