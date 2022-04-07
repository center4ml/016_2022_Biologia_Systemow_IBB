import torch
import torchvision as tv

source = tv.datasets.MNIST("../MNIST", train = False, download = True)
DATA = source.data.unsqueeze(1).float() / 255.
print(DATA.shape)

conv1 = torch.nn.Conv2d(1, 8, 5)
DATA = conv1(DATA)
print(DATA.shape)

relu1 = torch.nn.ReLU()
DATA = relu1(DATA)
print(DATA.shape)

