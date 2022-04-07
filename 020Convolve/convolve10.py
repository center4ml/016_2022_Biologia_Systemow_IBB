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

pool1 = torch.nn.MaxPool2d(2)
DATA = pool1(DATA)
print(DATA.shape)

conv2 = torch.nn.Conv2d(8, 16, 5)
DATA = conv2(DATA)
print(DATA.shape)

relu2 = torch.nn.ReLU()
DATA = relu2(DATA)
print(DATA.shape)

pool2 = torch.nn.MaxPool2d(2)
DATA = pool2(DATA)
print(DATA.shape)

flat = torch.nn.Flatten()
DATA = flat(DATA)
print(DATA.shape)
