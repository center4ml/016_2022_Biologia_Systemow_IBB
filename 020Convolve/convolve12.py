import torch
import torchvision as tv
import torchsummary

source = tv.datasets.MNIST("../MNIST", train = False, download = True)
DATA = source.data.unsqueeze(1).float() / 255.
print(DATA.shape)

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 8, 5),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(8, 16, 5),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Flatten(),
    torch.nn.Linear(256, 10))

torchsummary.summary(model, input_size = (1, 28, 28), device = 'cpu')
