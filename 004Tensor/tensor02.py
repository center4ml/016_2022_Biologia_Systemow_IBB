from sklearn import datasets
import torch

features = 4
classes = 3

source = datasets.load_iris()
data = source.data
target = source.target

DATA = torch.tensor(data)
TARGET = torch.tensor(target)

print(DATA)
print(TARGET)
