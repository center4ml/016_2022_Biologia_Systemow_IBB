from sklearn import datasets
import torch

features = 4
classes = 3

source = datasets.load_iris()
data = source.data
target = source.target

DATA = torch.tensor(data, dtype = torch.float32)
TARGET = torch.tensor(target, dtype = torch.int64)
DESIGN = torch.nn.functional.pad(DATA, (1, 0), 'constant', 1.)
ONEHOT = torch.nn.functional.one_hot(TARGET)

print(ONEHOT)
