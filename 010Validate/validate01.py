from sklearn import datasets
import torch

features = 64
classes = 10

source = datasets.load_digits()
data = source.data
target = source.target

DATA = torch.tensor(data, dtype = torch.float32)
TARGET = torch.tensor(target, dtype = torch.int64)
DESIGN = torch.nn.functional.pad(DATA, (1, 0), 'constant', 1.)

PARAM = torch.zeros(1 + features, classes, requires_grad = True)

optimizer = torch.optim.SGD([PARAM], lr = 0.0001)
loss = torch.nn.CrossEntropyLoss(reduction = "sum")
for epoch in range(1000):
    ACTIVATION = DESIGN @ PARAM
    LOSS = loss(ACTIVATION, TARGET)
    LOSS.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(epoch, LOSS.item(), flush = True)

ACTIVATION = DESIGN @ PARAM
LOSS = loss(ACTIVATION, TARGET)
LABEL = ACTIVATION.argmax(1)
ACCURACY = torch.eq(LABEL, TARGET).float().mean()
print(LOSS.item(), ACCURACY.item())
