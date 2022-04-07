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
ONEHOT = torch.nn.functional.one_hot(TARGET)

PARAM = torch.zeros(1 + features, classes, requires_grad = True)

optimizer = torch.optim.SGD([PARAM], lr = 0.0001)
for epoch in range(1000):
    ACTIVATION = DESIGN @ PARAM
    LOG = torch.nn.functional.log_softmax(ACTIVATION, 1)
    ENTROPY = -torch.sum(ONEHOT * LOG, 1)
    LOSS = ENTROPY.sum()
    LOSS.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(epoch, LOSS.item(), flush = True)

ACTIVATION = DESIGN @ PARAM
LABEL = ACTIVATION.argmax(1)
ACCURACY = torch.eq(LABEL, TARGET).float().mean()

print(ACCURACY.item())
