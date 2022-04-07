from sklearn import datasets
import torch

samples0, samples1 = 1500, 297
features = 64
classes = 10

source = datasets.load_digits()
data = source.data
target = source.target

DATA = torch.tensor(data, dtype = torch.float32)
TARGET = torch.tensor(target, dtype = torch.int64)
DESIGN = torch.nn.functional.pad(DATA, (1, 0), 'constant', 1.)

TARGET0, TARGET1 = TARGET[: samples0], TARGET[samples0:]
DESIGN0, DESIGN1 = DESIGN[: samples0], DESIGN[samples0:]

PARAM = torch.zeros(1 + features, classes, requires_grad = True)

batch = 100
optimizer = torch.optim.SGD([PARAM], lr = 0.0001)
loss = torch.nn.CrossEntropyLoss(reduction = "sum")
for epoch in range(1000):
    for index in range(0, samples0, batch):
        DESIGN = DESIGN0[index: index + batch]
        TARGET = TARGET0[index: index + batch]
        ACTIVATION = DESIGN @ PARAM
        LOSS = loss(ACTIVATION, TARGET)
        LOSS.backward()
    optimizer.step()
    optimizer.zero_grad()
    LOSS1 = torch.tensor(0.)
    ACCURACY1 = torch.tensor(0.)
    COUNT1 = torch.tensor(0)
    for index in range(0, samples1, batch):
        DESIGN = DESIGN1[index: index + batch]
        TARGET = TARGET1[index: index + batch]
        ACTIVATION = DESIGN @ PARAM
        LOSS1 += loss(ACTIVATION, TARGET)
        LABEL = ACTIVATION.argmax(1)
        ACCURACY1 += torch.eq(LABEL, TARGET).float().sum()
        COUNT1 += TARGET.size(0)
    ACCURACY1 /= COUNT1
    print("%4i %12.3f %12.3f" % (epoch, LOSS1, ACCURACY1), flush = True)

ACTIVATION0 = DESIGN0 @ PARAM
LOSS0 = loss(ACTIVATION0, TARGET0)
LABEL0 = ACTIVATION0.argmax(1)
ACCURACY0 = torch.eq(LABEL0, TARGET0).float().mean()
print(LOSS0.item(), ACCURACY0.item())

ACTIVATION1 = DESIGN1 @ PARAM
LOSS1 = loss(ACTIVATION1, TARGET1)
LABEL1 = ACTIVATION1.argmax(1)
ACCURACY1 = torch.eq(LABEL1, TARGET1).float().mean()
print(LOSS1.item(), ACCURACY1.item())
