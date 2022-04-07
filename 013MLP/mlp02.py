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

DATA0, DATA1 = DATA[: samples0], DATA[samples0:]
TARGET0, TARGET1 = TARGET[: samples0], TARGET[samples0:]

BIAS = torch.zeros(classes, requires_grad = True)
WEIGHT = torch.zeros(features, classes, requires_grad = True)

batch = 100
optimizer = torch.optim.SGD([BIAS, WEIGHT], lr = 0.1)
loss = torch.nn.CrossEntropyLoss()
for epoch in range(100):
    LOSS0 = torch.tensor(0.)
    ACCURACY0 = torch.tensor(0.)
    COUNT0 = torch.tensor(0)
    for index in range(0, samples0, batch):
        DATA = DATA0[index: index + batch]
        TARGET = TARGET0[index: index + batch]
        ACTIVATION = BIAS + DATA @ WEIGHT
        LOSS = loss(ACTIVATION, TARGET)
        LOSS.backward()
        LOSS0 += LOSS * TARGET.size(0)
        LABEL = ACTIVATION.argmax(1)
        ACCURACY0 += torch.eq(LABEL, TARGET).float().sum()
        COUNT0 += TARGET.size(0)
        optimizer.step()
        optimizer.zero_grad()
    LOSS0 /= COUNT0
    ACCURACY0 /= COUNT0
    LOSS1 = torch.tensor(0.)
    ACCURACY1 = torch.tensor(0.)
    COUNT1 = torch.tensor(0)
    for index in range(0, samples1, batch):
        DATA = DATA1[index: index + batch]
        TARGET = TARGET1[index: index + batch]
        ACTIVATION = BIAS + DATA @ WEIGHT
        LOSS1 += loss(ACTIVATION, TARGET) * TARGET.size(0)
        LABEL = ACTIVATION.argmax(1)
        ACCURACY1 += torch.eq(LABEL, TARGET).float().sum()
        COUNT1 += TARGET.size(0)
    LOSS1 /= COUNT1
    ACCURACY1 /= COUNT1
    print("%4i %12.3f %12.3f %12.3f %12.3f" % \
          (epoch, LOSS0, ACCURACY0, LOSS1, ACCURACY1), flush = True)
