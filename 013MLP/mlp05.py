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

size1 = 128
BIAS1 = torch.zeros(size1, requires_grad = True)
WEIGHT1 = torch.zeros(features, size1, requires_grad = True)
BIAS = torch.zeros(classes, requires_grad = True)
WEIGHT = torch.zeros(size1, classes, requires_grad = True)

torch.nn.init.xavier_normal_(WEIGHT1)

def model(DATA):
    ACTIVATION1 = BIAS1 + DATA @ WEIGHT1
    ACTIVITY1 = torch.sigmoid(ACTIVATION1)
    ACTIVATION = BIAS + ACTIVITY1 @ WEIGHT
    return ACTIVATION

batch = 100
optimizer = torch.optim.SGD([BIAS, WEIGHT, BIAS1, WEIGHT1], lr = 0.1)
loss = torch.nn.CrossEntropyLoss()
for epoch in range(100):
    LOSS0 = torch.tensor(0.)
    ACCURACY0 = torch.tensor(0.)
    COUNT0 = torch.tensor(0)
    for index in range(0, samples0, batch):
        DATA = DATA0[index: index + batch]
        TARGET = TARGET0[index: index + batch]
        ACTIVATION = model(DATA)
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
        ACTIVATION = model(DATA)
        LOSS1 += loss(ACTIVATION, TARGET) * TARGET.size(0)
        LABEL = ACTIVATION.argmax(1)
        ACCURACY1 += torch.eq(LABEL, TARGET).float().sum()
        COUNT1 += TARGET.size(0)
    LOSS1 /= COUNT1
    ACCURACY1 /= COUNT1
    print("%4i %12.3f %12.3f %12.3f %12.3f" % \
          (epoch, LOSS0, ACCURACY0, LOSS1, ACCURACY1), flush = True)
