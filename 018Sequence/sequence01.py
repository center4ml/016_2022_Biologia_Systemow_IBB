import torch
import torchvision as tv

samples0, samples1 = 60000, 10000
features = 784
classes = 10

source0 = tv.datasets.MNIST("../MNIST", train = True, download = True)
source1 = tv.datasets.MNIST("../MNIST", train = False, download = True)
DATA0 = source0.data.flatten(1).float().cuda()
DATA1 = source1.data.flatten(1).float().cuda()
TARGET0 = source0.targets.cuda()
TARGET1 = source1.targets.cuda()

size1 = 128
BIAS1 = torch.zeros(size1, requires_grad = True, device = "cuda")
WEIGHT1 = torch.zeros(features, size1, requires_grad = True, device = "cuda")
BIAS = torch.zeros(classes, requires_grad = True, device = "cuda")
WEIGHT = torch.zeros(size1, classes, requires_grad = True, device = "cuda")

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
    LOSS0 = torch.tensor(0., device = "cuda")
    ACCURACY0 = torch.tensor(0., device = "cuda")
    COUNT0 = torch.tensor(0, device = "cuda")
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
    LOSS1 = torch.tensor(0., device = "cuda")
    ACCURACY1 = torch.tensor(0., device = "cuda")
    COUNT1 = torch.tensor(0, device = "cuda")
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
    print("%4i %12.4f %12.4f %12.4f %12.4f" % \
          (epoch, LOSS0, ACCURACY0, LOSS1, ACCURACY1), flush = True)
