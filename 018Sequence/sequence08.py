import torch
import torchvision as tv
import torchsummary

samples0, samples1 = 60000, 10000

source0 = tv.datasets.MNIST("../MNIST", train = True, download = True)
source1 = tv.datasets.MNIST("../MNIST", train = False, download = True)
DATA0 = source0.data.flatten(1).float().cuda()
DATA1 = source1.data.flatten(1).float().cuda()
TARGET0 = source0.targets.cuda()
TARGET1 = source1.targets.cuda()

model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.Sigmoid(),
    torch.nn.Linear(256, 64),
    torch.nn.Sigmoid(),
    torch.nn.Linear(64, 10)).cuda()

torchsummary.summary(model, input_size = (784, ))

batch = 100
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
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

#params: 200960 + 16448 + 650 = 218058
