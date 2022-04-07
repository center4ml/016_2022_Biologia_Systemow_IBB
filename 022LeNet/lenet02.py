import torch
import torchvision as tv
import torchsummary

samples0, samples1 = 60000, 10000

source0 = tv.datasets.MNIST("../MNIST", train = True, download = True)
source1 = tv.datasets.MNIST("../MNIST", train = False, download = True)
DATA0 = source0.data.unsqueeze(1).float().cuda() / 255.
DATA1 = source1.data.unsqueeze(1).float().cuda() / 255.
TARGET0 = source0.targets.cuda()
TARGET1 = source1.targets.cuda()

model = torch.nn.Sequential(   #(1, 28, 28)
    torch.nn.Conv2d(1, 8, 5),  #(8, 24, 24)
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),     #(8, 12, 12)
    torch.nn.Conv2d(8, 16, 5), #(16, 8, 8)
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),     #(16, 4, 4)
    torch.nn.Flatten(),        #(256)
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10)).cuda()

torchsummary.summary(model, input_size = (1, 28, 28))

batch = 100
optimizer = torch.optim.Adam(model.parameters())
loss = torch.nn.CrossEntropyLoss()
for epoch in range(1000):
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

#params: 45226
#epoch: 123
#accuracy: 0.9926
