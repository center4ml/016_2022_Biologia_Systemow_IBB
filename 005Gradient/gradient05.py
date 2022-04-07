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

PARAM = torch.zeros(1 + features, classes, requires_grad = True)

for epoch in range(1000):
    ACTIVATION = DESIGN @ PARAM
    EXP = ACTIVATION.exp()
    SUM = EXP.sum(1, keepdim = True)
    ACTIVITY = EXP / SUM
    LOG = ACTIVITY.log()
    ENTROPY = -torch.sum(ONEHOT * LOG, 1)
    LOSS = ENTROPY.sum()
    LOSS.backward()
    GRAD = PARAM.grad
    with torch.no_grad():
        PARAM -= 0.001 * GRAD
    PARAM.grad.zero_()
    print(epoch, LOSS.item())