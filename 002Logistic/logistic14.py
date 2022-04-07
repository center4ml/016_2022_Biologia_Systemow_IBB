import numpy as np
from sklearn import datasets

source = datasets.load_iris()
data = source.data
target = source.target
design = np.insert(data, 0, 1., 1)
onehot = np.equal(np.arange(3), target[:, None])

param = np.zeros((5, 3))

for epoch in range(1000):
    activation = design @ param
    exp = np.exp(activation)
    sum = exp.sum(1, keepdims = True)
    activity = exp / sum
    log = np.log(activity)
    entropy = -np.sum(onehot * log, 1)
    loss = entropy.sum()
    grad = design.T @ (activity - onehot)
    param -= 0.001 * grad
    print(epoch, loss)
