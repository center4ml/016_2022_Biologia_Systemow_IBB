import numpy as np
from sklearn import datasets

source = datasets.load_iris()
data = source.data
target = source.target
design = np.insert(data, 0, 1., 1)

param = np.zeros((5, 3))

activation = design @ param
exp = np.exp(activation)
sum = exp.sum(1, keepdims = True)
activity = exp / sum
log = np.log(activity)

print(log)
