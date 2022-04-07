import numpy as np
from sklearn import datasets

source = datasets.load_iris()
data = source.data
target = source.target
design = np.insert(data, 0, 1., 1)

print(design)
