from matplotlib import pyplot as plt
from sklearn import datasets

source = datasets.load_digits()

print(source.DESCR)
print()

print(source.data.shape)
print(source.target)
print()

plt.imshow(source.data[0].reshape(8, 8))
print(source.target[0])
print()

plt.show()
