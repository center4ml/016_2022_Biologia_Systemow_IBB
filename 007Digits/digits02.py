from sklearn import datasets

source = datasets.load_digits()

print(source.DESCR)
print()

print(source.data.shape)
print(source.target)
print()
