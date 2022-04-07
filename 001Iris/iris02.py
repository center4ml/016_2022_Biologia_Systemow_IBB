from sklearn import datasets

source = datasets.load_iris()

print(source.DESCR)
print()

print(source.data)
print(source.target)
print()
